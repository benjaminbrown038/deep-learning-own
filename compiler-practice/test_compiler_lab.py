"""Tests for schedule costs, generated kernels, and the complete CLI workflow."""

import ctypes
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from codegen import emit_c
from graph import build_graph, cost_model, schedule
from runtime import Executable, compile_source, make_inputs, validate_output


ROOT = Path(__file__).resolve().parent


class GraphTests(unittest.TestCase):
    def test_fusion_preserves_arithmetic_and_reduces_materialization(self):
        graph = build_graph(32, 1)
        separate = cost_model(graph, schedule(graph, "separate"))
        fused = cost_model(graph, schedule(graph, "fused"))
        self.assertEqual(separate["matmul_flops_approx"], 2048)
        for key in ("matmul_flops_approx", "bias_additions", "relu_comparisons",
                    "parameter_values_untrained"):
            self.assertEqual(separate[key], fused[key])
        self.assertEqual(separate["temporary_tensor_bytes"], 256)
        self.assertEqual(fused["temporary_tensor_bytes"], 0)
        self.assertEqual(separate["logical_tensor_storage_bytes"], 4736)
        self.assertEqual(fused["logical_tensor_storage_bytes"], 4480)
        self.assertEqual(separate["ideal_tensor_boundary_bytes"], 4992)
        self.assertEqual(fused["ideal_tensor_boundary_bytes"], 4480)
        self.assertEqual(separate["kernel_calls_per_evaluation"], 3)
        self.assertEqual(fused["kernel_calls_per_evaluation"], 1)

    def test_batch_reuses_parameters(self):
        a = build_graph(16, 1)
        b = build_graph(16, 4)
        ca, cb = (cost_model(g, schedule(g, "fused")) for g in (a, b))
        self.assertEqual(ca["parameter_values_untrained"], cb["parameter_values_untrained"])
        self.assertEqual(cb["matmul_flops_approx"], 4 * ca["matmul_flops_approx"])
        ia, ib = make_inputs(16, 1, 3), make_inputs(16, 4, 3)
        self.assertEqual(list(ia["W"]), list(ib["W"]))
        self.assertEqual(list(ia["bias"]), list(ib["bias"]))

    def test_unsupported_graph_is_rejected(self):
        graph = build_graph(3, 2)
        changed = graph[:-1] + (replace(graph[-1], inputs=("Z",)),)
        with self.assertRaises(ValueError):
            schedule(changed, "fused")

    def test_inspection_needs_no_compiler(self):
        completed = subprocess.run([sys.executable, str(ROOT / "compiler_lab.py"),
                                    "--width", "32", "--inspect-only", "--show-code",
                                    "--compiler", "nonexistent-compiler-123"],
                                   capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("dense_relu", completed.stdout)
        self.assertIn("~2,048", completed.stdout)

    def test_invalid_arguments_and_missing_compiler(self):
        for arguments in (["--width", "0"], ["--batch", "-1"],
                          ["--iterations", "0"], ["--repeats", "0"],
                          ["--compiler", "nonexistent-compiler-123"]):
            completed = subprocess.run([sys.executable, str(ROOT / "compiler_lab.py"),
                                        *arguments], capture_output=True, text=True)
            self.assertNotEqual(completed.returncode, 0)


@unittest.skipUnless(shutil.which("cc") and platform.system() in ("Darwin", "Linux"),
                     "C execution requires cc on macOS or Linux")
class ExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.TemporaryDirectory()
        cls.graph = build_graph(3, 2)
        cls.binaries = {}
        for mode in ("separate", "fused"):
            kernels = schedule(cls.graph, mode)
            cls.binaries[mode], _ = compile_source(emit_c(cls.graph, kernels),
                                                    Path(cls.workspace.name), mode,
                                                    shutil.which("cc"))

    @classmethod
    def tearDownClass(cls):
        cls.workspace.cleanup()

    def test_known_batched_answer_both_schedules(self):
        inputs = {"X": (ctypes.c_float * 6)(-1, 2, 3, 4, -5, 6),
                  "W": (ctypes.c_float * 9)(1, 2, -1, 0, -2, 3, -1, 0, 0.5),
                  "bias": (ctypes.c_float * 3)(0.5, -1, 2)}
        for mode in self.binaries:
            executable = Executable(self.graph, schedule(self.graph, mode),
                                    self.binaries[mode], inputs)
            executable.run()
            self.assertEqual(list(executable.buffers["Y"]), [0.5, 4, 4.5, 0, 27, 1])
            self.assertEqual(validate_output(executable, 3, 2, 0)["output_check"], "passed")
        self.assertEqual(list(inputs["X"]), [-1, 2, 3, 4, -5, 6])

    def test_validator_detects_corruption(self):
        executable = Executable(self.graph, schedule(self.graph, "fused"),
                                self.binaries["fused"], make_inputs(3, 2, 0))
        executable.run()
        executable.buffers["Y"][0] = float("nan")
        executable.run = lambda: None
        with self.assertRaises(RuntimeError):
            validate_output(executable, 3, 2, 0)

    def test_complete_cli_saves_sources_samples_and_unique_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            command = [sys.executable, str(ROOT / "compiler_lab.py"),
                       "--width", "8", "--batch", "2", "--warmup", "1",
                       "--iterations", "3", "--repeats", "2", "--emit-assembly",
                       "--output", directory]
            for _ in range(2):
                completed = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(completed.returncode, 0, completed.stderr)
            runs = list(Path(directory).iterdir())
            self.assertEqual(len(runs), 2)
            for run in runs:
                details = json.loads((run / "details.json").read_text())
                self.assertEqual(len(details["results"]), 2)
                self.assertTrue((run / "summary.csv").exists())
                self.assertTrue((run / "graph_and_schedule.json").exists())
                for mode in ("separate", "fused"):
                    self.assertTrue((run / f"{mode}.c").exists())
                    self.assertTrue((run / f"{mode}.s").exists())
                    self.assertEqual(len(details["sample_total_seconds"][mode]), 2)
                    self.assertTrue(all(s > 0 for s in details["sample_total_seconds"][mode]))


if __name__ == "__main__":
    unittest.main()
