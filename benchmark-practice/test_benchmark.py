"""Run: python -m unittest -v test_benchmark.py"""
import argparse
import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark import add_settings, matrix_cost, new_run_directory, run_case, save_results


class MathAndInterfaceTests(unittest.TestCase):
    def test_32_by_32_fp32(self):
        result = matrix_cost(32, 1, 4)
        self.assertEqual(result["weight_values"], 1024)
        self.assertEqual(result["flops_per_call_approx"], 2048)
        self.assertEqual(result["weight_bytes"], 4096)
        self.assertEqual(result["logical_tensor_bytes"], 4352)

    def test_batch_reuses_weights(self):
        one, four = matrix_cost(32, 1, 4), matrix_cost(32, 4, 4)
        self.assertEqual(one["weight_values"], four["weight_values"])
        self.assertEqual(one["weight_bytes"], four["weight_bytes"])
        self.assertEqual(one["flops_per_call_approx"] * 4, four["flops_per_call_approx"])

    def test_fp16_halves_storage_not_work(self):
        a, b = matrix_cost(32, 1, 4), matrix_cost(32, 1, 2)
        self.assertEqual(a["logical_tensor_bytes"], 2 * b["logical_tensor_bytes"])
        self.assertEqual(a["flops_per_call_approx"], b["flops_per_call_approx"])

    def test_doubling_width_quadruples_weights(self):
        self.assertEqual(matrix_cost(64, 1, 4)["weight_values"], 4 * matrix_cost(32, 1, 4)["weight_values"])

    def test_reject_invalid_settings(self):
        parser = argparse.ArgumentParser()
        add_settings(parser)
        for flag in ("--width", "--batch", "--repeats", "--iterations", "--threads"):
            with self.subTest(flag=flag), patch("sys.stderr"), self.assertRaises(SystemExit):
                parser.parse_args([flag, "0"])
        with self.assertRaises(ValueError):
            matrix_cost(-1, 1, 4)

    def test_new_run_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = new_run_directory(tmp, "test")
            second = new_run_directory(tmp, "test")
            self.assertNotEqual(first, second)

    def test_help_without_model_imports(self):
        for script in ("benchmark.py", "sweep.py"):
            proc = subprocess.run([sys.executable, str(Path(__file__).parent / script), "--help"], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("--device", proc.stdout)


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch not installed")
class ExecutionTests(unittest.TestCase):
    def test_tiny_cpu_execution_and_saved_samples(self):
        parser = argparse.ArgumentParser()
        add_settings(parser)
        args = parser.parse_args(["--width", "32", "--batch", "2", "--repeats", "3", "--iterations", "2"])
        result = run_case(args)
        self.assertTrue(result["check_passed"])
        self.assertEqual(len(result["samples"]), 3)
        self.assertGreater(result["median_ms_per_call"], 0)
        self.assertEqual(result["flops_per_call_approx"], 4096)
        with tempfile.TemporaryDirectory() as tmp, patch("builtins.print"):
            folder = save_results([result], tmp, "test")
            with (folder / "summary.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            with (folder / "details.json").open() as handle:
                details = json.load(handle)
            self.assertEqual(len(rows), 1)
            self.assertEqual(len(details[0]["samples"]), 3)
            self.assertNotIn("samples", rows[0])

    def test_sweep_commands(self):
        for variable in ("width", "batch"):
            with self.subTest(variable=variable), tempfile.TemporaryDirectory() as tmp:
                cmd = [sys.executable, str(Path(__file__).parent / "sweep.py"), variable,
                       "--values", "2", "4", "--width", "8", "--warmup", "1",
                       "--repeats", "2", "--iterations", "2", "--output", tmp]
                proc = subprocess.run(cmd, capture_output=True, text=True)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                paths = list(Path(tmp).glob("*/summary.csv"))
                self.assertEqual(len(paths), 1)
                with paths[0].open() as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual([int(row[variable]) for row in rows], [2, 4])


if __name__ == "__main__":
    unittest.main()
