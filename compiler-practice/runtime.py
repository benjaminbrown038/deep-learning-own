"""Compile C, allocate buffers, load machine code, and dispatch CPU kernels."""

import ctypes
import math
import platform
import random
import shutil
import subprocess
import time
from pathlib import Path


def find_compiler(requested=None):
    compiler = shutil.which(requested or "cc")
    if compiler is None:
        raise RuntimeError("C compiler not found. On macOS run: xcode-select --install. "
                           "On Linux install a C compiler such as gcc or clang.")
    return compiler


def compile_source(source: str, destination: Path, name: str,
                   compiler: str, emit_assembly: bool = False):
    system = platform.system()
    if system not in ("Darwin", "Linux"):
        raise RuntimeError("This teaching runtime supports macOS and Linux.")
    c_file = destination / f"{name}.c"
    c_file.write_text(source)
    binary = destination / (name + (".dylib" if system == "Darwin" else ".so"))
    common = [compiler, "-O3", "-std=c99", "-fPIC", "-ffp-contract=off"]
    command = common + ["-dynamiclib" if system == "Darwin" else "-shared",
                        str(c_file), "-o", str(binary)]
    start = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True)
    seconds = time.perf_counter() - start
    if completed.returncode:
        raise RuntimeError(f"C compilation failed:\n{completed.stderr.strip()}")
    if emit_assembly:
        assembly = subprocess.run(common + ["-S", str(c_file), "-o",
                                            str(destination / f"{name}.s")],
                                  capture_output=True, text=True)
        if assembly.returncode:
            raise RuntimeError(f"Assembly generation failed:\n{assembly.stderr.strip()}")
    version = subprocess.run([compiler, "--version"], capture_output=True, text=True)
    return binary, {"compiler": compiler,
                    "compiler_version": version.stdout.splitlines()[0]
                    if version.stdout else "unknown",
                    "compile_command": command, "compile_seconds": seconds,
                    "compiler_messages": completed.stderr.strip()}


def make_inputs(width: int, batch: int, seed: int):
    """W and bias are generated first so changing batch preserves parameters."""
    rng = random.Random(seed)
    scale = 1.0 / math.sqrt(width)
    weight = (ctypes.c_float * (width * width))()
    for i in range(len(weight)):
        weight[i] = rng.uniform(-1.0, 1.0) * scale
    bias = (ctypes.c_float * width)(*(rng.uniform(-0.25, 0.25) for _ in range(width)))
    inputs = (ctypes.c_float * (batch * width))()
    for i in range(len(inputs)):
        inputs[i] = rng.uniform(-1.0, 1.0)
    return {"X": inputs, "W": weight, "bias": bias}


class Executable:
    def __init__(self, graph, kernels, binary, inputs):
        self.library = ctypes.CDLL(str(binary.resolve()))
        self.buffers = dict(inputs)
        self.calls = []
        shapes = {node.name: node.shape for node in graph}
        pointer = ctypes.POINTER(ctypes.c_float)
        for kernel in kernels:
            elements = math.prod(shapes[kernel.output])
            self.buffers[kernel.output] = (ctypes.c_float * elements)()
            function = getattr(self.library, kernel.name)
            function.argtypes = [pointer] * (len(kernel.inputs) + 1)
            function.restype = None
            arguments = tuple(self.buffers[name]
                              for name in (*kernel.inputs, kernel.output))
            self.calls.append((function, arguments))

    def run(self):
        # Synchronous CPU calls: on return, the output is ready.
        for function, arguments in self.calls:
            function(*arguments)

    def sample(self, iterations):
        start = time.perf_counter()
        for _ in range(iterations):
            self.run()
        return time.perf_counter() - start


def validate_output(executable, width, batch, seed):
    """Independent Python double-precision reference at up to 64 positions."""
    executable.run()
    count = width * batch
    if count <= 64:
        indices = list(range(count))
    else:
        indices = [0, count - 1] + random.Random(seed).sample(range(1, count - 1), 62)
    buffers = executable.buffers
    maximum_error = 0.0
    for index in indices:
        b, row = divmod(index, width)
        products = [float(buffers["X"][b * width + col])
                    * float(buffers["W"][row * width + col]) for col in range(width)]
        expected = max(0.0, math.fsum(products) + float(buffers["bias"][row]))
        actual = float(buffers["Y"][index])
        error = abs(actual - expected)
        # Bound permits normal FP32 accumulation error, including cancellation.
        tolerance = 1e-5 + 2e-6 * math.fsum(abs(value) for value in products)
        if not math.isfinite(actual) or error > tolerance:
            raise RuntimeError(f"Output check failed at {index}: {actual} vs {expected}")
        maximum_error = max(maximum_error, error)
    return {"output_check": "passed", "checked_outputs": len(indices),
            "total_outputs": count, "max_abs_error": maximum_error}
