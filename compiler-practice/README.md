# Compiler Practice

**The additional layer to understand is the compiler and runtime.**

This small working compiler builds a computation graph, schedules its operations
into kernels, generates C, compiles that C into CPU machine code, and runs it.
You can inspect every stage and compare execution times.

Add this `compiler-practice` folder at the root of your `deep-learning-own`
repository, beside `benchmark-practice` and `llm-inference-lab`.

This is a teaching implementation of one dense layer with a bias and ReLU. It
uses the same broad stages described in [tinygrad's execution architecture](https://docs.tinygrad.org/developer/developer/).
It does not use tinygrad, PyTorch, pretrained weights, or a GPU. Its graph and
one fusion rule are deliberately limited to this equation; it is not a general
tensor compiler or a tinygrad performance reproduction.

## Run it on your Mac

In Terminal, enter the folder you just added. Adjust the first path if your repo
has a different name or location:

```bash
cd ~/Documents/GitHub/deep-learning-own/compiler-practice
python3 compiler_lab.py --width 32 --inspect-only
```

This first command needs only Python 3.9 or newer. It shows the graph, two kernel
schedules, arithmetic counts, and modeled bytes. It does not run the calculation.
An existing virtual environment is fine; there are no pip packages to install.

Then compile and benchmark:

```bash
python3 compiler_lab.py --width 32
```

Execution also needs a C compiler. If the command reports that one is missing on
macOS, install Apple's command-line developer tools:

```bash
xcode-select --install
```

Finish that installer, then rerun the benchmark. On Linux, install GCC or Clang
through your system package manager. Windows execution is not implemented.

## One equation, five stages

$$
Y=\operatorname{ReLU}(XW^{\mathsf T}+b),\qquad
X\in\mathbb{R}^{B\times D},\quad W\in\mathbb{R}^{D\times D},\quad
b\in\mathbb{R}^{D},\quad Y\in\mathbb{R}^{B\times D}.
$$

ReLU means replacing negative values with zero. The same bias vector is added
to every input in the batch. All stored numbers are FP32 (4 bytes per value).

| Stage | What it does | Where to look |
|---|---|---|
| Frontend / graph | Records inputs and operations without computing them | `graph.py`: `build_graph()` |
| Scheduler | Chooses three separate kernels or one fused kernel | `graph.py`: `schedule()` |
| Code generation | Converts each scheduled kernel into C loops | `codegen.py`: `emit_c()` |
| Machine-code compilation | Asks Clang/GCC to turn C into a loadable binary | `runtime.py`: `compile_source()` |
| Runtime | Allocates buffers, loads the binary, and calls its functions | `runtime.py`: `Executable` |

`compiler_lab.py` connects the stages and measures them. Start reading `graph.py`,
then open a generated `.c` file from a completed run.

## What fusion changes

The separate schedule calculates and stores:

$$
Z=XW^{\mathsf T},\qquad A=Z+b,\qquad Y=\max(A,0).
$$

The fused kernel finishes each output's dot product, adds its bias, applies ReLU,
and writes directly to Y. Z and A do not become full intermediate tensors.
The dot-product accumulator is a local variable; the C compiler controls its
register allocation and any spills.

For width 32, batch 1:

| Quantity | Separate | Fused |
|---|---:|---:|
| Weight + bias values (untrained) | 1,056 | 1,056 |
| Approximate matmul FLOPs | 2,048 | 2,048 |
| Bias additions | 32 | 32 |
| ReLU comparisons | 32 | 32 |
| Kernel calls per evaluation | 3 | 1 |
| Full temporary tensor bytes | 256 | 0 |
| Logical tensor storage bytes | 4,736 | 4,480 |
| Ideal tensor-boundary bytes | 4,992 | 4,480 |

Fewer calls and intermediate tensors can save time. There is no guaranteed
speedup: matrix-loop efficiency, compiler choices, caching, dispatch, and noise
also affect the result. This code uses simple loops, not optimized BLAS.

## Understand the byte counts

Logical storage counts each materialized tensor once. Separate execution keeps
X, W, b, Z, A, and Y; fused execution keeps X, W, b, and Y:

$$
S_{\rm separate}=4(D^2+D+4BD),\qquad S_{\rm fused}=4(D^2+D+2BD).
$$

The tensor-boundary model counts every kernel input once and every output once:

$$
Q_{\rm separate}=4(D^2+D+6BD),\qquad Q_{\rm fused}=4(D^2+D+2BD).
$$

**These Q values are idealized bookkeeping, not measured memory bandwidth.**
The C loops can load values repeatedly; CPU caches can serve repeated accesses.
Neither formula counts register spills, cache-line effects, compiler workspace,
or total process memory. Because the comparison retains both executables and
their buffers, per-schedule storage also does not equal combined process RAM.

## Experiments, one change at a time

Inspect C without compiling:

```bash
python3 compiler_lab.py --width 32 --inspect-only --show-code
```

Increase width with batch fixed:

```bash
python3 compiler_lab.py --width 32
python3 compiler_lab.py --width 128
python3 compiler_lab.py --width 512
```

At small sizes, the Python-to-C calls may dominate. At larger sizes, the dot
products account for more time. These are hypotheses to investigate, not
bottleneck diagnoses from a single timing result.

Increase batch with width fixed:

```bash
python3 compiler_lab.py --width 128 --batch 1
python3 compiler_lab.py --width 128 --batch 8
```

The weights and bias stay identical for a fixed seed; input/output sizes and
arithmetic increase. This basic implementation loops over batch items and does
not implement an optimized matrix-multiplication tiling strategy.

Save assembly as well as C and machine-code binaries:

```bash
python3 compiler_lab.py --width 32 --emit-assembly
```

Optional controls:

```bash
python3 compiler_lab.py --width 128 --mode fused
python3 compiler_lab.py --width 128 --iterations 1000 --repeats 10
python3 compiler_lab.py --help
```

## What the timer includes

Compilation is measured separately. Steady execution excludes graph creation,
compilation, buffer allocation, correctness checks, warmup, and saving.

Each sample runs the whole layer repeatedly. The reported number is the median
of those per-evaluation averages. The timer **includes Python dispatch, ctypes
calls, and synchronous single-thread CPU kernel execution**. It is not a
GPU-kernel duration, a GPU-launch-overhead measurement, or LLM tokens/second.

The two schedules share identical input values. Sample order alternates after a
seeded initial shuffle, and buffers are reused. This is a warm-cache experiment;
repeat the command to assess variability. More repeated samples do not remove
systematic measurement bias.

Output checks compare up to 64 positions against an independent Python
double-precision calculation with FP32 error tolerance. These checks run before
timing and abort the benchmark on failure.

## Files saved from each run

A unique directory under `results/` contains:

- `graph_and_schedule.json`: graph, selected kernels, and modeled costs.
- `separate.c` and/or `fused.c`: the exact generated source.
- `.so` on Linux or `.dylib` on macOS: locally compiled machine code.
- `.s`: assembly, when `--emit-assembly` is requested.
- `summary.csv`: timings, arithmetic counts, and modeled bytes.
- `details.json`: raw sample times, sample order, compiler command/version,
  Python and platform information.

An execution failure may leave the generated sources in its run directory for
inspection. The included `.gitignore` excludes generated results and binaries.
To share an experiment, copy selected CSV/JSON/C files to a named tracked folder;
compiled binaries are specific to the local platform.

## Check and commit

Run from `compiler-practice`:

```bash
python3 -m unittest -v test_compiler_lab.py
```

Execution tests require a C compiler; they report a skip if one is unavailable.
The suite checks known batched numerical answers, fusion's arithmetic/storage
counts, input validation, corruption detection, and saved results.

Validation performed for this package: all eight tests passed on Linux x86-64
with Python 3.12.14 and a real C compiler. A width-128, batch-8 run also compiled,
generated assembly, passed its output checks, and saved measurements for both
schedules. All Python sources pass a Python 3.9 syntax check. macOS execution
is implemented but was not available for testing here.

Then, from your repository root:

```bash
git add compiler-practice
git commit -m "Add compiler and runtime lab with separate and fused CPU kernels"
```

## Connection to the Qwen investigation

Your earlier benchmark changed the amount of arithmetic by changing width.
This experiment holds the equation fixed and changes how it is executed.
That distinction helps explain why two systems can run the same model at
different speeds.

This CPU experiment does not reproduce the reported GH200/Q8 performance gap.
That would require quantized kernels, representative model shapes, GPU timing,
and a trace of the actual model execution. Here you can first see, change, and
test the compiler/runtime decisions on a small calculation.
