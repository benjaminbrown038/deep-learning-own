# Benchmark Practice

One operation. One variable at a time. No pretrained model and no `llm-lab` command.

This kit isolates the core question: **how does the time to multiply a matrix change when its size or batch changes?**

## Start here on your Mac

Unzip `benchmark-practice.zip`. Open a terminal in the resulting `benchmark-practice` folder (the one containing `benchmark.py`). Keep it separate from `llm-inference-lab`.

If your active virtual environment already has PyTorch, you can use it. Otherwise:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Python 3.9 or newer is supported by these scripts; pip must also find a PyTorch build compatible with your Python and OS. A current supported Python is preferable for a new environment. If installation fails, save the actual error; changing the CLI command cannot fix a missing dependency.

Your first run:

```bash
python benchmark.py --width 32
```

CPU, FP32, and batch 1 are the defaults. No network access is needed after installing PyTorch.

## What each file does

| File | Purpose |
|---|---|
| `benchmark.py` | Runs one configuration, prints a small summary, saves results |
| `sweep.py` | Repeats the same benchmark while changing width or batch |
| `test_benchmark.py` | Checks arithmetic, input validation, execution and saved data |
| `requirements.txt` | Just PyTorch; no Transformers, pandas or plotting library |

Read the numbered sections inside `run_case()` in `benchmark.py`: create data, warm up, time it, check the answer.

## The mathematics

We calculate:

$$
Y=XW^{\mathsf T},\qquad X\in\mathbb{R}^{B\times D},\quad W\in\mathbb{R}^{D\times D},\quad Y\in\mathbb{R}^{B\times D}.
$$

Here $D$ is width and $B$ is batch size. There is no bias or activation in this test.

$$
\text{weight values}=D^2,\qquad \text{FLOPs per call}\approx2BD^2.
$$

The usual FLOP convention counts one multiply-add as two operations. The exact dot-product count without bias is $BD(2D-1)$; the benchmark uses the conventional approximation.

For $s$ bytes per value:

$$
\text{weight bytes}=D^2s,\qquad \text{logical tensor bytes}=(D^2+2BD)s.
$$

At width 32, batch 1, FP32, predict 1,024 weight values, about 2,048 FLOPs/call, and 4,096 weight bytes. X and Y add another 256 bytes.

These are randomly initialized weights, not trained parameters. They have the same shapes and arithmetic as a bias-free dense layer.

## Three practice exercises

### 1. Repeat a fixed test

```bash
python benchmark.py --width 512
```

Run that command three times. Each invocation saves a new directory.

- What stays constant: shapes, precision, thread count, input seed.
- What changes: measured execution time.
- Explain why a stopwatch result varies even when FLOPs do not.

### 2. Change only width

```bash
python sweep.py width --values 32 64 128 256 512 1024
```

Before running, predict: doubling width quadruples weights and approximately quadruples FLOPs. Does time also quadruple? Small operations may be dominated by overhead; cache behavior and hardware efficiency can change with shape.

### 3. Change only batch

```bash
python sweep.py batch --width 512 --values 1 2 4 8 16
```

Before running, predict: weights stay fixed while FLOPs and input/output storage grow with batch. Compare both time per call and examples per second. A slower batch can still complete more examples per second.

Sweeps shuffle execution order with a fixed seed to reduce a simple size-versus-order bias, and sort results by the changed variable when saving. Repeat the entire sweep to check consistency.

## Optional: precision or GPU

Hold width and batch fixed; compare these separate runs:

```bash
python benchmark.py --width 512 --dtype fp32
python benchmark.py --width 512 --dtype fp16
```

FP16 halves logical storage but does not halve the FLOP count. FP16 speed depends on backend support; CPU FP16 may be slower or unsupported on some systems. This is floating-point precision testing, not INT8/INT4 quantization.

On a supported Mac, compare the CPU and Apple GPU explicitly:

```bash
python benchmark.py --width 512 --device cpu
python benchmark.py --width 512 --device mps
```

On a machine with a CUDA-enabled PyTorch installation and NVIDIA GPU:

```bash
python benchmark.py --width 512 --device cuda
```

The scripts stop clearly if a requested accelerator is unavailable; they do not silently switch devices. No large models are loaded.

## Reading the measurements

| Output | Meaning |
|---|---|
| Weight values | Number of coefficients in W |
| Approximate FLOPs/call | Arithmetic count, not a hardware-counter reading |
| Logical tensor storage | Bytes for X, W and Y, not peak process/GPU memory |
| Median time/call | Median of the average per-call times in the repeated samples |
| Sample range | Variation among those per-call averages |
| Examples/s | Batch size divided by measured time per call; NOT LLM tokens/s |
| Output check | Up to eight output columns checked against a CPU float64 reference |

Default timing: 5 warm-up calls, then 5 samples of 50 matrix calls each. Each sample has a start/end synchronization on GPU. The scripts exclude tensor creation, device transfers, printing, validation and file saving from the timer. They reuse a preallocated output buffer.

The clock includes Python dispatch and backend overhead. Grouped calls measure amortized operation time, not isolated request latency or a GPU-only kernel time. For an exploratory single-call measurement, use `--iterations 1`; its synchronization overhead is more prominent. Increase `--iterations 200` to make very short samples less sensitive to timer resolution.

The same tensors are reused and can remain in caches. **Logical bytes are not measured DRAM traffic, and this benchmark cannot directly predict tokens/sec or diagnose a bandwidth bottleneck.** It also does not measure total memory capacity needed by a full language model.

CPU threads are fixed at 1 by default; adjust `--threads` only as a separate experiment. CUDA matmul TF32 is disabled in this kit. Keep the laptop plugged in and avoid unrelated heavy work during comparisons. Do not compare these CPU results to earlier GPU/Qwen tests as though the workloads were identical.

## Saved results and your notes

Each command writes a new timestamped subfolder of `results/`:

- `summary.csv`: one row per configuration, including device/software metadata and timing summaries.
- `details.json`: the same metadata plus every raw timing sample.

Old runs are preserved. `--output another_folder` chooses a different parent directory. If a sweep stops partway through, completed cases are saved in a directory marked `partial`.

Open the CSV in a spreadsheet application if desired. Record four short notes for every experiment:

1. I changed ...
2. I expected ...
3. I measured ...
4. One possible explanation is ...; to test it I would ...

No reference speed is promised: your hardware produces your measurements. No fabricated benchmark results are bundled.

## Check the files

```bash
python benchmark.py --help
python sweep.py --help
python -m unittest -v test_benchmark.py
```

Help and the math tests work without PyTorch. Execution tests are skipped if it is absent. A passing math test alone does not validate GPU behavior.

Validation for this kit: all nine tests passed on Linux, Python 3.12, PyTorch 2.5.1 CPU. FP32 and FP16 single-case commands and default width/batch sweeps also executed successfully. MPS and CUDA execution could not be tested here; use the same output sanity checks on your hardware. PyTorch can emit an optional NumPy-initialization warning if NumPy is absent; these scripts do not use NumPy.

## Further reading

This explicit teaching timer follows the warm-up, repeated-sample and synchronization principles described in [PyTorch benchmarking utilities](https://docs.pytorch.org/docs/stable/benchmark_utils.html). The operation uses [torch.mm](https://docs.pytorch.org/docs/stable/generated/torch.mm.html); Mac GPU completion uses [torch.mps.synchronize](https://docs.pytorch.org/docs/stable/generated/torch.mps.synchronize.html).
