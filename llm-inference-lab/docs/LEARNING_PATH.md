# From Arithmetic to Tokens per Second

The purpose of this repository is to understand what makes LLM inference fast
or slow. A tokens-per-second number is useful only when we can explain where
the time and memory went.

## 1. A small neural-network calculation

A neuron begins with a dot product:

$$
y=\sum_{i=1}^{K}x_iw_i+b.
$$

Each input-weight pair requires a multiplication and contributes to a sum. A
dense matrix-vector operation with a matrix of shape $M\times K$ requires
approximately

$$
2MK\ \text{FLOPs}.
$$

Run:

```bash
llm-lab learn --stage arithmetic
```

## 2. Memory tracking

Arithmetic cannot begin until inputs and weights reach the processor. For
$y=Wx$, the ideal data traffic is

$$
B=MKs+Ks+Ms,
$$

where $s$ is bytes per value. Arithmetic intensity is

$$
I=\frac{2MK}{B}.
$$

The roofline lower bound is

$$
t\geq\max\left(\frac{F}{C},\frac{B}{\beta}\right),
$$

where $F$ is FLOPs, $C$ is compute throughput, $B$ is bytes moved, and $\beta$
is memory bandwidth.

Run:

```bash
llm-lab learn --stage memory
```

## 3. Quantization and batching

Quantization changes the bytes per stored weight:

$$
B_W=MK\frac{b}{8}.
$$

Batching lets several inputs reuse the same weight matrix. The weight traffic
per sequence ideally falls as

$$
\frac{B_W}{N},
$$

where $N$ is batch size. Real speedup depends on kernel support, dequantization,
cache behavior, and whether the workload becomes compute-bound.

Run:

```bash
llm-lab learn --stage scaling
```

## 4. Transformer layer and KV cache

One decode token performs projections, attention, an MLP, and vocabulary
scoring. Approximate attention work per layer is

$$
F_{\text{attention}}\approx4TD,
$$

while grouped-query KV-cache storage is

$$
B_{KV}=2LTH_{KV}d_hs.
$$

Longer context therefore increases KV storage and KV reads linearly during
decode. The projection and MLP weights remain a large fixed cost per token.

Run:

```bash
llm-lab learn --stage transformer
```

## Final connection

The learning track builds the explanation:

```text
calculation -> FLOPs -> bytes -> hardware limits -> transformer operations
            -> measured prefill/decode time -> tokens per second
```

After running these examples, use `llm-lab benchmark`, `llm-lab profile`, and
`llm-lab roofline` to compare the estimates against real execution.

