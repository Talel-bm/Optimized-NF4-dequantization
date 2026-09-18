# A Triton kernel for NF4 dequantization

This is a single Triton kernel that turns a 4-bit quantized weight matrix back into `fp16` or `bf16`. It produces **byte-for-byte identical output** to Unsloth's `fast_dequantize` while running **1.30x faster** (median of 30 runs) on a Tesla T4.

Written for the [Unsloth NF4 challenge](https://github.com/unslothai/unsloth): one Triton kernel, must run on a T4, no `torch.compile`, no hand-written CUDA, beat the baseline by at least 1.15x.

---

## What problem this solves

When you load a model in 4-bit with QLoRA, the weights sit in memory as NF4 — about 0.5 bytes per parameter instead of 2. But you can't multiply with 4-bit values directly. Every forward pass has to expand them back to `fp16` first, do the matmul, and throw the expanded copy away.

That expansion is pure overhead. It happens constantly during fine-tuning, it's memory-bandwidth bound, and it's on the critical path of every layer. Making it faster makes 4-bit training faster, with no change to the math and no loss of accuracy.

This kernel does that expansion in one GPU launch.

---

## The format you're decoding

NF4 with `compress_statistics=True` is **quantization applied twice**. That's the "double" in double dequantization, and it's the part that makes the kernel interesting.

**First level — the weights.** Split the tensor into blocks of 64. For each block, record `absmax` (the largest magnitude in it), divide every value by it to get the range into [-1, 1], then snap each to one of 16 predefined levels. Store the level as a 4-bit index. Two indices pack into one byte.

**Second level — the scales.** You now have one `fp32` absmax per 64 weights. For a 4096×14336 layer that's 917,504 floats, 3.5 MB of pure overhead. So those get quantized too: subtract their mean (the `offset`), take them in groups of 256, and quantize each group to 8 bits against its own scale.

So what bitsandbytes hands you is:

| tensor | dtype | count (for `n` weights) | what it is |
|---|---|---|---|
| `weight` | `uint8` | `n/2` | two 4-bit weight indices per byte |
| `quant_state.absmax` | `uint8` | `n/64` | 8-bit index for each block's scale |
| `state2.code` | `fp32` | 256 | lookup table for those 8-bit indices |
| `state2.absmax` | `fp32` | `n/16384` | scale for each group of 256 blocks |
| `quant_state.offset` | `fp32` | 1 | mean subtracted before quantizing scales |

And decoding runs the two levels in reverse:

```
absmax[b] = code2[absmax1[b]] * absmax2[b // 256] + offset     # rebuild the scale
W[i]      = NF4[nibble(i)] * absmax[i // 64]                   # rebuild the weight
```

### Two things that will bite you

**`state2.blocksize == 256` counts scales, not weights.** One `absmax2` entry covers 256 blocks × 64 weights = **16,384 weights**. If you index it as `absmax2[i // 256]` you're off by a factor of 64 and every scale you read is wrong.

**`absmax1` holds indices, not numbers.** It's `uint8`, and the values are positions in the 256-entry `code2` table. You must gather through the table. Treating them as numeric scales silently produces garbage.

**And the 16 NF4 levels are not evenly spaced.** They're `-1.0, -0.6962, -0.5251, -0.3949, …, 0.7230, 1.0` — clustered near zero, where neural network weights actually live. This spacing is what makes NF4 better than plain 4-bit integers at the same bit width. There's no formula for it. It's a table, and you have to look values up in it.

---

## How the kernel works

Everything below is in `_dequant_kernel_join`. It's one launch — the scales and the weights are both reconstructed inside it, and nothing intermediate is ever written to memory.

### The work assignment

Each program instance handles **16 blocks of 64 weights** — 1024 weights, from 512 packed bytes.

Picking 16 blocks as the unit (rather than, say, "4096 weights") is deliberate: a block of 64 is exactly the span that shares one scale. Aligning the tile to block boundaries means every scale is loaded once and used 64 times, with no straddling and no recomputation.

### Step 1: rebuild the scales, once per block

```python
blk    = pid * BLOCKS + tl.arange(0, BLOCKS)          # 16 block indices
c1     = tl.load(absmax1_ptr + blk, mask=bmask)       # 16 uint8 indices
a1     = tl.load(code2_ptr + c1, mask=bmask)          # 16 LUT lookups
a2     = tl.load(absmax2_ptr + (blk >> 8), mask=bmask)
absmax = _add_rn(_mul_rn(a1, a2), offset)             # 16 scales
```

Sixteen values, not 1024. The naive version of this kernel does the whole chain per weight, which is 64x the arithmetic and 64x the table lookups for results that are identical within each block. Here the scale is computed once and broadcast across its row.

(`blk >> 8` is `blk // 256` — the `absmax2` index. See the "two things that will bite you" note above.)

### Step 2: decode the 4-bit values without touching memory

The obvious way to map a 4-bit index to its NF4 value is `tl.load(nf4_table + q)`. On a T4 that's a bad idea: each lane wants a different table entry, so it compiles to a divergent gather, and you pay memory latency to read a 16-element array that every thread needs.

Instead the table is compiled into a branchless select tree over the 4 bits of the index:

```python
hi = q >= 8                      # which half of the table
b2, b1, b0 = (q & 4) != 0, (q & 2) != 0, (q & 1) != 0
# 4 levels of tl.where, all 16 constants inlined
```

Four levels of `tl.where` with the constants baked in. No loads, no divergence, no table in memory at all. The 16 NF4 values become immediates in the instruction stream.

### Step 3: read each byte once, then un-interleave

This is where the performance actually is, and it took three attempts to get right.

The packing is interleaved: weight `2k` is the **high** nibble of byte `k`, weight `2k+1` is the **low** nibble. So the byte you read and the weights you write are in different orders, and you have to reconcile that somewhere. Two natural approaches both fail:

**Tile over bytes** → each program owns a contiguous run of bytes, unpacks them, and writes weights at `out[2k]` and `out[2k+1]`. Both writes are **stride-2**. On a T4 every 32-byte memory sector ends up half-used, so the write path costs roughly double. Since the `fp16` output is 4x the size of the packed input, writes dominate — this version ran at **0.202x**, five times *slower* than the baseline.

**Tile over weights** → fix the writes by having each program own a contiguous run of weights, deriving the byte index as `row*32 + (col >> 1)`. Writes are now perfect. But each byte gets loaded **twice**, once per nibble, and the stride-½ pattern can't be vectorized.

**What the kernel does** → read bytes contiguously, split both nibbles in registers, then interleave the *results*:

```python
packed = tl.load(packed_ptr + row * 32 + bcol, mask=bmask2)   # (16, 32) contiguous
hi_q   = (packed >> 4) & 0x0F        # weights 0, 2, 4, ...
lo_q   = packed & 0x0F               # weights 1, 3, 5, ...

v_hi   = _nf4_lookup(hi_q) * scale   # (16, 32)
v_lo   = _nf4_lookup(lo_q) * scale   # (16, 32)

vals   = tl.reshape(tl.join(v_hi, v_lo), (BLOCKS, 64))        # (16, 64) in order
tl.store(out_ptr + row * 64 + col, vals.to(odt), mask=bmask2) # contiguous
```

`tl.join` stacks the two into `(16, 32, 2)` with `[hi, lo]` as the last axis. Reshaping to `(16, 64)` flattens that axis in place — which is exactly the interleave you need, because `[hi[0], lo[0], hi[1], lo[1], …]` *is* weight order. The un-interleaving costs nothing: it's a register shuffle, not memory traffic.

Result: 512 loads and 1024 stores per program, all contiguous, all vectorizable.

### Step 4: don't sync with the CPU

`offset` is passed as a **pointer** and read inside the kernel. Calling `.item()` on it looks harmless but forces a device-to-host copy, which blocks until the GPU finishes everything queued. The benchmark makes 9000 dequantization calls; doing that once per call serializes the CPU and GPU completely.

---

## Matching the reference bit-for-bit

The challenge harness compares your output against Unsloth's with `torch.testing.assert_close`. Getting *close* isn't the hard part. Getting **identical** is.

The first working version was off by 1 ULP on 28 weights out of 50 million. The cause was this line:

```python
absmax = a1 * a2 + offset
```

That's a multiply followed by an add, so LLVM fuses it into a single FMA instruction — which rounds **once**, at the end. Unsloth gets the same value in two separate steps:

```python
out_absmax = cdequantize_blockwise_fp32(...)   # CUDA kernel writes fp32 — rounds
out_absmax += offset                           # separate torch op — rounds again
```

Two roundings. The FMA version is *more accurate* — one rounding beats two — but it lands on a different `fp32` value about a quarter of the time. And a 1-ULP difference in a scale changes the `fp16` rounding of weights in that entire 64-weight block.

The fix is to force two roundings via inline PTX:

```python
@triton.jit
def _mul_rn(a, b):
    return tl.inline_asm_elementwise(
        asm="mul.rn.f32 $0, $1, $2;", constraints="=f,f,f",
        args=[a, b], dtype=tl.float32, is_pure=True, pack=1)
```

`.rn` is round-to-nearest-even, and an explicit PTX instruction can't be contracted into an FMA by the compiler. It runs once per block, so the cost is nothing.

Worth being clear: this is deliberately choosing the *less* accurate computation, because the goal is reproducing a reference implementation exactly, not maximizing precision.

After the fix: **0 differing elements** out of 16,777,216 per projection, in both `fp16` and `bf16`, `max_ulp = 0`.

---

## Results

30 consecutive runs of the challenge harness on a Colab T4:

| | unsloth | this kernel | speedup |
|---|---|---|---|
| **median** | 4.348 s | 3.315 s | **1.300x** |
| mean | 4.444 s | 3.418 s | 1.307x |
| range | 4.116 – 5.411 s | 3.276 – 5.248 s | 1.020x – 1.642x |
| p10 – p90 | | | 1.230x – 1.377x |

**29 of 30 runs beat the 1.15x target.** The one exception (run 18, 1.020x) is a machine hiccup, not a kernel effect — both implementations slowed down together, each about 1.6 s above its own median. Drop that run and the median is 1.306x with a 1.217x floor.

The kernel is also the *steadier* of the two. Excluding run 18, its runtime varies by 2.3% (3.276 – 3.504 s) against Unsloth's 5.5% (4.116 – 5.411 s). The 1.642x and 1.493x peaks are Unsloth having bad runs (5.411 s, 4.951 s) — this kernel was at 3.296 s and 3.317 s on those same runs, right at its median. So the speedup *range* looks noisy mostly because the baseline is noisy. The median is the honest number.

### Why it's faster

Measured in isolation the kernel sustains roughly **200 GB/s** effective bandwidth, counting packed bytes read plus `fp16` bytes written. A T4 tops out around 220–240 GB/s in practice (320 GB/s on paper), so this is close to the memory ceiling — there isn't much left to win without changing what gets moved.

Unsloth's throughput varies widely by shape (106–178 GB/s), which points at launch and dispatch overhead rather than bandwidth: it calls into bitsandbytes twice, once for the scales and once for the weights, and materializes the intermediate `absmax` array in between. Fusing both levels into one launch removes that round trip.

---

## Notebook contents

1. **Setup** — challenge helpers, `assert_same`, `MLP`, `test_dequantize`
2. **The kernel** — `_dequant_kernel_join` and its wrapper, plus a `_dequant_kernel_dup` fallback for Triton versions without `tl.join`
3. **Correctness** — bit-exactness check against Unsloth per projection, with ULP counts
4. **Benchmark** — 30 harness runs, median speedup

Run top to bottom on any T4 instance. The correctness cell should print `bit_exact=True` for all three projections before the benchmark means anything.

---

## Tuning

```python
_BLOCKS = 16        # 16 blocks x 64 weights = 1024 weights per program
_NUM_WARPS = 1
_NUM_STAGES = 2
```

Found by sweeping `BLOCKS ∈ {2,4,8,16,32,64} × warps ∈ {1,2,4} × stages ∈ {2,3,4,6}`, discarding any config that wasn't bit-exact, and ranking by total time across all three benchmark shapes (the harness sums them, so that's what matters).

Small tiles with a single warp win because this is a pure streaming kernel — many small programs keep more loads in flight than a few large ones, and 16 scales stay comfortably in registers.

The best config **shifts when the memory access pattern changes**. It moved between every version of this kernel. Re-sweep after any change to loads or stores, and re-sweep on non-T4 hardware.

---

## Requirements

`torch`, `triton`, `bitsandbytes`, `transformers`, and `unsloth` (baseline only).

Developed on a Tesla T4 (sm_75). Nothing in the kernel is architecture-specific, but the launch config is tuned for that card. `tl.join` needs a recent Triton; the fallback kernel is selected automatically if it's missing.

---

## Limits

- Hardcoded to `blocksize=64` and `state2.blocksize=256`, the QLoRA configuration. Other sizes need the `>> 8`, `* 32`, and `* 64` constants changed.
- Assumes the weight count is a multiple of 64. bitsandbytes pads to this, so the masked tail path exists but is never exercised in practice.
- `nf4` only. `fp4` uses different levels and would need its own select tree.
- The comparison is specifically against `fast_dequantize`. A kernel that fused dequantization *into* the matmul would beat both by never writing `fp16` weights to memory at all — but the harness requires a standalone dequantized tensor, so that's out of scope here.
