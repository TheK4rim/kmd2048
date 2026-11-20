# KMD-2048: an experimental hash construction in pure Python

[![tests](https://github.com/TheK4rim/kmd2048/actions/workflows/tests.yml/badge.svg)](https://github.com/TheK4rim/kmd2048/actions/workflows/tests.yml)
![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)
![dependencies: none](https://img.shields.io/badge/dependencies-none-brightgreen)
![status: experimental](https://img.shields.io/badge/status-experimental-orange)

> **Warning.** KMD-2048 is an experimental construction that has **not been
> independently reviewed**. It is **not more secure than SHA-512, SHA3-512 or
> SHAKE256**; its realistic security level is at most about 256 bits, the same
> as theirs, at more than a thousand times their cost. Do not use it to
> protect real data. See [Security analysis](#security-analysis).

This repository is a complete, dependency-free implementation of KMD-2048
in pure Python. Every cryptographic primitive is written from scratch with
integer arithmetic: the Keccak-f[1600] permutation, the SHAKE256 sponge and
SHA-512. Nothing comes from `hashlib`, `cryptography`, PyCryptodome, PyNaCl
or any other cryptographic library, and no cryptographic library was used to
produce or check any value in this project.

| File | Contents |
|---|---|
| `kmd2048.py` | Keccak-f[1600], SHAKE256, SHA-512, KMD-2048, streaming file hashing, command-line tool |
| `test_kmd2048.py` | 71 `unittest` tests: published NIST/Keccak-team vectors, independent reference implementations, file hashing |
| `benchmark.py` | `time.perf_counter()` benchmarks of every primitive and of KMD-2048 |
| `README.md` | This document |
| `.github/workflows/tests.yml` | GitHub Actions: test suite, benchmark smoke run and CLI check on CPython 3.9–3.14 |

## Contents

- [Quick start](#quick-start)
- [Specification](#specification)
- [Implementation](#implementation)
- [Streaming and file-hashing format](#streaming-and-file-hashing-format)
- [Testing and validation](#testing-and-validation)
- [Benchmarks](#benchmarks)
- [Performance limits of pure Python](#performance-limits-of-pure-python)
- [Security analysis](#security-analysis)

## Quick start

Requirements: Python 3.8 or newer, with no packages to install. The test
suite passes on CPython 3.8, 3.9, 3.10, 3.11, 3.12, 3.13 and 3.14
(Linux x86-64).

Get the code:

```bash
git clone https://github.com/TheK4rim/kmd2048.git
cd kmd2048
```

Run the tests (about 25–45 s, depending on the interpreter):

```bash
python -m unittest -v test_kmd2048
```

Run the benchmarks (about 30 s; `--quick` takes about 5 s):

```bash
python benchmark.py
```

Hash files from the command line (output format like `sha512sum`; use `-` or
no argument for standard input):

```bash
python kmd2048.py path/to/file another/file
```

Python API:

```python
import kmd2048

kmd2048.kmd2048(b"abc")            # 256-byte digest (bytes)
kmd2048.kmd2048_hex(b"abc")        # 512 lowercase hex digits
kmd2048.kmd2048_file("disk.img")   # streamed in 1 MiB chunks, constant memory

h = kmd2048.KMD2048()              # incremental, hashlib-like
h.update(b"hello ")
h.update(b"world")
h.hexdigest()                      # same as kmd2048_hex(b"hello world")

# The primitives can be used on their own:
kmd2048.sha512(b"abc")             # 64 bytes, FIPS 180-4
kmd2048.shake256(b"abc", 100)      # any output length, FIPS 202
kmd2048.keccak_f1600([0] * 25)     # 25 lanes in, 25 lanes out
```

Inputs must be bytes-like (`bytes`, `bytearray`, `memoryview`, ...). Text is
rejected with a `TypeError` and must be encoded explicitly, for example
`text.encode("utf-8")`, so that the hashed bytes are never ambiguous.

## Specification

```
H_0 = M                                          (the message, any length)
for i = 1 .. 1024:
    K_i = SHAKE256(b"KMD-K" || H_(i-1), 64)       64 bytes
    H_i = SHA512(b"KMD-M" || K_i)                 64 bytes
D = SHAKE256(b"KMD-FINAL" || H_1024, 256)         256 bytes = the digest
```

This implementation interprets the specification as follows:

* All values are byte strings and `||` is concatenation. The prefixes are the
  ASCII bytes `KMD-K` (5), `KMD-M` (5) and `KMD-FINAL` (9).
* `SHAKE256(X, n)` is the first `n` bytes of SHAKE256 as defined in FIPS 202:
  KECCAK[c = 512] with the domain suffix `1111` and `pad10*1` padding. The
  rate is 136 bytes. `SHA512` is SHA-512 as defined in FIPS 180-4.
* The iteration count (1024) and output size (256 bytes) are fixed. No
  parameter can change them.
* Cost per digest of an `L`-byte message: `floor((L + 5) / 136) + 1`
  permutations to absorb the message, 1023 more permutations for iterations
  2–1024, and 2 for the 256-byte final output (136 bytes per squeeze). On top
  of that come 1024 single-block SHA-512 compressions. In total that is
  1026 permutations plus 1024 compressions for any message of at most
  130 bytes.

Example: `kmd2048_hex(b"abc")` begins `6cd0f026fb16efd3e0f0653a6cf89d80...`.
KMD-2048 has no independently published test vectors. This value is a
regression value produced by this implementation; see
[Testing and validation](#testing-and-validation) for how it was checked.

## Implementation

All public functions validate their inputs. The hot paths are internal
functions that the public API calls after validation.

### Keccak-f[1600]: `keccak_f1600()`, internal core `_keccak_rounds()`

The state is 25 Python ints (lanes). Lane `A[x, y]` has index `x + 5y`, and
lanes map to bytes little-endian (FIPS 202 §3.1.2, Appendix B.1). The core
optimisations are:

* **One fully unrolled round on 25 local variables**, executed once per round
  constant. There are no lists, no indexing and no inner loops. The round
  constant tuple is a parameter, so the tests can run any number of rounds.
* **θ is fused into ρ and π.** All 25 `B[y, 2x+3y] = ROT(A[x,y] ^ D[x], r[x,y])`
  are computed directly, with the rotation amounts as literal constants.
* **Rotation in 3 operations instead of 4.** For `0 <= x < 2**64`,
  `x * (2**64 + 1)` is the 128-bit value `x||x`, so
  `ROTL(x, n) == (x * (2**64 + 1) >> (64 - n)) & M`. This replaces two
  shifts, an OR and a mask.
* **The lane-complementing transform** (Bertoni, Daemen, Peeters, Van Assche,
  *Keccak implementation overview*, §2.2). Six lanes, `A[1,0]`, `A[2,0]`,
  `A[3,1]`, `A[2,2]`, `A[2,3]` and `A[0,4]`, are stored complemented while the
  rounds run. The pattern is preserved by every round, and χ can then be
  written with AND/OR and only **one NOT per plane**: 5 per round instead
  of 25. NOT is computed as `x ^ (2**64 - 1)`, because Python's `~x` creates
  negative integers, which would break the shift-based rotations. The chi
  formulas were derived by hand for this pattern and are verified against
  the reference implementation for every round count from 0 to 24.

The result is 193 integer operations per round, or 4,644 per permutation,
counted from the compiled bytecode.

### SHA-512: `sha512()`, `Sha512`, internal `_sha512_compress()`

* **16 rounds are unrolled with role renaming.** Each round writes the new
  `e` into the variable that held `d`, and the new `a` into the one that held
  `h`. The working variables are therefore never shuffled. A loop of 5
  iterations gives the 80 rounds.
* **The message schedule is computed in place** in 16 locals, 16 words at a
  time, between the unrolled blocks.
* **Σ and σ use the same `x * (2**64 + 1)` trick**: `ROTR(x, n)` is the low
  64 bits of `(x||x) >> n`, so `Σ1(e)` costs one multiplication, three shifts
  and two XORs. **Masks are deferred.** Σ/σ results feed only sums that are
  reduced mod 2^64, so the bits above bit 63 cannot affect the result and are
  never masked. Only the new `a`, the new `e` and each new `W[t]` are reduced.
* `Ch(e,f,g) = g ^ (e & (f ^ g))` and `Maj(a,b,c) = b ^ ((a ^ b) & (b ^ c))`.
  `b ^ c` is the previous round's `a ^ b`, so Maj costs 3 operations.
* Byte-to-word conversion uses precompiled `struct` objects (`>16Q`, `>8Q`).

The result is 27 operations per round and 16 per schedule word, or 3,201 per
compression, counted from the bytecode.

### SHAKE256 sponge: `shake256()`, `Shake256`

The rate is 136 bytes. Padding is `0x1F, 0x00 ... 0x00, 0x80`, or the single
byte `0x9F` when only one byte of padding is needed. Blocks are read straight
from a `memoryview` of the caller's buffer with `struct.unpack_from('<17Q')`,
without copying the input. Partial blocks are buffered, so `update()`
accepts pieces of any size. `digest(n)` pads a copy of the state, so it does
not consume the object and can be called repeatedly. `copy()` is supported.

### KMD-2048 fused chain: `_kmd2048_chain()`

From iteration 2 on, every stage hashes a 69-byte input that fits in one
block (`KMD-K` + 64 bytes for SHAKE256, `KMD-M` + 64 bytes for SHA-512). The
two padded blocks are therefore built once, from the generic padding
functions. Each iteration then does the following:

1. write `H_(i-1)` at offset 5 with `struct.pack_into`;
2. unpack 17 lanes, append the 8 zero capacity lanes, and run one
   permutation;
3. write lanes 0–7, which are `K_i`, at offset 5;
4. unpack 16 words and run one compression.

No bytes objects are created inside the loop. The message itself only
enters the first SHAKE256 call, through the ordinary incremental sponge. The
tests check the fused loop against a literal transcription of the
specification, built both on the generic `shake256()`/`sha512()` and on
independent reference primitives.

### How the code shapes were chosen

The code shapes above were chosen by measurement. During development,
throwaway prototypes of each variant were timed on the measurement VM
described under [Benchmarks](#benchmarks): Keccak
variants as the minimum of 25 interleaved trials of 300 permutations, SHA-512
variants as the minimum of 20 interleaved trials of 300 blocks. The
prototypes are not shipped, so this table cannot be reproduced with the
delivered files. Compare numbers only within a column. `benchmark.py`
reproduces the overall comparison against the plain reference
implementations (see [Benchmarks](#benchmarks)).

| Keccak-f[1600], µs per permutation | 3.11 | 3.12 | 3.13 |
|---|---:|---:|---:|
| Unrolled; rotation by shifts + OR + mask; χ = `b ^ (~c & d)` | 107.1 | 137.1 | 108.9 |
| Unrolled; rotation by shifts + OR + mask; lane complementing | 100.2 | 114.8 | 101.0 |
| Unrolled; rotation by multiply + shift + mask; χ = `b ^ (~c & d)` | 98.6 | 113.8 | 96.8 |
| **Unrolled; rotation by multiply + shift + mask; lane complementing (shipped)** | **94.1** | **102.3** | **92.1** |

| SHA-512 compression, µs per block | 3.11 | 3.12 | 3.13 |
|---|---:|---:|---:|
| Textbook: 80-round loop, `W` list, `rotr()` calls, tuple shuffle | 106.8 | 142.2 | 161.4 |
| 16-round unroll with renaming; shift/XOR Σ with deferred masks | 61.4 | 82.3 | 92.3 |
| **16-round unroll with renaming; `x·(2^64+1)` Σ (shipped)** | **47.1** | **61.9** | **69.8** |

Other approaches were considered and rejected:

* **Packing a plane or the whole state into one large int**, so that θ and χ
  run on 320- or 1600-bit integers. By operation count, unpacking and
  repacking for ρ/π costs more than it saves.
* **Computing only lanes 0–7 in the last Keccak round of each iteration, or
  constant-folding the fixed bytes of the 69-byte blocks.** Either would save
  a few percent of one permutation, and would need special-case code that
  is harder to audit.
* **Generating code at import time with `exec`.** It would make the executed
  code invisible in the source.
* **`multiprocessing`.** A single digest is inherently sequential: each
  iteration needs the previous one, and the sponge absorbs blocks in order.
  Processes cannot speed it up. They would only help when hashing many
  independent files, which is an application-level decision, so
  `multiprocessing` is not used.

## Streaming and file-hashing format

`kmd2048_file(path, chunk_size=1 MiB)` and `kmd2048_fileobj(fileobj,
chunk_size)` use this exact, deterministic format:

* **Message.** The message is the exact bytes of the file, from offset 0 to
  end-of-file. For `kmd2048_fileobj`, it runs from the stream's current
  position to end-of-stream. The file is opened in binary mode: there is no
  text decoding and no newline translation (a test hashes `\r\n`, `\r`,
  `\x00` and `\x1a` content).
* **Framing.** There is none: no length prefix, no chunk headers, no
  per-chunk hashing and no tree structure.
* **Chunk boundaries do not affect the digest.** The message enters
  KMD-2048 only through the first SHAKE256 call, and a sponge absorbs a byte
  string block by block. This implementation buffers partial 136-byte blocks
  across `update()` calls and applies the padding exactly once, after the
  last byte, when the digest is requested.
* **Length encoding.** The length is never stored explicitly. It is implied
  by the SHAKE256 padding: suffix `1111`, then `pad10*1`. This padding is
  injective, so two different files always give different sponge inputs,
  even when they differ only by trailing zero bytes. SHA-512's own 128-bit
  length field is also present, but it only ever covers the fixed 69-byte
  stage-2 inputs.

The input encoding is therefore identical, and for every file

```python
kmd2048_file(path) == kmd2048(pathlib.Path(path).read_bytes())
```

This is not an assumption. The tests check it for:

* chunk sizes 1, 2, 7, 64, 135, 136, 137, 4096 and 1 MiB;
* file sizes 0, 1, 135, 136, 137, 272, 273, 5,000 and 1 MiB + 17 bytes;
* a file object that returns random short reads;
* a stream read from a non-zero position.

**Memory** is bounded by `chunk_size` plus a 136-byte buffer. A test
verifies that no read requests more than `chunk_size` bytes. An opt-in test
(`KMD2048_TEST_STREAM_MIB=N`) streams N MiB of generated data that never
exists in memory as a whole. The incremental `KMD2048().update()` API gives
the same digest for any split of the message.

**Caveats.** If a file changes while it is being hashed, the digest covers
the bytes that were actually read; the file is not locked or snapshotted.
Text-mode file objects are rejected with `TypeError`.

## Testing and validation

```bash
python -m unittest -v test_kmd2048
```

The suite has 71 tests, one of them opt-in, and takes about 25–45 s on the
machine described under [Benchmarks](#benchmarks), depending on the
interpreter. It uses only `unittest` (including `unittest.mock`), `os`,
`pathlib`, `struct` and `time`. It
deliberately avoids `tempfile`, `random`, `math` and `io`, so it relies on
nothing beyond the modules allowed for this project. One test statically
audits the `import` statements of all three Python files and fails if any
other module, such as `hashlib`, appears.

**Continuous integration.** GitHub Actions runs the suite, a quick
benchmark and the command-line tool on CPython 3.9–3.14 for every push to
`main` and every pull request. Python 3.8 has reached end of life and is
tested only locally.

### Where the expected values come from

| Source | Used for |
|---|---|
| NIST CAVP SHA-512 byte-oriented vectors (SHAVS, CAVS 11.0/11.1) | `SHA512ShortMsg.rsp`: 29 of 129 vectors (0–128 bytes, including every padding boundary and the 69-byte length KMD-2048 uses). `SHA512LongMsg.rsp`: 2 vectors. `SHA512Monte.rsp`: 3 checkpoints, 3,000 chained hashes. |
| NIST CAVP SHAKE256 byte-oriented vectors (SHA3VS, CAVS 19.0) | `SHAKE256ShortMsg.rsp`: 30 vectors (0–272 bytes, around the 136-byte rate). `SHAKE256LongMsg.rsp`: 2. `SHAKE256VariableOut.rsp`: 10 (outputs of 16–2000 bits). `SHAKE256Monte.rsp`: 2 checkpoints, 2,000 chained calls with varying output lengths. |
| FIPS 180-2, Appendix C | SHA-512 of `"abc"`, of the 896-bit two-block message, and of one million `"a"` (also hashed incrementally). |
| Keccak team, XKCP `tests/TestVectors/KeccakF-1600-IntermediateValues.txt` | The state after **every step** (θ, ρ, π, χ, ι) of round 1 of the zero-state example and of round 0 of the second example. The state after 8 different round counts, both full permutation outputs, and the published 200-byte output, which checks the lane byte order. |
| Keccak team, XKCP `tests/TestVectors/ShortMsgKAT_SHAKE256.txt` | 4 messages (0, 69, 136 and 255 bytes) with 512-byte outputs, plus all their prefixes at block boundaries. |

The NIST CAVP `.rsp` files were obtained from the verbatim copy in the
pyca/cryptography repository (`vectors/cryptography_vectors/hashes/`), with
their CAVS headers unchanged, because `csrc.nist.gov` was not reachable
from the development environment. The selected vectors are embedded in
`test_kmd2048.py` as fixed hex strings.

### Independent reference implementations (in the test file)

`ref_keccak_f1600`, `ref_shake256` and `ref_sha512` are deliberately plain
transcriptions of the standards:

* a 5×5 state array with one function per step mapping;
* a bit-level sponge with `pad10*1` (FIPS 202 Algorithm 9) and the FIPS 202
  bit-ordering conventions;
* a textbook 80-round SHA-512.

Their constants are re-derived from first principles: the round constants
from the `rc(t)` LFSR (Algorithms 5–6), the ρ offsets from the (x, y) walk
(Algorithm 2), and the SHA-512 constants from exact integer square and cube
roots of the first 8 and 80 primes. They share no code with `kmd2048.py`, are
themselves checked against the published vectors, and are then used as
oracles for inputs that no published vector covers:

* 140 structured and random Keccak states, including all-zero, all-one and
  single-bit lanes. All-one lanes matter because of the lane-complementing
  transform.
* Every round count from 0 to 24.
* SHAKE256 for every message length from 0 to 140 bytes, plus lengths around
  the 136-byte rate boundary and a 40,000-byte message.
* SHA-512 for every length from 0 to 300 bytes.
* The SHA-512 compression function on arbitrary states.

### KMD-2048 itself

KMD-2048 has no published vectors. Its correctness rests on three checks:

1. `kmd2048()`, with its fused fast path, equals a **literal transcription of
   the specification** (`kmd2048_by_spec`) that calls the module's generic
   `shake256()`/`sha512()`. This is tested for empty, single-byte, binary,
   69/135/136/137-byte and 1 MiB inputs.
2. It equals the same transcription running on the **independent reference
   primitives**, all 1024 iterations through the bit-level reference sponge.
3. It passes property tests: the output is always 256 bytes, the function is
   deterministic, streaming equals one-shot, digests are repeatable and
   non-destructive, `copy()` works, bytes-like types are equivalent and text
   is rejected.

`KMD2048_REGRESSION` holds digests of `b""`, `b"\x00"`, `b"abc"` and
`bytes(range(256))`. These are **produced by this implementation and are not
independent**. Before being recorded, each was reproduced with the reference
primitives. They only guard against accidental changes.

## Benchmarks

```bash
python benchmark.py            # full run, about 30 s
python benchmark.py --quick    # smaller inputs, 3 rounds
python benchmark.py --markdown # Markdown table output
python benchmark.py --file X   # also time kmd2048_file(X) once
```

**Method.** All times come from `time.perf_counter()`. After a warm-up, the
benchmarks run in 7 **interleaved trial rounds**: each round runs every
benchmark once. Each row reports the best (minimum) and median time per
operation. Interleaving matters on a shared machine. The measurements below
were taken on a shared cloud VM. On it, an identical busy loop did between
512 and 1,278 units of work per 100 ms slice (a 60% spread), while the
kernel reported only 0.2% steal time. Every number in this section was
measured; nothing is extrapolated.

### Measured results

The following is one full run on CPython 3.13, copied verbatim (the first of
three consecutive runs):

```
Python    : cpython 3.13.18 (3.13.18 (main, Oct 26 2026, 14:08:19) [Clang 23.1.3 ])
Platform  : Linux 6.18.44-fc-v726 x86_64
CPU       : Intel(R) Xeon(R) Processor @ 2.10GHz (4 logical CPUs; the benchmark uses one)
kmd2048.py: version 0.1.0, 1024 iterations, 256-byte digest
Mode      : full, 7 interleaved trial rounds, timer time.perf_counter()
```

| Benchmark | Best per op | Median per op | Throughput (best) | Ops/s (best) |
|---|---:|---:|---:|---:|
| Keccak-f[1600] keccak_f1600() (validating) | 101.6 us | 123.4 us | 1.878 MiB/s | 9,846.6 permutation/s |
| Keccak-f[1600] internal core | 98.8 us | 124.9 us | 1.931 MiB/s | 10,121.4 permutation/s |
| baseline: textbook FIPS 202 transcription | 273.1 us | 328.0 us | 0.698 MiB/s | 3,661.7 permutation/s |
| SHA-512 compression function | 52.0 us | 61.9 us | 2.349 MiB/s | 19,246.9 128-byte block/s |
| baseline: textbook FIPS 180-4 transcription | 124.2 us | 151.8 us | 0.983 MiB/s | 8,054.5 128-byte block/s |
| SHA-512 sha512(), 64 KiB message | 28.47 ms | 33.97 ms | 2.196 MiB/s | 35.1 message/s |
| SHA-512 sha512(), 69-byte message | 53.9 us | 64.5 us | 1.220 MiB/s | 18,546.1 message/s |
| SHAKE256 absorb 64 KiB, 64-byte output | 46.99 ms | 56.36 ms | 1.330 MiB/s | 21.3 call/s |
| SHAKE256 squeeze 64 KiB from empty input | 48.52 ms | 57.90 ms | 1.288 MiB/s | 20.6 call/s |
| SHAKE256 69-byte input, 64-byte output | 97.5 us | 116.3 us | 0.675 MiB/s | 10,254.7 call/s |
| KMD-2048 chain H_1 .. H_1024 (fast path) | 164.96 ms | 173.95 ms | - | 6.1 chain/s |
| KMD-2048 kmd2048(b'') | 163.72 ms | 217.38 ms | - | 6.1 digest/s |
| KMD-2048 kmd2048(), 1 KiB message | 165.40 ms | 222.10 ms | 0.006 MiB/s | 6.0 digest/s |
| KMD-2048 kmd2048(), 1 MiB message | 1.044 s | 1.203 s | 0.958 MiB/s | 1.0 digest/s |
| KMD-2048 kmd2048_file(), 1 MiB file | 1.034 s | 1.228 s | 0.967 MiB/s | 1.0 digest/s |

Summary lines printed by the same run:

```
One KMD-2048 iteration (chain time / 1024, best): 161.1 us
Keccak-f[1600] internal core: 2.8x faster than the textbook FIPS 202 transcription (best vs best)
SHA-512 compression function: 2.4x faster than the textbook FIPS 180-4 transcription (best vs best)
```

The Keccak throughput column counts the full 200-byte state per permutation.
Hashing absorbs only 136 bytes per permutation, which is why the
`SHAKE256 absorb` row is lower.

**Run-to-run variation.** These are best-per-op values from three
consecutive full runs on CPython 3.13:

| Benchmark | Run 1 | Run 2 | Run 3 | Spread |
|---|---:|---:|---:|---:|
| Keccak-f[1600] internal core | 98.8 us | 127.0 us | 105.9 us | 29% |
| SHA-512 compression function | 52.0 us | 69.0 us | 53.7 us | 33% |
| SHAKE256 absorb 64 KiB | 46.99 ms | 59.46 ms | 46.66 ms | 27% |
| KMD-2048 chain H_1 .. H_1024 | 164.96 ms | 210.36 ms | 183.65 ms | 28% |
| KMD-2048 kmd2048(b'') | 163.72 ms | 189.79 ms | 197.85 ms | 21% |
| KMD-2048 kmd2048(), 1 MiB message | 1.044 s | 1.103 s | 1.106 s | 6% |
| KMD-2048 kmd2048_file(), 1 MiB file | 1.034 s | 1.118 s | 1.178 s | 14% |

On that VM, treat differences under about 30% as noise.

**Across interpreters.** Best per op, one full run each, measured
consecutively on the same VM:

| Benchmark | 3.8 | 3.9 | 3.10 | 3.11 | 3.12 | 3.13 |
|---|---:|---:|---:|---:|---:|---:|
| Keccak-f[1600] internal core | 99.4 us | 124.6 us | 124.4 us | 100.9 us | 105.3 us | 98.8 us |
| baseline: textbook FIPS 202 | 446.8 us | 500.1 us | 451.6 us | 402.5 us | 278.9 us | 273.1 us |
| SHA-512 compression function | 56.4 us | 63.8 us | 53.9 us | 58.2 us | 54.5 us | 52.0 us |
| baseline: textbook FIPS 180-4 | 149.2 us | 168.0 us | 145.9 us | 125.6 us | 122.1 us | 124.2 us |
| KMD-2048 kmd2048(b'') | 163.70 ms | 168.97 ms | 202.12 ms | 213.06 ms | 193.16 ms | 163.72 ms |
| KMD-2048 kmd2048(), 1 MiB | 945.47 ms | 1.141 s | 1.168 s | 1.144 s | 1.177 s | 1.044 s |

### Interpretation

* Every digest has a **fixed cost of about 0.16–0.21 s** on the measurement VM,
  from the 1024-iteration chain. On top of that comes roughly **0.8–1.0 s
  per additional MiB** of input, because SHAKE256 absorbs about
  1.1–1.3 MiB/s. For large files, absorption dominates. At that rate, a
  1 GiB file would take roughly 14–17 minutes; that figure is arithmetic
  from the measured rate, not a measurement.
* One KMD-2048 iteration (one permutation plus one compression on 69-byte
  inputs) costs about 155–210 µs across all runs and interpreters.
* The optimised cores are 2.2–4.5× faster than the plain transcriptions in
  the test file, depending on the interpreter and run. The ratio is largest
  on 3.8–3.11, where the textbook code's list comprehensions are slower;
  CPython 3.12 inlined comprehensions (PEP 709).
* A digest costs about 1,700 times one SHAKE256 call on a short input
  (163.7 ms against 97.5 µs in the run above).

## Performance limits of pure Python

* **Every 64-bit operation is a heap-allocated object.** CPython integers are
  arbitrary-precision objects with 30-bit digits. A 64-bit lane is a 3-digit
  object, and the 128-bit `x||x` values have 5 digits. Each `^`, `&`, `>>` or
  `*` dispatches a bytecode, checks types, allocates a new integer object and
  frees an old one. The measured cost is about **21 ns per operation** for
  Keccak (4,644 operations in about 99 µs) and about **16 ns** for SHA-512
  (3,201 in about 52 µs). That is dozens of CPU cycles for what native code
  does in about one instruction.
* **No access to the hardware.** There are no unboxed machine words, no
  rotate instruction, no SIMD and no register allocation. CPython's
  specialising interpreter (3.11+) has no fast path for bitwise operations
  on multi-digit integers. Its integer fast paths target small ("compact")
  values below 2^30, which hardly ever occur here.
* **No parallelism for one digest.** The chain is strictly sequential, the
  sponge absorbs blocks in order, and the GIL serialises threads. See
  [How the code shapes were chosen](#how-the-code-shapes-were-chosen) for why
  `multiprocessing` is not used.
* **Native implementations** of Keccak and SHA-512 (OpenSSL, XKCP) are orders
  of magnitude faster. They were not measured here, because this project
  deliberately does not import `hashlib`.
* **PyPy.** A JIT such as PyPy may run this code much faster. PyPy could not
  be installed in the development environment, so no figure is given.
* **Not constant-time.** CPython big-integer operations take time that
  depends on the values (a lane below 2^30 has fewer digits than one above
  it), and intermediate values stay in memory until garbage collection. This
  implementation leaks timing information and cannot erase secrets. Do not
  hash secrets where an attacker can measure timing or read memory.

## Security analysis

**KMD-2048 is not more secure than SHA-512, SHA3-512 or SHAKE256.** It
inherits every assumption of both primitives, adds structure of its own that
has never been analysed, and has no security proof. A single call of
SHAKE256 or SHA3-512 with a 512-bit output gives the same 256-bit collision
security at less than a thousandth of the cost.

### Why 2048 output bits do not mean 2048-bit security

An *ideal* 2048-bit hash would need about 2^1024 work to find a collision and
about 2^2048 to find a preimage. KMD-2048 is far from that:

* **The digest is an expansion of 512 bits.**
  `D = SHAKE256("KMD-FINAL" || H_1024, 256)` is a deterministic function of
  the 512-bit value `H_1024`. At most 2^512 distinct digests can exist, and
  the last 192 bytes carry no information that is not already determined by
  `H_1024`.
* **Collision resistance is at most about 2^256.** That is the birthday bound
  of any 512-bit chaining value. The cheapest generic attack never runs the
  iterations at all: it searches for two messages with equal
  `K_1 = SHAKE256("KMD-K" || M, 64)`, which takes about 2^256 SHAKE256 calls.
  Any such pair collides in every later value, and therefore in `D`.
* **Preimage resistance cannot be argued above 256 bits.** FIPS 202 claims
  SHAKE256 preimage resistance of at least min(d, 256) bits, never more, and
  KMD-2048 relies on SHAKE256 at both ends.
* **Conclusion.** At best, KMD-2048 offers the 256-bit security level of its
  components. The extra 1,536 output bits make digests longer, not stronger.

### Why the 512-bit intermediate state matters

After the first SHAKE256 call, everything about the message has been
compressed into `K_1`, which is 512 bits. Every later value is 512 bits.
Iterations 2–1024 apply one fixed public function
`F(h) = SHA512("KMD-M" || SHAKE256("KMD-K" || h, 64))`, so
`H_1024 = F^1023(H_1)`. This has several consequences:

* **A collision at any position is final.** If two messages reach the same
  chain value at any position, their digests are identical.
* **Entropy is capped.** The digest's entropy is at most the smaller of the
  message's entropy and the size of the reachable `H_1024` set. Using `D` as
  "2048 bits of key material" gives at most about 512 bits, and usually much
  less.
* **Iteration shrinks the output space.** Iterating a random function
  shrinks its image. After *k* steps, about 2·2^512/*k* values remain
  reachable (random-mapping statistics, Flajolet and Odlyzko 1990). For
  *k* = 1023 that is about 2^503, so roughly 9 bits are lost. This is
  harmless at this size, but it shows that iteration can only lose
  information, never add it.

### Why 1024 iterations do not increase collision resistance

* **Collisions persist.** Each iteration is a deterministic function of the
  previous value, so a collision present after the first stage persists
  through all 1023 later stages. Iteration can only *add* collision
  opportunities (two chains merging later), never remove them.
* **The generic attack avoids the iterations.** The generic collision attack
  above targets only the first SHAKE256 call, so the iterations add no
  collision resistance at all.
* **What iteration does buy.** Iteration makes one evaluation of KMD-2048
  cost about 2,050 primitive calls, roughly 2^11 times the cost of one hash.
  That is a constant factor. It slows exhaustive guessing of *low-entropy*
  inputs by 2^11, which is tiny compared with real password hashes. There is
  no salt, so precomputed tables work. There is no memory hardness and no
  tunable cost, and GPUs and ASICs evaluate Keccak and SHA-512 extremely
  cheaply.

### Potential structural weaknesses

1. **Cascade, not a robust combiner.** Chaining two hash families does not
   give "the stronger of the two". Collision resistance depends on SHAKE256
   at the first stage: a collision attack on SHAKE256 with 512-bit outputs
   would immediately give KMD-2048 collisions, whatever the strength of
   SHA-512. The design depends on *both* primitives behaving well on these
   exact input formats.
2. **No iteration counter (slide property).** Every iteration applies the same
   `F`, and for 64-byte messages the first stage has exactly the same form as
   the chaining stages. If a 64-byte message `X` equals a chain value
   `H_j(M)` of another message, then `K_1(X) = K_(j+1)(M)`. The chain of `X`
   is the chain of `M` shifted by `j` steps, so
   `KMD-2048(X) = SHAKE256("KMD-FINAL" || H_(1024+j)(M))`. A random oracle has
   no such computable relations, and one long chain yields the digests of
   many related inputs at amortised cost.
3. **No salt, key or personalisation.** The function is fully deterministic
   and public. It is not a MAC, KDF or password hash, and it is open to
   precomputation.
4. **A misleading output length.** A 2048-bit output invites false
   confidence and wastes space. Truncating `D` to 64 bytes loses nothing that
   matters for security.
5. **No proofs.** There is no collision-resistance, preimage or
   indifferentiability proof for this composition. The analysis in this
   README is informal. On the positive side, the classic Merkle–Damgård
   length-extension attack does not apply directly: the message is absorbed
   by a sponge, the SHA-512 inputs have fixed length, and the output comes
   from a sponge. That is an observation, not a proof.
6. **Implementation-level weaknesses.** This implementation is not
   constant-time and cannot wipe memory (see above). It is also slow: about
   1 s per MiB, plus 0.2 s per digest. A service that hashes
   attacker-supplied input with it can easily be overloaded.

### Security assumptions of SHAKE256 and SHA-512

**SHAKE256 (FIPS 202).**

* The assumption is that Keccak-f[1600] has no exploitable structural
  property (the "hermetic sponge strategy"). Under that assumption, a sponge
  with capacity c = 512 is indifferentiable from a random oracle up to about
  2^(c/2) = 2^256 queries (Bertoni, Daemen, Peeters, Van Assche, 2008).
* FIPS 202 claims, for a *d*-bit output: collision resistance of min(d/2, 256)
  bits, preimage resistance of at least min(d, 256) bits, and second-preimage
  resistance of min(d, 256) bits. That gives 256 bits for the 512-bit `K_i`,
  and still 256 bits for the 2048-bit `D`.
* Extensive public cryptanalysis has so far reached only reduced-round
  variants. The full 24-round permutation keeps a large security margin.

**SHA-512 (FIPS 180-4).**

* SHA-512 is a Merkle–Damgård hash with a Davies–Meyer-style compression
  function built from a dedicated block cipher. Its collision resistance
  follows from that of the compression function, together with the length
  padding (MD strengthening).
* Claimed strengths (NIST SP 800-107r1): collision resistance of 256 bits,
  preimage resistance of 512 bits, and second-preimage resistance of
  394–512 bits, depending on message length.
* Known non-ideal properties are length extension and easily found
  compression-function fixed points. Neither applies here: KMD-2048 uses
  SHA-512 only on fixed 69-byte, single-block inputs from the standard IV,
  and never exposes a SHA-512 output directly.
* Published attacks reach only reduced-step variants. No practical attack on
  the full 80 steps is known.

**KMD-2048** additionally assumes that both primitives behave like
independent random functions on these specific input formats, and that the
composition introduces no exploitable structure. Nobody has tested either
assumption.

### Why custom cryptographic constructions require independent review

* **Designers are poor judges of their own designs.** Almost anyone can
  design a cipher or hash they cannot break themselves; that says nothing
  about whether others can.
* **Standard functions have been examined publicly for years.** Keccak was
  selected in the SHA-3 competition (2007–2012, 64 submissions) after years
  of public cryptanalysis. Its sponge mode has security proofs, many
  independent implementations exist, and its test vectors come from several
  sources. KMD-2048 has none of this. It has one implementation,
  self-generated regression values and an informal analysis.
* **Subtle flaws hide in the details** this document can only point at:
  domain separation, entropy loss, related-input properties, encoding
  ambiguities, side channels and misuse in contexts it was not designed for.
* **A passing test suite does not mean a secure design.** The tests here show
  that the code matches FIPS 202, FIPS 180-4 and the stated specification on
  the tested inputs. They say nothing about whether the specification is
  secure.

### What to use instead

| Need | Use |
|---|---|
| General-purpose hash | SHA-512, SHA3-512, SHAKE256, BLAKE2b |
| Password storage | Argon2id, scrypt (or PBKDF2-HMAC where mandated), each with a per-password salt |
| Message authentication | HMAC-SHA-512, KMAC256 |
| Key derivation | HKDF, KMAC256 |

Treat KMD-2048 as an educational and experimental artifact.
