"""Test suite for kmd2048.py, using only the standard-library ``unittest``.

Run (takes about half a minute on CPython)::

    python -m unittest -v test_kmd2048

Optional: stream N MiB of generated data through ``kmd2048_fileobj`` without
storing it (skipped unless the variable is set)::

    KMD2048_TEST_STREAM_MIB=64 python -m unittest -v test_kmd2048

Where the expected values come from
-----------------------------------
No expected value in this file was computed with hashlib or any other
cryptographic library.  The fixed values at the end of the file are:

* NIST CAVP SHA-512 byte-oriented vectors (SHAVS; CAVS 11.0 and 11.1):
  ``SHA512ShortMsg.rsp``, ``SHA512LongMsg.rsp``, ``SHA512Monte.rsp``.
* NIST CAVP SHAKE256 byte-oriented vectors (SHA3VS; CAVS 19.0):
  ``SHAKE256ShortMsg.rsp``, ``SHAKE256LongMsg.rsp``,
  ``SHAKE256VariableOut.rsp``, ``SHAKE256Monte.rsp``.
  The .rsp files were taken verbatim (CAVS headers intact) from the copy
  kept in the pyca/cryptography repository under
  ``vectors/cryptography_vectors/hashes/``.
* FIPS 180-2 Appendix C SHA-512 examples: "abc", the 896-bit message, and
  one million repetitions of "a".
* The Keccak team's XKCP repository, ``tests/TestVectors/``:
  ``KeccakF-1600-IntermediateValues.txt`` (state after every step of every
  round) and ``ShortMsgKAT_SHAKE256.txt`` (512-byte outputs).
* KMD-2048 itself has no published vectors.  ``KMD2048_REGRESSION`` was
  produced by this implementation and only guards against accidental
  changes.  Correctness of KMD-2048 is established by comparing it with a
  literal transcription of the specification evaluated with (a) the
  module's generic primitives and (b) the independent reference primitives
  defined in this file.

Independent reference implementations
-------------------------------------
``ref_keccak_f1600``, ``ref_shake256`` and ``ref_sha512`` are deliberately
plain transcriptions of FIPS 202 and FIPS 180-4: a 5x5 state array with one
function per step mapping, a bit-level sponge with pad10*1, and constants
derived from their mathematical definitions (LFSR, rho walk, integer roots
of primes).  They share no code with kmd2048.py, are themselves checked
against the published vectors, and are then used for differential testing
on many inputs that no published vector covers.
"""

from __future__ import annotations

import os
import struct
import time
import unittest
from pathlib import Path
from unittest import mock

import kmd2048
from kmd2048 import (
    DEFAULT_CHUNK_SIZE,
    DIGEST_SIZE,
    ITERATIONS,
    KECCAK_ROUND_CONSTANTS,
    KMD2048,
    SHA512_INITIAL_HASH,
    SHA512_ROUND_CONSTANTS,
    Sha512,
    Shake256,
    keccak_f1600,
    kmd2048_file,
    kmd2048_fileobj,
    kmd2048_hex,
    sha512,
    shake256,
)

M64 = (1 << 64) - 1
HERE = Path(__file__).resolve().parent


def unhex(text: str) -> bytes:
    return bytes.fromhex(text)


def parse_lanes(text: str) -> list:
    """Parse 25 hexadecimal lanes listed in lane order x + 5*y."""
    lanes = [int(word, 16) for word in text.split()]
    assert len(lanes) == 25
    return lanes


class SplitMix64:
    """Small deterministic PRNG for reproducible test data (not cryptographic)."""

    def __init__(self, seed: int) -> None:
        self.state = seed & M64

    def next64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & M64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
        return z ^ (z >> 31)

    def below(self, n: int) -> int:
        return self.next64() % n

    def randbytes(self, n: int) -> bytes:
        words = (n + 7) // 8
        return struct.pack(f"<{words}Q", *(self.next64() for _ in range(words)))[:n]


# ===========================================================================
# Reference implementations (plain transcriptions of the standards)
# ===========================================================================

# ---- Keccak-f[1600], FIPS 202 section 3 ------------------------------------

def ref_rc_bit(t: int) -> int:
    """FIPS 202 Algorithm 5: rc(t), the output of an 8-bit LFSR."""
    if t % 255 == 0:
        return 1
    r = [1, 0, 0, 0, 0, 0, 0, 0]
    for _ in range(t % 255):
        r = [0] + r
        r[0] ^= r[8]
        r[4] ^= r[8]
        r[5] ^= r[8]
        r[6] ^= r[8]
        r = r[:8]
    return r[0]


def ref_round_constants() -> list:
    """FIPS 202 Algorithm 6 (l = 6): RC[2**j - 1] = rc(j + 7*i_r)."""
    constants = []
    for ir in range(24):
        rc = 0
        for j in range(7):
            rc |= ref_rc_bit(j + 7 * ir) << ((1 << j) - 1)
        constants.append(rc)
    return constants


def ref_rho_offsets() -> list:
    """FIPS 202 Algorithm 2: offsets[x][y] = (t+1)(t+2)/2 along the (x,y) walk."""
    offsets = [[0] * 5 for _ in range(5)]
    x, y = 1, 0
    for t in range(24):
        offsets[x][y] = ((t + 1) * (t + 2) // 2) % 64
        x, y = y, (2 * x + 3 * y) % 5
    return offsets


REF_RC = ref_round_constants()
REF_RHO = ref_rho_offsets()


def ref_rotl(v: int, n: int) -> int:
    n %= 64
    return ((v << n) | (v >> (64 - n))) & M64


def ref_lanes_to_array(lanes):
    """A[x][y] = lane x + 5*y (FIPS 202 section 3.1.2)."""
    return [[lanes[x + 5 * y] for y in range(5)] for x in range(5)]


def ref_array_to_lanes(a):
    return [a[i % 5][i // 5] for i in range(25)]


def ref_theta(a):
    c = [a[x][0] ^ a[x][1] ^ a[x][2] ^ a[x][3] ^ a[x][4] for x in range(5)]
    d = [c[(x - 1) % 5] ^ ref_rotl(c[(x + 1) % 5], 1) for x in range(5)]
    return [[a[x][y] ^ d[x] for y in range(5)] for x in range(5)]


def ref_rho(a):
    return [[ref_rotl(a[x][y], REF_RHO[x][y]) for y in range(5)] for x in range(5)]


def ref_pi(a):
    return [[a[(x + 3 * y) % 5][x] for y in range(5)] for x in range(5)]


def ref_chi(a):
    return [[a[x][y] ^ ((a[(x + 1) % 5][y] ^ M64) & a[(x + 2) % 5][y])
             for y in range(5)] for x in range(5)]


def ref_iota(a, ir: int):
    out = [column[:] for column in a]
    out[0][0] ^= REF_RC[ir]
    return out


REF_STEPS = (("theta", ref_theta), ("rho", ref_rho), ("pi", ref_pi), ("chi", ref_chi))


def ref_keccak_rounds(lanes, rounds: int):
    """The first ``rounds`` rounds of Keccak-f[1600] (Rnd = iota o chi o pi o rho o theta)."""
    a = ref_lanes_to_array(lanes)
    for ir in range(rounds):
        a = ref_iota(ref_chi(ref_pi(ref_rho(ref_theta(a)))), ir)
    return ref_array_to_lanes(a)


def ref_keccak_f1600(lanes):
    return ref_keccak_rounds(lanes, 24)


# ---- SHAKE256, FIPS 202 sections 4, 5.1, 5.2 and 6.2 (bit level) ----------

def ref_bits(data: bytes) -> list:
    """Byte string to bit string; bit i is bit (i mod 8) of byte i//8 (FIPS 202 B.1)."""
    return [(byte >> i) & 1 for byte in data for i in range(8)]


def ref_bytes(bits) -> bytes:
    assert len(bits) % 8 == 0
    return bytes(sum(bits[8 * k + i] << i for i in range(8)) for k in range(len(bits) // 8))


def ref_pad10star1(x: int, m: int) -> list:
    """FIPS 202 Algorithm 9."""
    j = (-m - 2) % x
    return [1] + [0] * j + [1]


def ref_shake256(message: bytes, out_bytes: int) -> bytes:
    """SHAKE256(M, d) = KECCAK[512](M || 1111, d), straight from FIPS 202."""
    r = 1600 - 512
    n = ref_bits(message) + [1, 1, 1, 1]
    p = n + ref_pad10star1(r, len(n))
    assert len(p) % r == 0
    lanes = [0] * 25
    for start in range(0, len(p), r):
        for i, bit in enumerate(p[start:start + r]):   # S ^= P_i || 0^c
            lanes[i // 64] ^= bit << (i % 64)        # S[64*(5y+x) + z] = A[x,y,z]
        lanes = ref_keccak_f1600(lanes)
    d = 8 * out_bytes
    z = []
    while True:
        z += [(lanes[i // 64] >> (i % 64)) & 1 for i in range(r)]   # Trunc_r(S)
        if len(z) >= d:
            return ref_bytes(z[:d])
        lanes = ref_keccak_f1600(lanes)


# ---- SHA-512, FIPS 180-4 ----------------------------------------------------

def ref_primes(count: int) -> list:
    primes = []
    n = 2
    while len(primes) < count:
        if all(n % p for p in primes if p * p <= n):
            primes.append(n)
        n += 1
    return primes


def ref_iroot(n: int, k: int) -> int:
    """floor(n ** (1/k)) by integer Newton iteration (exact, no floats)."""
    x = 1 << ((n.bit_length() + k - 1) // k)      # x >= the true root
    while True:
        y = ((k - 1) * x + n // x ** (k - 1)) // k
        if y >= x:
            break
        x = y
    assert x ** k <= n < (x + 1) ** k
    return x


def ref_fraction_bits(p: int, k: int) -> int:
    """First 64 bits of the fractional part of p ** (1/k)."""
    return ref_iroot(p << (64 * k), k) & M64


REF_SHA512_IV = [ref_fraction_bits(p, 2) for p in ref_primes(8)]
REF_SHA512_K = [ref_fraction_bits(p, 3) for p in ref_primes(80)]


def ref_rotr(x: int, n: int) -> int:
    return ((x >> n) | (x << (64 - n))) & M64


def ref_sha512_compress(h, block: bytes):
    """FIPS 180-4 section 6.4.2, one block."""
    w = [int.from_bytes(block[8 * t:8 * t + 8], "big") for t in range(16)]
    for t in range(16, 80):
        s0 = ref_rotr(w[t - 15], 1) ^ ref_rotr(w[t - 15], 8) ^ (w[t - 15] >> 7)
        s1 = ref_rotr(w[t - 2], 19) ^ ref_rotr(w[t - 2], 61) ^ (w[t - 2] >> 6)
        w.append((s1 + w[t - 7] + s0 + w[t - 16]) & M64)
    a, b, c, d, e, f, g, hh = h
    for t in range(80):
        big_s1 = ref_rotr(e, 14) ^ ref_rotr(e, 18) ^ ref_rotr(e, 41)
        ch = (e & f) ^ ((e ^ M64) & g)
        t1 = (hh + big_s1 + ch + REF_SHA512_K[t] + w[t]) & M64
        big_s0 = ref_rotr(a, 28) ^ ref_rotr(a, 34) ^ ref_rotr(a, 39)
        maj = (a & b) ^ (a & c) ^ (b & c)
        t2 = (big_s0 + maj) & M64
        hh, g, f, e, d, c, b, a = g, f, e, (d + t1) & M64, c, b, a, (t1 + t2) & M64
    return [(x + y) & M64 for x, y in zip(h, (a, b, c, d, e, f, g, hh))]


def ref_sha512(message: bytes) -> bytes:
    """FIPS 180-4: pad (section 5.1.2), parse, compress each block."""
    length = 8 * len(message)
    k = (896 - (length + 1)) % 1024        # l + 1 + k = 896 (mod 1024)
    padded = message + b"\x80" + bytes((k - 7) // 8) + length.to_bytes(16, "big")
    assert len(padded) % 128 == 0
    h = list(REF_SHA512_IV)
    for start in range(0, len(padded), 128):
        h = ref_sha512_compress(h, padded[start:start + 128])
    return b"".join(x.to_bytes(8, "big") for x in h)


# ---- KMD-2048, transcribed literally from the specification ---------------

def kmd2048_by_spec(message: bytes, shake, sha) -> bytes:
    """H_0 = M; K_i = SHAKE256("KMD-K"||H_(i-1), 64); H_i = SHA512("KMD-M"||K_i);
    D = SHAKE256("KMD-FINAL"||H_1024, 256)."""
    h = bytes(message)
    for _ in range(1, 1024 + 1):
        k = shake(b"KMD-K" + h, 64)
        h = sha(b"KMD-M" + k)
    return shake(b"KMD-FINAL" + h, 256)


# ===========================================================================
# Helpers for file tests
# ===========================================================================

def make_temp_dir() -> Path:
    """Create a fresh private directory (the tempfile module is not used)."""
    bases = [os.environ.get(name) for name in ("TMPDIR", "TEMP", "TMP")]
    bases += ["/tmp", str(HERE)]
    for base in bases:
        if base and os.path.isdir(base) and os.access(base, os.W_OK):
            break
    else:
        raise unittest.SkipTest("no writable directory for temporary files")
    for attempt in range(1000):
        path = Path(base) / f"kmd2048-tests-{os.getpid()}-{time.perf_counter_ns()}-{attempt}"
        try:
            path.mkdir(mode=0o700)
        except FileExistsError:
            continue
        return path
    raise RuntimeError("could not create a temporary directory")


def remove_tree(path: Path) -> None:
    for child in path.iterdir():
        if child.is_dir() and not child.is_symlink():
            remove_tree(child)
        else:
            child.unlink()
    path.rmdir()


class ShortReader:
    """Binary file-like object that returns random short reads."""

    def __init__(self, data: bytes, seed: int) -> None:
        self.data = data
        self.pos = 0
        self.rng = SplitMix64(seed)
        self.requested = []

    def read(self, n: int = -1) -> bytes:
        self.requested.append(n)
        if n is None or n < 0:
            n = len(self.data) - self.pos
        n = min(n, 1 + self.rng.below(max(1, n)))       # 1 .. n bytes
        chunk = self.data[self.pos:self.pos + n]
        self.pos += len(chunk)
        return chunk


class GeneratedStream:
    """Deterministic pseudo-random byte stream of a given length, never stored.

    The stream is the concatenation of whole 64-bit PRNG words, so its
    content does not depend on how it is read (leftover bytes are kept).
    """

    def __init__(self, length: int, seed: int) -> None:
        self.remaining = length
        self.rng = SplitMix64(seed)
        self.pending = b""

    def read(self, n: int) -> bytes:
        n = min(n, self.remaining)
        if len(self.pending) < n:
            words = (n - len(self.pending) + 7) // 8
            self.pending += self.rng.randbytes(8 * words)
        chunk, self.pending = self.pending[:n], self.pending[n:]
        self.remaining -= n
        return chunk


class Collector:
    def __init__(self) -> None:
        self.parts = []

    def write(self, text: str) -> int:
        self.parts.append(text)
        return len(text)

    def getvalue(self) -> str:
        return "".join(self.parts)


# ===========================================================================
# Tests
# ===========================================================================

class TestConstants(unittest.TestCase):
    """Module constant tables against their mathematical definitions."""

    def test_keccak_round_constants_match_lfsr_definition(self):
        self.assertEqual(list(KECCAK_ROUND_CONSTANTS), REF_RC)

    def test_keccak_rho_offsets_match_algorithm_2(self):
        self.assertEqual([list(row) for row in kmd2048.KECCAK_RHO_OFFSETS], REF_RHO)

    def test_sha512_initial_hash_is_square_roots_of_first_8_primes(self):
        self.assertEqual(list(SHA512_INITIAL_HASH), REF_SHA512_IV)

    def test_sha512_round_constants_are_cube_roots_of_first_80_primes(self):
        self.assertEqual(list(SHA512_ROUND_CONSTANTS), REF_SHA512_K)

    def test_kmd2048_parameters(self):
        self.assertEqual(DIGEST_SIZE, 256)
        self.assertEqual(ITERATIONS, 1024)
        self.assertEqual(kmd2048.PREFIX_K, b"KMD-K")
        self.assertEqual(kmd2048.PREFIX_M, b"KMD-M")
        self.assertEqual(kmd2048.PREFIX_FINAL, b"KMD-FINAL")

    def test_precomputed_fast_path_blocks(self):
        # The fused loop hashes 69-byte inputs; its padded block templates must
        # equal the textbook paddings for a 69-byte (552-bit) message.
        self.assertEqual(kmd2048._SHAKE_BLOCK_KMD_K,
                         b"KMD-K" + bytes(64) + b"\x1f" + bytes(65) + b"\x80")
        self.assertEqual(kmd2048._SHA512_BLOCK_KMD_M,
                         b"KMD-M" + bytes(64) + b"\x80" + bytes(42) + (552).to_bytes(16, "big"))


class TestKeccakF1600(unittest.TestCase):

    def test_zero_state_gives_published_output(self):
        # Example 1 of KeccakF-1600-IntermediateValues.txt; its output is the
        # input printed for example 2.
        self.assertEqual(keccak_f1600([0] * 25), parse_lanes(KECCAK_EX2_INPUT))

    def test_published_output_bytes_use_little_endian_lanes(self):
        lanes = keccak_f1600([0] * 25)
        self.assertEqual(struct.pack("<25Q", *lanes), unhex(KECCAK_EX1_OUTPUT_BYTES))

    def test_second_permutation_gives_published_output(self):
        self.assertEqual(keccak_f1600(parse_lanes(KECCAK_EX2_INPUT)),
                         parse_lanes(KECCAK_EX2_AFTER_ROUND[23]))

    def test_optimized_rounds_match_published_state_after_each_round(self):
        start = parse_lanes(KECCAK_EX2_INPUT)
        for rnd, text in KECCAK_EX2_AFTER_ROUND.items():
            with self.subTest(example=2, round=rnd):
                got = kmd2048._keccak_rounds(start, KECCAK_ROUND_CONSTANTS[:rnd + 1])
                self.assertEqual(got, parse_lanes(text))
        for (rnd, step), text in KECCAK_EX1_STEPS.items():
            if step == "iota":
                with self.subTest(example=1, round=rnd):
                    got = kmd2048._keccak_rounds([0] * 25, KECCAK_ROUND_CONSTANTS[:rnd + 1])
                    self.assertEqual(got, parse_lanes(text))

    def test_reference_step_mappings_match_published_intermediate_values(self):
        # Example 1, round 1 (sparse state, easy to follow by hand).
        a = ref_lanes_to_array(parse_lanes(KECCAK_EX1_STEPS[(0, "iota")]))
        for name, step in REF_STEPS:
            a = step(a)
            with self.subTest(example=1, round=1, step=name):
                self.assertEqual(ref_array_to_lanes(a), parse_lanes(KECCAK_EX1_STEPS[(1, name)]))
        a = ref_iota(a, 1)
        self.assertEqual(ref_array_to_lanes(a), parse_lanes(KECCAK_EX1_STEPS[(1, "iota")]))
        # Example 2, round 0 (dense state).
        a = ref_lanes_to_array(parse_lanes(KECCAK_EX2_INPUT))
        for name, step in REF_STEPS:
            a = step(a)
            with self.subTest(example=2, round=0, step=name):
                self.assertEqual(ref_array_to_lanes(a), parse_lanes(KECCAK_EX2_ROUND0_STEPS[name]))
        a = ref_iota(a, 0)
        self.assertEqual(ref_array_to_lanes(a), parse_lanes(KECCAK_EX2_ROUND0_STEPS["iota"]))

    def test_reference_permutation_matches_published_outputs(self):
        self.assertEqual(ref_keccak_f1600([0] * 25), parse_lanes(KECCAK_EX2_INPUT))
        self.assertEqual(ref_keccak_f1600(parse_lanes(KECCAK_EX2_INPUT)),
                         parse_lanes(KECCAK_EX2_AFTER_ROUND[23]))

    def test_matches_reference_on_structured_and_random_states(self):
        states = [[0] * 25, [M64] * 25, [0x5555555555555555] * 25,
                  [0xAAAAAAAAAAAAAAAA] * 25, list(range(25))]
        for i in range(25):
            for value in (1, 1 << 63):
                state = [0] * 25
                state[i] = value
                states.append(state)
            state = [M64] * 25
            state[i] = 0
            states.append(state)
        rng = SplitMix64(0x4B4D44)
        states += [[rng.next64() for _ in range(25)] for _ in range(60)]
        for index, state in enumerate(states):
            with self.subTest(state=index):
                self.assertEqual(keccak_f1600(state), ref_keccak_f1600(state))

    def test_any_number_of_rounds_matches_reference(self):
        # The lane-complementing transform must be exact after every round.
        rng = SplitMix64(7)
        for rounds in range(25):
            state = [rng.next64() for _ in range(25)]
            with self.subTest(rounds=rounds):
                self.assertEqual(
                    kmd2048._keccak_rounds(state, KECCAK_ROUND_CONSTANTS[:rounds]),
                    ref_keccak_rounds(state, rounds))

    def test_input_is_not_modified_and_tuples_are_accepted(self):
        state = list(range(25))
        out = keccak_f1600(state)
        self.assertEqual(state, list(range(25)))
        self.assertIsInstance(out, list)
        self.assertEqual(keccak_f1600(tuple(state)), out)
        self.assertTrue(all(isinstance(v, int) and 0 <= v <= M64 for v in out))

    def test_rejects_invalid_states(self):
        for bad in ([0] * 24, [0] * 26, []):
            with self.assertRaises(ValueError):
                keccak_f1600(bad)
        for lane in (1 << 64, -1):
            with self.assertRaises(ValueError):
                keccak_f1600([0] * 24 + [lane])
        for lane in (1.0, "1", None, True):
            with self.assertRaises(TypeError):
                keccak_f1600([0] * 24 + [lane])


class TestSHAKE256(unittest.TestCase):

    def test_nist_short_messages(self):
        for msg_hex, out_hex in SHAKE256_SHORT_MSG:
            msg, expected = unhex(msg_hex), unhex(out_hex)
            with self.subTest(length=len(msg)):
                self.assertEqual(shake256(msg, len(expected)), expected)

    def test_nist_long_messages(self):
        for msg_hex, out_hex in SHAKE256_LONG_MSG:
            msg, expected = unhex(msg_hex), unhex(out_hex)
            with self.subTest(length=len(msg)):
                self.assertEqual(shake256(msg, len(expected)), expected)

    def test_nist_variable_output_lengths(self):
        for msg_hex, bits, out_hex in SHAKE256_VARIABLE_OUT:
            self.assertEqual(bits % 8, 0)
            with self.subTest(output_bits=bits):
                self.assertEqual(shake256(unhex(msg_hex), bits // 8), unhex(out_hex))

    def test_nist_monte_carlo(self):
        # SHA3VS section 6.3.3: 1000 chained SHAKE256 calls per checkpoint; the
        # next input is the leftmost 128 bits of the previous output (zero-padded
        # when shorter) and the next output length is derived from its last
        # 16 bits.
        min_bytes = SHAKE256_MONTE_MIN_OUT_BITS // 8
        max_bytes = SHAKE256_MONTE_MAX_OUT_BITS // 8
        span = max_bytes - min_bytes + 1
        output = unhex(SHAKE256_MONTE_MSG)
        out_len = max_bytes
        for index, (bits, out_hex) in enumerate(SHAKE256_MONTE_CHECKPOINTS):
            for _ in range(1000):
                msg = output[:16].ljust(16, b"\x00")
                output = shake256(msg, out_len)
                out_len = min_bytes + int.from_bytes(output[-2:], "big") % span
            with self.subTest(checkpoint=index):
                self.assertEqual(len(output) * 8, bits)
                self.assertEqual(output, unhex(out_hex))

    def test_xkcp_long_outputs_and_prefixes(self):
        for msg_hex, out_hex in XKCP_SHAKE256_KAT:
            msg, expected = unhex(msg_hex), unhex(out_hex)
            with self.subTest(length=len(msg)):
                self.assertEqual(shake256(msg, 512), expected)
                for n in (0, 1, 64, 135, 136, 137, 256, 271, 272, 273, 408, 511):
                    self.assertEqual(shake256(msg, n), expected[:n])

    def test_reference_matches_published_vectors(self):
        # Validates the independent reference before it is used as an oracle.
        for msg_hex, out_hex in SHAKE256_SHORT_MSG + SHAKE256_LONG_MSG:
            msg, expected = unhex(msg_hex), unhex(out_hex)
            with self.subTest(length=len(msg)):
                self.assertEqual(ref_shake256(msg, len(expected)), expected)
        msg_hex, out_hex = XKCP_SHAKE256_KAT[-1]
        self.assertEqual(ref_shake256(unhex(msg_hex), 512), unhex(out_hex))

    def test_matches_reference_around_the_rate_boundary(self):
        rng = SplitMix64(136)
        lengths = list(range(0, 141)) + [200, 271, 272, 273, 407, 408, 409, 544]
        out_lengths = (1, 32, 64, 135, 136, 137, 256, 300)
        for i, length in enumerate(lengths):
            msg = rng.randbytes(length)
            out_len = out_lengths[i % len(out_lengths)]
            with self.subTest(length=length, out_len=out_len):
                self.assertEqual(shake256(msg, out_len), ref_shake256(msg, out_len))

    def test_large_input_matches_reference(self):
        msg = SplitMix64(99).randbytes(40_000)
        self.assertEqual(shake256(msg, 300), ref_shake256(msg, 300))

    def test_output_is_prefix_consistent(self):
        msg = b"prefix property"
        full = shake256(msg, 700)
        self.assertEqual(len(full), 700)
        for n in list(range(0, 300)) + [407, 408, 409, 544, 699, 700]:
            self.assertEqual(shake256(msg, n), full[:n])

    def test_incremental_absorption_matches_one_shot(self):
        rng = SplitMix64(5)
        data = rng.randbytes(2000)
        expected = shake256(data, 64)
        plans = [[1] * 300, [135] * 15, [136] * 15, [137] * 15, [0, 5, 0, 131, 0, 1]]
        plans += [[1 + rng.below(300) for _ in range(40)] for _ in range(5)]
        for plan in plans:
            sponge = Shake256()
            pos = 0
            for size in plan:
                sponge.update(data[pos:pos + size])
                pos += size
            sponge.update(data[pos:])
            with self.subTest(plan=plan[:6]):
                self.assertEqual(sponge.digest(64), expected)

    def test_digest_is_repeatable_and_does_not_finalize(self):
        sponge = Shake256(b"abc")
        first = sponge.digest(200)
        self.assertEqual(sponge.digest(200), first)
        self.assertEqual(sponge.hexdigest(10), first[:10].hex())
        sponge.update(b"def")
        self.assertEqual(sponge.digest(64), shake256(b"abcdef", 64))

    def test_copy_is_independent(self):
        original = Shake256(b"x" * 150)
        clone = original.copy()
        clone.update(b"y")
        self.assertEqual(original.digest(32), shake256(b"x" * 150, 32))
        self.assertEqual(clone.digest(32), shake256(b"x" * 150 + b"y", 32))

    def test_accepts_bytes_like_objects(self):
        data = bytes(range(200))
        expected = shake256(data, 48)
        self.assertEqual(shake256(bytearray(data), 48), expected)
        self.assertEqual(shake256(memoryview(data), 48), expected)
        self.assertEqual(shake256(memoryview(b"\xff" + data)[1:], 48), expected)
        self.assertEqual(shake256(memoryview(data).cast("I"), 48), expected)

    def test_rejects_text_and_invalid_lengths(self):
        with self.assertRaises(TypeError):
            shake256("text", 32)
        with self.assertRaises(TypeError):
            shake256(12345, 32)
        with self.assertRaises(ValueError):
            shake256(b"", -1)
        for bad in (1.5, "32", None, True):
            with self.assertRaises(TypeError):
                shake256(b"", bad)
        self.assertEqual(shake256(b"abc", 0), b"")


class TestSHA512(unittest.TestCase):

    def test_nist_short_messages(self):
        for msg_hex, md_hex in SHA512_SHORT_MSG:
            msg = unhex(msg_hex)
            with self.subTest(length=len(msg)):
                self.assertEqual(sha512(msg), unhex(md_hex))

    def test_nist_long_messages(self):
        for msg_hex, md_hex in SHA512_LONG_MSG:
            msg = unhex(msg_hex)
            with self.subTest(length=len(msg)):
                self.assertEqual(sha512(msg), unhex(md_hex))

    def test_nist_monte_carlo(self):
        # SHAVS section 6.4: MD_i = SHA512(MD_(i-3) || MD_(i-2) || MD_(i-1)),
        # 1000 iterations per checkpoint, starting from MD_0 = MD_1 = MD_2 = Seed.
        seed = unhex(SHA512_MONTE_SEED)
        for index, md_hex in enumerate(SHA512_MONTE_CHECKPOINTS):
            a = b = c = seed
            for _ in range(1000):
                a, b, c = b, c, sha512(a + b + c)
            seed = c
            with self.subTest(checkpoint=index):
                self.assertEqual(seed, unhex(md_hex))

    def test_fips_180_2_examples(self):
        self.assertEqual(sha512(b"abc").hex(), FIPS180_SHA512_ABC)
        self.assertEqual(sha512(FIPS180_896_BIT_MESSAGE).hex(), FIPS180_SHA512_896)

    def test_one_million_a(self):
        data = b"a" * 1_000_000
        self.assertEqual(sha512(data).hex(), FIPS180_SHA512_MILLION_A)
        hasher = Sha512()
        for start in range(0, len(data), 4099):           # odd piece size
            hasher.update(data[start:start + 4099])
        self.assertEqual(hasher.hexdigest(), FIPS180_SHA512_MILLION_A)

    def test_reference_matches_published_vectors(self):
        for msg_hex, md_hex in SHA512_SHORT_MSG + SHA512_LONG_MSG:
            msg = unhex(msg_hex)
            with self.subTest(length=len(msg)):
                self.assertEqual(ref_sha512(msg), unhex(md_hex))
        self.assertEqual(ref_sha512(b"abc").hex(), FIPS180_SHA512_ABC)

    def test_matches_reference_for_every_length_up_to_300_bytes(self):
        rng = SplitMix64(512)
        for length in range(301):
            msg = rng.randbytes(length)
            with self.subTest(length=length):
                self.assertEqual(sha512(msg), ref_sha512(msg))

    def test_compression_function_matches_reference_for_arbitrary_states(self):
        rng = SplitMix64(80)
        for _ in range(40):
            state = [rng.next64() for _ in range(8)]
            block = rng.randbytes(128)
            words = struct.unpack(">16Q", block)
            self.assertEqual(list(kmd2048._sha512_compress(state, words)),
                             ref_sha512_compress(state, block))
        extremes = [[0] * 8, [M64] * 8]
        for state in extremes:
            for block in (bytes(128), b"\xff" * 128):
                words = struct.unpack(">16Q", block)
                self.assertEqual(list(kmd2048._sha512_compress(state, words)),
                                 ref_sha512_compress(state, block))

    def test_incremental_updates_match_one_shot(self):
        rng = SplitMix64(3)
        data = rng.randbytes(1500)
        expected = sha512(data)
        plans = [[1] * 260, [127] * 11, [128] * 11, [129] * 11, [0, 3, 0, 125, 0]]
        plans += [[1 + rng.below(260) for _ in range(30)] for _ in range(5)]
        for plan in plans:
            hasher = Sha512()
            pos = 0
            for size in plan:
                hasher.update(data[pos:pos + size])
                pos += size
            hasher.update(data[pos:])
            with self.subTest(plan=plan[:6]):
                self.assertEqual(hasher.digest(), expected)

    def test_digest_is_repeatable_and_copy_is_independent(self):
        hasher = Sha512(b"ab")
        self.assertEqual(hasher.digest(), hasher.digest())
        clone = hasher.copy()
        hasher.update(b"c")
        self.assertEqual(hasher.hexdigest(), FIPS180_SHA512_ABC)
        self.assertEqual(clone.digest(), sha512(b"ab"))

    def test_bytes_like_inputs_and_text_rejection(self):
        self.assertEqual(sha512(bytearray(b"abc")).hex(), FIPS180_SHA512_ABC)
        self.assertEqual(sha512(memoryview(b"xabc")[1:]).hex(), FIPS180_SHA512_ABC)
        with self.assertRaises(TypeError):
            sha512("abc")


class TestKMD2048(unittest.TestCase):

    def test_output_length_is_256_bytes(self):
        for msg in (b"", b"\x00", bytes(range(256)), b"z" * 1000):
            with self.subTest(length=len(msg)):
                self.assertEqual(len(kmd2048.kmd2048(msg)), DIGEST_SIZE)
        self.assertEqual(KMD2048.digest_size, 256)

    def test_empty_input(self):
        digest = kmd2048.kmd2048(b"")
        self.assertEqual(digest, kmd2048_by_spec(b"", shake256, sha512))
        self.assertEqual(digest.hex(), KMD2048_REGRESSION[""])

    def test_single_byte_inputs(self):
        digests = {}
        for value in (0x00, 0x01, 0x7F, 0x80, 0xFF):
            digests[value] = kmd2048.kmd2048(bytes([value]))
            self.assertEqual(len(digests[value]), 256)
        self.assertEqual(len(set(digests.values())), len(digests))
        self.assertEqual(digests[0x00], kmd2048_by_spec(b"\x00", shake256, sha512))
        self.assertEqual(digests[0x00].hex(), KMD2048_REGRESSION["00"])

    def test_binary_input(self):
        data = bytes(range(256)) + b"\x00" * 7 + b"\xff\xfe\x80\x7f"
        digest = kmd2048.kmd2048(data)
        self.assertEqual(digest, kmd2048_by_spec(data, shake256, sha512))
        self.assertEqual(kmd2048.kmd2048(bytes(range(256))).hex(),
                         KMD2048_REGRESSION[bytes(range(256)).hex()])

    def test_large_input(self):
        data = SplitMix64(2048).randbytes((1 << 20) + 3)      # 1 MiB + 3 bytes
        digest = kmd2048.kmd2048(data)
        self.assertEqual(len(digest), 256)
        hasher = KMD2048()
        for start in range(0, len(data), 100_003):
            hasher.update(data[start:start + 100_003])
        self.assertEqual(hasher.digest(), digest)
        self.assertEqual(digest, kmd2048_by_spec(data, shake256, sha512))

    def test_matches_specification_with_module_primitives(self):
        # Exercises the fused fast path (kmd2048) against the literal
        # specification built on the generic shake256()/sha512() functions.
        for msg in (b"abc", b"\x00" * 64, b"m" * 69, b"r" * 135, b"r" * 136, b"r" * 137):
            with self.subTest(length=len(msg)):
                self.assertEqual(kmd2048.kmd2048(msg), kmd2048_by_spec(msg, shake256, sha512))

    def test_matches_specification_with_independent_reference_primitives(self):
        # 1024 iterations through the bit-level reference SHAKE256 and the
        # textbook SHA-512 (a few seconds).
        self.assertEqual(kmd2048.kmd2048(b"abc"),
                         kmd2048_by_spec(b"abc", ref_shake256, ref_sha512))

    def test_deterministic(self):
        msg = b"determinism"
        first = kmd2048.kmd2048(msg)
        self.assertEqual(kmd2048.kmd2048(msg), first)
        self.assertEqual(KMD2048(msg).digest(), first)

    def test_hex_output(self):
        msg = b"abc"
        text = kmd2048_hex(msg)
        self.assertEqual(len(text), 512)
        self.assertEqual(text, text.lower())
        self.assertEqual(bytes.fromhex(text), kmd2048.kmd2048(msg))
        self.assertEqual(text, KMD2048_REGRESSION["616263"])
        self.assertEqual(KMD2048(msg).hexdigest(), text)

    def test_streaming_matches_one_shot(self):
        data = SplitMix64(11).randbytes(3000)
        expected = kmd2048.kmd2048(data)
        rng = SplitMix64(12)
        for _ in range(3):
            hasher = KMD2048()
            pos = 0
            while pos < len(data):
                size = rng.below(400)
                hasher.update(data[pos:pos + size])
                pos += size
            self.assertEqual(hasher.digest(), expected)

    def test_digest_is_repeatable_and_update_after_digest_works(self):
        hasher = KMD2048(b"ab")
        first = hasher.digest()
        self.assertEqual(hasher.digest(), first)
        hasher.update(b"c")
        self.assertEqual(hasher.hexdigest(), KMD2048_REGRESSION["616263"])

    def test_copy_is_independent(self):
        hasher = KMD2048(b"a")
        clone = hasher.copy()
        clone.update(b"bc")
        self.assertEqual(clone.hexdigest(), KMD2048_REGRESSION["616263"])
        self.assertEqual(hasher.digest(), kmd2048.kmd2048(b"a"))

    def test_bytes_like_inputs_give_identical_digests(self):
        expected = KMD2048_REGRESSION["616263"]
        self.assertEqual(kmd2048_hex(bytearray(b"abc")), expected)
        self.assertEqual(kmd2048_hex(memoryview(b"--abc")[2:]), expected)

    def test_rejects_text_and_non_buffer_inputs(self):
        for bad in ("abc", 123, None, ["a"]):
            with self.assertRaises(TypeError):
                kmd2048.kmd2048(bad)
        with self.assertRaises(TypeError):
            KMD2048().update("abc")

    def test_regression_vectors(self):
        for msg_hex, digest_hex in KMD2048_REGRESSION.items():
            with self.subTest(message=msg_hex[:16]):
                self.assertEqual(kmd2048.kmd2048(unhex(msg_hex)).hex(), digest_hex)


class TestFileHashing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = make_temp_dir()

    @classmethod
    def tearDownClass(cls):
        remove_tree(cls.tmp)

    def write(self, name: str, data: bytes) -> Path:
        path = self.tmp / name
        path.write_bytes(data)
        return path

    def test_empty_file(self):
        path = self.write("empty.bin", b"")
        self.assertEqual(kmd2048_file(path).hex(), KMD2048_REGRESSION[""])

    def test_file_equals_in_memory_hash_at_block_boundaries(self):
        rng = SplitMix64(21)
        for size in (1, 135, 136, 137, 272, 273):
            data = rng.randbytes(size)
            path = self.write(f"size-{size}.bin", data)
            with self.subTest(size=size):
                self.assertEqual(kmd2048_file(path), kmd2048.kmd2048(data))

    def test_chunk_size_does_not_change_the_digest(self):
        data = SplitMix64(22).randbytes(5000)
        path = self.write("chunks.bin", data)
        expected = kmd2048.kmd2048(data)
        for chunk_size in (1, 2, 7, 64, 135, 136, 137, 4096, DEFAULT_CHUNK_SIZE):
            with self.subTest(chunk_size=chunk_size):
                self.assertEqual(kmd2048_file(path, chunk_size=chunk_size), expected)

    def test_large_file_spanning_many_chunks(self):
        data = SplitMix64(23).randbytes((1 << 20) + 17)
        path = self.write("large.bin", data)
        self.assertEqual(kmd2048_file(path, chunk_size=65536), kmd2048.kmd2048(data))

    def test_path_types(self):
        path = self.write("abc.txt", b"abc")
        expected = KMD2048_REGRESSION["616263"]
        self.assertEqual(kmd2048_file(path).hex(), expected)
        self.assertEqual(kmd2048_file(str(path)).hex(), expected)
        self.assertEqual(kmd2048_file(os.fsencode(str(path))).hex(), expected)

    def test_binary_content_is_not_translated(self):
        data = b"line1\r\nline2\rline3\n\x00\x1a"
        path = self.write("newlines.bin", data)
        self.assertEqual(kmd2048_file(path), kmd2048.kmd2048(data))

    def test_file_object_with_short_reads(self):
        data = SplitMix64(24).randbytes(4000)
        reader = ShortReader(data, seed=25)
        self.assertEqual(kmd2048_fileobj(reader, chunk_size=300), kmd2048.kmd2048(data))
        self.assertEqual(reader.pos, len(data))

    def test_reads_are_bounded_by_chunk_size(self):
        data = SplitMix64(26).randbytes(10_000)
        reader = ShortReader(data, seed=27)
        kmd2048_fileobj(reader, chunk_size=512)
        self.assertTrue(reader.requested)
        self.assertTrue(all(n == 512 for n in reader.requested))
        self.assertGreaterEqual(len(reader.requested), len(data) // 512)

    def test_file_object_is_read_from_its_current_position(self):
        data = b"0123456789abc"
        path = self.write("offset.bin", data)
        with open(path, "rb") as fileobj:
            fileobj.seek(10)
            self.assertEqual(kmd2048_fileobj(fileobj).hex(), KMD2048_REGRESSION["616263"])

    def test_text_mode_file_object_is_rejected(self):
        path = self.write("text.txt", b"abc")
        with open(path, "r", encoding="ascii") as fileobj:
            with self.assertRaises(TypeError):
                kmd2048_fileobj(fileobj)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            kmd2048_file(self.tmp / "does-not-exist")

    def test_invalid_chunk_size(self):
        path = self.write("chunk-check.bin", b"x")
        for bad in (0, -1):
            with self.assertRaises(ValueError):
                kmd2048_file(path, chunk_size=bad)
        for bad in (1.5, "1", None, True):
            with self.assertRaises(TypeError):
                kmd2048_file(path, chunk_size=bad)

    def test_command_line_interface(self):
        path = self.write("cli.bin", b"abc")
        out, err = Collector(), Collector()
        with mock.patch("sys.stdout", out), mock.patch("sys.stderr", err):
            status = kmd2048.main([str(path), str(self.tmp / "missing")])
        self.assertEqual(status, 1)
        self.assertEqual(out.getvalue(), f"{KMD2048_REGRESSION['616263']}  {path}\n")
        self.assertIn("missing", err.getvalue())

    @unittest.skipUnless(os.environ.get("KMD2048_TEST_STREAM_MIB"),
                         "set KMD2048_TEST_STREAM_MIB=N to stream N MiB")
    def test_stream_of_many_mebibytes(self):
        size = int(os.environ["KMD2048_TEST_STREAM_MIB"]) << 20
        digest = kmd2048_fileobj(GeneratedStream(size, seed=31))
        hasher = KMD2048()
        source = GeneratedStream(size, seed=31)
        rng = SplitMix64(32)
        while source.remaining:
            hasher.update(source.read(1 + rng.below(3 << 20)))
        self.assertEqual(hasher.digest(), digest)


class TestNoCryptographicDependencies(unittest.TestCase):
    """Static audit of the import statements in the deliverables."""

    # kmd2048 and test_kmd2048 are this project's own modules.
    ALLOWED = {"__future__", "dataclasses", "kmd2048", "os", "pathlib", "struct",
               "sys", "test_kmd2048", "time", "typing", "unittest"}

    @staticmethod
    def imported_modules(path: Path):
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("import "):
                for name in stripped[len("import "):].split(","):
                    yield line, name.strip().split(" as ")[0].split(".")[0]
            elif stripped.startswith("from ") and " import " in stripped:
                yield line, stripped[len("from "):].split(" import ")[0].strip().split(".")[0]

    def test_only_allowed_standard_library_modules_are_imported(self):
        for name in ("kmd2048.py", "test_kmd2048.py", "benchmark.py"):
            path = HERE / name
            if not path.exists():
                continue
            for line, module in self.imported_modules(path):
                with self.subTest(file=name, line=line.strip()):
                    self.assertIn(module, self.ALLOWED)


# ===========================================================================
# Fixed expected values
# ===========================================================================

# ---- FIPS 180-2 Appendix C (SHA-512 examples) ------------------------------

FIPS180_SHA512_ABC = (
    "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a"
    "2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f")
FIPS180_896_BIT_MESSAGE = (
    b"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmn"
    b"hijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu")
FIPS180_SHA512_896 = (
    "8e959b75dae313da8cf4f72814fc143f8f7779c6eb9f7fa17299aeadb6889018"
    "501d289e4900f7e4331b99dec4b5433ac7d329eeb6dd26545e96e55b874be909")
FIPS180_SHA512_MILLION_A = (
    "e718483d0ce769644e2e42c7bc15b4638e1f98b13b2044285632a803afa973eb"
    "de0ff244877ea60a4cb0432ce577c31beb009c5c2c49aa2e4eadb217ad8cc09b")

# ---- Keccak team, XKCP tests/TestVectors/KeccakF-1600-IntermediateValues.txt
# Lanes are listed in lane order x + 5*y, five per line, as in the source file.

# Example 1: all-zero input.  Round 0 maps zero to RC[0] in A[0,0], so
# only its iota output is listed; every step of round 1 is listed.
KECCAK_EX1_STEPS = {
    (0, 'iota'): """
        0000000000000001 0000000000000000 0000000000000000 0000000000000000 0000000000000000
        0000000000000000 0000000000000000 0000000000000000 0000000000000000 0000000000000000
        0000000000000000 0000000000000000 0000000000000000 0000000000000000 0000000000000000
        0000000000000000 0000000000000000 0000000000000000 0000000000000000 0000000000000000
        0000000000000000 0000000000000000 0000000000000000 0000000000000000 0000000000000000
        """,
    (1, 'theta'): """
        0000000000000001 0000000000000001 0000000000000000 0000000000000000 0000000000000002
        0000000000000000 0000000000000001 0000000000000000 0000000000000000 0000000000000002
        0000000000000000 0000000000000001 0000000000000000 0000000000000000 0000000000000002
        0000000000000000 0000000000000001 0000000000000000 0000000000000000 0000000000000002
        0000000000000000 0000000000000001 0000000000000000 0000000000000000 0000000000000002
        """,
    (1, 'rho'): """
        0000000000000001 0000000000000002 0000000000000000 0000000000000000 0000000010000000
        0000000000000000 0000100000000000 0000000000000000 0000000000000000 0000000000200000
        0000000000000000 0000000000000400 0000000000000000 0000000000000000 0000010000000000
        0000000000000000 0000200000000000 0000000000000000 0000000000000000 0000000000000200
        0000000000000000 0000000000000004 0000000000000000 0000000000000000 0000000000008000
        """,
    (1, 'pi'): """
        0000000000000001 0000100000000000 0000000000000000 0000000000000000 0000000000008000
        0000000000000000 0000000000200000 0000000000000000 0000200000000000 0000000000000000
        0000000000000002 0000000000000000 0000000000000000 0000000000000200 0000000000000000
        0000000010000000 0000000000000000 0000000000000400 0000000000000000 0000000000000000
        0000000000000000 0000000000000000 0000010000000000 0000000000000000 0000000000000004
        """,
    (1, 'chi'): """
        0000000000000001 0000100000000000 0000000000008000 0000000000000001 0000100000008000
        0000000000000000 0000200000200000 0000000000000000 0000200000000000 0000000000200000
        0000000000000002 0000000000000200 0000000000000000 0000000000000202 0000000000000000
        0000000010000400 0000000000000000 0000000000000400 0000000010000000 0000000000000000
        0000010000000000 0000000000000000 0000010000000004 0000000000000000 0000000000000004
        """,
    (1, 'iota'): """
        0000000000008083 0000100000000000 0000000000008000 0000000000000001 0000100000008000
        0000000000000000 0000200000200000 0000000000000000 0000200000000000 0000000000200000
        0000000000000002 0000000000000200 0000000000000000 0000000000000202 0000000000000000
        0000000010000400 0000000000000000 0000000000000400 0000000010000000 0000000000000000
        0000010000000000 0000000000000000 0000010000000004 0000000000000000 0000000000000004
        """,
}

# Example 2: input = output of example 1.
KECCAK_EX2_INPUT = """
    F1258F7940E1DDE7 84D5CCF933C0478A D598261EA65AA9EE BD1547306F80494D 8B284E056253D057
    FF97A42D7F8E6FD4 90FEE5A0A44647C4 8C5BDA0CD6192E76 AD30A6F71B19059C 30935AB7D08FFC64
    EB5AA93F2317D635 A9A6E6260D712103 81A57C16DBCF555F 43B831CD0347C826 01F22F1A11A5569F
    05E5635A21D9AE61 64BEFEF28CC970F2 613670957BC46611 B87C5A554FD00ECB 8C3EE88A1CCF32C8
    940C7922AE3A2614 1841F924A2C509E4 16F53526E70465C2 75F644E97F30A13B EAF1FF7B5CECA249
    """

# Example 2: every step of round 0 (dense state).
KECCAK_EX2_ROUND0_STEPS = {
    'theta': """
        AF463273CA4D877D AF9FDF84CEC209D0 28C573DB9CDDA7BA ABBCDA349E794C02 FD3CB094025A23B6
        A1F41927F522354E BBB4F6DD5944099E 71068FC9EC9E2022 BB993BF3EAE000D3 4687A426B0860F85
        B5391435A9BB8CAF 82ECF55BF0736F59 7CF829D3E1485B0B 5511ACC9F2BECD69 77E6D18B71ACA57E
        5B86DE50AB75F4FB 4FF4ED8F71CB3EA8 9C6B255041436845 AED5C751BE290B84 FA2A161B7CC6C129
        CA6FC42824967C8E 330BEA595FC747BE EBA860E3DD836B96 635FD9ED8EC9A474 9CE501EA3CE551A8
        """,
    'rho': """
        AF463273CA4D877D 5F3FBF099D8413A1 8A315CF6E73769EE 49E794C02ABBCDA3 A012D11DB7E9E584
        522354EA1F41927F 4099EBBB4F6DD594 41A3F27B2788089C 69DDCC9DF9F57000 426B0860F854687A
        A9C8A1AD4DDC657D B3D56FC1CDBD660B 42D85BE7C14E9F0A 93E57D9AD2AA2359 D652BF3BF368C5B8
        EBE9F6B70DBCA156 67D509FE9DB1EE39 92A820A1B422CE35 EA37C5217095DAB8 2A161B7CC6C129FA
        10A09259F23B29BF CC2FA9657F1D1EF8 DD750C1C7BB06D72 74635FD9ED8EC9A4 407A8F39546A2739
        """,
    'pi': """
        AF463273CA4D877D 4099EBBB4F6DD594 42D85BE7C14E9F0A EA37C5217095DAB8 407A8F39546A2739
        49E794C02ABBCDA3 426B0860F854687A A9C8A1AD4DDC657D 67D509FE9DB1EE39 DD750C1C7BB06D72
        5F3FBF099D8413A1 41A3F27B2788089C 93E57D9AD2AA2359 2A161B7CC6C129FA 10A09259F23B29BF
        A012D11DB7E9E584 522354EA1F41927F B3D56FC1CDBD660B 92A820A1B422CE35 74635FD9ED8EC9A4
        8A315CF6E73769EE 69DDCC9DF9F57000 D652BF3BF368C5B8 EBE9F6B70DBCA156 CC2FA9657F1D1EF8
        """,
    'chi': """
        AD0622374A4F8D77 E8BE6FBB7FFC9524 429051FFC524BA0B 4533F563FA905AFC 00E346B1514A77B9
        E067354D2F33C8A6 047E00326875E27A 31E8A5AD2FDC643F 6757993E9DBA6EB8 DF7D043CABF44D2A
        CD7BB2894DA630E0 69B1F01F23C9003E 8345FD9BE290235C 6509367CCB453BFA 1020D22BD03321A3
        01C6FA1C77558184 520B54CA2F431A4B D79630998431678B 12B8A0A5A643EA35 26425B3BE58EDBDF
        1C336FD4E53FEC56 40748C19F5615046 D254B67B8169DB10 E9F9A2258D9EC050 ADE3296C67DD0EF8
        """,
    'iota': """
        AD0622374A4F8D76 E8BE6FBB7FFC9524 429051FFC524BA0B 4533F563FA905AFC 00E346B1514A77B9
        E067354D2F33C8A6 047E00326875E27A 31E8A5AD2FDC643F 6757993E9DBA6EB8 DF7D043CABF44D2A
        CD7BB2894DA630E0 69B1F01F23C9003E 8345FD9BE290235C 6509367CCB453BFA 1020D22BD03321A3
        01C6FA1C77558184 520B54CA2F431A4B D79630998431678B 12B8A0A5A643EA35 26425B3BE58EDBDF
        1C336FD4E53FEC56 40748C19F5615046 D254B67B8169DB10 E9F9A2258D9EC050 ADE3296C67DD0EF8
        """,
}

# Example 2: state after rounds 0..k ('After iota' of round k).
KECCAK_EX2_AFTER_ROUND = {
    0: """
        AD0622374A4F8D76 E8BE6FBB7FFC9524 429051FFC524BA0B 4533F563FA905AFC 00E346B1514A77B9
        E067354D2F33C8A6 047E00326875E27A 31E8A5AD2FDC643F 6757993E9DBA6EB8 DF7D043CABF44D2A
        CD7BB2894DA630E0 69B1F01F23C9003E 8345FD9BE290235C 6509367CCB453BFA 1020D22BD03321A3
        01C6FA1C77558184 520B54CA2F431A4B D79630998431678B 12B8A0A5A643EA35 26425B3BE58EDBDF
        1C336FD4E53FEC56 40748C19F5615046 D254B67B8169DB10 E9F9A2258D9EC050 ADE3296C67DD0EF8
        """,
    1: """
        672D4E0D264DB92C 1A84BFA087E4FB8B B2EC97619AECE89C CEBE13CAC549DE27 54FC146106194E84
        8B23BDEA5F2F1AFE 8E8C9267B1555861 3C46FF954DED256C C86854F37CD6FBB0 03244A0A2C4D75B8
        7D7792A1F6691ECB 3FD486E0A64AF877 67B7914C6C118AB9 81276E3FC0DFC5A2 8C3A2CC762ED79F4
        3874CB197C10E7F9 D291F63B2EC58375 0BE9C36C033DEF63 5347708E98801482 0F173D486F940319
        2B633DFC491DD734 3CED4BFDF9681BDF 310AD7CBBDB65F99 BC0BB0F982245E36 491ECBD0E4A725C5
        """,
    2: """
        C1C6A2F1696A84BC CCAE2AD5FCEAA094 709CDB89E6827F69 CF8502CD29577FD2 8150B4D331AEE95F
        D85C2EAA40187EE8 43DBBE0C4C06E63C 5E63FAA4DAF6CD8A DC7C92E2873C95B4 9739B23D0EA3B935
        EEAFE43E4DF93B64 4235BBBB518CF60B 9284A174F0DADB16 59FC9A6A55313133 51681169C53C3328
        A7AFF3E947AEF6BA 201D89C9C8010E51 B36DD6596D6BB009 133C44795C6F0B0B AC542A551701AA20
        CBE99AF8AB9C8FB2 791B20D5507FAAA2 839477F02C55592A 660EC205385DC468 71E839816106C5F0
        """,
    3: """
        F1F7E99BB752608F E53ABC28D9CD8DFC EF3D1A4049934BBA B7FBB26A4815780C E9DC6FC7A0E4C37E
        8BEDDA621FF7C281 DCD89BC9974AEFC2 2DF84DBB3B889E3A 1430960D4B6F196A 8927BCB26D7B40A0
        F4B241905F243240 96BD2EEF54320B2C 33DAC91325C18924 110405194A2E77A5 4B567C3DAE1DAACB
        173DBE60957812D0 C52FA9E420C2E4F1 903159A1419B7A5C 200A1D52425AA422 7BDFF77D702FC15E
        F6A464C12C158C74 6BB6C25EF9B96054 37BB3EAC5CC79C29 4BA9424B1FB59C73 ECA30B1FDD142CDD
        """,
    11: """
        32BDB29984DD0136 D007CCF81390F00D 255F7D637DE64FAB 1E68C397BD8851B5 42650061D07BBD1D
        8904BC882D358CD1 59861999977D5625 3F58254B6D09B4B3 D16CA2220ABB1D9C 15F1DA0B4C67B71B
        D65DA28284EE7FB3 286B5606107B116B EC032B5ABF13AF97 B917B75D3D045350 673D897D6F671820
        28D0476C209DF0F0 0C6182B878CE4246 B83F00CD9DB7FEFF 1A1419F9A37814C8 29E53860A4AF645F
        6922DB03F53E69E8 36C30F12E8BB4606 1A30B0200C06B2E2 EA5DC7C65491C36A C1DE6860D0D10358
        """,
    12: """
        9096AE130B1FD899 A1D6B9F878FB944D 6C288BDEFDE1CA82 559E4FF8CB26F76B 2013C6D937A666CE
        B87D1A760FD734E2 135C1E3E9300A36D 1245E498292364A9 0B41BB1A073B3FFE D3B990E0EE639A13
        7A1CF3362D4724CF 284FC20CACA78DF8 DE322692C2B870E5 3B555E2C3F3F5CBF 173BE7F17B9A9DA3
        FA617D1BF5E495EB 3B29DE529E6BB11A 9B8F5053016C4EA1 CF373C6C6EA59EAB BE17F066766EF621
        B5A9C8CC044E1AC7 A0DEADB2C91EBC7B B98B739CB7F01953 9B67B50DA41DC919 C918C2727B01158A
        """,
    22: """
        B651C43A0473AEB8 17B17FA3BE010EA8 003BF869BE63BD2E 1C329530D06B9927 B10E910A10961F48
        0A2F07C85C10355D 701BD0131F22AD86 3128151D2057FEDC 756AB22962016607 D58237CB6D5F917E
        FCCBC4342466AB94 DE0A3D95BBA9685B 4E29B4ABFC1914DC 3437C0026B533EB5 578BD2F8DAE3DB5F
        BA1F6454C1486521 DB6B309F246ECA65 0F06335B93251FE5 2C688ACDE357856D 3A53EBD8BF860788
        BE4508CAD6F1AD6A BB406CE4DC458547 0BE5DC9864AD218E 0B6B54BA5342C697 A5CED12020F2F5C0
        """,
    23: """
        2D5C954DF96ECB3C 6A332CD07057B56D 093D8D1270D76B6C 8A20D9B25569D094 4F9C4F99E5E7F156
        F957B9A2DA65FB38 85773DAE1275AF0D FAF4F247C3D810F7 1F1B9EE6F79A8759 E4FECC0FEE98B425
        68CE61B6B9CE68A1 DEEA66C4BA8F974F 33C43D836EAFB1F5 E00654042719DBD9 7CF8A9F009831265
        FD5449A6BF174743 97DDAD33D8994B40 48EAD5FC5D0BE774 E3B8C8EE55B7B03C 91A0226E649E42E9
        900E3129E7BADD7B 202A9EC5FAA3CCE8 5B3402464E1C3DB6 609F4E62A44C1059 20D06CD26A8FBF5C
        """,
}

# Example 1: output state printed as 200 bytes (lane byte order check).
KECCAK_EX1_OUTPUT_BYTES = ("e7dde140798f25f18a47c033f9ccd584eea95aa61e2698d54d49806f304715bd"
                           "57d05362054e288bd46f8e7f2da497ffc44746a4a0e5fe90762e19d60cda5b8c"
                           "9c05191bf7a630ad64fc8fd0b75a933035d617233fa95aeb0321710d26e6a6a9"
                           "5f55cfdb167ca58126c84703cd31b8439f56a5111a2ff20161aed9215a63e505"
                           "f270c98cf2febe641166c47b95703661cb0ed04f555a7cb8c832cf1c8ae83e8c"
                           "14263aae22790c94e409c5a224f94118c26504e72635f5163ba1307fe944f675"
                           "49a2ec5c7bfff1ea")

# ---- NIST CAVP, SHA-512 byte-oriented vectors (SHAVS) ---------------------
# SHA512ShortMsg.rsp and SHA512LongMsg.rsp (CAVS 11.0): (message, digest).
# SHA512Monte.rsp (CAVS 11.1): seed and the first checkpoints.
SHA512_SHORT_MSG = [
    # Len = 0 bits (0 bytes)
    ("",
     ("cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce"
      "47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e")),
    # Len = 8 bits (1 bytes)
    ("21",
     ("3831a6a6155e509dee59a7f451eb35324d8f8f2df6e3708894740f98fdee2388"
      "9f4de5adb0c5010dfb555cda77c8ab5dc902094c52de3278f35a75ebc25f093a")),
    # Len = 16 bits (2 bytes)
    ("9083",
     ("55586ebba48768aeb323655ab6f4298fc9f670964fc2e5f2731e34dfa4b0c09e"
      "6e1e12e3d7286b3145c61c2047fb1a2a1297f36da64160b31fa4c8c2cddd2fb4")),
    # Len = 24 bits (3 bytes)
    ("0a55db",
     ("7952585e5330cb247d72bae696fc8a6b0f7d0804577e347d99bc1b11e52f3849"
      "85a428449382306a89261ae143c2f3fb613804ab20b42dc097e5bf4a96ef919b")),
    # Len = 32 bits (4 bytes)
    ("23be86d5",
     ("76d42c8eadea35a69990c63a762f330614a4699977f058adb988f406fb0be8f2"
      "ea3dce3a2bbd1d827b70b9b299ae6f9e5058ee97b50bd4922d6d37ddc761f8eb")),
    # Len = 56 bits (7 bytes)
    ("b39f71aaa8a108",
     ("258b8efa05b4a06b1e63c7a3f925c5ef11fa03e3d47d631bf4d474983783d8c0"
      "b09449009e842fc9fa15de586c67cf8955a17d790b20f41dadf67ee8cdcdfce6")),
    # Len = 64 bits (8 bytes)
    ("6f8d58b7cab1888c",
     ("a3941def2803c8dfc08f20c06ba7e9a332ae0c67e47ae57365c243ef40059b11"
      "be22c91da6a80c2cff0742a8f4bcd941bdee0b861ec872b215433ce8dcf3c031")),
    # Len = 120 bits (15 bytes)
    ("9bcd5262868cd9c8a96c9e82987f03",
     ("2e07662a001b9755ae922c8e8a95756db5341dc0f2e62ae1cf827038f33ce055"
      "f63ad5c00b65391428434ddc01e5535e7fecbf53db66d93099b8e0b7e44e4b25")),
    # Len = 128 bits (16 bytes)
    ("cd67bd4054aaa3baa0db178ce232fd5a",
     ("0d8521f8f2f3900332d1a1a55c60ba81d04d28dfe8c504b6328ae787925fe018"
      "8f2ba91c3a9f0c1653c4bf0ada356455ea36fd31f8e73e3951cad4ebba8c6e04")),
    # Len = 136 bits (17 bytes)
    ("6ba004fd176791efb381b862e298c67b08",
     ("112e19144a9c51a223a002b977459920e38afd4ca610bd1c532349e9fa7c0d50"
      "3215c01ad70e1b2ac5133cf2d10c9e8c1a4c9405f291da2dc45f706761c5e8fe")),
    # Len = 248 bits (31 bytes)
    ("f2c66efbf2a76c5b041860ea576103cd8c6b25e50eca9ff6a2fa88083fe9ac",
     ("7978f93ef7ed02c4a24abecba124d14dd214e1492ff1e168304c0eab89637da0"
      "f7a569c43dc4562bdb9404a018b6314fe0eebaccfb25ba76506aa7e9dcd956a7")),
    # Len = 256 bits (32 bytes)
    ("8ccb08d2a1a282aa8cc99902ecaf0f67a9f21cffe28005cb27fcf129e963f99d",
     ("4551def2f9127386eea8d4dae1ea8d8e49b2add0509f27ccbce7d9e950ac7db0"
      "1d5bca579c271b9f2d806730d88f58252fd0c2587851c3ac8a0e72b4e1dc0da6")),
    # Len = 264 bits (33 bytes)
    (("9f8c49320af9370cd3db20e9b50d3eaa59a6232d7a86fb7d472f124508d7968b"
      "05"),
     ("81b002f15c4d48be8517f7ed89df302fb1435c9435efefed58f3eb8ea1191062"
      "3f1eb9028a66e02121a7f08a7c604226f2324f483e91548dbbd2c441ab704ce5")),
    # Len = 440 bits (55 bytes)
    (("ec2a92e47f692b53c1355475c71ceff0b0952a8b3541b2938270247d44e7c5cc"
      "04e17236b353da028674eab4047d89ec5dad868cfd91ce"),
     ("c83aca6147bfcbbc72c377efa8d53654ba0830c5a6a89e1d2a19b713e68fb534"
      "640deb833ca512247166dd273b5897e57d526f88eef58f6ff97baee0b4ee5644")),
    # Len = 448 bits (56 bytes)
    (("c99e31ad4e23ac68e15e605d0b02437f8147c44f5445a55b68a10905276cce86"
      "76481c33e8cd3efe322bb13fe0107bb546ccbec7b8b38d10"),
     ("52992d45a88221d972958e9f2854adaa9a21d2bf7051e1f1019ae78004da50c5"
      "b55c144a02afffe539d753949a2b056534f5b4c21f248a05baa52a6c38c7f5dd")),
    # Len = 456 bits (57 bytes)
    (("9aa3e8ad92777dfeb121a646ce2e918d1e12b30754bc09470d6da4af6cc9642b"
      "012f041ff046569d4fd8d0dccfe448e59feefc908d9ad5af6f"),
     ("994d1cda4de40aff4713237cf9f78f7033af83369ac9c64e504091ea2f1caff6"
      "c5152d6a0c5608f82886c0093b3d7fbadd49dfd1f9e0f85accf23bc7dad48904")),
    # Len = 504 bits (63 bytes)
    (("ebb3e2ad7803508ba46e81e220b1cff33ea8381504110e9f8092ef085afef84d"
      "b0d436931d085d0e1b06bd218cf571c79338da31a83b4cb1ec6c06d6b98768"),
     ("f33428d8fc67aa2cc1adcb2822f37f29cbd72abff68190483e415824f0bcecd4"
      "47cb4f05a9c47031b9c50e0411c552f31cd04c30cea2bc64bcf825a5f8a66028")),
    # Len = 512 bits (64 bytes)
    (("c1ca70ae1279ba0b918157558b4920d6b7fba8a06be515170f202fafd36fb7f7"
      "9d69fad745dba6150568db1e2b728504113eeac34f527fc82f2200b462ecbf5d"),
     ("046e46623912b3932b8d662ab42583423843206301b58bf20ab6d76fd47f1cbb"
      "cf421df536ecd7e56db5354e7e0f98822d2129c197f6f0f222b8ec5231f3967d")),
    # Len = 520 bits (65 bytes)
    (("d3ddddf805b1678a02e39200f6440047acbb062e4a2f046a3ca7f1dd6eb03a18"
      "be00cd1eb158706a64af5834c68cf7f105b415194605222c99a2cbf72c50cb14"
      "bf"),
     ("bae7c5d590bf25a493d8f48b8b4638ccb10541c67996e47287b984322009d27d"
      "1348f3ef2999f5ee0d38e112cd5a807a57830cdc318a1181e6c4653cdb8cf122")),
    # Len = 552 bits (69 bytes)
    (("2249d698c4d807a8e7b4de21c485738959a0d67e5d2ca6f77983dfccb5dbf479"
      "31261e1f1537f3cbca253afb6bf4fe5e7672e1dcc860b3d6c8d243afe2d9758b"
      "375e955692"),
     ("6af44563fc468d51182f6c3be58d45932af1d985c6f283976c91a9ff421f383f"
      "e21dc7322f397ccead583e26b3e3fda067976a7f34665df25a2ced7b4b09cdec")),
    # Len = 800 bits (100 bytes)
    (("52167de2d6c502d99fa10c27b2ab6203bdebc2cafbbfdef15872a43dd610c236"
      "2f796ad9bcb5528d95870058fa454453f1e6065b315d410a3f2650e5d71e69d7"
      "8d9767dfb4accc057fd2069266b0f180cb319e30ded7535bbe52d24be151de4b"
      "b598fc5c"),
     ("25cb3ed3983a91b4cf37a65193916c5e3e211b63e943e2f7b50a85d349a463b9"
      "41aad33eff16561bdfdc92fda06a4e1d94b162de48f06d3c626940b31020925f")),
    # Len = 880 bits (110 bytes)
    (("10f2be77a4055771a67007cd8630e3230e38288499cb160380290174d66da574"
      "55b6baaa9785c84c8a663de41ed3bd544055b9170cec43cb3eb120eceaba1fe3"
      "6e3eaa3fa4f99b425cd2519f09bc0282bada52d14ce625b1ded3b24d86b1dad3"
      "42d2b7be322b775b04fc6b86afb4"),
     ("a872fa33d463b3343cec57c20c66979c33e1ad067bfc703454696aab5dd0003b"
      "c194318f4a8ebbc74503feb7211a472dadee991efe3e38f21a1310f8a76eac80")),
    # Len = 888 bits (111 bytes)
    (("324533e685f1852e358eea8ea8b81c288b3f3beb1f2bc2b8d3fdbac318382e3d"
      "7120de30c9c237aa0a34831deb1e5e060a7969cd3a9742ec1e64b354f7eb290c"
      "ba1c681c66cc7ea994fdf5614f604d1a2718aab581c1c94931b1387e4b7dc736"
      "35bf3a7301174075fa70a9227d85d3"),
     ("3b26c5170729d0814153becb95f1b65cd42f9a6d0649d914e4f69d938b5e9dc0"
      "41cd0f5c8da0b484d7c7bc7b1bdefb08fe8b1bfedc81109345bc9e9a399feedf")),
    # Len = 896 bits (112 bytes)
    (("518985977ee21d2bf622a20567124fcbf11c72df805365835ab3c041f4a9cd8a"
      "0ad63c9dee1018aa21a9fa3720f47dc48006f1aa3dba544950f87e627f369bc2"
      "793ede21223274492cceb77be7eea50e5a509059929a16d33a9f54796cde5770"
      "c74bd3ecc25318503f1a41976407aff2"),
     ("c00926a374cde55b8fbd77f50da1363da19744d3f464e07ce31794c5a61b6f9c"
      "85689fa1cfe136553527fd876be91673c2cac2dd157b2defea360851b6d92cf4")),
    # Len = 904 bits (113 bytes)
    (("9159767275ba6f79cbb3d58c0108339d8c6a41138991ab7aa58b14793b545b04"
      "bda61dd255127b12cc501d5aaad476e09fa14aec21626e8d57b7d08c36cdb79e"
      "ea314bdd77e65779a0b54eab08c48ceb976adf631f4246a33f7ef896887ea8b5"
      "dfa2087a225c8c180f8970696101fc283b"),
     ("3cd3380a90868de17dee4bd4d7f90d7512696f0a92b2d089240d61a9d20cd3af"
      "094c78bf466c2d404dd2f662ec5f4a299be2adeadf627b98e50e1c072b769d62")),
    # Len = 952 bits (119 bytes)
    (("36af17595494ef793c42f48410246df07d05936a918afe74cd005e537c586b28"
      "43701f5df8952242b74586f83339b48f4ba3a66bdeb457ecdf61784eac6765cd"
      "9b8c570dd628dbba6ae5836b9ac3dbcd795f9efdb8742a35bca232abf36eb3b6"
      "698b2933965802277ba953a6edcacaf330c1e4e8c7d45f"),
     ("158bfc348a30b4fabbe355a7d44bdc2122a4c850444c03f289003ce01bfc1ebf"
      "3ecc0febb6a8ff523d25db7681b05bdce048d11943ab476c1967cf6556c4a120")),
    # Len = 960 bits (120 bytes)
    (("42d66edc5f22e0c13c25504c5101a5d172d2db7209e461efa323c0bfaed27e5f"
      "808042ea9c3838ea31f9b76de465225ccfbd0c09ca0d9f07e9a43e3e46c7693e"
      "00a7e1d483900ddb0a629d5563456dbbf299ac91f92c3d3c17b05d180e6c87c6"
      "c93194c39d90273fcf4a482c56084f95e34c04311fa80438"),
     ("061afb119a3c60876e04c10f12ad0f4b977593dc5a2d21096a57e7d3f7d4d44f"
      "def934b2c17d7530674e4f4a1c176dbdcc54811a22e1b8712e4192fc2d4bf8e8")),
    # Len = 1016 bits (127 bytes)
    (("c13e6ca3abb893aa5f82c4a8ef754460628af6b75af02168f45b72f8f09e45ed"
      "127c203bc7bb80ff0c7bd96f8cc6d8110868eb2cfc01037d8058992a6cf2effc"
      "bfe498c842e53a2e68a793867968ba18efc4a78b21cdf6a11e5de821dcabab14"
      "921ddb33625d48a13baffad6fe8272dbdf4433bd0f7b813c981269c388f001"),
     ("6e56f77f6883d0bd4face8b8d557f144661989f66d51b1fe4b8fc7124d66d9d2"
      "0218616fea1bcf86c08d63bf8f2f21845a3e519083b937e70aa7c358310b5a7c")),
    # Len = 1024 bits (128 bytes)
    (("fd2203e467574e834ab07c9097ae164532f24be1eb5d88f1af7748ceff0d2c67"
      "a21f4e4097f9d3bb4e9fbf97186e0db6db0100230a52b453d421f8ab9c9a6043"
      "aa3295ea20d2f06a2f37470d8a99075f1b8a8336f6228cf08b5942fc1fb4299c"
      "7d2480e8e82bce175540bdfad7752bc95b577f229515394f3ae5cec870a4b2f8"),
     ("a21b1077d52b27ac545af63b32746c6e3c51cb0cb9f281eb9f3580a6d4996d5c"
      "9917d2a6e484627a9d5a06fa1b25327a9d710e027387fc3e07d7c4d14c6086cc")),
]

SHA512_LONG_MSG = [
    # Len = 1816 bits (227 bytes)
    (("4f05600950664d5190a2ebc29c9edb89c20079a4d3e6bc3b27d75e34e2fa3d02"
      "768502bd69790078598d5fcf3d6779bfed1284bbe5ad72fb456015181d9587d6"
      "e864c940564eaafb4f2fead4346ea09b6877d9340f6b82eb1515880872213da3"
      "ad88feba9f4f13817a71d6f90a1a17c43a15c038d988b5b29edffe2d6a062813"
      "cedbe852cde302b3e33b696846d2a8e36bd680efcc6cd3f9e9a4c1ae8cac10cc"
      "5244d131677140399176ed46700019a004a163806f7fa467fc4e17b4617bbd76"
      "41aaff7ff56396ba8c08a8be100b33a20b5daf134a2aefa5e1c3496770dcf6ba"
      "a4f7bb"),
     ("a9db490c708cc72548d78635aa7da79bb253f945d710e5cb677a474efc7c65a2"
      "aab45bc7ca1113c8ce0f3c32e1399de9c459535e8816521ab714b2a6cd200525")),
    # Len = 2608 bits (326 bytes)
    (("d5e378ae9fc2648f4a13bbec4b0935afb4f822f5fe0d5063053d2fbd547b33b4"
      "a32e7a009ee2afafe83d2ebd603568e4a38189b5d24d59e8953260f15f654ed4"
      "f42f9a39299d68c3eb78b09e83779d5718b433f1765d35350eac46493d194e84"
      "d1ce1f81c95b59725cab8ab73d369ab01e7967cf73a3acf1789227ee75fdfb6e"
      "40f353ff0484486542be053115db2896bab86c774f8985c4dbcc4c078f7b1c3a"
      "4c867cdc6580fe44a598673494cc0fb1f6598b1295768a584041fdbd14fa7b90"
      "fa6fe33f71b743b68e23f8e7407217aad9440cc8cad28152aedb82388be2de16"
      "5496d051b292de6303460273a43508296b6237c07804335d2e81229f7c9a0e77"
      "61e38a3aaf7799f40fe9cb00457ea9d5b59953232676681fc71b261a6f8cd359"
      "293f5b21f0cf3a11b7f49cb5adb3c357bed2aa185d8fe8408192d6d3ed1ff465"
      "b590892efe03"),
     ("a70c75b9b1f0ac2ed2c2797763ac9a6601d95f46889b00fc3ddae4d0ac692375"
      "0a108d79eb764e77ac07b7cb5c01cb4b3747dcf69ba3b35c51fb995da2632e70")),
]

SHA512_MONTE_SEED = ("5c337de5caf35d18ed90b5cddfce001ca1b8ee8602f367e7c24ccca6f893802f"
                     "b1aca7a3dae32dcd60800a59959bc540d63237876b799229ae71a2526fbc52cd")
SHA512_MONTE_CHECKPOINTS = [
    ("ada69add0071b794463c8806a177326735fa624b68ab7bcab2388b9276c036e4"
     "eaaff87333e83c81c0bca0359d4aeebcbcfd314c0630e0c2af68c1fb19cc470e"),
    ("ef219b37c24ae507a2b2b26d1add51b31fb5327eb8c3b19b882fe38049433dbe"
     "ccd63b3d5b99ba2398920bcefb8aca98cd28a1ee5d2aaf139ce58a15d71b06b4"),
    ("c3d5087a62db0e5c6f5755c417f69037308cbce0e54519ea5be8171496cc6d18"
     "023ba15768153cfd74c7e7dc103227e9eed4b0f82233362b2a7b1a2cbcda9daf"),
]

# ---- NIST CAVP, SHAKE256 byte-oriented vectors (SHA3VS, CAVS 19.0) --------
# SHAKE256ShortMsg.rsp / SHAKE256LongMsg.rsp ([Outputlen = 256]): (message,
# output); SHAKE256VariableOut.rsp and SHAKE256Monte.rsp below.
SHAKE256_SHORT_MSG = [
    # Len = 0 bits (0 bytes)
    ("",
     "46b9dd2b0ba88d13233b3feb743eeb243fcd52ea62b81b82b50c27646ed5762f"),
    # Len = 8 bits (1 bytes)
    ("0f",
     "aabb07488ff9edd05d6a603b7791b60a16d45093608f1badc0c9cc9a9154f215"),
    # Len = 16 bits (2 bytes)
    ("0dc1",
     "8e2df9d379bb034aee064e965f960ebb418a9bb535025fb96427f678cf207877"),
    # Len = 24 bits (3 bytes)
    ("21eda6",
     "f7d02b4512be5ddcc25d148c71664dfd34e16abea26d6e7287f45e08ed6fcd87"),
    # Len = 32 bits (4 bytes)
    ("4a71964b",
     "7b7e12d2a520e232fde6c41dbbb2b8b74c2912fb3f15404f7304fe46691430c9"),
    # Len = 56 bits (7 bytes)
    ("c5f21a07c8dcc7",
     "569b1a32fb32f1a7fc9e35baad586f35645c5366cdfdfed394076bcd1801371f"),
    # Len = 64 bits (8 bytes)
    ("587cb398fe82ffda",
     "54f5dddb85f62dba7dc4727d502bdee959fb665bd482bd0ce31cbdd1a042e4b5"),
    # Len = 128 bits (16 bytes)
    ("fdbff9bf3fb910a92d25651ebbfba365",
     "89f2373e131a899b4ba27f6da606716a5e289fd530ae066bb8b11dc023dacbd6"),
    # Len = 248 bits (31 bytes)
    ("e34dc086477a69347b5d85d9e55345edeeb3fa8e05bc361ae6cb2efc1cd163",
     "9076470ef15650b75ad59780b1fb4265838dc3ed50bf17f2dd714d90a02d7d8e"),
    # Len = 256 bits (32 bytes)
    ("ef896cdcb363a6159178a1bb1c993946c50402095cdaea4fd4d419aa47321c88",
     "7abba4e8b8dd766bbabe98f8f169cb6208674de19a51d73c92b7dc04a4b5ee3d"),
    # Len = 264 bits (33 bytes)
    (("e68f2d765dd0b3450db083cae53026b4ea16de9186ed97ec224d1ab9f75c84f4"
      "20"),
     "ab529d0f7c3fb4b49e88731d2e4c1a226d328e44829e749dff710970a20d2520"),
    # Len = 504 bits (63 bytes)
    (("4cfaa2476de91c3f90a1d4b3a06cbd5ac9dcf075945ff40554133d3bc2dcf230"
      "e4b3f4d633411b2ffcacc3bc07b1873977f34c87f19282ea0000a674b8100b"),
     "073e19f7f56765e8e0170fb597f9533abada89b805d232d69c94354d1fcfbe15"),
    # Len = 512 bits (64 bytes)
    (("47c7ee751edd20c2cb566df1403463a3a8c0b21f401698f468e39ce0fa6e7ff1"
      "dad570eace997fe8905d544522eb21bd6278d5cfde299366e9d662b9ef0d2670"),
     "701bfdc149b2c347220b8ee2b64ff6b3fac3cb0d7c0286f3f476428973e5a40b"),
    # Len = 520 bits (65 bytes)
    (("bad35d2ea10451fd7312e54d9f763810615475184eb9027f2e21fa46093cde3c"
      "85f187216fa45f0c97b4d10ae148fe08994cb2f307b6961d430ea7e34b2de68d"
      "53"),
     "4f69084fd2a053ae48d6d4f3f72246d809037125adaa8b7b4a5e1a2cddb8d1d5"),
    # Len = 552 bits (69 bytes)
    (("ab5e11c2c23f3f07c9e26895415ba0466582afcbf8a300fc4077ba6d53532c42"
      "91a06e44728b453482fd485fa9ce4172b036d690018469a2ae123ceeb578d9ec"
      "8300b154b6"),
     "c9b66f15d7075cfe2b90f89547992986f9f4d474ca850cbe9c40ad67cc9161c2"),
    # Len = 568 bits (71 bytes)
    (("1c744d510d32a4def51189ecd2a395c2a0893628d7e6517936142cfabf968c6a"
      "034e1ada9ff00bbf07b7004682572be77632b43de6ce5b1539875ec53d819dbf"
      "6be1fc656d3d09"),
     "b21bc7a5fde822156691e11a8ba2040f2e9b403b0917c7ab860a28028807bb98"),
    # Len = 576 bits (72 bytes)
    (("7ab4397ee403e4fa3cdbfffe6ec208df114c5868513db11e2cee16c8e2f2df2f"
      "a6f80c239af3dabefa7994ed9769ac2b88d10a901c0fa40b13364fec0db3d4a5"
      "423047d174b89475"),
     "4eafd931f87c4130050a4d7e3bb35fcd09e1879762163690abe37d6b414e5c59"),
    # Len = 584 bits (73 bytes)
    (("584f50ce46b7c5a84b2f0c0ac03ae9b72ecfc976120aa964dd22348e3f9b2bbf"
      "a5bd0a970144e9ddc959a01642c12e57fe555c57a3b29dc8ef3139e8f91aaa53"
      "ba14461bcbe01f1a70"),
     "045554935fc713437e807694d3eef5e957f1e87a3b5ebb79b1370519af5b4c15"),
    # Len = 800 bits (100 bytes)
    (("14c841c658760d214eb913e1d9fae8d7d1c3c1f9a6d0341346683eb941ee8a80"
      "2bb3f06e430e3dae51af3288d530cf915dedd11f3716790048028f6c3cc6a737"
      "e007a34bcb0c7956441736c060f643e06bd5e76f89560e3d9de449226a441538"
      "f96c6047"),
     "f2d42ad8670de2f17e8b7faf1f080b07a72be6a3e86bf32961209e78a05c8b0f"),
    # Len = 1016 bits (127 bytes)
    (("cae515304b67bbf41ae3c599212f01c53b47ca783b422d9c2f844adde634fcba"
      "aff1bcdc64c0c2a7a7496e106c44e7aeb82e0213a740655cb4a1ec0735db4b05"
      "258cf510265e3d3fc638bbd802b6f26509d72f0a42f6b4158ffc16d7d4d3e4b3"
      "b31e89749a2774e30018769aeb6f6b13b6ac1c04114c64ef49fb92f4ca060f"),
     "0be4f9bff900ef43c9894e0efd761321067055ade3b6d107e07a4058e8764688"),
    # Len = 1024 bits (128 bytes)
    (("a1081e6994e1ba5f966f958158ee0a8c369b73466918f7077e11a3790620bc9c"
      "90ee540dd49c1c87316179aa5a74dafb4c2c482d83653f9f6d27bc251f82970b"
      "47bc135761afb44f2e6c503b6224696c9d49ba7d9df010d9b27a5b7d72730500"
      "adba82d0a4e229be75d942bddac760e61339f1c0217f13bf3552a90a68bbae81"),
     "8e9a891592447c3461ea0a2373caeeaac8b9e5a981d7742f692779379964eb1e"),
    # Len = 1072 bits (134 bytes)
    (("719effd45ed3a8394bf6c49b43f35879176a598601bd6f598867f966a38f512d"
      "21dc51b1488c162cbdc00301a41a09f2078a26937c652cfe02b8c4c92ddbb235"
      "83495ba825ae845eb2425c5b6856bda48c2cafae0c0c2e1764942d94be50da2b"
      "5d8b24a23b647a37f124d691d8cefbf76ef8fbc0fbdafb0a74a53aaf9f165075"
      "784ab485d4d4"),
     "6881babbb48e9eea72eeb3524db56e4efc323f3350b6be3cdb1f9c6826e359da"),
    # Len = 1080 bits (135 bytes)
    (("362f1eb00b37a9613b1ae82b90452579d42f8b1f9ede95f86badc6cdf04c9b79"
      "af08be4bc94d7cac136979026b92a2d44d2b642ea1431b47d75fce61367919f1"
      "71486a007cc271d19de0d1c4c6a11c7a2251fe3aee0bb8938a7dd043d0eb0758"
      "a4768c95cc9f6f1703075839487879b47c29c10b2c3e5326ac8f363c65aa4ef7"
      "6f1b8bd363eb60"),
     "c6ce60c1852ea780ed845aac4ca6a30e09f5c0064c9675865178717cfeb1dc97"),
    # Len = 1088 bits (136 bytes)
    (("d8f12b97f81d47aebbfb7314ff04172cf2be71c3778e238bcccdeecb691fbd54"
      "2b00e5b7b1a0abb507f107f781fea700ea7e375fdea9e029754a0ea62216774b"
      "da3c59e8783d022360fe9625621c0d93e27f7bc03632942150716f019d048a75"
      "2ccc0f93139c55df0f4aaa066a0550cf22e8c54e47d0475ba56b9842a392ffbc"
      "6bd98f1e4b64abd1"),
     "e2e1c432dd07c2ee89a78f31211c92eeb5306c4fa4db93c4e5cd43080d6079e4"),
    # Len = 1096 bits (137 bytes)
    (("a10d05d7e51e75dc150f640ec4722837220b86df2a3580ca1c826ec22ea25097"
      "7e8663634cc4f212663e6f22e3ffc2a81465e194b885a1356fcbcc0072e1738d"
      "80d285e21c70a1f4f5f3296ba6e298a69f3715ff63be4850f5be6cb68cdba594"
      "8e3b94dbbce82989aa75b97073e55139aac849a894a71c2294a2776ce6588fb5"
      "9007b8d796f434da6e"),
     "02f17bf86dc7b7f9c3fb96e4b3a10ca574cd0f8dedda50f3dda8008ce9e8fec9"),
    # Len = 1144 bits (143 bytes)
    (("353111e447fee6f0bd05d562f30626ab9fb06384a620c49034a5eb3c0bc6d1eb"
      "1b86015053e6041ab8ac1cd7b4633512b0a318bfe592e2da6eabb44aa2bead0b"
      "a238158c2ea5db56bd7342efccf9d7fe76b8a6af45e0ad594816915f65749054"
      "f1d1b7627e4355ecf4e3af72e4d0f5b51877751c6f110f57e86ce942fcef640c"
      "31d94e98ecc959238683cb28a3f178"),
     "11b27034db724b46882a3086815a835947d19322885e08595be271c511ef783d"),
    # Len = 1600 bits (200 bytes)
    (("d00df64c4bb9e2fd16fb6f9ca746d6cf162015ec7326e41a5d51e9b3d0792fed"
      "3f17d5bae34f03ec522e229d53304dcef105024ece941edeba410892846b2c7a"
      "1039ab82aa9750979a7bc70bf96d093bc3461b6f2d38f801380eccc286b56299"
      "6cfce06d4a98b245176bc4ae4006f45eb36cc71636185acdfe429c0a7d5fbb92"
      "7be7dc43685a0f40f185824ed102f57eeafe6d0d943e2d883564e233126f1eac"
      "648207ccafe651ce4f5169b35369f3e48f84771aedb2577b04fd0506ecef7230"
      "5055cacfc4435e38"),
     "67302648e0082254d8d342b4eb8070ef9a44e0fc55c3d9a3f20613e4824aff21"),
    # Len = 2040 bits (255 bytes)
    (("74c91ed7756b88ea869c0c569d87fcb04b8911639e8868a54786caa9f8753016"
      "104d86b97a5db076d2decb85caceb5dbcdfccfe5e25f017cd2abff168d26cb39"
      "cab87c36ee423d132f3198c9f953f5b090e95d37f8fd84d812f6074b0769f2fa"
      "e93285b698c438a8e3bdfc8af0bec417de7944bfa80bf8a04d8d69b22ee5ac41"
      "b7e22ba14189a81d37a97e3078b62af8fdd30be7bc3166bccc98adb1c94301f6"
      "7af1f55fb139955a731ea72359e3303b36213a16fa16d513d7405cb2a22af902"
      "7b876b58b07d0746df43fdcb5beeddc8d1885ab32c634d8fb5acfd1c7f439d4b"
      "02ca16f406fd07cbc3410a91467f3a871632f22818bfd717b00618572936f4"),
     "b5958bcadd7cd116e01c66a2b23533ef2b67685aae75f21fe8de658f51ed9ef5"),
    # Len = 2168 bits (271 bytes)
    (("a1ce406d09c02ca1c3cc97f25c9b96eeb9d3480c053b67edee201ce111f718ad"
      "b243e08cb1b96873b9a2e499bb82db379bf77d8c29e3362552cd835b9885051d"
      "bf07d4f0c9a21533255acfa8248afa30acced9d6301f8a0fdf40dc401c5a6081"
      "2fe3c3a13ac35a6a6ebaff267efc1c62541f05f104378b34fe40ce9987bc5262"
      "6a9c55a8ea64609ffc8b1d1bb3206853fcb4a8e58b73601b1594016dc0f40347"
      "d8fd330cc3cd4f5a3794d090aa3e078d0a536bbbaf1b324d8e051ec4af499ab8"
      "e1cd05d5ac464e24879fc18b4b9e2098e8c5f67a56762850cf5bcda73c053f3d"
      "edd3720b8c168583547297177e88dcfafcd1f02a6acd6bae425eb51c4f6e1c5f"
      "64d823b74d861d0106d7fb392b4363"),
     "a7835b81740824ebfc4a0dda40da7a70a66f9f3a8ea77cf857664ff955c5c5fe"),
    # Len = 2176 bits (272 bytes)
    (("104fefe89f08d15d36a2233f42a7defa917c5ad2642e06cac56d5cc51ad914ec"
      "fb7d984f4199b9cf5fa5a03bf69207b9a353a9681c9cf6437bea0c49d9c3e3db"
      "1f3fc76519c70c40cc1dfdd70a9c150943c272cf9eeb861f485f10100c8f4a3e"
      "259c6470501932782512225ba64d70b219cf9d5013a21d25d6d65062dcc6b3de"
      "b49d58b90d18933f118df70ff42c807ccc851233a34a221eca56b38971ef8584"
      "75488988794a975d3894633a19c1ae2f05e9b9c0756affd3cfe823ccf29228f6"
      "0fa7e025bc39a79943325126409460926b057a3fb28a1b098b938872883804fd"
      "2bc245d7fd6d29bcda6ca6198f2eff6ea7e03ef78133de8ba65fc8c45a688160"
      "719fa1e7646d878ea44c4b5c2e16f48b"),
     "46293a63c235750d58a24edca5ba637b96cae74325c6c8122c4155c0d15805e6"),
]

SHAKE256_LONG_MSG = [
    # Len = 2184 bits (273 bytes)
    (("dc5a100fa16df1583c79722a0d72833d3bf22c109b8889dbd35213c6bfce2058"
      "13edae3242695cfd9f59b9a1c203c1b72ef1a5423147cb990b5316a852666758"
      "94e2644c3f9578cebe451a09e58c53788fe77a9e850943f8a275f830354b0593"
      "a762bac55e984db3e0661eca3cb83f67a6fb348e6177f7dee2df40c4322602f0"
      "94953905681be3954fe44c4c902c8f6bba565a788b38f13411ba76ce0f9f6756"
      "a2a2687424c5435a51e62df7a8934b6e141f74c6ccf539e3782d22b5955d3baf"
      "1ab2cf7b5c3f74ec2f9447344e937957fd7f0bdfec56d5d25f61cde18c0986e2"
      "44ecf780d6307e313117256948d4230ebb9ea62bb302cfe80d7dfebabc4a51d7"
      "687967ed5b416a139e974c005fff507a96"),
     "2bac5716803a9cda8f9e84365ab0a681327b5ba34fdedfb1c12e6e807f45284b"),
    # Len = 3280 bits (410 bytes)
    (("16caf60da14b4fa9174a6d40c23cff93ed8fc9279990f749718db1500036ef22"
      "22498ffab86fa568a0611299e54e58d83281ac558d3f4d2541ee158b1c7d4d76"
      "dbffc64ae39925e3329f7fd894fa26fc1acdc22bc858a3438e1c55707a3f75ad"
      "2b33c48789937a24b34ddd85390611088cba3231b2a3a0a93e5d9a8780470fcf"
      "f92cb03811234a330db353283b3bc3036f9125efb3eaed613bfa0c59975cc2e5"
      "2c33b3e6e5123e1626190a4a0261e1f5ad9bc2ee34f331736b3bd26d274536f5"
      "ae90f5186c27fdd7e8c72972f64016e72d1d32b59b8715e5b867154b99cb140a"
      "668b9d560e2c307e3904d9297f9f07dfd7629ccc526e41c109c8fc7c53b60429"
      "3c6cd42933e77e11031a42f605485fe893b129bcbf705c0f45a4b087bfcead5c"
      "187ac1174322909a2d4f8b61f001c4074951000c4c550ed5564458f444dab8aa"
      "e2fe8daaa6a30d209fecddf2a893df46e18b4b4460e4011d23f01d4c49a4cc1c"
      "82405f6ac5339eac41385f3295c657ac43a72fed62e6daee94ef271638f292b8"
      "e18860de0699eb45fb7d3aa81f61d44158edd68ebc244451918b"),
     "21a48efd949c3f785179a0e340756a23f77d29a7625229a71a05731c7fbd5aa9"),
]

# (message, output length in bits, output); 256-bit messages.
SHAKE256_VARIABLE_OUT = [
    # COUNT = 0
    ("c61a9188812ae73994bc0d6d4021e31bf124dc72669749111232da7ac29e61c4", 16,
     "23ce"),
    # COUNT = 5
    ("29eade8f428791131625a160f95d97e8654f5ff93fdca8915c158c7344f2747a", 24,
     "6d0e57"),
    # COUNT = 30
    ("dc886df3f69c49513de3627e9481db5871e8ee88eb9f99611541930a8bc885e0", 64,
     "00648afbc5e65164"),
    # COUNT = 150
    ("dc886df3f69c49513de3627e9481db5871e8ee88eb9f99611541930a8bc885e0", 256,
     "00648afbc5e651649db1fd82936b00dbbc122fb4c877860d385c4950d56de7e0"),
    # COUNT = 310
    ("dc886df3f69c49513de3627e9481db5871e8ee88eb9f99611541930a8bc885e0", 512,
     ("00648afbc5e651649db1fd82936b00dbbc122fb4c877860d385c4950d56de7e0"
      "96d613d7a3f27ed8f26334b0ccc1407b41dccb23dfaa529818d1125cd5348092")),
    # COUNT = 665
    ("8d8001e2c096f1b88e7c9224a086efd4797fbf74a8033a2d422a2b6b8f6747e4", 1080,
     ("2e975f6a8a14f0704d51b13667d8195c219f71e6345696c49fa4b9d08e9225d3"
      "d39393425152c97e71dd24601c11abcfa0f12f53c680bd3ae757b8134a9c10d4"
      "29615869217fdd5885c4db174985703a6d6de94a667eac3023443a8337ae1bc6"
      "01b76d7d38ec3c34463105f0d3949d78e562a039e4469548b609395de5a4fd43"
      "c46ca9fd6ee29a")),
    # COUNT = 670
    ("dc886df3f69c49513de3627e9481db5871e8ee88eb9f99611541930a8bc885e0", 1088,
     ("00648afbc5e651649db1fd82936b00dbbc122fb4c877860d385c4950d56de7e0"
      "96d613d7a3f27ed8f26334b0ccc1407b41dccb23dfaa529818d1125cd5348092"
      "524366b85fabb97c6cd1e6066f459bcc566da87ec9b7ba36792d118ac39a4cce"
      "f6192bbf3a54af18e57b0c146101f6aeaa822bc4b4c9708b09f0b3bab41bcce9"
      "64d999d1107bd7c2")),
    # COUNT = 675
    ("8d8001e2c096f1b88e7c9224a086efd4797fbf74a8033a2d422a2b6b8f6747e4", 1096,
     ("2e975f6a8a14f0704d51b13667d8195c219f71e6345696c49fa4b9d08e9225d3"
      "d39393425152c97e71dd24601c11abcfa0f12f53c680bd3ae757b8134a9c10d4"
      "29615869217fdd5885c4db174985703a6d6de94a667eac3023443a8337ae1bc6"
      "01b76d7d38ec3c34463105f0d3949d78e562a039e4469548b609395de5a4fd43"
      "c46ca9fd6ee29ada5e")),
    # COUNT = 1235
    ("8d8001e2c096f1b88e7c9224a086efd4797fbf74a8033a2d422a2b6b8f6747e4", 1992,
     ("2e975f6a8a14f0704d51b13667d8195c219f71e6345696c49fa4b9d08e9225d3"
      "d39393425152c97e71dd24601c11abcfa0f12f53c680bd3ae757b8134a9c10d4"
      "29615869217fdd5885c4db174985703a6d6de94a667eac3023443a8337ae1bc6"
      "01b76d7d38ec3c34463105f0d3949d78e562a039e4469548b609395de5a4fd43"
      "c46ca9fd6ee29ada5efc07d84d553249450dab4a49c483ded250c9338f85cd93"
      "7ae66bb436f3b4026e859fda1ca571432f3bfc09e7c03ca4d183b741111ca048"
      "3d0edabc03feb23b17ee48e844ba2408d9dcfd0139d2e8c7310125aee801c61a"
      "b7900d1efc47c078281766f361c5e6111346235e1dc3832566")),
    # COUNT = 1240
    ("dc886df3f69c49513de3627e9481db5871e8ee88eb9f99611541930a8bc885e0", 2000,
     ("00648afbc5e651649db1fd82936b00dbbc122fb4c877860d385c4950d56de7e0"
      "96d613d7a3f27ed8f26334b0ccc1407b41dccb23dfaa529818d1125cd5348092"
      "524366b85fabb97c6cd1e6066f459bcc566da87ec9b7ba36792d118ac39a4cce"
      "f6192bbf3a54af18e57b0c146101f6aeaa822bc4b4c9708b09f0b3bab41bcce9"
      "64d999d1107bd7c27af989ebe1e104a35478df362ec4c9628f5ea29cc1164b92"
      "960d42068a59b1ddcb8875722ed59df36ea654d97b6bf39e5075121ec93f8a8e"
      "a6ee5d923997ae8a9d4c315a3d074e3ff83dfa26490f0f5b00a3cc3f110f0f76"
      "1eaac885a3d1eee302014191733bf77b4d58cca49258897d6460")),
]

SHAKE256_MONTE_MSG = "48a0321b3653e4e86446d00f6a036efd"
SHAKE256_MONTE_MIN_OUT_BITS = 16
SHAKE256_MONTE_MAX_OUT_BITS = 2000
# (output length in bits, output) after each block of 1000 iterations
SHAKE256_MONTE_CHECKPOINTS = [
    (1880,
     ("5410fae0cd08b96ac832890a2f4caa8b3d404e9f71e5ced4f3ef34f15c5aeb7b"
      "4c69ab3526dbe59118d5db5eb90f11a0e7f0aa5e10f648dace0e043732f6e948"
      "c8cf02af090107e3f9e8e07be835d0cdce6343248df67c669a36c6d3d0dcbbbb"
      "7b135e1751f0ed01f275840e0c20309d77ac802bbb0bf991373e7560ceb7d332"
      "bfad7ae827c97233fb79b66f01619759e2c7dab42a2ef3dfdda8a18fe4a4047d"
      "e51279f513b318e7765b52f71575df51e6720936f68d9e4cd8342a8aafdd82d4"
      "7a2f08de05956a5d97c6e72d2c77b74f5a3cc6d68c3549206c8485fc87b408e1"
      "9ebbb7f9c0027a492120bc")),
    (88,
     "c5e5196c3ab04ecc27892a"),
]

# ---- Keccak team, XKCP tests/TestVectors/ShortMsgKAT_SHAKE256.txt ---------
# (message, first 512 output bytes)
XKCP_SHAKE256_KAT = [
    # Len = 0 bits (0 bytes)
    ("",
     ("46b9dd2b0ba88d13233b3feb743eeb243fcd52ea62b81b82b50c27646ed5762f"
      "d75dc4ddd8c0f200cb05019d67b592f6fc821c49479ab48640292eacb3b7c4be"
      "141e96616fb13957692cc7edd0b45ae3dc07223c8e92937bef84bc0eab862853"
      "349ec75546f58fb7c2775c38462c5010d846c185c15111e595522a6bcd16cf86"
      "f3d122109e3b1fdd943b6aec468a2d621a7c06c6a957c62b54dafc3be87567d6"
      "77231395f6147293b68ceab7a9e0c58d864e8efde4e1b9a46cbe854713672f5c"
      "aaae314ed9083dab4b099f8e300f01b8650f1f4b1d8fcf3f3cb53fb8e9eb2ea2"
      "03bdc970f50ae55428a91f7f53ac266b28419c3778a15fd248d339ede785fb7f"
      "5a1aaa96d313eacc890936c173cdcd0fab882c45755feb3aed96d477ff96390b"
      "f9a66d1368b208e21f7c10d04a3dbd4e360633e5db4b602601c14cea737db3dc"
      "f722632cc77851cbdde2aaf0a33a07b373445df490cc8fc1e4160ff118378f11"
      "f0477de055a81a9eda57a4a2cfb0c83929d310912f729ec6cfa36c6ac6a75837"
      "143045d791cc85eff5b21932f23861bcf23a52b5da67eaf7baae0f5fb1369db7"
      "8f3ac45f8c4ac5671d85735cdddb09d2b1e34a1fc066ff4a162cb263d6541274"
      "ae2fcc865f618abe27c124cd8b074ccd516301b91875824d09958f341ef274bd"
      "ab0bae316339894304e35877b0c28a9b1fd166c796b9cc258a064a8f57e27f2a")),
    # Len = 552 bits (69 bytes)
    (("abc87763cae1ca98bd8c5b82caba54ac83286f87e9610128ae4de68ac95df5e3"
      "29c360717bd349f26b872528492ca7c94c2c1e1ef56b74dbb65c2ac351981fdb"
      "31d06c77a4"),
     ("cd85deb9da581af66c0b3125b697371f16ee34606a577ad6d8ac7e4e4a948b36"
      "c844ec9ea85eb168d7a5a1eb4b72a685aeb80a75075b420c9b53b963d960f7bc"
      "88dced631c734114c28b2e20f0657d9256ab01b1afedda9b9f85fd30d0de3b98"
      "db38ab90de60045836cfb12c41e5bdae57b937b637c11ed47f7de242eb9f72d3"
      "253c5d883b22333f181ffad60a541da08601791fc0f8d9f44a94b551b1a879b7"
      "6fc2b7931a2f4301d121a4e6861f5c84ffcb0139fc37268b5f8a4c3fd490809c"
      "f44f68bc1e665b369d4d74dd0bdea71ed0514c37a47124ce146927274d95067c"
      "80036ed9f1a5b0a10ae71e837a09dbe4dc358df4687392d99b2ace8beada9656"
      "677518b1607c8e13b7f45100bfcefa1d4e38b9bb29eb23d17b9cc66f70635a6c"
      "531be9cb890ae833cd7ce35498cc9f81c576493913bad87532b711f3f88873e9"
      "ed48a83b6b2c503e096a33f824feb4ccc702e82cef00fbd938ff9bef6e3f80e1"
      "49eb3433816a1d3fe7f005734192ccb5a8b0d7d43327b535547a9cc7f5fa286f"
      "9eac26e9e6a1cfb86db35831f75a9902a3e78f6bf9b4728836c81b3c614923c3"
      "ea88d6c5f55449a83eba0f5ff7b4f07084206d4590bf817c1feb43595462037a"
      "fb6969a91eeb963bd244adb1b654fc98a0b0be99029b3d5bdd69d2158939d677"
      "b054ce55decf0f332851e0a74eaf2bf3eb672d4cb1f467d0e5391f98501fec2e")),
    # Len = 1088 bits (136 bytes)
    (("b32d95b0b9aad2a8816de6d06d1f86008505bd8c14124f6e9a163b5a2ade55f8"
      "35d0ec3880ef50700d3b25e42cc0af050ccd1be5e555b23087e04d7bf9813622"
      "780c7313a1954f8740b6ee2d3f71f768dd417f520482bd3a08d4f222b4ee9dbd"
      "015447b33507dd50f3ab4247c5de9a8abd62a8decea01e3b87c8b927f5b08beb"
      "37674c6f8e380c04"),
     ("cc2eaa04eef8479cdae8566eb8ffa1100a407995bf999ae97ede526681dc3490"
      "616f28442d20da92124ce081588b81491aedf65caaf0d27e82a4b0e1d1cab238"
      "33328f1b8da430c8a08766a86370fa848a79b5998db3cffd057b96e1e2ee0ef2"
      "29eca133c15548f9839902043730e44bc52c39fadc1ddeead95f9939f220ca30"
      "0661540df7edd9af378a5d4a19b2b93e6c78f49c353343a0b5f119132b5312d0"
      "04831d01769a316d2f51bf64ccb20a21c2cf7ac8fb6f6e90706126bdae0611dd"
      "13962e8b53d6eae26c7b0d2551daf6248e9d65817382b04d23392d108e4d3443"
      "de5adc7273c721a8f8320ecfe8177ac067ca8a50169a6e73000ebcdc1e4ee633"
      "9fc867c3d7aeab84146398d7bade121d1989fa457335564e975770a3a00259ca"
      "08706108261aa2d34de00f8cac7d45d35e5aa63ea69e1d1a2f7dab3900d51e0b"
      "c65348a25554007039a52c3c309980d17cad20f1156310a39cd393760cfe58f6"
      "f8ade42131288280a35e1db8708183b91cfaf5827e96b0f774c45093b417aff9"
      "dd6417e59964a01bd2a612ffcfba18a0f193db297b9a6cc1d270d97aae8f8a3a"
      "6b26695ab66431c202e139d63dd3a24778676cefe3e21b02ec4e8f5cfd66587a"
      "12b44078fcd39eee44bbef4a949a63c0dfd58cf2fb2cd5f002e2b0219266cfc0"
      "31817486de70b4285a8a70f3d38a61d3155d99aaf4c25390d73645ab3e8d80f0")),
    # Len = 2040 bits (255 bytes)
    (("3a3a819c48efde2ad914fbf00e18ab6bc4f14513ab27d0c178a188b61431e7f5"
      "623cb66b23346775d386b50e982c493adbbfc54b9a3cd383382336a1a0b2150a"
      "15358f336d03ae18f666c7573d55c4fd181c29e6ccfde63ea35f0adf5885cfc0"
      "a3d84a2b2e4dd24496db789e663170cef74798aa1bbcd4574ea0bba40489d764"
      "b2f83aadc66b148b4a0cd95246c127d5871c4f11418690a5ddf01246a0c80a43"
      "c70088b6183639dcfda4125bd113a8f49ee23ed306faac576c3fb0c1e256671d"
      "817fc2534a52f5b439f72e424de376f4c565cca82307dd9ef76da5b7c4eb7e08"
      "5172e328807c02d011ffbf33785378d79dc266f6a5be6bb0e4a92eceebaeb1"),
     ("8a5199b4a7e133e264a86202720655894d48cff344a928cf8347f48379cef347"
      "dfc5bcffab99b27b1f89aa2735e23d30088ffa03b9edb02b9635470ab9f10389"
      "85d55f9ca774572dd006470ea65145469609f9fa0831bf1ffd842dc24acade27"
      "bd9816e3b5bf2876cb112232a0eb4475f1dff9f5c713d9ffd4ccb89ae5607fe3"
      "5731df06317949eef646e9591cf3be53add6b7dd2b6096e2b3fb06e662ec8b2d"
      "77422daad9463cd155204acdbd38e319613f39f99b6dfb35ca9365160066db19"
      "835888c2241ff9a731a4acbb5663727aac34a401247fbaa7499e7d5ee5b69d31"
      "025e63d04c35c798bca1262d5673a9cf0930b5ad89bd485599dc184528da4790"
      "f088ebd170b635d9581632d2ff90db79665ced430089af13c9f21f6d443a8180"
      "64f17aec9e9c5457001fa8dc6afbadbe3138f388d89d0e6f22f66671255b2107"
      "54ed63d81dce75ce8f189b534e6d6b3539aa51e837c42df9df59c71e6171cd49"
      "02fe1bdc73fb1775b5c754a1ed4ea7f3105fc543ee0418dad256f3f6118ea771"
      "14a16c15355b42877a1db2a7df0e155ae1d8670abcec3450f4e2eec9838f8954"
      "23ef63d261138baaf5d9f104cb5a957aea06c0b9b8c78b0d441796dc0350ddea"
      "bb78a33b6f1f9e68ede3d1805c7b7e2cfd54e0fad62f0d8ca67a775dc4546af9"
      "096f2edb221db42843d65327861282dc946a0ba01a11863ab2d1dfd16e3973d4")),
]


# ---- KMD-2048 regression values ------------------------------------------
# Produced by this implementation; they are NOT independently published.
# Before being recorded, every value was also reproduced by
# kmd2048_by_spec() running on the independent reference primitives
# (ref_shake256 / ref_sha512) defined above.  Keys are the message in hex.
KMD2048_REGRESSION = {
    "": (
        "c44be042aea64dfd9e994c98953ef7d39421819172e401c2c70db815d89547a2"
        "167e7151c4221902ce60453db99c4290a8d3ea3c7f9926997e3ae6f9016c07fc"
        "3b2183b46d6b88605de25028bb1f1f0c90ee7f4c4d28709cb1ed8f6c04a4fc1a"
        "475448c2642a65b6e18ab10069076ad31b69324fb1bcc1eb76db0540290c67e8"
        "23fe2018253d8b59478789549f2e4fe48faa6c28e34175578fe729d2c5e7ffa1"
        "ab46443024ed79da67d357b5c6cea9b89e711f2332d77f664ece61015bae0eb5"
        "b23e97bf7ba72c1130cb3f2fa4f9d93c66d88d8c4dcedb684d42bc4b02c20cbc"
        "8a82752c0bfefeb8db547908ad9caa8520f1108bd4d6455ef77f096f4db1b9c3"),
    "00": (
        "1ef40dcc3c9e591ab38b2acc80cdaeb3cdb9c1bcacaac07269980ee07c4c8ae2"
        "de5375d79b9892cc44f90799d3904baf5ab8a45a15fd1d0afa8fa808dbae1cad"
        "ee23b7c0997a859af528f2d6f16a31ee1dbb8b30a0cad828de8597455fbb1ab4"
        "a359277a5679877f1a00ccb13c0a3b24e7c232ba5db5c8de9d6c59ddc997e376"
        "6e8713732b69d33ab37906a905a0ac965c50e3618e42ce15a38116eda6e267e4"
        "afc71a05e52e80f42e1b421ffda6824ed7fe2d07a907233f3406e983aba1ca00"
        "2224aefd955662c5b68be7e7800ab4bb55aa9ae456bd7b8cc39143a1fd2a7ec6"
        "1d867cfe94bed13a668ebcf0a5bb3e684f1c323c099ce49cac5cc089a3204416"),
    "616263": (
        "6cd0f026fb16efd3e0f0653a6cf89d80b4bd143e9eb50e7feebc102682069f8e"
        "007ed0fe888a6eae2c142a8fe6c74513672452c93ab4af6387950d39bb724c90"
        "33e0f7f0656e42d8391a99cecba1ef1ec2d13dffc5ea0ddbd5a17944019adc7f"
        "de2b3605e173efa42ba420293b7a03c89b985d6ab0d700c1a32110ab2e81ab80"
        "db99d0cce7452fa5fe63df6c5867fb0760edaaad3b3d47896c5e83a402ed478e"
        "ba1c9937ea375386ca769295c9e5ebba79732c738496ec88cb70326e5bc7379f"
        "aced590738bd344e881a0e64f846deb77f0bb2f5dd20825ef4df22fe55bbdea9"
        "47df245fa1605f9219895b873d86f82fac63eb97fa7d2a515eea2bda9eb58366"),
    bytes(range(256)).hex(): (
        "d01d4cabba8a134d44ddb7c992d31e401be2f900cff37efeb0fa8e19ce4afe67"
        "eed05718c92862bbc4ab596aac6c6cb125c8f8a160123c662e10b3b200f077c1"
        "a96837a415da620f1d338e4f1837d9777ae431468f9dabcff47b308fa50b3940"
        "cde5968ff1490b0794cf148c6a2f21118ed18199f9d0eafa6210dc0bf78fca4f"
        "c658fce1be86dda2cd02b3dc1c1f53c8d32e94d16269571fc782694f01bde4a0"
        "c9ccd96bd36c4ff0325ceec0fd7a147ad101244d597ffb81d9e11054af17530e"
        "782a8f89dc41570b9fbc8e84aba2af31ce082da425c8fb90cc9c5a7d9a394be5"
        "a68acd6e6382754253d30fdc70429c1534a5c50760da13e4c21995a314ced332"),
}


if __name__ == "__main__":
    unittest.main()
