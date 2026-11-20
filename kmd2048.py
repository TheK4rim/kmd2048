"""KMD-2048: an experimental iterated SHAKE256 / SHA-512 hash, in pure Python.

    *** EXPERIMENTAL CONSTRUCTION.  NOT INDEPENDENTLY REVIEWED.          ***
    *** It is NOT more secure than SHA-512, SHA3-512 or SHAKE256.        ***
    *** Do not use it to protect real data.  See README.md, "Security". ***

Specification (all values are byte strings, ``||`` is concatenation)::

    H_0 = M                                              (the input message)
    for i = 1 .. 1024:
        K_i = SHAKE256(b"KMD-K" || H_(i-1), 64 bytes)    # stage 1, Keccak sponge
        H_i = SHA-512(b"KMD-M" || K_i)                   # stage 2, 64-byte output
    D = SHAKE256(b"KMD-FINAL" || H_1024, 256 bytes)      # 2048-bit digest

Everything below is implemented from scratch with integer arithmetic.  The
module imports only ``struct``, ``os`` and ``typing`` (plus ``sys`` for the
command-line entry point); it does not use ``hashlib`` or any other
cryptographic library.

Public API
----------
``keccak_f1600(lanes)``               Keccak-f[1600] permutation (FIPS 202, 24 rounds)
``shake256(data, n)``, ``Shake256``   SHAKE256 XOF (FIPS 202)
``sha512(data)``, ``Sha512``          SHA-512 (FIPS 180-4)
``kmd2048(data)``, ``KMD2048``        KMD-2048 (256-byte digest)
``kmd2048_hex(data)``                 KMD-2048 as 512 lowercase hex digits
``kmd2048_file(path)``                KMD-2048 of a file, streamed in chunks
``kmd2048_fileobj(f)``                KMD-2048 of a binary file object

Streaming / file format
-----------------------
The message hashed by ``kmd2048_file`` is exactly the sequence of bytes stored
in the file, from offset 0 to end-of-file: no length prefix, no chunk framing,
no text or newline translation.  The message enters the construction only
through the first SHAKE256 call, and a sponge absorbs a byte string
incrementally, applying its padding once, after the final byte.  Chunk
boundaries are therefore invisible, and for every file::

    kmd2048_file(path) == kmd2048(pathlib.Path(path).read_bytes())

The test suite checks this equality for many chunk sizes, including sizes
that do not divide the 136-byte SHAKE256 rate.  The message length is never
encoded explicitly; distinct byte strings get distinct sponge inputs because
the SHAKE256 padding (domain bits 1111, then pad10*1) is injective.
"""

from __future__ import annotations

import os
import struct
from typing import BinaryIO, List, Optional, Sequence, Tuple, Union

__all__ = [
    "DIGEST_SIZE",
    "ITERATIONS",
    "DEFAULT_CHUNK_SIZE",
    "KECCAK_ROUND_CONSTANTS",
    "KECCAK_RHO_OFFSETS",
    "SHA512_INITIAL_HASH",
    "SHA512_ROUND_CONSTANTS",
    "keccak_f1600",
    "Shake256",
    "shake256",
    "Sha512",
    "sha512",
    "KMD2048",
    "kmd2048",
    "kmd2048_hex",
    "kmd2048_file",
    "kmd2048_fileobj",
]

__version__ = "0.1.0"

BytesLike = Union[bytes, bytearray, memoryview]
PathLike = Union[str, bytes, os.PathLike]

# ---------------------------------------------------------------------------
# KMD-2048 parameters
# ---------------------------------------------------------------------------

DIGEST_SIZE = 256            # bytes (2048 bits)
ITERATIONS = 1024            # iterations of the (SHAKE256, SHA-512) pair
CHAIN_SIZE = 64              # bytes in every K_i and H_i (512 bits)
PREFIX_K = b"KMD-K"          # stage-1 domain prefix
PREFIX_M = b"KMD-M"          # stage-2 domain prefix
PREFIX_FINAL = b"KMD-FINAL"  # final-expansion domain prefix
DEFAULT_CHUNK_SIZE = 1 << 20  # bytes read per call by kmd2048_file()

_MASK64 = 0xFFFFFFFFFFFFFFFF
# For 0 <= x < 2**64:  x * _DUP == (x << 64) | x, i.e. the 128-bit value x||x.
# Shifting x||x right by k and keeping the low 64 bits rotates x right by k.
_DUP = 0x10000000000000001

# ---------------------------------------------------------------------------
# Keccak-f[1600] (FIPS 202, section 3)
# ---------------------------------------------------------------------------
#
# The 1600-bit state is 25 lanes of 64 bits.  Lane A[x, y] (0 <= x, y < 5)
# has index x + 5*y, and the state bytes map to lanes little-endian, 8 bytes
# per lane (FIPS 202 section 3.1.2 with the byte/bit conventions of
# Appendix B.1).

# Round constants RC[i] for the iota step (FIPS 202 Algorithms 5 and 6).
# The test suite re-derives them from the rc(t) LFSR.
KECCAK_ROUND_CONSTANTS: Tuple[int, ...] = (
    0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000,
    0x000000000000808B, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
    0x000000000000008A, 0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
    0x000000008000808B, 0x800000000000008B, 0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080, 0x000000000000800A, 0x800000008000000A,
    0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
)

# Rotation offsets of the rho step, KECCAK_RHO_OFFSETS[x][y] (FIPS 202 Table 2,
# reduced mod 64).  The unrolled permutation below uses these values as
# literal shift amounts; the test suite re-derives the table from Algorithm 2.
KECCAK_RHO_OFFSETS: Tuple[Tuple[int, ...], ...] = (
    (0, 36, 3, 41, 18),
    (1, 44, 10, 45, 2),
    (62, 6, 43, 15, 61),
    (28, 55, 25, 21, 56),
    (27, 20, 39, 8, 14),
)


def _keccak_rounds(state: Sequence[int], round_constants: Sequence[int]) -> List[int]:
    """Apply one Keccak-f[1600] round per entry of ``round_constants``.

    ``state`` holds 25 lanes (ints in [0, 2**64), index x + 5*y).  Returns
    a new list; the input is not modified and is NOT validated (this is the
    internal hot path; see ``keccak_f1600`` for the checked public wrapper).
    Passing all 24 ``KECCAK_ROUND_CONSTANTS`` gives Keccak-f[1600]; a
    prefix of them gives the state after that many rounds (used by tests).

    Optimisations (each one is checked against a plain FIPS 202 reference
    implementation and the published intermediate values in the tests):

    * Fully unrolled round on 25 local variables; no lists or indexing.
    * theta is fused into rho/pi: B[y, 2x+3y] = ROT(A[x,y] ^ D[x], r[x,y]).
    * 64-bit rotation in 3 integer operations:
      ROT(v, n) == ((v * (2**64 + 1)) >> (64 - n)) & M.
    * Lane-complementing transform (Bertoni et al., "Keccak implementation
      overview", section 2.2): lanes A[1,0], A[2,0], A[3,1], A[2,2], A[2,3]
      and A[0,4] are kept complemented while the rounds run.  The pattern is
      stable under one round, and chi then needs one NOT per plane, i.e. 5
      per round instead of 25.  NOT is computed as ``x ^ M`` so that every
      value stays a non-negative 64-bit integer.
    """
    M = _MASK64
    DUP = _DUP
    (a00, a10, a20, a30, a40,
     a01, a11, a21, a31, a41,
     a02, a12, a22, a32, a42,
     a03, a13, a23, a33, a43,
     a04, a14, a24, a34, a44) = state

    # Enter the lane-complemented representation.
    a10 ^= M
    a20 ^= M
    a31 ^= M
    a22 ^= M
    a23 ^= M
    a04 ^= M

    for rc in round_constants:
        # theta: column parities C[x] and D[x] = C[x-1] ^ ROT(C[x+1], 1).
        c0 = a00 ^ a01 ^ a02 ^ a03 ^ a04
        c1 = a10 ^ a11 ^ a12 ^ a13 ^ a14
        c2 = a20 ^ a21 ^ a22 ^ a23 ^ a24
        c3 = a30 ^ a31 ^ a32 ^ a33 ^ a34
        c4 = a40 ^ a41 ^ a42 ^ a43 ^ a44
        d0 = c4 ^ ((c1 * DUP >> 63) & M)
        d1 = c0 ^ ((c2 * DUP >> 63) & M)
        d2 = c1 ^ ((c3 * DUP >> 63) & M)
        d3 = c2 ^ ((c4 * DUP >> 63) & M)
        d4 = c3 ^ ((c0 * DUP >> 63) & M)

        # theta (XOR of D) + rho (rotate by r[x,y]) + pi (move to B[y, 2x+3y]).
        # Equivalently B[X,Y] = ROT(A[x,y] ^ D[x], r[x,y]) with x = X + 3Y mod 5,
        # y = X.  A rotation by r is a right shift of v||v by 64 - r.
        b00 = a00 ^ d0                        # A[0,0], r = 0
        b10 = ((a11 ^ d1) * DUP >> 20) & M    # A[1,1], r = 44
        b20 = ((a22 ^ d2) * DUP >> 21) & M    # A[2,2], r = 43
        b30 = ((a33 ^ d3) * DUP >> 43) & M    # A[3,3], r = 21
        b40 = ((a44 ^ d4) * DUP >> 50) & M    # A[4,4], r = 14

        b01 = ((a30 ^ d3) * DUP >> 36) & M    # A[3,0], r = 28
        b11 = ((a41 ^ d4) * DUP >> 44) & M    # A[4,1], r = 20
        b21 = ((a02 ^ d0) * DUP >> 61) & M    # A[0,2], r = 3
        b31 = ((a13 ^ d1) * DUP >> 19) & M    # A[1,3], r = 45
        b41 = ((a24 ^ d2) * DUP >> 3) & M     # A[2,4], r = 61

        b02 = ((a10 ^ d1) * DUP >> 63) & M    # A[1,0], r = 1
        b12 = ((a21 ^ d2) * DUP >> 58) & M    # A[2,1], r = 6
        b22 = ((a32 ^ d3) * DUP >> 39) & M    # A[3,2], r = 25
        b32 = ((a43 ^ d4) * DUP >> 56) & M    # A[4,3], r = 8
        b42 = ((a04 ^ d0) * DUP >> 46) & M    # A[0,4], r = 18

        b03 = ((a40 ^ d4) * DUP >> 37) & M    # A[4,0], r = 27
        b13 = ((a01 ^ d0) * DUP >> 28) & M    # A[0,1], r = 36
        b23 = ((a12 ^ d1) * DUP >> 54) & M    # A[1,2], r = 10
        b33 = ((a23 ^ d2) * DUP >> 49) & M    # A[2,3], r = 15
        b43 = ((a34 ^ d3) * DUP >> 8) & M     # A[3,4], r = 56

        b04 = ((a20 ^ d2) * DUP >> 2) & M     # A[2,0], r = 62
        b14 = ((a31 ^ d3) * DUP >> 9) & M     # A[3,1], r = 55
        b24 = ((a42 ^ d4) * DUP >> 25) & M    # A[4,2], r = 39
        b34 = ((a03 ^ d0) * DUP >> 23) & M    # A[0,3], r = 41
        b44 = ((a14 ^ d1) * DUP >> 62) & M    # A[1,4], r = 2

        # chi: A[x,y] = B[x,y] ^ (~B[x+1,y] & B[x+2,y]), rewritten for the
        # complemented representation (De Morgan), one NOT (n) per plane.
        # iota: XOR the round constant into A[0,0] (never complemented).
        n = b20 ^ M
        a00 = b00 ^ (b10 | b20) ^ rc
        a10 = b10 ^ (n | b30)
        a20 = b20 ^ (b30 & b40)
        a30 = b30 ^ (b40 | b00)
        a40 = b40 ^ (b00 & b10)

        n = b41 ^ M
        a01 = b01 ^ (b11 | b21)
        a11 = b11 ^ (b21 & b31)
        a21 = b21 ^ (b31 | n)
        a31 = b31 ^ (b41 | b01)
        a41 = b41 ^ (b01 & b11)

        n = b32 ^ M
        a02 = b02 ^ (b12 | b22)
        a12 = b12 ^ (b22 & b32)
        a22 = b22 ^ (n & b42)
        a32 = n ^ (b42 | b02)
        a42 = b42 ^ (b02 & b12)

        n = b33 ^ M
        a03 = b03 ^ (b13 & b23)
        a13 = b13 ^ (b23 | b33)
        a23 = b23 ^ (n | b43)
        a33 = n ^ (b43 & b03)
        a43 = b43 ^ (b03 | b13)

        n = b14 ^ M
        a04 = b04 ^ (n & b24)
        a14 = n ^ (b24 | b34)
        a24 = b24 ^ (b34 & b44)
        a34 = b34 ^ (b44 | b04)
        a44 = b44 ^ (b04 & b14)

    # Leave the lane-complemented representation.
    a10 ^= M
    a20 ^= M
    a31 ^= M
    a22 ^= M
    a23 ^= M
    a04 ^= M

    return [a00, a10, a20, a30, a40,
            a01, a11, a21, a31, a41,
            a02, a12, a22, a32, a42,
            a03, a13, a23, a33, a43,
            a04, a14, a24, a34, a44]


def keccak_f1600(state: Sequence[int]) -> List[int]:
    """Return Keccak-f[1600](state): the 24-round permutation of FIPS 202.

    ``state`` is a sequence of 25 unsigned 64-bit integers ("lanes"); lane
    A[x, y] is ``state[x + 5*y]``.  To convert a 200-byte state ``s`` use
    ``list(struct.unpack('<25Q', s))`` and ``struct.pack('<25Q', *lanes)``.

    A new list is returned; ``state`` is left unchanged.
    """
    lanes = list(state)
    if len(lanes) != 25:
        raise ValueError(f"Keccak-f[1600] needs 25 lanes, got {len(lanes)}")
    for lane in lanes:
        if not isinstance(lane, int) or isinstance(lane, bool):
            raise TypeError("Keccak-f[1600] lanes must be int")
        if not 0 <= lane <= _MASK64:
            raise ValueError("Keccak-f[1600] lanes must be in [0, 2**64)")
    return _keccak_rounds(lanes, KECCAK_ROUND_CONSTANTS)


# ---------------------------------------------------------------------------
# SHAKE256 (FIPS 202, sections 4, 5.1 and 6.2)
# ---------------------------------------------------------------------------
#
# SHAKE256(M, d) = KECCAK[c=512](M || 1111, d): a sponge over Keccak-f[1600]
# with capacity 512 bits and rate r = 1600 - 512 = 1088 bits = 136 bytes.
# Byte-oriented padding: the 4 domain-separation bits 1111 and the first bit
# of pad10*1 form the byte 0x1F; the final 1 bit of pad10*1 is 0x80 in the
# last byte of the block (0x1F ^ 0x80 = 0x9F if both land in the same byte).

SHAKE256_RATE = 136

_LE17 = struct.Struct("<17Q")   # one 136-byte rate block as 17 lanes
_LE8 = struct.Struct("<8Q")     # 64 bytes as 8 lanes


def _shake256_padding(message_length: int) -> bytes:
    """SHAKE256 padding (domain bits 1111, then pad10*1) for a message length."""
    pad_length = SHAKE256_RATE - message_length % SHAKE256_RATE  # 1 .. 136
    if pad_length == 1:
        return b"\x9f"
    return b"\x1f" + bytes(pad_length - 2) + b"\x80"


def _byte_view(data: BytesLike) -> memoryview:
    """Return a flat unsigned-byte memoryview of ``data`` (bytes-like only)."""
    if isinstance(data, str):
        raise TypeError("Unicode text must be encoded before hashing, "
                        "e.g. text.encode('utf-8')")
    try:
        view = memoryview(data)
    except TypeError:
        raise TypeError("a bytes-like object is required, not "
                        f"{type(data).__name__!r}") from None
    return view.cast("B")


def _check_output_length(length: int) -> int:
    if isinstance(length, bool) or not isinstance(length, int):
        raise TypeError("output length must be an int")
    if length < 0:
        raise ValueError("output length must be non-negative")
    return length


class Shake256:
    """Incremental SHAKE256 extendable-output function (FIPS 202).

    >>> Shake256(b"").hexdigest(32)
    '46b9dd2b0ba88d13233b3feb743eeb243fcd52ea62b81b82b50c27646ed5762f'

    ``update()`` absorbs bytes; ``digest(n)`` returns the first ``n`` output
    bytes without disturbing the object, so more data can be absorbed after
    a digest and ``digest`` can be called repeatedly.
    """

    __slots__ = ("_lanes", "_buffer")

    name = "shake_256"
    block_size = SHAKE256_RATE

    def __init__(self, data: BytesLike = b"") -> None:
        self._lanes: List[int] = [0] * 25
        self._buffer = bytearray()          # pending bytes, always < 136
        self.update(data)

    def update(self, data: BytesLike) -> None:
        """Absorb ``data`` (any bytes-like object) into the sponge."""
        view = _byte_view(data)
        size = len(view)
        if not size:
            return
        rate = SHAKE256_RATE
        buffer = self._buffer
        lanes = self._lanes
        rounds = _keccak_rounds
        constants = KECCAK_ROUND_CONSTANTS
        unpack = _LE17.unpack_from
        start = 0
        if buffer:
            need = rate - len(buffer)
            if size < need:
                buffer += view
                return
            buffer += view[:need]
            words = unpack(buffer)
            lanes = rounds([x ^ y for x, y in zip(lanes, words)] + lanes[17:], constants)
            buffer.clear()
            start = need
        end = size - (size - start) % rate
        for offset in range(start, end, rate):
            words = unpack(view, offset)
            lanes = rounds([x ^ y for x, y in zip(lanes, words)] + lanes[17:], constants)
        if end < size:
            buffer += view[end:]
        self._lanes = lanes

    def digest(self, length: int) -> bytes:
        """Return the first ``length`` bytes of SHAKE256 output."""
        length = _check_output_length(length)
        if not length:
            return b""
        buffer = self._buffer
        words = _LE17.unpack(bytes(buffer) + _shake256_padding(len(buffer)))
        lanes = self._lanes
        lanes = _keccak_rounds([x ^ y for x, y in zip(lanes, words)] + lanes[17:],
                               KECCAK_ROUND_CONSTANTS)
        output = bytearray(_LE17.pack(*lanes[:17]))
        while len(output) < length:
            lanes = _keccak_rounds(lanes, KECCAK_ROUND_CONSTANTS)
            output += _LE17.pack(*lanes[:17])
        del output[length:]
        return bytes(output)

    def hexdigest(self, length: int) -> str:
        """Return ``digest(length)`` as lowercase hexadecimal."""
        return self.digest(length).hex()

    def copy(self) -> "Shake256":
        """Return an independent copy of the current sponge state."""
        other = Shake256.__new__(Shake256)
        other._lanes = list(self._lanes)
        other._buffer = bytearray(self._buffer)
        return other


def shake256(data: BytesLike, output_length: int) -> bytes:
    """Return SHAKE256(data) truncated to ``output_length`` bytes (FIPS 202)."""
    return Shake256(data).digest(output_length)


# ---------------------------------------------------------------------------
# SHA-512 (FIPS 180-4, sections 4.1.3, 4.2.3, 5.1.2, 5.3.5 and 6.4)
# ---------------------------------------------------------------------------

# Initial hash value H(0): the first 64 bits of the fractional parts of the
# square roots of the first 8 primes (FIPS 180-4 section 5.3.5).
SHA512_INITIAL_HASH: Tuple[int, ...] = (
    0x6A09E667F3BCC908, 0xBB67AE8584CAA73B, 0x3C6EF372FE94F82B, 0xA54FF53A5F1D36F1,
    0x510E527FADE682D1, 0x9B05688C2B3E6C1F, 0x1F83D9ABFB41BD6B, 0x5BE0CD19137E2179,
)

# Round constants K[t]: the first 64 bits of the fractional parts of the cube
# roots of the first 80 primes (FIPS 180-4 section 4.2.3).
SHA512_ROUND_CONSTANTS: Tuple[int, ...] = (
    0x428A2F98D728AE22, 0x7137449123EF65CD, 0xB5C0FBCFEC4D3B2F, 0xE9B5DBA58189DBBC,
    0x3956C25BF348B538, 0x59F111F1B605D019, 0x923F82A4AF194F9B, 0xAB1C5ED5DA6D8118,
    0xD807AA98A3030242, 0x12835B0145706FBE, 0x243185BE4EE4B28C, 0x550C7DC3D5FFB4E2,
    0x72BE5D74F27B896F, 0x80DEB1FE3B1696B1, 0x9BDC06A725C71235, 0xC19BF174CF692694,
    0xE49B69C19EF14AD2, 0xEFBE4786384F25E3, 0x0FC19DC68B8CD5B5, 0x240CA1CC77AC9C65,
    0x2DE92C6F592B0275, 0x4A7484AA6EA6E483, 0x5CB0A9DCBD41FBD4, 0x76F988DA831153B5,
    0x983E5152EE66DFAB, 0xA831C66D2DB43210, 0xB00327C898FB213F, 0xBF597FC7BEEF0EE4,
    0xC6E00BF33DA88FC2, 0xD5A79147930AA725, 0x06CA6351E003826F, 0x142929670A0E6E70,
    0x27B70A8546D22FFC, 0x2E1B21385C26C926, 0x4D2C6DFC5AC42AED, 0x53380D139D95B3DF,
    0x650A73548BAF63DE, 0x766A0ABB3C77B2A8, 0x81C2C92E47EDAEE6, 0x92722C851482353B,
    0xA2BFE8A14CF10364, 0xA81A664BBC423001, 0xC24B8B70D0F89791, 0xC76C51A30654BE30,
    0xD192E819D6EF5218, 0xD69906245565A910, 0xF40E35855771202A, 0x106AA07032BBD1B8,
    0x19A4C116B8D2D0C8, 0x1E376C085141AB53, 0x2748774CDF8EEB99, 0x34B0BCB5E19B48A8,
    0x391C0CB3C5C95A63, 0x4ED8AA4AE3418ACB, 0x5B9CCA4F7763E373, 0x682E6FF3D6B2B8A3,
    0x748F82EE5DEFB2FC, 0x78A5636F43172F60, 0x84C87814A1F0AB72, 0x8CC702081A6439EC,
    0x90BEFFFA23631E28, 0xA4506CEBDE82BDE9, 0xBEF9A3F7B2C67915, 0xC67178F2E372532B,
    0xCA273ECEEA26619C, 0xD186B8C721C0C207, 0xEADA7DD6CDE0EB1E, 0xF57D4F7FEE6ED178,
    0x06F067AA72176FBA, 0x0A637DC5A2C898A6, 0x113F9804BEF90DAE, 0x1B710B35131C471B,
    0x28DB77F523047D84, 0x32CAAB7B40C72493, 0x3C9EBE0A15C9BEBC, 0x431D67C49C100D4C,
    0x4CC5D4BECB3E42B6, 0x597F299CFC657E2A, 0x5FCB6FAB3AD6FAEC, 0x6C44198C4A475817,
)

# The same constants grouped by 16, one tuple per unrolled 16-round block.
_SHA512_K_BLOCKS = tuple(tuple(SHA512_ROUND_CONSTANTS[i:i + 16]) for i in range(0, 80, 16))

SHA512_BLOCK_SIZE = 128
SHA512_DIGEST_SIZE = 64

_BE16 = struct.Struct(">16Q")   # one 1024-bit message block as 16 big-endian words
_BE8 = struct.Struct(">8Q")     # the 512-bit hash value as 8 big-endian words


def _sha512_padding(message_length: int) -> bytes:
    """SHA-512 padding (FIPS 180-4 section 5.1.2) for a length in bytes.

    One 1 bit (0x80), zero bits up to 896 mod 1024, then the message length
    in bits as a 128-bit big-endian integer.
    """
    return (b"\x80" + bytes((111 - message_length) % 128)
            + (8 * message_length).to_bytes(16, "big"))


def _sha512_compress(state: Sequence[int], block: Sequence[int]) -> Tuple[int, ...]:
    """SHA-512 compression (FIPS 180-4 section 6.4.2) of one 16-word block.

    ``state`` is the 8-word intermediate hash H(i-1); ``block`` is the 16
    big-endian 64-bit words of the message block.  Returns H(i).

    Optimisations:

    * 16 rounds are unrolled with the working-variable roles renamed in the
      code (so no a..h shuffling); 5 iterations give the 80 rounds.
    * The message schedule is computed in place in 16 locals, 16 words at a
      time, between the unrolled blocks.
    * Sigma and sigma rotations use v = x * (2**64 + 1) (= x||x): ROTR(x, n)
      is the low 64 bits of v >> n.  The bits above 2**64 are left in the
      Sigma values because they only feed sums that are reduced mod 2**64.
    * Ch(e,f,g) = g ^ (e & (f ^ g)); Maj(a,b,c) = b ^ ((a ^ b) & (b ^ c)),
      where b ^ c is the previous round's a ^ b (kept in x / y).
    """
    M = _MASK64
    DUP = _DUP
    k_blocks = _SHA512_K_BLOCKS
    a, b, c, d, e, f, g, h = state
    (w0, w1, w2, w3, w4, w5, w6, w7,
     w8, w9, w10, w11, w12, w13, w14, w15) = block
    x = b ^ c
    for i in range(5):
        (k0, k1, k2, k3, k4, k5, k6, k7,
         k8, k9, k10, k11, k12, k13, k14, k15) = k_blocks[i]

        # Each round:  T1 = h + Sigma1(e) + Ch(e,f,g) + K[t] + W[t]
        #              d  = d + T1             (becomes the new e)
        #              h  = T1 + Sigma0(a) + Maj(a,b,c)   (becomes the new a)
        # Sigma1(e) = ROTR14 ^ ROTR18 ^ ROTR41,  Sigma0(a) = ROTR28 ^ ROTR34 ^ ROTR39.

        # round 16*i + 0: (a..h) = (a, b, c, d, e, f, g, h)
        t = e * DUP
        t1 = h + ((t >> 14) ^ (t >> 18) ^ (t >> 41)) + (g ^ (e & (f ^ g))) + k0 + w0
        d = (d + t1) & M
        t = a * DUP
        y = a ^ b
        h = (t1 + ((t >> 28) ^ (t >> 34) ^ (t >> 39)) + (b ^ (y & x))) & M
        # round 16*i + 1: (a..h) = (h, a, b, c, d, e, f, g)
        t = d * DUP
        t1 = g + ((t >> 14) ^ (t >> 18) ^ (t >> 41)) + (f ^ (d & (e ^ f))) + k1 + w1
        c = (c + t1) & M
        t = h * DUP
        x = h ^ a
        g = (t1 + ((t >> 28) ^ (t >> 34) ^ (t >> 39)) + (a ^ (x & y))) & M
        # round 16*i + 2: (a..h) = (g, h, a, b, c, d, e, f)
        t = c * DUP
        t1 = f + ((t >> 14) ^ (t >> 18) ^ (t >> 41)) + (e ^ (c & (d ^ e))) + k2 + w2
        b = (b + t1) & M
        t = g * DUP
        y = g ^ h
        f = (t1 + ((t >> 28) ^ (t >> 34) ^ (t >> 39)) + (h ^ (y & x))) & M
        # round 16*i + 3: (a..h) = (f, g, h, a, b, c, d, e)
        t = b * DUP
        t1 = e + ((t >> 14) ^ (t >> 18) ^ (t >> 41)) + (d ^ (b & (c ^ d))) + k3 + w3
        a = (a + t1) & M
        t = f * DUP
        x = f ^ g
        e = (t1 + ((t >> 28) ^ (t >> 34) ^ (t >> 39)) + (g ^ (x & y))) & M
        # round 16*i + 4: (a..h) = (e, f, g, h, a, b, c, d)
        t = a * DUP
        t1 = d + ((t >> 14) ^ (t >> 18) ^ (t >> 41)) + (c ^ (a & (b ^ c))) + k4 + w4
        h = (h + t1) & M
        t = e * DUP
        y = e ^ f
        d = (t1 + ((t >> 28) ^ (t >> 34) ^ (t >> 39)) + (f ^ (y & x))) & M
        # round 16*i + 5: (a..h) = (d, e, f, g, h, a, b, c)
        t = h * DUP
        t1 = c + ((t >> 14) ^ (t >> 18) ^ (t >> 41)) + (b ^ (h & (a ^ b))) + k5 + w5
        g = (g + t1) & M
        t = d * DUP
        x = d ^ e
        c = (t1 + ((t >> 28) ^ (t >> 34) ^ (t >> 39)) + (e ^ (x & y))) & M
        # round 16*i + 6: (a..h) = (c, d, e, f, g, h, a, b)
        t = g * DUP
        t1 = b + ((t >> 14) ^ (t >> 18) ^ (t >> 41)) + (a ^ (g & (h ^ a))) + k6 + w6
        f = (f + t1) & M
        t = c * DUP
        y = c ^ d
        b = (t1 + ((t >> 28) ^ (t >> 34) ^ (t >> 39)) + (d ^ (y & x))) & M
        # round 16*i + 7: (a..h) = (b, c, d, e, f, g, h, a)
        t = f * DUP
        t1 = a + ((t >> 14) ^ (t >> 18) ^ (t >> 41)) + (h ^ (f & (g ^ h))) + k7 + w7
        e = (e + t1) & M
        t = b * DUP
        x = b ^ c
        a = (t1 + ((t >> 28) ^ (t >> 34) ^ (t >> 39)) + (c ^ (x & y))) & M
        # round 16*i + 8: (a..h) = (a, b, c, d, e, f, g, h)
        t = e * DUP
        t1 = h + ((t >> 14) ^ (t >> 18) ^ (t >> 41)) + (g ^ (e & (f ^ g))) + k8 + w8
        d = (d + t1) & M
        t = a * DUP
        y = a ^ b
        h = (t1 + ((t >> 28) ^ (t >> 34) ^ (t >> 39)) + (b ^ (y & x))) & M
        # round 16*i + 9: (a..h) = (h, a, b, c, d, e, f, g)
        t = d * DUP
        t1 = g + ((t >> 14) ^ (t >> 18) ^ (t >> 41)) + (f ^ (d & (e ^ f))) + k9 + w9
        c = (c + t1) & M
        t = h * DUP
        x = h ^ a
        g = (t1 + ((t >> 28) ^ (t >> 34) ^ (t >> 39)) + (a ^ (x & y))) & M
        # round 16*i + 10: (a..h) = (g, h, a, b, c, d, e, f)
        t = c * DUP
        t1 = f + ((t >> 14) ^ (t >> 18) ^ (t >> 41)) + (e ^ (c & (d ^ e))) + k10 + w10
        b = (b + t1) & M
        t = g * DUP
        y = g ^ h
        f = (t1 + ((t >> 28) ^ (t >> 34) ^ (t >> 39)) + (h ^ (y & x))) & M
        # round 16*i + 11: (a..h) = (f, g, h, a, b, c, d, e)
        t = b * DUP
        t1 = e + ((t >> 14) ^ (t >> 18) ^ (t >> 41)) + (d ^ (b & (c ^ d))) + k11 + w11
        a = (a + t1) & M
        t = f * DUP
        x = f ^ g
        e = (t1 + ((t >> 28) ^ (t >> 34) ^ (t >> 39)) + (g ^ (x & y))) & M
        # round 16*i + 12: (a..h) = (e, f, g, h, a, b, c, d)
        t = a * DUP
        t1 = d + ((t >> 14) ^ (t >> 18) ^ (t >> 41)) + (c ^ (a & (b ^ c))) + k12 + w12
        h = (h + t1) & M
        t = e * DUP
        y = e ^ f
        d = (t1 + ((t >> 28) ^ (t >> 34) ^ (t >> 39)) + (f ^ (y & x))) & M
        # round 16*i + 13: (a..h) = (d, e, f, g, h, a, b, c)
        t = h * DUP
        t1 = c + ((t >> 14) ^ (t >> 18) ^ (t >> 41)) + (b ^ (h & (a ^ b))) + k13 + w13
        g = (g + t1) & M
        t = d * DUP
        x = d ^ e
        c = (t1 + ((t >> 28) ^ (t >> 34) ^ (t >> 39)) + (e ^ (x & y))) & M
        # round 16*i + 14: (a..h) = (c, d, e, f, g, h, a, b)
        t = g * DUP
        t1 = b + ((t >> 14) ^ (t >> 18) ^ (t >> 41)) + (a ^ (g & (h ^ a))) + k14 + w14
        f = (f + t1) & M
        t = c * DUP
        y = c ^ d
        b = (t1 + ((t >> 28) ^ (t >> 34) ^ (t >> 39)) + (d ^ (y & x))) & M
        # round 16*i + 15: (a..h) = (b, c, d, e, f, g, h, a)
        t = f * DUP
        t1 = a + ((t >> 14) ^ (t >> 18) ^ (t >> 41)) + (h ^ (f & (g ^ h))) + k15 + w15
        e = (e + t1) & M
        t = b * DUP
        x = b ^ c
        a = (t1 + ((t >> 28) ^ (t >> 34) ^ (t >> 39)) + (c ^ (x & y))) & M

        if i == 4:
            break

        # Message schedule for the next 16 rounds (t = 16*(i+1) + j):
        #   W[t] = sigma1(W[t-2]) + W[t-7] + sigma0(W[t-15]) + W[t-16]
        # w_j holds W[t-16]; W[t-15], W[t-7], W[t-2] are w_(j+1), w_(j+9), w_(j+14)
        # (indices mod 16), already updated where needed by the sequential order.
        # sigma0(v) = ROTR1 ^ ROTR8 ^ SHR7,  sigma1(v) = ROTR19 ^ ROTR61 ^ SHR6.
        u = w1 * DUP
        v = w14 * DUP
        w0 = (w0 + ((u >> 1) ^ (u >> 8) ^ (w1 >> 7))
              + w9 + ((v >> 19) ^ (v >> 61) ^ (w14 >> 6))) & M
        u = w2 * DUP
        v = w15 * DUP
        w1 = (w1 + ((u >> 1) ^ (u >> 8) ^ (w2 >> 7))
              + w10 + ((v >> 19) ^ (v >> 61) ^ (w15 >> 6))) & M
        u = w3 * DUP
        v = w0 * DUP
        w2 = (w2 + ((u >> 1) ^ (u >> 8) ^ (w3 >> 7))
              + w11 + ((v >> 19) ^ (v >> 61) ^ (w0 >> 6))) & M
        u = w4 * DUP
        v = w1 * DUP
        w3 = (w3 + ((u >> 1) ^ (u >> 8) ^ (w4 >> 7))
              + w12 + ((v >> 19) ^ (v >> 61) ^ (w1 >> 6))) & M
        u = w5 * DUP
        v = w2 * DUP
        w4 = (w4 + ((u >> 1) ^ (u >> 8) ^ (w5 >> 7))
              + w13 + ((v >> 19) ^ (v >> 61) ^ (w2 >> 6))) & M
        u = w6 * DUP
        v = w3 * DUP
        w5 = (w5 + ((u >> 1) ^ (u >> 8) ^ (w6 >> 7))
              + w14 + ((v >> 19) ^ (v >> 61) ^ (w3 >> 6))) & M
        u = w7 * DUP
        v = w4 * DUP
        w6 = (w6 + ((u >> 1) ^ (u >> 8) ^ (w7 >> 7))
              + w15 + ((v >> 19) ^ (v >> 61) ^ (w4 >> 6))) & M
        u = w8 * DUP
        v = w5 * DUP
        w7 = (w7 + ((u >> 1) ^ (u >> 8) ^ (w8 >> 7))
              + w0 + ((v >> 19) ^ (v >> 61) ^ (w5 >> 6))) & M
        u = w9 * DUP
        v = w6 * DUP
        w8 = (w8 + ((u >> 1) ^ (u >> 8) ^ (w9 >> 7))
              + w1 + ((v >> 19) ^ (v >> 61) ^ (w6 >> 6))) & M
        u = w10 * DUP
        v = w7 * DUP
        w9 = (w9 + ((u >> 1) ^ (u >> 8) ^ (w10 >> 7))
              + w2 + ((v >> 19) ^ (v >> 61) ^ (w7 >> 6))) & M
        u = w11 * DUP
        v = w8 * DUP
        w10 = (w10 + ((u >> 1) ^ (u >> 8) ^ (w11 >> 7))
               + w3 + ((v >> 19) ^ (v >> 61) ^ (w8 >> 6))) & M
        u = w12 * DUP
        v = w9 * DUP
        w11 = (w11 + ((u >> 1) ^ (u >> 8) ^ (w12 >> 7))
               + w4 + ((v >> 19) ^ (v >> 61) ^ (w9 >> 6))) & M
        u = w13 * DUP
        v = w10 * DUP
        w12 = (w12 + ((u >> 1) ^ (u >> 8) ^ (w13 >> 7))
               + w5 + ((v >> 19) ^ (v >> 61) ^ (w10 >> 6))) & M
        u = w14 * DUP
        v = w11 * DUP
        w13 = (w13 + ((u >> 1) ^ (u >> 8) ^ (w14 >> 7))
               + w6 + ((v >> 19) ^ (v >> 61) ^ (w11 >> 6))) & M
        u = w15 * DUP
        v = w12 * DUP
        w14 = (w14 + ((u >> 1) ^ (u >> 8) ^ (w15 >> 7))
               + w7 + ((v >> 19) ^ (v >> 61) ^ (w12 >> 6))) & M
        u = w0 * DUP
        v = w13 * DUP
        w15 = (w15 + ((u >> 1) ^ (u >> 8) ^ (w0 >> 7))
               + w8 + ((v >> 19) ^ (v >> 61) ^ (w13 >> 6))) & M

    s0, s1, s2, s3, s4, s5, s6, s7 = state
    return ((s0 + a) & M, (s1 + b) & M, (s2 + c) & M, (s3 + d) & M,
            (s4 + e) & M, (s5 + f) & M, (s6 + g) & M, (s7 + h) & M)


class Sha512:
    """Incremental SHA-512 (FIPS 180-4).

    >>> Sha512(b"abc").hexdigest()[:32]
    'ddaf35a193617abacc417349ae204131'
    """

    __slots__ = ("_state", "_buffer", "_length")

    name = "sha512"
    digest_size = SHA512_DIGEST_SIZE
    block_size = SHA512_BLOCK_SIZE

    def __init__(self, data: BytesLike = b"") -> None:
        self._state: Tuple[int, ...] = SHA512_INITIAL_HASH
        self._buffer = bytearray()          # pending bytes, always < 128
        self._length = 0                    # total bytes absorbed
        self.update(data)

    def update(self, data: BytesLike) -> None:
        """Process ``data`` (any bytes-like object)."""
        view = _byte_view(data)
        size = len(view)
        if not size:
            return
        self._length += size
        block = SHA512_BLOCK_SIZE
        buffer = self._buffer
        state = self._state
        compress = _sha512_compress
        unpack = _BE16.unpack_from
        start = 0
        if buffer:
            need = block - len(buffer)
            if size < need:
                buffer += view
                return
            buffer += view[:need]
            state = compress(state, unpack(buffer))
            buffer.clear()
            start = need
        end = size - (size - start) % block
        for offset in range(start, end, block):
            state = compress(state, unpack(view, offset))
        if end < size:
            buffer += view[end:]
        self._state = state

    def digest(self) -> bytes:
        """Return the 64-byte SHA-512 digest of everything processed so far."""
        tail = bytes(self._buffer) + _sha512_padding(self._length)
        state = self._state
        for offset in range(0, len(tail), SHA512_BLOCK_SIZE):
            state = _sha512_compress(state, _BE16.unpack_from(tail, offset))
        return _BE8.pack(*state)

    def hexdigest(self) -> str:
        """Return the digest as 128 lowercase hexadecimal digits."""
        return self.digest().hex()

    def copy(self) -> "Sha512":
        """Return an independent copy of the current hash state."""
        other = Sha512.__new__(Sha512)
        other._state = self._state
        other._buffer = bytearray(self._buffer)
        other._length = self._length
        return other


def sha512(data: BytesLike) -> bytes:
    """Return the 64-byte SHA-512 digest of ``data`` (FIPS 180-4)."""
    return Sha512(data).digest()


# ---------------------------------------------------------------------------
# KMD-2048
# ---------------------------------------------------------------------------

# From iteration 2 on, both primitives always hash 69-byte inputs:
#   stage 1: b"KMD-K" || H_(i-1)  -> one 136-byte SHAKE256 block
#   stage 2: b"KMD-M" || K_i      -> one 128-byte SHA-512 block
# Stage 2 hashes 69 bytes in iteration 1 as well.  The fixed parts of both
# padded blocks are built once from the generic padding functions above;
# only bytes 5..68 (the 64-byte chaining value) change between iterations.
_SHAKE_BLOCK_KMD_K = PREFIX_K + bytes(CHAIN_SIZE) + _shake256_padding(len(PREFIX_K) + CHAIN_SIZE)
_SHA512_BLOCK_KMD_M = PREFIX_M + bytes(CHAIN_SIZE) + _sha512_padding(len(PREFIX_M) + CHAIN_SIZE)
_CHAIN_OFFSET = 5            # len(PREFIX_K) == len(PREFIX_M) == 5
_CAPACITY_LANES = (0,) * 8   # lanes 17..24 of a freshly initialised sponge


def _kmd2048_chain(k1: bytes) -> bytes:
    """Given K_1 (64 bytes), return H_1024 (64 bytes).

    Computes H_1 = SHA-512(b"KMD-M" || K_1), then iterations i = 2 .. 1024.
    This is the fused fast path of the specification; the tests check it
    against a direct composition of the generic ``shake256``/``sha512``.
    """
    kmd_k = bytearray(_SHAKE_BLOCK_KMD_K)   # b"KMD-K" || H || SHAKE256 padding
    kmd_m = bytearray(_SHA512_BLOCK_KMD_M)  # b"KMD-M" || K || SHA-512 padding
    kmd_m[_CHAIN_OFFSET:_CHAIN_OFFSET + CHAIN_SIZE] = k1
    rounds = _keccak_rounds
    constants = KECCAK_ROUND_CONSTANTS
    compress = _sha512_compress
    iv = SHA512_INITIAL_HASH
    capacity = _CAPACITY_LANES
    load_block = _BE16.unpack
    load_rate = _LE17.unpack
    store_h = _BE8.pack_into
    store_k = _LE8.pack_into
    offset = _CHAIN_OFFSET

    h = compress(iv, load_block(kmd_m))                 # H_1
    for _ in range(ITERATIONS - 1):                     # i = 2 .. 1024
        store_h(kmd_k, offset, *h)                      # H_(i-1) -> stage-1 block
        # SHAKE256: absorb the single padded block into the all-zero state,
        # permute once, squeeze the first 64 bytes (lanes 0..7) as K_i.
        lanes = rounds(load_rate(kmd_k) + capacity, constants)
        store_k(kmd_m, offset, *lanes[:8])              # K_i -> stage-2 block
        h = compress(iv, load_block(kmd_m))             # H_i = SHA-512(b"KMD-M" || K_i)
    return _BE8.pack(*h)


def _check_chunk_size(chunk_size: int) -> None:
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise TypeError("chunk_size must be an int")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")


class KMD2048:
    """Incremental KMD-2048 hasher with a hashlib-like interface.

    >>> h = KMD2048()
    >>> h.update(b"hello ")
    >>> h.update(b"world")
    >>> h.digest() == kmd2048(b"hello world")
    True

    Only the first SHAKE256 call sees the message, so ``update`` simply
    absorbs into that sponge (the prefix b"KMD-K" is absorbed at creation).
    ``digest()`` runs the 1024 iterations and the final expansion each time
    it is called (about 0.2 s in CPython) and does not modify the object.
    """

    __slots__ = ("_sponge",)

    name = "kmd2048"
    digest_size = DIGEST_SIZE
    iterations = ITERATIONS

    def __init__(self, data: BytesLike = b"") -> None:
        self._sponge = Shake256(PREFIX_K)
        self.update(data)

    def update(self, data: BytesLike) -> None:
        """Append ``data`` (any bytes-like object) to the message."""
        self._sponge.update(data)

    def digest(self) -> bytes:
        """Return the 256-byte KMD-2048 digest of the message so far."""
        k1 = self._sponge.digest(CHAIN_SIZE)                  # K_1
        h1024 = _kmd2048_chain(k1)                            # H_1024
        return shake256(PREFIX_FINAL + h1024, DIGEST_SIZE)    # D

    def hexdigest(self) -> str:
        """Return the digest as 512 lowercase hexadecimal digits."""
        return self.digest().hex()

    def copy(self) -> "KMD2048":
        """Return an independent copy of the current state."""
        other = KMD2048.__new__(KMD2048)
        other._sponge = self._sponge.copy()
        return other


def kmd2048(data: BytesLike) -> bytes:
    """Return the 256-byte KMD-2048 digest of ``data`` (any bytes-like object)."""
    return KMD2048(data).digest()


def kmd2048_hex(data: BytesLike) -> str:
    """Return the KMD-2048 digest of ``data`` as 512 lowercase hex digits."""
    return KMD2048(data).digest().hex()


def kmd2048_fileobj(fileobj: BinaryIO, chunk_size: int = DEFAULT_CHUNK_SIZE) -> bytes:
    """Return the KMD-2048 digest of everything readable from ``fileobj``.

    ``fileobj`` must be a blocking binary stream (``read(n)`` returning
    bytes, and ``b""`` at end of stream).  It is read from its current
    position in pieces of at most ``chunk_size`` bytes, so memory use is
    bounded by ``chunk_size`` plus a 136-byte sponge buffer.  Short reads
    are fine.  The result equals ``kmd2048(all_bytes_read)``.
    """
    _check_chunk_size(chunk_size)
    hasher = KMD2048()
    read = fileobj.read
    while True:
        chunk = read(chunk_size)
        if not chunk:
            break
        hasher.update(chunk)
    return hasher.digest()


def kmd2048_file(path: PathLike, chunk_size: int = DEFAULT_CHUNK_SIZE) -> bytes:
    """Return the 256-byte KMD-2048 digest of the file at ``path``.

    The file is streamed in ``chunk_size`` pieces and is never loaded into
    memory as a whole.  The hashed message is the file's exact byte
    content (no framing, no length prefix, no newline translation), so the
    result is identical to ``kmd2048(Path(path).read_bytes())`` for any
    ``chunk_size``; see the module docstring for why chunking cannot
    change the result.
    """
    _check_chunk_size(chunk_size)
    with open(path, "rb") as fileobj:
        return kmd2048_fileobj(fileobj, chunk_size)


# ---------------------------------------------------------------------------
# Command-line interface:  python kmd2048.py [FILE ...]
# ---------------------------------------------------------------------------

_USAGE = """\
usage: python kmd2048.py [FILE ...]

Print the KMD-2048 digest (512 hex digits) of each FILE, like sha512sum.
With no FILE, or when FILE is -, read standard input.
EXPERIMENTAL construction: do not use it to protect real data.
"""


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point; returns the process exit status."""
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    if any(arg in ("-h", "--help") for arg in args):
        sys.stdout.write(_USAGE)
        return 0
    if "--version" in args:
        sys.stdout.write(f"kmd2048.py {__version__}\n")
        return 0
    status = 0
    for name in args or ["-"]:
        try:
            if name == "-":
                digest = kmd2048_fileobj(sys.stdin.buffer)
            else:
                digest = kmd2048_file(name)
        except OSError as exc:
            sys.stderr.write(f"kmd2048: {name}: {exc.strerror or exc}\n")
            status = 1
            continue
        sys.stdout.write(f"{digest.hex()}  {name}\n")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
