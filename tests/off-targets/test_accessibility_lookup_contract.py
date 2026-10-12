"""What an opening-energy lookup must return, checked from first principles.

Every other accessibility test compares one lookup path against another, or uses
a single u column on the plus strand only, so a wrong row or column on the minus
strand passes all of them as long as every path makes the same mistake. These
tests state the expected value in chromosome coordinates, independently of how
any path stores or indexes a profile:

* the RNAplfold convention: ``u<k>`` at position ``i`` is the opening energy of
  the ``k`` unpaired bases ENDING at ``i`` on the molecule that was folded;
* a hit ``[s, e]`` (1-based, inclusive, length ``L = e - s + 1``, column clamped
  to the widest ``u``) needs the stretch that covers exactly ``[s, e]`` on its
  own transcript;
* plus strand: the forward sequence, stretch ending at ``e`` -> ``(e, L)``;
* minus strand: the reverse complement, where chromosome position ``c`` is
  RC position ``N - c + 1``; the stretch ``[s, e]`` ends at RC position
  ``N - s + 1`` -> ``(N - s + 1, L)`` in the minus rows of the Parquet, which
  ``_fold_full_chromosome`` writes in RC coordinates.

The minus-strand cases are xfail(strict=True): every lookup path currently reads
RC row ``s`` (chromosome position ``N - s + 1``) instead. Remove the markers with
the fix; strict makes an unannounced fix fail loudly.
"""

from __future__ import annotations

import math
import random

import numpy as np
import polars as pl
import pytest

from sioff.services.accessibility import (
    GenomeAccessibilityService,
    _fold_full_chromosome,
    gather_opening_energies,
)

MINUS_BUG = pytest.mark.xfail(
    strict=True,
    reason="minus-strand lookup reads RC row s (chromosome N-s+1) instead of RC row N-s+1",
)

CHROM = "chrT"

# --------------------------------------------------------------------------
# Lookup paths. Each takes (parquet_dir, path_to_parquet, cache_dir, hits) and
# returns float32 energies in hit order. hits = list of (strand, start, end).
# --------------------------------------------------------------------------


def _hits_frame(hits):
    return pl.DataFrame(
        {
            "chrom": [CHROM] * len(hits),
            "start": [h[1] for h in hits],
            "end": [h[2] for h in hits],
            "strand": [h[0] for h in hits],
        },
        schema={
            "chrom": pl.Utf8,
            "start": pl.Int32,
            "end": pl.Int32,
            "strand": pl.Utf8,
        },
    )


def _annotate(service, hits):
    out = service.annotate_opening_energy_vectorized(_hits_frame(hits))
    return out["opening_energy"].to_numpy()


def _path_eager(pdir, ppath, cdir, hits):
    return _annotate(GenomeAccessibilityService(pdir), hits)


def _path_frames(pdir, ppath, cdir, hits):
    frames = {CHROM: pl.read_parquet(ppath)}
    return _annotate(GenomeAccessibilityService.from_frames(frames), hits)


def _path_memmap(pdir, ppath, cdir, hits):
    return _annotate(GenomeAccessibilityService(pdir, cache_dir=cdir), hits)


def _path_query_single(pdir, ppath, cdir, hits):
    service = GenomeAccessibilityService(pdir)
    return np.array(
        [service.query_single(CHROM, s, e, strand) for strand, s, e in hits],
        dtype=np.float32,
    )


def _path_chrom_major(pdir, ppath, cdir, hits):
    energies, _ = gather_opening_energies(
        ppath,
        np.array([h[0] for h in hits]),
        np.array([h[1] for h in hits], dtype=np.int64),
        np.array([h[2] for h in hits], dtype=np.int64),
    )
    return energies


PATHS = {
    "eager": _path_eager,
    "from_frames": _path_frames,
    "memmap": _path_memmap,
    "query_single": _path_query_single,
    "chrom_major": _path_chrom_major,
}


# --------------------------------------------------------------------------
# A. Layout contract on a synthetic profile with a unique value per cell.
# --------------------------------------------------------------------------

N = 12  # contig length
N_U = 5  # u1..u5
MINUS_OFFSET = 13.0


def _cell(strand: str, file_pos: int, u: int) -> float:
    """Unique, <= 25.5 (fits the uint8 cache without the sidecar), 0.1-exact."""
    base = 0.0 if strand == "+" else MINUS_OFFSET
    return round(base + file_pos + u / 10.0, 1)


@pytest.fixture
def synthetic(tmp_path):
    rows = []
    for strand in ("+", "-"):
        for p in range(1, N + 1):
            rows.append([p, strand] + [_cell(strand, p, u) for u in range(1, N_U + 1)])
    df = pl.DataFrame(
        rows,
        schema={"position": pl.Int32, "strand": pl.Utf8}
        | {f"u{u}": pl.Float32 for u in range(1, N_U + 1)},
        orient="row",
    )
    pdir = tmp_path / "acc"
    pdir.mkdir()
    ppath = pdir / f"{CHROM}.accessibility.parquet"
    df.write_parquet(ppath)
    return pdir, ppath, tmp_path / "cache"


def _expected(strand: str, s: int, e: int) -> float:
    if s < 1 or e > N:
        return 10.0
    u = min(e - s + 1, N_U)
    if strand == "+":
        return _cell("+", e, u)
    return _cell("-", N - s + 1, u)


# (strand, start, end, case id); in-bounds minus cases carry the xfail marker.
_CASES = [
    ("L1", 5, 5),
    ("L_eq_nu", 3, 7),
    ("L_gt_nu_clamped", 2, 9),
    ("start_at_1", 1, 4),
    ("end_at_N", 9, 12),
]


def _contract_params():
    params = []
    for path in PATHS:
        for name, s, e in _CASES:
            params.append(pytest.param(path, "+", s, e, id=f"{path}-plus-{name}"))
            params.append(
                pytest.param(
                    path, "-", s, e, id=f"{path}-minus-{name}", marks=MINUS_BUG
                )
            )
        for strand in ("+", "-"):
            params.append(
                pytest.param(path, strand, 10, 13, id=f"{path}-{strand}-out_of_bounds")
            )
    return params


@pytest.mark.parametrize("path,strand,s,e", _contract_params())
def test_lookup_returns_the_cell_covering_the_site(synthetic, path, strand, s, e):
    pdir, ppath, cdir = synthetic
    got = PATHS[path](pdir, ppath, cdir, [(strand, s, e)])
    assert got[0] == pytest.approx(_expected(strand, s, e), abs=1e-4), (
        f"{path} {strand} [{s},{e}]: got {got[0]}, expected {_expected(strand, s, e)}"
    )


# --------------------------------------------------------------------------
# B. Physics oracle: fold the sequence here, independently of the writer's
#    coordinate handling, and compare with what the lookup returns.
# --------------------------------------------------------------------------

W, SPAN, U, TEMP = 80, 40, 30, 37.0
_RT = 0.001987 * (TEMP + 273.15)
_COMPLEMENT = {"A": "U", "C": "G", "G": "C", "U": "A"}
SEQ_LEN = 200

# (s, e) sites, well inside the sequence so every u <= 30 is defined.
SITES = [(61, 81), (90, 108), (120, 142), (150, 160)]


def _random_rna(n: int, seed: int = 20261012) -> str:
    rng = random.Random(seed)
    seq = "".join(rng.choice("ACGU") for _ in range(n))
    rc = "".join(_COMPLEMENT[b] for b in reversed(seq))
    assert seq != rc, "need a non-palindromic test sequence"
    return seq


def _fold_up(rna: str) -> list:
    """Per position i (1-based), the list v with v[u] = P(u bases ending at i unpaired)."""
    RNA = pytest.importorskip("RNA")
    md = RNA.md()
    md.temperature = TEMP
    md.window_size = min(W, len(rna))
    md.max_bp_span = min(SPAN, len(rna))
    fc = RNA.fold_compound(rna, md, RNA.OPTION_WINDOW)
    up: list = [None] * (len(rna) + 2)

    def cb(v, v_size, i, maxsize, what, data):
        if (what & RNA.PROBS_WINDOW_UP) and v is not None:
            up[i] = list(v)

    fc.probs_window(U, RNA.PROBS_WINDOW_UP, cb)
    return up


def _energy(up, i: int, u: int) -> float:
    p = up[i][u]
    assert p is not None and p > 0, f"no probability for u={u} ending at {i}"
    return round(-_RT * math.log(p) * 10.0) / 10.0


@pytest.fixture(scope="module")
def folded(tmp_path_factory):
    seq = _random_rna(SEQ_LEN)
    pdir = tmp_path_factory.mktemp("acc_real")
    ppath = pdir / f"{CHROM}.accessibility.parquet"
    _fold_full_chromosome(seq.replace("U", "T"), CHROM, W, SPAN, U, str(ppath))
    rc = "".join(_COMPLEMENT[b] for b in reversed(seq))
    return {
        "pdir": pdir,
        "ppath": ppath,
        "cdir": tmp_path_factory.mktemp("acc_real_cache"),
        "up_fwd": _fold_up(seq),
        "up_rc": _fold_up(rc),
    }


def _oracle(folded, strand: str, s: int, e: int) -> float:
    L = min(e - s + 1, U)
    if strand == "+":
        return _energy(folded["up_fwd"], e, L)
    return _energy(folded["up_rc"], SEQ_LEN - s + 1, L)


def _oracle_params():
    params = []
    for path in PATHS:
        params.append(pytest.param(path, "+", id=f"{path}-plus"))
        params.append(pytest.param(path, "-", id=f"{path}-minus", marks=MINUS_BUG))
    return params


@pytest.mark.parametrize("path,strand", _oracle_params())
def test_lookup_matches_rnaplfold_on_the_sites_own_transcript(folded, path, strand):
    hits = [(strand, s, e) for s, e in SITES]
    got = PATHS[path](folded["pdir"], folded["ppath"], folded["cdir"], hits)
    want = [_oracle(folded, strand, s, e) for s, e in SITES]
    np.testing.assert_allclose(got, want, atol=1e-4, err_msg=f"{path} {strand}")


def test_plus_and_minus_differ_for_the_same_window(folded):
    """Folding the reverse complement is a different molecule: if plus and minus
    gave the same energies for the same chromosome window, the minus lookup would
    be reading the plus profile."""
    plus = [_oracle(folded, "+", s, e) for s, e in SITES]
    minus = [_oracle(folded, "-", s, e) for s, e in SITES]
    assert plus != minus


def test_oracle_sees_what_the_writer_wrote(folded):
    """Guard for the oracle itself: plus-strand rows of the written Parquet equal
    an independent fold of the forward sequence, cell by cell, so a failure of
    the minus cases above cannot be blamed on a mismatched oracle setup."""
    df = pl.read_parquet(folded["ppath"]).filter(pl.col("strand") == "+")
    up = folded["up_fwd"]
    for row in df.filter(pl.col("position").is_between(40, 160)).iter_rows(named=True):
        i = row["position"]
        for u in (1, 10, 20, 30):
            assert row[f"u{u}"] == pytest.approx(_energy(up, i, u), abs=1e-4)
    rc_df = pl.read_parquet(folded["ppath"]).filter(pl.col("strand") == "-")
    up_rc = folded["up_rc"]
    for row in rc_df.filter(pl.col("position").is_between(40, 160)).iter_rows(
        named=True
    ):
        i = row["position"]
        for u in (1, 10, 20, 30):
            assert row[f"u{u}"] == pytest.approx(_energy(up_rc, i, u), abs=1e-4)
