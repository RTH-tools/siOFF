"""Exons with expression <= 0 are dropped right after the annotation is loaded.

A site on a transcript with TPM 0 has Boltzmann weight 0, so it changes no
partition function, but on GENCODE v47 + H1299 TPM more than half the exon
lines (1.21M of 2.16M) are such transcripts: about half of every siRNA's
site x transcript rows went through intersection, accessibility lookup and
scoring for nothing. The legacy pipeline's GTF path skips them too.
"""

from pathlib import Path

import polars as pl
import pytest

from sioff.core.off_targets import _drop_unexpressed, compute_off_targets_directory


def test_drop_unexpressed_keeps_only_positive_expression():
    df = pl.DataFrame(
        {"exp_value": [5.0, 0.0, -1.0, 0.25]}, schema={"exp_value": pl.Float32}
    )
    assert _drop_unexpressed(df)["exp_value"].to_list() == [5.0, 0.25]


def _gtf(path: Path, with_zero: bool) -> Path:
    lines = [
        'chr1\tt\texon\t100\t300\t.\t+\t.\tgene_id "g1"; transcript_id "t1"; TPM "4.0";',
        'chr1\tt\texon\t150\t400\t.\t+\t.\tgene_id "g1"; transcript_id "t2"; TPM "2.5";',
        'chr1\tt\texon\t500\t700\t.\t-\t.\tgene_id "g2"; transcript_id "t3"; TPM "7.0";',
    ]
    if with_zero:
        lines += [
            'chr1\tt\texon\t100\t300\t.\t+\t.\tgene_id "g1"; transcript_id "t4"; TPM "0.000000";',
            'chr1\tt\texon\t480\t720\t.\t-\t.\tgene_id "g3"; transcript_id "t5"; TPM "0";',
        ]
    path.write_text("\n".join(lines) + "\n")
    return path


def _hits(d: Path) -> Path:
    d.mkdir()
    (d / "risearch_s1.out").write_text(
        "s1\t1\t19\tchr1\t160\t180\t+\t-22.5\n"
        "s1\t1\t19\tchr1\t290\t310\t+\t-18.0\n"
        "s1\t2\t18\tchr1\t600\t620\t-\t-15.0\n"
    )
    (d / "risearch_s2.out").write_text(
        "s2\t1\t19\tchr1\t120\t140\t+\t-25.0\ns2\t1\t19\tchr1\t490\t505\t-\t-12.0\n"
    )
    return d


def _run(hits: Path, gtf: Path):
    metas, rows = {}, 0
    for df, meta in compute_off_targets_directory(
        input_dir=hits,
        gtf_file=gtf,
        expression_metric="TPM",
        alpha="0.8;1",
        gamma="0.9;1",
    ):
        rows += df.height
        metas.update(meta["z_per_sirna"])
    return metas, rows


def test_zero_tpm_transcripts_change_no_partition_function(tmp_path):
    hits = _hits(tmp_path / "hits")
    with_zero, rows_with = _run(hits, _gtf(tmp_path / "a.gtf", with_zero=True))
    without, rows_without = _run(hits, _gtf(tmp_path / "b.gtf", with_zero=False))
    assert rows_with == rows_without  # the zero-TPM rows never reach scoring
    assert set(with_zero) == set(without) == {"s1", "s2"}
    for sid in without:
        for k, v in without[sid].items():
            if k != "sirna_id":
                assert with_zero[sid][k] == pytest.approx(v, rel=1e-9), (sid, k)
