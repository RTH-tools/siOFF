"""The one off-targets engine: every input form takes the same per-siRNA path.

A file, a folder holding that file and a folder of per-siRNA files (TSV or
Parquet) must give the same output. The siRNA-table checks fire in the parent
before anything is scored, the warnings are logged by the parent (loguru
handlers do not reach spawned workers), a run that needs no parallelism never
starts a pool, and the temporary directories never outlive the run.
"""

import os
import shutil
import tempfile
from pathlib import Path

import polars as pl
import pytest
from loguru import logger
from polars.testing import assert_frame_equal

import sioff.core.off_targets as core
from sioff.core.off_targets import WorkerFailureError, compute_off_targets
from sioff.core.split import split_frame
from sioff.services.risearch_parser import RIsearchParser
from sioff.services.risearch_service import RIsearchService

EX = Path(__file__).resolve().parents[2] / "examples" / "data"
PREDICTIONS = EX / "predictions.out"
SIRNAS = ["siRNA_1", "siRNA_2", "siRNA_3", "siRNA_4", "siRNA_5"]
SORT_KEYS = ["sirna_id", "chrom", "start", "end", "strand", "transcript_id"]
NO_T_WARNING = (
    "--sirna-table on-target IDs are matched after intersection; "
    "without -t/--transcriptome no row can be on-target"
)


def _sequences() -> dict[str, str]:
    lines = (EX / "sirnas.fa").read_text().split()
    return {lines[i][1:]: lines[i + 1] for i in range(0, len(lines), 2)}


def _table(tmp_path: Path, rows: dict[str, str]) -> Path:
    """A siRNA table with ``{sirna_id: on_target_id}`` rows, sequences from the FASTA."""
    seqs = _sequences()
    path = tmp_path / "sirna_table.tsv"
    path.write_text(
        "sirna_id\ton_target_id\tsequence\n"
        + "".join(f"{sid}\t{on}\t{seqs[sid]}\n" for sid, on in rows.items())
    )
    return path


def _collect(results) -> pl.DataFrame:
    df = pl.concat([df for df, _ in results])
    return df.sort([k for k in SORT_KEYS if k in df.columns])  # no -t: no transcript_id


@pytest.fixture
def perfect_match(monkeypatch):
    """Fake perfect-match energies (no risearch), checked against the table."""
    seqs = {sid: s.replace("U", "T") for sid, s in _sequences().items()}
    calls: list[str] = []

    def fake(self, sequence, sirna_id="query", matrix="t04"):
        assert sequence == seqs[sirna_id] and matrix == "t04"
        calls.append(sirna_id)
        return -25.0 - SIRNAS.index(sirna_id)

    monkeypatch.setattr(RIsearchService, "self_hybridization_emin", fake)
    return calls


@pytest.fixture
def scored(monkeypatch):
    """Record `_score_part` calls instead of scoring (seen in-process only)."""
    calls: list[tuple] = []
    monkeypatch.setattr(core, "_score_part", lambda *args: calls.append(args))
    return calls


@pytest.fixture
def info_seen():
    seen: list[str] = []
    handler = logger.add(lambda m: seen.append(m.record["message"]), level="INFO")
    yield seen
    logger.remove(handler)


def _forms(tmp_path: Path) -> dict[str, Path]:
    """The same predictions as a file, a folder, and folders of per-siRNA files."""
    one_dir = tmp_path / "one"
    one_dir.mkdir()
    shutil.copy(PREDICTIONS, one_dir / PREDICTIONS.name)

    parquet_dir = tmp_path / "per_sirna_parquet"
    parts = split_frame(RIsearchParser().load(PREDICTIONS), parquet_dir)
    tsv_dir = tmp_path / "per_sirna_tsv"
    tsv_dir.mkdir()
    for sid, part in parts.items():
        # The `sioff search` layout: tab-separated, header = RISEARCH_COLUMNS.
        pl.read_parquet(part).write_csv(tsv_dir / f"{sid}.tsv", separator="\t")
    return {
        "file": PREDICTIONS,
        "folder": one_dir,
        "per_sirna_tsv": tsv_dir,
        "per_sirna_parquet": parquet_dir,
    }


@pytest.mark.parametrize("n_workers", [1, 2])
def test_file_folder_and_split_folder_agree(tmp_path, perfect_match, n_workers):
    table = _table(tmp_path, {sid: "" for sid in SIRNAS})
    runs = {}
    for name, source in _forms(tmp_path).items():
        results = list(
            compute_off_targets(
                risearch_file=source,
                gtf_file=EX / "annotation.gtf",
                accessibility_dir=EX / "accessibility",
                sirna_table=table,
                alpha="0.8",
                gamma="0.8",
                n_workers=n_workers,
            )
        )
        z = {s: r for _, meta in results for s, r in meta["z_per_sirna"].items()}
        runs[name] = (_collect(results), z)

    assert sorted(set(perfect_match)) == SIRNAS
    reference_df, reference_z = runs.pop("file")
    assert sorted(reference_z) == SIRNAS
    assert "P_off_target:alpha=0.8,gamma=0.8" in reference_df.columns
    for name, (df, z) in runs.items():
        assert_frame_equal(df, reference_df)
        assert z == reference_z, name


def test_meta_carries_the_per_sirna_keys(tmp_path):
    results = list(
        compute_off_targets(risearch_file=PREDICTIONS, gtf_file=EX / "annotation.gtf")
    )
    assert len(results) == len(SIRNAS)
    for df, meta in results:
        (sid,) = df["sirna_id"].unique().to_list()
        assert meta["n_sirnas"] == 1 and meta["n_rows"] == df.height
        assert list(meta["z_per_sirna"]) == [sid]
        assert meta["n_on_target_rows"] == {sid: 0}
        assert {"on_target_weights", "on_target_count"} <= meta.keys()
        assert {"load_s", "intersect_s", "prob_s", "total_s"} <= meta["_timings"].keys()


def test_on_split_gets_the_sirna_count_before_scoring(tmp_path):
    calls: list[int] = []
    gen = compute_off_targets(risearch_file=PREDICTIONS, on_split=calls.append)
    assert calls == []
    next(gen)
    assert calls == [len(SIRNAS)]
    gen.close()


@pytest.mark.parametrize("n_workers", [1, 2])
def test_sirna_in_two_files_stops_before_scoring(tmp_path, scored, n_workers):
    d = tmp_path / "in"
    d.mkdir()
    shutil.copy(PREDICTIONS, d / "x.out")
    RIsearchParser().load(PREDICTIONS).write_parquet(d / "x.parquet")
    with pytest.raises(ValueError, match=r"siRNA_1.*x\.out.*x\.parquet"):
        list(compute_off_targets(risearch_file=d, n_workers=n_workers))
    if n_workers == 1:  # a spawned worker would import the unpatched module
        assert scored == []


@pytest.mark.parametrize("n_workers", [1, 2])
def test_missing_table_row_is_an_error_with_alpha_gamma(
    tmp_path, scored, perfect_match, n_workers
):
    table = _table(tmp_path, {sid: "" for sid in SIRNAS[:4]})
    with pytest.raises(ValueError, match="no row in --sirna-table") as info:
        list(
            compute_off_targets(
                risearch_file=PREDICTIONS,
                sirna_table=table,
                alpha="0.8",
                gamma="0.8",
                n_workers=n_workers,
            )
        )
    assert str(info.value).startswith("1 siRNA(s) have predictions")
    assert str(info.value).endswith("E_min: siRNA_5")
    assert perfect_match == [], "perfect-match energies come after the checks"
    if n_workers == 1:
        assert scored == []


def test_no_table_is_an_error_with_alpha_gamma(tmp_path, scored):
    with pytest.raises(ValueError, match=r"5 siRNA\(s\) have predictions but no row"):
        list(compute_off_targets(risearch_file=PREDICTIONS, alpha="0.8", gamma="0.9"))
    assert scored == []


@pytest.mark.parametrize("n_workers", [1, 2])
def test_missing_table_row_is_a_warning_at_baseline(tmp_path, warnings_seen, n_workers):
    table = _table(tmp_path, {sid: "" for sid in SIRNAS[:4]})
    results = list(
        compute_off_targets(
            risearch_file=PREDICTIONS, sirna_table=table, n_workers=n_workers
        )
    )
    assert len(results) == len(SIRNAS), "the siRNA is still scored"
    hits = [m for m in warnings_seen if "no row in --sirna-table" in m]
    assert len(hits) == 1 and hits[0].startswith("1 siRNA(s)")
    assert hits[0].endswith("siRNA_5")


def test_no_table_at_baseline_is_silent(tmp_path, warnings_seen):
    list(compute_off_targets(risearch_file=PREDICTIONS, gtf_file=EX / "annotation.gtf"))
    assert warnings_seen == []


def test_unused_table_rows_are_counted(tmp_path, info_seen):
    one = tmp_path / "one.out"
    one.write_text(
        "".join(
            line + "\n"
            for line in PREDICTIONS.read_text().splitlines()
            if line.startswith("siRNA_1\t")
        )
    )
    table = _table(tmp_path, {sid: "" for sid in SIRNAS})
    list(compute_off_targets(risearch_file=one, sirna_table=table))
    assert "4 siRNA(s) in --sirna-table have no predictions" in info_seen


@pytest.mark.parametrize("n_workers", [1, 2])
def test_missing_on_target_row_is_warned_in_the_parent(
    tmp_path, warnings_seen, n_workers
):
    table = _table(
        tmp_path,
        {sid: "no_such_transcript" if sid == "siRNA_1" else "" for sid in SIRNAS},
    )
    list(
        compute_off_targets(
            risearch_file=PREDICTIONS,
            gtf_file=EX / "annotation.gtf",
            sirna_table=table,
            n_workers=n_workers,
        )
    )
    hits = [m for m in warnings_seen if "no on-target row" in m]
    assert hits == ["1 siRNA(s) have no on-target row (W_on = 0): siRNA_1"]


@pytest.mark.parametrize("n_workers", [1, 2])
def test_table_without_transcriptome_warns(tmp_path, warnings_seen, n_workers):
    table = _table(tmp_path, {"siRNA_1": "transcript_1", "siRNA_2": ""})
    list(
        compute_off_targets(
            risearch_file=PREDICTIONS, sirna_table=table, n_workers=n_workers
        )
    )
    assert warnings_seen.count(NO_T_WARNING) == 1
    # One warning says it all: no per-siRNA "no on-target row" on top of it.
    assert not [m for m in warnings_seen if "no on-target row" in m]


def _raise_if_called(*args, **kwargs):
    raise AssertionError("a process pool was started")


def test_one_worker_never_builds_a_pool(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "ProcessPoolExecutor", _raise_if_called)
    assert len(_collect(compute_off_targets(risearch_file=PREDICTIONS, n_workers=1)))


def test_one_sirna_never_builds_a_pool(tmp_path, monkeypatch):
    one = tmp_path / "one.out"
    one.write_text(
        "".join(
            line + "\n"
            for line in PREDICTIONS.read_text().splitlines()
            if line.startswith("siRNA_3\t")
        )
    )
    monkeypatch.setattr(core, "ProcessPoolExecutor", _raise_if_called)
    df = _collect(compute_off_targets(risearch_file=one, n_workers=8))
    assert df["sirna_id"].unique().to_list() == ["siRNA_3"]


def test_in_process_run_leaves_no_worker_state(tmp_path):
    list(compute_off_targets(risearch_file=PREDICTIONS, gtf_file=EX / "annotation.gtf"))
    assert core._WORKER_DF_TRANS is None and core._WORKER_PROB_SERVICE is None


def test_interleaved_in_process_runs_keep_their_own_state(tmp_path):
    """Two lazy in-process runs consumed in turn must not share worker state."""
    alone_t = _collect(
        compute_off_targets(risearch_file=PREDICTIONS, gtf_file=EX / "annotation.gtf")
    )
    alone_no_t = _collect(compute_off_targets(risearch_file=PREDICTIONS))
    gens = [
        compute_off_targets(risearch_file=PREDICTIONS, gtf_file=EX / "annotation.gtf"),
        compute_off_targets(risearch_file=PREDICTIONS),
    ]
    outs: list[list] = [[], []]
    while gens[0] is not None or gens[1] is not None:
        for i, gen in enumerate(gens):
            item = None if gen is None else next(gen, None)
            if item is None:
                gens[i] = None
            else:
                outs[i].append(item)
    assert_frame_equal(_collect(outs[0]), alone_t)
    assert_frame_equal(_collect(outs[1]), alone_no_t)


@pytest.mark.parametrize("n_workers", [1, 2])
def test_failed_input_is_reported_after_the_good_sirnas(tmp_path, n_workers):
    d = tmp_path / "in"
    d.mkdir()
    shutil.copy(PREDICTIONS, d / "good.out")
    (d / "broken.parquet").write_bytes(b"this is not a parquet file")
    frames = []
    with pytest.raises(WorkerFailureError) as info:
        for df, _ in compute_off_targets(risearch_file=d, n_workers=n_workers):
            frames.append(df)
    assert len(frames) == len(SIRNAS)
    assert info.value.failed_names == ["broken.parquet"]
    assert info.value.n_total == 1 + len(SIRNAS)


def test_failed_sirna_is_reported_after_the_others(tmp_path, monkeypatch):
    real = core._score_part

    def flaky(sirna_id, *args):
        if sirna_id == "siRNA_3":
            raise RuntimeError("boom")
        return real(sirna_id, *args)

    monkeypatch.setattr(core, "_score_part", flaky)
    frames = []
    with pytest.raises(WorkerFailureError, match=r"1/5 .*siRNA_3: boom"):
        for df, _ in compute_off_targets(risearch_file=PREDICTIONS, n_workers=1):
            frames.append(df)
    assert len(frames) == 4


def test_a_broken_pool_fails_the_later_phase_too():
    """A worker killed in the split phase (say, OOM) breaks the shared pool."""
    executor = core._Executor(
        2,
        {
            "ipc_path": "",
            "accessibility_dir": "",
            "on_target_map": {},
            "self_hyb_emin": {},
        },
    )
    kw = {"threads_per_worker": 1, "uses_worker_state": False}
    try:
        assert list(executor.run(os._exit, {"a": (1,), "b": (1,)}, **kw)) == []
        assert list(executor.run(abs, {"c": (-1,)}, **kw)) == []
    finally:
        executor.close()
    assert sorted(name for name, _ in executor.failures) == ["a", "b", "c"]


@pytest.fixture
def tmp_root(tmp_path, monkeypatch):
    """Point every tempfile call (this process and spawned workers) at one dir."""
    root = tmp_path / "tmp"
    root.mkdir()
    monkeypatch.setenv("TMPDIR", str(root))
    monkeypatch.setattr(tempfile, "tempdir", str(root))
    return root


def _engine_dirs(root: Path) -> list[str]:
    return sorted(
        p.name
        for p in root.iterdir()
        if p.name.startswith(("sioff_split_", "risearch_ipc_"))
    )


def test_failure_mid_split_leaves_no_temp_dirs(tmp_path, tmp_root):
    d = tmp_path / "in"
    d.mkdir()
    shutil.copy(PREDICTIONS, d / "good.out")
    (d / "corrupt.out.gz").write_bytes(b"\x1f\x8b not really gzip")
    during: list[str] = []
    with pytest.raises(WorkerFailureError, match="corrupt.out.gz"):
        list(
            compute_off_targets(
                risearch_file=d,
                gtf_file=EX / "annotation.gtf",
                n_workers=2,
                on_split=lambda n: during.extend(_engine_dirs(tmp_root)),
            )
        )
    assert [n.split("_")[1] for n in during] == ["ipc", "split"]
    assert not list(tmp_root.glob("sioff_split_*"))
    assert not list(tmp_root.glob("risearch_ipc_*"))


@pytest.mark.parametrize("ending", ["success", "duplicate", "closed early"])
def test_temp_dirs_are_removed_on_every_path(tmp_path, tmp_root, ending):
    d = tmp_path / "in"
    d.mkdir()
    shutil.copy(PREDICTIONS, d / "x.out")
    if ending == "duplicate":
        RIsearchParser().load(PREDICTIONS).write_parquet(d / "x.parquet")
    gen = compute_off_targets(
        risearch_file=d, gtf_file=EX / "annotation.gtf", n_workers=2
    )
    if ending == "duplicate":
        with pytest.raises(ValueError):
            list(gen)
    elif ending == "closed early":
        next(gen)
        assert len(_engine_dirs(tmp_root)) == 2
        gen.close()
    else:
        assert len(list(gen)) == len(SIRNAS)
    assert _engine_dirs(tmp_root) == []


def test_perfect_match_energy_comes_from_the_table_sequence(tmp_path):
    pytest.importorskip("risearch")
    table = _table(tmp_path, {sid: "" for sid in SIRNAS})
    df = _collect(
        compute_off_targets(
            risearch_file=PREDICTIONS,
            gtf_file=EX / "annotation.gtf",
            sirna_table=table,
            alpha="0.8",
            gamma="0.8",
            detailed_report=True,
        )
    )
    seqs = _sequences()
    for sid in SIRNAS:
        rows = df.filter(pl.col("sirna_id") == sid)
        perfect = RIsearchService().self_hybridization_emin(seqs[sid], sid)
        expected = min(perfect, float(rows["energy"].min()))
        assert rows["E_min"].unique().to_list() == [pytest.approx(expected, abs=1e-5)]
