"""The one off-targets engine: every input form takes the same per-siRNA path.

A file, a folder holding that file and a folder of per-siRNA files (TSV or
Parquet) must give the same output. The siRNA-table checks fire in the parent
before anything is scored, the warnings are logged by the parent (loguru
handlers do not reach spawned workers), a run that needs no parallelism never
starts a pool, and the temporary directories never outlive the run.
"""

import os
import re
import shutil
import tempfile
from pathlib import Path

import polars as pl
import pytest
from loguru import logger
from polars.testing import assert_frame_equal

import sioff
import sioff.core.off_targets as core
from sioff.core.off_targets import WorkerFailureError, compute_off_targets
from sioff.core.split import split_frame
from sioff.services.risearch_parser import RIsearchParser
from sioff.services.risearch_service import RIsearchService

EX = Path(__file__).resolve().parents[2] / "examples" / "data"
PREDICTIONS = EX / "predictions.out"
GENOME = EX / "genome.fa"
SIRNAS = ["siRNA_1", "siRNA_2", "siRNA_3", "siRNA_4", "siRNA_5"]
# energy breaks ties: a search can report two hits on one site.
SORT_KEYS = ["sirna_id", "chrom", "start", "end", "strand", "transcript_id", "energy"]
NO_T_WARNING = (
    "--sirna-table on-target IDs are matched after intersection; "
    "without -t/--transcriptome no row can be on-target"
)
SIRNA_FASTA_ERROR = (
    "sirna_fasta is the query for the in-process search and cannot be combined "
    "with risearch_file or predictions; the perfect-match energy for E_min now "
    "comes from the siRNA table"
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
    # Scored in place, not split: the engine must still read only what `load`
    # keeps (sirna_id as text, no extra columns).
    raw_dir = tmp_path / "per_sirna_parquet_raw"
    raw_dir.mkdir()
    for sid, part in parts.items():
        # The `sioff search` layout: tab-separated, header = RISEARCH_COLUMNS.
        pl.read_parquet(part).write_csv(tsv_dir / f"{sid}.tsv", separator="\t")
        pl.read_parquet(part).with_columns(
            pl.col("sirna_id").cast(pl.Categorical), raw_e_min=pl.lit(-1.0)
        ).write_parquet(raw_dir / f"{sid}.parquet")
    return {
        "file": PREDICTIONS,
        "folder": one_dir,
        "per_sirna_tsv": tsv_dir,
        "per_sirna_parquet": parquet_dir,
        "per_sirna_parquet_raw": raw_dir,
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


def test_on_split_gets_the_sirna_count_before_scoring(tmp_path, monkeypatch):
    events: list = []
    real = core._score_part

    def recording(*args):
        events.append("scored")
        return real(*args)

    monkeypatch.setattr(core, "_score_part", recording)  # in-process: n_workers=1
    gen = compute_off_targets(
        risearch_file=PREDICTIONS, n_workers=1, on_split=events.append
    )
    assert events == []
    next(gen)
    assert events == [len(SIRNAS), "scored"]
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
        if p.name.startswith(
            ("sioff_split_", "risearch_ipc_", "sioff_acc_", "sioff_fold_")
        )
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


@pytest.mark.parametrize(
    "ending", ["success", "duplicate", "closed early", "interrupted"]
)
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
    elif ending == "interrupted":
        # Ctrl-C while the caller holds a frame: not an Exception, so the
        # engine must not swallow it, and its finally must still run.
        next(gen)
        assert len(_engine_dirs(tmp_root)) == 2
        with pytest.raises(KeyboardInterrupt):
            gen.throw(KeyboardInterrupt)
    else:
        assert len(list(gen)) == len(SIRNAS)
    assert _engine_dirs(tmp_root) == []


@pytest.mark.parametrize("n_workers", [1, 2])
def test_per_sirna_parquet_is_scored_in_place(tmp_path, tmp_root, n_workers):
    # A Parquet file that holds one siRNA is its own part: the split copies nothing.
    copied: list[Path] = []
    results = list(
        compute_off_targets(
            risearch_file=_forms(tmp_path)["per_sirna_parquet_raw"],
            n_workers=n_workers,
            on_split=lambda n: copied.extend(tmp_root.glob("sioff_split_*/**/*")),
        )
    )
    assert len(results) == len(SIRNAS)
    assert copied == []


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


# ---------------------------------------------------------------------------
# Review findings carried into B5
# ---------------------------------------------------------------------------
def _without_rows_for_sirna_1(tmp_path: Path, cause: str) -> Path:
    """The predictions, with every siRNA_1 row lost to --sense-only or to -t."""
    is_1 = pl.col("sirna_id") == "siRNA_1"
    column, value = ("strand", "-") if cause == "sense_only" else ("chrom", "nowhere")
    path = tmp_path / "predictions.parquet"
    RIsearchParser().load(PREDICTIONS).with_columns(
        pl.when(is_1).then(pl.lit(value)).otherwise(pl.col(column)).alias(column)
    ).write_parquet(path)
    return path


@pytest.mark.parametrize("n_workers", [1, 2])
@pytest.mark.parametrize("cause", ["sense_only", "intersection"])
def test_on_target_sirna_with_no_rows_left_is_warned(
    tmp_path, warnings_seen, cause, n_workers
):
    table = _table(
        tmp_path, {sid: "transcript_1" if sid == "siRNA_1" else "" for sid in SIRNAS}
    )
    results = list(
        compute_off_targets(
            risearch_file=_without_rows_for_sirna_1(tmp_path, cause),
            gtf_file=EX / "annotation.gtf",
            sirna_table=table,
            sense_only=cause == "sense_only",
            n_workers=n_workers,
        )
    )
    scored_ids = {sid for df, _ in results for sid in df["sirna_id"].unique()}
    assert "siRNA_1" not in scored_ids and scored_ids
    hits = [m for m in warnings_seen if "no on-target row" in m]
    assert hits == ["1 siRNA(s) have no on-target row (W_on = 0): siRNA_1"]


def test_failed_in_process_init_still_resets_the_worker_state(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("no probability service")

    # _init_worker sets the transcriptome globals, then fails on this.
    monkeypatch.setattr(core, "ProbabilityService", broken)
    # Put back at teardown even if the reset under test is missing.
    monkeypatch.setattr(core, "_WORKER_DF_TRANS", None)
    monkeypatch.setattr(core, "_WORKER_INTERSECTOR", None)
    with pytest.raises(WorkerFailureError, match="no probability service"):
        list(
            compute_off_targets(
                risearch_file=PREDICTIONS, gtf_file=EX / "annotation.gtf", n_workers=1
            )
        )
    assert core._WORKER_DF_TRANS is None
    assert core._WORKER_INTERSECTOR is None


def test_reset_worker_state_does_not_share_the_defaults():
    list(compute_off_targets(risearch_file=PREDICTIONS, n_workers=1))
    for name, default in core._WORKER_STATE_DEFAULTS.items():
        if isinstance(default, dict):
            live = getattr(core, name)
            assert live == {} and live is not default, name


def test_temp_dirs_are_removed_when_close_fails(tmp_root, monkeypatch):
    real_close = core._Executor.close

    def failing_close(self):
        real_close(self)
        raise RuntimeError("close failed")

    monkeypatch.setattr(core._Executor, "close", failing_close)
    during: list[str] = []
    with pytest.raises(RuntimeError, match="close failed"):
        list(
            compute_off_targets(
                risearch_file=PREDICTIONS,
                gtf_file=EX / "annotation.gtf",
                on_split=lambda n: during.extend(_engine_dirs(tmp_root)),
            )
        )
    assert len(during) == 2
    assert _engine_dirs(tmp_root) == []


# ---------------------------------------------------------------------------
# In-memory predictions, the in-process search, in-memory profiles, folding
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def profiles() -> dict[str, pl.DataFrame]:
    """The example genome folded in memory, with the engine's fold defaults."""
    return sioff.accessibility(genome=GENOME)


@pytest.mark.parametrize("source", ["risearch_file", "predictions"])
def test_sirna_fasta_is_only_the_search_query(tmp_root, scored, source):
    given = (
        {"risearch_file": PREDICTIONS}
        if source == "risearch_file"
        else {"predictions": RIsearchParser().load(PREDICTIONS)}
    )
    gen = compute_off_targets(sirna_fasta=EX / "sirnas.fa", **given)
    with pytest.raises(ValueError) as info:
        next(gen)
    assert str(info.value) == SIRNA_FASTA_ERROR
    assert scored == [] and _engine_dirs(tmp_root) == []


# ---------------------------------------------------------------------------
# Arguments the run would ignore, and missing files, fail before any work
# ---------------------------------------------------------------------------
def _refused(tmp_root, scored, **kwargs) -> str:
    """The ValueError of the first ``next()``, after checking nothing ran."""
    with pytest.raises(ValueError) as info:
        next(compute_off_targets(**kwargs))
    assert scored == [] and _engine_dirs(tmp_root) == []
    return str(info.value)


@pytest.mark.parametrize("given", ["accessibility", "accessibility_dir"])
def test_genome_file_with_profiles_is_refused(tmp_root, scored, profiles, given):
    source = profiles if given == "accessibility" else EX / "accessibility"
    message = _refused(
        tmp_root,
        scored,
        risearch_file=PREDICTIONS,
        genome_file=GENOME,
        **{given: source},
    )
    assert message == (
        f"genome_file and {given} are two sources for the same profiles; drop "
        f"genome_file to use {given}, or drop {given} to fold the genome"
    )


CACHE_DIR_IGNORED = {
    "accessibility": "accessibility is packed in a temporary directory, so drop "
    "accessibility_cache_dir",
    "genome_file": "genome_file is folded in a temporary directory, so drop "
    "accessibility_cache_dir",
    "nothing": "without accessibility_dir it is unused, so pass accessibility_dir "
    "or drop accessibility_cache_dir",
}


@pytest.mark.parametrize("given", list(CACHE_DIR_IGNORED))
def test_cache_dir_without_accessibility_dir_is_refused(
    tmp_path, tmp_root, scored, profiles, given
):
    source = {
        "accessibility": {"accessibility": profiles},
        "genome_file": {"genome_file": GENOME},
        "nothing": {},
    }[given]
    message = _refused(
        tmp_root,
        scored,
        risearch_file=PREDICTIONS,
        accessibility_cache_dir=tmp_path / "cache",
        **source,
    )
    assert message == (
        "accessibility_cache_dir only holds the cache of accessibility_dir; "
        + CACHE_DIR_IGNORED[given]
    )
    assert not (tmp_path / "cache").exists()


def test_missing_sirna_table_fails_before_the_profiles(tmp_path, tmp_root):
    """A typo in the table path fails before -a is checked or packed."""
    empty = tmp_path / "empty"  # would raise AccessibilityError if packed first
    empty.mkdir()
    ghost = tmp_path / "missing.tsv"
    with pytest.raises(
        FileNotFoundError, match=re.escape(f"siRNA table not found: {ghost}")
    ):
        next(
            compute_off_targets(
                risearch_file=PREDICTIONS, accessibility_dir=empty, sirna_table=ghost
            )
        )
    assert _engine_dirs(tmp_root) == []


@pytest.mark.parametrize("target", ["target_fasta", "target_index"])
@pytest.mark.parametrize("source", ["risearch_file", "predictions"])
def test_search_target_with_predictions_is_refused(tmp_root, scored, target, source):
    given = (
        {"risearch_file": PREDICTIONS}
        if source == "risearch_file"
        else {"predictions": RIsearchParser().load(PREDICTIONS)}
    )
    message = _refused(tmp_root, scored, **given, **{target: GENOME})
    assert message == (
        f"{target} is only used by the in-process search (sirna_fasta) and "
        f"cannot be combined with {source}; drop {target}"
    )


@pytest.mark.parametrize("missing", ["gtf_file", "target_fasta", "target_index"])
def test_missing_file_fails_before_the_fold(tmp_path, tmp_root, missing):
    ghost = tmp_path / "missing"
    given = {
        "gtf_file": {"risearch_file": PREDICTIONS, "gtf_file": ghost},
        "target_fasta": {"sirna_fasta": EX / "sirnas.fa", "target_fasta": ghost},
        "target_index": {
            "sirna_fasta": EX / "sirnas.fa",
            "target_fasta": GENOME,
            "target_index": ghost,
        },
    }[missing]
    progress: list[dict] = []
    with pytest.raises(FileNotFoundError, match=re.escape(str(ghost))):
        next(
            compute_off_targets(
                genome_file=GENOME,
                accessibility_progress_callback=lambda **kw: progress.append(kw),
                **given,
            )
        )
    assert progress == [], "the genome was folded first"
    assert _engine_dirs(tmp_root) == []


def test_transcriptome_is_read_before_the_fold(tmp_path, tmp_root):
    bed = tmp_path / "annotation.bed"
    bed.write_text("chr1\t100\t200\tt1\t0\t+\n")
    progress: list[dict] = []
    with pytest.raises(ValueError, match="BED annotations are not supported"):
        next(
            compute_off_targets(
                risearch_file=PREDICTIONS,
                gtf_file=bed,
                genome_file=GENOME,
                accessibility_progress_callback=lambda **kw: progress.append(kw),
            )
        )
    assert progress == [], "the genome was folded first"
    assert _engine_dirs(tmp_root) == []


@pytest.mark.parametrize("n_workers", [1, 2])
def test_predictions_frame_matches_the_file(tmp_path, perfect_match, n_workers):
    table = _table(tmp_path, {sid: "" for sid in SIRNAS})

    def run(**source):
        results = list(
            compute_off_targets(
                **source,
                gtf_file=EX / "annotation.gtf",
                accessibility_dir=EX / "accessibility",
                sirna_table=table,
                alpha="0.8",
                gamma="0.8",
                n_workers=n_workers,
            )
        )
        z = {s: r for _, meta in results for s, r in meta["z_per_sirna"].items()}
        return _collect(results), z

    from_file, z_file = run(risearch_file=PREDICTIONS)
    from_frame, z_frame = run(predictions=RIsearchParser().load(PREDICTIONS))
    assert sorted(z_file) == SIRNAS
    assert_frame_equal(from_frame, from_file)
    assert z_frame == z_file


@pytest.mark.parametrize("prebuilt_index", [True, False])
def test_in_process_search_feeds_the_engine(tmp_path, prebuilt_index):
    pytest.importorskip("risearch")
    # A copy, so an index built next to the target never lands in examples/.
    genome = tmp_path / "genome.fa"
    shutil.copy(GENOME, genome)
    hits = sioff.search(
        EX / "sirnas.fa", sioff.index(genome, tmp_path / "ref.idx"), target=genome
    )
    assert hits.height > 0
    kw = {"gtf_file": EX / "annotation.gtf", "accessibility_dir": EX / "accessibility"}
    expected = _collect(compute_off_targets(predictions=hits, **kw))

    index = sioff.index(genome, tmp_path / "prebuilt.idx") if prebuilt_index else None
    got = _collect(
        compute_off_targets(
            sirna_fasta=EX / "sirnas.fa", target_fasta=genome, target_index=index, **kw
        )
    )
    assert_frame_equal(got, expected)
    # Without target_index the index is built next to the target.
    assert (tmp_path / "genome.idx").exists() is not prebuilt_index


def test_in_memory_accessibility_matches_the_directory(tmp_path, profiles):
    acc_dir = tmp_path / "acc"
    acc_dir.mkdir()
    for chrom, frame in profiles.items():
        frame.write_parquet(acc_dir / f"{chrom}.accessibility.parquet")

    def run(**source):
        return _collect(
            compute_off_targets(
                risearch_file=PREDICTIONS,
                gtf_file=EX / "annotation.gtf",
                n_workers=2,
                **source,
            )
        )

    in_memory = run(accessibility=profiles)
    assert (in_memory["opening_energy"] != 0).any(), "the profiles were not used"
    assert_frame_equal(in_memory, run(accessibility_dir=acc_dir), rel_tol=1e-6)


def test_genome_fold_matches_precomputed_profiles(profiles):
    progress: list[dict] = []

    def run(**source):
        return _collect(
            compute_off_targets(
                risearch_file=PREDICTIONS,
                gtf_file=EX / "annotation.gtf",
                n_workers=2,
                **source,
            )
        )

    folded = run(
        genome_file=GENOME,
        accessibility_progress_callback=lambda **kw: progress.append(kw),
    )
    assert progress, "the fold reports its progress"
    assert (folded["opening_energy"] != 0).any(), "the fold was not used"
    assert_frame_equal(folded, run(accessibility=profiles), rel_tol=1e-6)


@pytest.mark.parametrize("ending", ["success", "error", "closed early"])
@pytest.mark.parametrize("source", ["accessibility", "genome_file"])
def test_spills_are_removed(tmp_root, profiles, source, ending):
    prefix = "sioff_acc_" if source == "accessibility" else "sioff_fold_"
    given = (
        {"accessibility": profiles}
        if source == "accessibility"
        else {"genome_file": GENOME}
    )
    during: list[str] = []

    def on_split(n):
        during.extend(_engine_dirs(tmp_root))
        if ending == "error":
            raise RuntimeError("stop")

    gen = compute_off_targets(risearch_file=PREDICTIONS, on_split=on_split, **given)
    if ending == "error":
        with pytest.raises(RuntimeError, match="stop"):
            list(gen)
    elif ending == "closed early":
        next(gen)
        gen.close()
    else:
        assert len(list(gen)) == len(SIRNAS)
    assert [n for n in during if n.startswith(prefix)], "no spill seen mid-run"
    assert _engine_dirs(tmp_root) == []


@pytest.mark.parametrize("bad", ["", ".", "..", "chr/1", "../escape", "nul\0byte"])
def test_profile_key_that_cannot_be_a_file_name_is_refused(
    tmp_path, tmp_root, profiles, bad
):
    (frame,) = profiles.values()
    with pytest.raises(ValueError) as info:
        next(
            compute_off_targets(
                risearch_file=PREDICTIONS, accessibility={"chr1": frame, bad: frame}
            )
        )
    assert repr(bad) in str(info.value) and "'chr1'" not in str(info.value)
    assert _engine_dirs(tmp_root) == []
    assert not list(tmp_path.rglob("*.accessibility.parquet")), "a profile escaped"


def test_every_bad_profile_key_is_named(tmp_root, profiles):
    (frame,) = profiles.values()
    with pytest.raises(ValueError) as info:
        next(
            compute_off_targets(
                risearch_file=PREDICTIONS,
                accessibility={"a/b": frame, "chr1": frame, "..": frame},
            )
        )
    assert str(info.value).endswith("'a/b', '..'")


def test_empty_profiles_are_refused(tmp_root):
    with pytest.raises(ValueError) as info:
        next(compute_off_targets(risearch_file=PREDICTIONS, accessibility={}))
    assert str(info.value) == (
        "accessibility holds no profiles; pass one per chromosome, as "
        "sioff.accessibility returns"
    )
    assert _engine_dirs(tmp_root) == []
