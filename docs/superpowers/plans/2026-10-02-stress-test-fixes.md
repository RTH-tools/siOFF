# Stress-Test Findings Fix Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the six defects found by the 2026-10-02 stress test of siOFF: silent accessibility fallbacks, unvalidated folding parameters, unsanitized siRNA IDs in filenames, predictions-format mismatch, undocumented sweep constraint, and stale bundled example data.

**Architecture:** All fixes are small, local changes in `src/sioff/` plus tests. No new modules, no new dependencies. Failure behavior changes from "silent default" to "warn for legitimately-missing data, raise for corrupt data".

**Tech Stack:** Python ≥3.11, polars, typer, loguru, pytest, uv.

**Spec:** No separate spec document. The authority is the findings list embedded in each task below (from the stress-test session).

## Global Constraints

- Run tests with `uv run pytest` from the repo root. The full suite (232 tests at baseline) must pass after every task.
- No new runtime dependencies.
- Logging uses `from loguru import logger` (already imported in the touched modules).
- CLI errors must exit nonzero (`typer.Exit(1)` or an exception); library (`sioff.*` API) errors must raise ordinary Python exceptions (`ValueError`, `FileNotFoundError`, `AccessibilityError`), never `typer.Exit`.
- Commit messages: Conventional Commits (`fix:`, `test:`, `docs:`, `chore:`), subject ≤72 chars.
- Legacy numeric semantics are frozen: the 10.0 kcal/mol default opening energy for *missing* profiles and the 25.5 sentinel for unfolded positions stay as-is. Only *error visibility* changes.

## Review Focus

1. A predictions file for chromosomes that have **no accessibility profile at all** (user folded a subset) must still complete with default 10.0 energies and exactly one warning per (chrom, strand) — not raise, not warn per-row (Task 1).
2. A **truncated/corrupt** `.accessibility.parquet` (e.g. a 4-byte placeholder left by a crashed folding run) must abort the run with a clear error naming the file (Task 1).
3. A gzipped legacy predictions file (`.out.gz`) must still load through the format sniffer (Task 4).
4. A siRNA FASTA whose ID sanitizes to an empty string (e.g. `>...`) must still produce a usable summary filename (Task 3).
5. `sioff accessibility` invoked through the YAML config path (not just CLI flags) must also reject invalid W/L/u/T (Task 2 — validation lives in the service entry points, not the CLI parser).

---

### Task 1: Loud failure on unreadable accessibility profiles

**Finding:** `src/sioff/services/accessibility.py:493-497` — `annotate_opening_energy_vectorized` wraps `_ensure_profile` in `except Exception: continue`. A corrupt per-chromosome parquet (reproduced with a 4-byte file) silently leaves every site on that chromosome at the prefilled 10.0 default; run exits 0 with no warning.

**Files:**
- Modify: `src/sioff/services/accessibility.py` (exception classes near line 15; `_ensure_profile` ~line 359; `annotate_opening_energy_vectorized` ~line 476)
- Test: `tests/off-targets/test_accessibility.py`

**Interfaces:**
- Produces: `AccessibilityProfileMissingError(AccessibilityError)` — raised when a profile is absent (file not found / chrom not in in-memory frames). `AccessibilityError` (plain) now also wraps parquet read failures. `GenomeAccessibilityService` gains private attr `self._warned_missing: set[tuple[str, str]]`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/off-targets/test_accessibility.py` (follow the file's existing fixture style for constructing the service; the two tests below show the required behavior — adapt constructor/call names to the ones already used in that file):

```python
def test_corrupt_profile_parquet_raises(tmp_path):
    """A truncated parquet must abort annotation, not silently default."""
    acc_dir = tmp_path / "acc"
    acc_dir.mkdir()
    (acc_dir / "chr9.accessibility.parquet").write_bytes(b"PAR1")  # truncated
    svc = GenomeAccessibilityService(data_dir=acc_dir)
    df = pl.DataFrame(
        {"chrom": ["chr9"], "start": [10], "end": [30],
         "strand": ["+"], "energy": [-20.0]}
    )
    with pytest.raises(AccessibilityError, match="chr9"):
        svc.annotate_opening_energy_vectorized(df)


def test_missing_profile_warns_once_and_defaults(tmp_path):
    """No profile file at all → default 10.0 + exactly one warning per chrom/strand."""
    acc_dir = tmp_path / "acc"
    acc_dir.mkdir()  # empty: no profiles
    svc = GenomeAccessibilityService(data_dir=acc_dir)
    df = pl.DataFrame(
        {"chrom": ["chrX", "chrX"], "start": [10, 50], "end": [30, 70],
         "strand": ["+", "+"], "energy": [-20.0, -21.0]}
    )
    messages: list[str] = []
    handler_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        out = svc.annotate_opening_energy_vectorized(df)
    finally:
        logger.remove(handler_id)
    assert out["opening_energy"].to_list() == [10.0, 10.0]
    warned = [m for m in messages if "chrX" in m]
    assert len(warned) == 1
```

Required imports at the top of the test file (add only what's missing): `import polars as pl`, `import pytest`, `from loguru import logger`, and `AccessibilityError`, `AccessibilityProfileMissingError`, `GenomeAccessibilityService` from `sioff.services.accessibility`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest "tests/off-targets/test_accessibility.py" -k "corrupt_profile or missing_profile_warns" -v`
Expected: FAIL — `ImportError` for `AccessibilityProfileMissingError`, and (after a stub) the corrupt test failing because no error propagates.

- [ ] **Step 3: Implement**

In `src/sioff/services/accessibility.py`:

(a) Below the existing `AccessibilityError` class (~line 14), add:

```python
class AccessibilityProfileMissingError(AccessibilityError):
    """No accessibility profile exists for the requested chromosome."""
```

(b) In `_ensure_profile`, change the two "not found" raises (~lines 384 and 390) from `AccessibilityError` to `AccessibilityProfileMissingError` (messages unchanged), and wrap the parquet read (~line 395):

```python
            source = str(path)
            try:
                full = pl.read_parquet(path)
            except Exception as e:
                raise AccessibilityError(
                    f"Failed to read accessibility profile {path}: {e}. "
                    "The file may be truncated (e.g. left by a crashed "
                    "`sioff accessibility` run) — regenerate it."
                ) from e
```

(c) In `__init__`, add `self._warned_missing: set[tuple[str, str]] = set()` next to the existing cache attributes.

(d) In `annotate_opening_energy_vectorized`, replace the blanket handler (~lines 493-497):

```python
        for (chrom, strand), row_indices in groups.items():
            try:
                profile_key = self._ensure_profile(chrom, strand)
            except AccessibilityProfileMissingError as e:
                if (chrom, strand) not in self._warned_missing:
                    self._warned_missing.add((chrom, strand))
                    logger.warning(
                        f"No accessibility profile for {chrom} ({strand}): {e} "
                        f"Using default opening energy 10.0 for "
                        f"{len(row_indices)} site(s)."
                    )
                continue
```

Corrupt-file `AccessibilityError`s now propagate. Check the rest of the module for other `except Exception` blocks around `_ensure_profile` or `read_parquet` (lines ~78, ~142, ~303 are folding-side and out of scope — leave them) and apply the same missing-vs-corrupt split only where profiles are *read*.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest "tests/off-targets/test_accessibility.py" tests/off-targets/test_accessibility_in_memory.py -v`
Expected: all PASS (new and existing).

- [ ] **Step 5: Full suite, then commit**

Run: `uv run pytest -q` — expected: all pass.

```bash
git add src/sioff/services/accessibility.py "tests/off-targets/test_accessibility.py"
git commit -m "fix(accessibility): raise on corrupt profiles, warn once on missing"
```

---

### Task 2: Validate folding parameters (W, L, u, T)

**Finding:** `sioff accessibility -u 100 -W 80` (u > W, invalid for RNAplfold) and `-T -500` both exit 0 and write profiles in which every value is the 25.5 sentinel — folding silently produced nothing usable.

**Files:**
- Modify: `src/sioff/services/accessibility.py` (new module-level function; call it from `fold_sequence` and from the genome/parallel folding entry point ~line 280)
- Modify: `src/sioff/commands/accessibility.py` (catch `ValueError` → friendly message + `typer.Exit(1)`)
- Test: `tests/off-targets/test_accessibility.py`

**Interfaces:**
- Produces: `validate_fold_params(window_size: int, max_span: int, unpaired_prob: int, temperature: float) -> None` in `sioff.services.accessibility` — raises `ValueError` on invalid combinations. Task 2 is independent of Task 1's changes.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize(
    ("w", "ll", "u", "t", "msg"),
    [
        (80, 40, 100, 37.0, "u"),        # u > W
        (80, 100, 30, 37.0, "span"),     # L > W
        (0, 40, 30, 37.0, "window"),     # W < 1
        (80, 40, 0, 37.0, "u"),          # u < 1
        (80, 40, 30, -500.0, "temperature"),
        (80, 40, 30, 150.0, "temperature"),
    ],
)
def test_validate_fold_params_rejects(w, ll, u, t, msg):
    with pytest.raises(ValueError, match=msg):
        validate_fold_params(w, ll, u, t)


def test_validate_fold_params_accepts_defaults():
    validate_fold_params(80, 40, 30, 37.0)  # must not raise
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest "tests/off-targets/test_accessibility.py" -k validate_fold_params -v`
Expected: FAIL with `ImportError: cannot import name 'validate_fold_params'`.

- [ ] **Step 3: Implement**

In `src/sioff/services/accessibility.py`, module level (near the top, after imports):

```python
def validate_fold_params(
    window_size: int, max_span: int, unpaired_prob: int, temperature: float
) -> None:
    """Reject RNAplfold parameter combinations that produce no usable output.

    RNAplfold requires L <= W and u <= W; out-of-range values leave every
    position at the 25.5 sentinel, which silently corrupts downstream
    opening energies.
    """
    if window_size < 1:
        raise ValueError(f"window size W must be >= 1, got {window_size}")
    if max_span < 1 or max_span > window_size:
        raise ValueError(
            f"max span L must be in [1, W={window_size}], got {max_span}"
        )
    if unpaired_prob < 1 or unpaired_prob > window_size:
        raise ValueError(
            f"unpaired length u must be in [1, W={window_size}], got {unpaired_prob}"
        )
    if not (0.0 <= temperature <= 100.0):
        raise ValueError(
            f"temperature must be in [0, 100] °C, got {temperature}"
        )
```

Call `validate_fold_params(window_size, max_span, unpaired_prob, temperature)` as the first statement of (a) the module-level `fold_sequence` function and (b) the service method that drives genome folding (the one `sioff accessibility` reaches — it submits per-chromosome futures around line 280; validate once before submitting). In `src/sioff/commands/accessibility.py`, wrap the service invocation (~line 110) so a `ValueError` prints `console.print(f"[red]Error:[/red] {e}")` (match the file's existing console/error style) and raises `typer.Exit(1)`.

- [ ] **Step 4: Run tests + CLI check**

Run: `uv run pytest "tests/off-targets/test_accessibility.py" -v` — expected: PASS.
Run: `uv run sioff accessibility -f examples/data/genome.fa -o /tmp/sioff_acc_test -u 100 -W 80; echo $?` — expected: error message, exit code 1, no output directory contents written.

- [ ] **Step 5: Full suite, then commit**

Run: `uv run pytest -q` — expected: all pass.

```bash
git add src/sioff/services/accessibility.py src/sioff/commands/accessibility.py "tests/off-targets/test_accessibility.py"
git commit -m "fix(accessibility): validate W/L/u/T before folding"
```

---

### Task 3: Sanitize siRNA IDs in summary filenames

**Finding:** Summary files are written as `f"{sid}.summary"` with the raw FASTA ID at `src/sioff/commands/off_targets.py:469` and `:645`. ID `siRNA/with/slash` crashes with `FileNotFoundError`; ID `../traversal` escapes the output directory (wrote `traversal.summary` one level up).

**Files:**
- Modify: `src/sioff/services/helpers.py` (new function)
- Modify: `src/sioff/commands/off_targets.py:469` and `:645` (wrap `sid`)
- Test: `tests/off-targets/test_cli.py` (or a new `tests/off-targets/test_helpers.py` if test_cli.py has no unit-style section)

**Interfaces:**
- Produces: `sanitize_filename(name: str) -> str` in `sioff.services.helpers`.

- [ ] **Step 1: Write the failing tests**

```python
from sioff.services.helpers import sanitize_filename


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("siRNA_1", "siRNA_1"),
        ("hsa-miR-21.5p", "hsa-miR-21.5p"),
        ("siRNA/with/slash", "siRNA_with_slash"),
        ("..\\win\\path", "win_path"),
        ("../traversal", "traversal"),
        ("...", "unnamed"),
        ("", "unnamed"),
    ],
)
def test_sanitize_filename(raw, expected):
    assert sanitize_filename(raw) == expected
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/off-targets -k sanitize_filename -v`
Expected: FAIL with `ImportError: cannot import name 'sanitize_filename'`.

- [ ] **Step 3: Implement**

In `src/sioff/services/helpers.py`:

```python
import re

_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_filename(name: str) -> str:
    """Make a siRNA ID safe to embed in an output filename.

    FASTA headers are user-controlled; '/' would create subdirectories and
    '..' would escape the output directory entirely.
    """
    safe = _UNSAFE_FILENAME_CHARS.sub("_", name)
    safe = safe.strip("._")
    return safe or "unnamed"
```

Note `"../traversal"` → sub gives `.._traversal` → `strip("._")` gives `traversal`. In `src/sioff/commands/off_targets.py`, import `sanitize_filename` from `sioff.services.helpers` and change both write sites:

```python
(summary_out_dir / f"{sanitize_filename(sid)}.summary").write_text(text)   # line ~469
(out_dir / f"{sanitize_filename(sid)}.summary").write_text(summary_text)   # line ~645
```

Grep for any other `f"{sid}.summary"` / `.summary"` write sites and wrap them identically (legacy-format writer in the formatting/probability service may also build per-siRNA paths — `rg '\.summary|\.results' src/sioff` and wrap every filename built from a siRNA ID).

- [ ] **Step 4: Run tests + reproduce the original crash**

Run: `uv run pytest tests/off-targets -k sanitize_filename -v` — expected: PASS.
Then end-to-end: write `/tmp/sioff_weird.fa` containing `>../traversal\nAGCUAGCUAGCUAGGCUUAG\n` plus `>siRNA/with/slash\nUUCGAAGUACUCAGCGUAAG\n`, run `uv run sioff off-targets -s /tmp/sioff_weird.fa --target-fasta examples/data/genome.fa -t examples/data/annotation.gtf -o /tmp/sioff_weird_out/out.tsv` — expected: exit 0, summaries only inside `/tmp/sioff_weird_out/`, nothing written to `/tmp/`.

- [ ] **Step 5: Full suite, then commit**

Run: `uv run pytest -q` — expected: all pass.

```bash
git add src/sioff/services/helpers.py src/sioff/commands/off_targets.py tests/off-targets
git commit -m "fix(output): sanitize siRNA IDs in summary filenames"
```

---

### Task 4: Accept `sioff search` TSV in `off-targets -r`

**Finding:** `sioff search -o hits.tsv` writes a 6-column TSV with header (`sirna_id  chrom  start  end  strand  energy`), but `RIsearchParser.load` (`src/sioff/services/risearch_parser.py:30`) only parses the legacy headerless 8-column RIsearch2 format. Feeding the pipeline's own stage-1 output back fails with the cryptic `projection index: 7 is out of bounds for csv schema with length: 6`.

**Files:**
- Modify: `src/sioff/services/risearch_parser.py`
- Test: `tests/off-targets/test_risearch_parser.py`

**Interfaces:**
- Consumes: `RISEARCH_COLUMNS`, `RISEARCH_SCHEMA` from `sioff.models` (already imported).
- Produces: `RIsearchParser.load` auto-detects the two formats; raises `ValueError` naming both formats on anything else. Signature unchanged.

- [ ] **Step 1: Write the failing tests**

Add to `tests/off-targets/test_risearch_parser.py` (reuse its existing fixtures/imports):

```python
def test_load_sioff_search_tsv(tmp_path):
    """Stage-1 `sioff search` output must load through -r."""
    p = tmp_path / "hits.tsv"
    p.write_text(
        "sirna_id\tchrom\tstart\tend\tstrand\tenergy\n"
        "siRNA_1\tchr1\t400\t420\t+\t-38.9\n"
        "siRNA_2\tchr1\t1200\t1220\t-\t-32.3\n"
    )
    df = RIsearchParser().load(p)
    assert df.columns == ["sirna_id", "chrom", "start", "end", "strand", "energy"]
    assert df.height == 2
    assert df["start"].dtype == pl.Int32


def test_load_unrecognised_format_raises(tmp_path):
    p = tmp_path / "junk.tsv"
    p.write_text("just one column\nno tabs here\n")
    with pytest.raises(ValueError, match="sioff search"):
        RIsearchParser().load(p)


def test_load_sioff_search_tsv_gzipped(tmp_path):
    import gzip

    p = tmp_path / "hits.tsv.gz"
    with gzip.open(p, "wt") as f:
        f.write(
            "sirna_id\tchrom\tstart\tend\tstrand\tenergy\n"
            "siRNA_1\tchr1\t400\t420\t+\t-38.9\n"
        )
    df = RIsearchParser().load(p)
    assert df.height == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest "tests/off-targets/test_risearch_parser.py" -k "sioff_search or unrecognised" -v`
Expected: FAIL — the first with the projection-index polars error, the second because no `ValueError` is raised.

- [ ] **Step 3: Implement**

In `src/sioff/services/risearch_parser.py`:

```python
import gzip


def _first_line(path: Path) -> str:
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt") as f:  # type: ignore[operator]
            return f.readline()
    except (OSError, gzip.BadGzipFile):
        return ""
```

Rewrite `load`:

```python
    def load(self, path: Path) -> pl.DataFrame:
        """Load predictions: legacy 8-column RIsearch TSV or `sioff search` output."""
        _check_file(path)
        first = _first_line(path)
        if first.startswith("sirna_id\t"):
            df = pl.read_csv(
                path,
                separator="\t",
                schema_overrides=RISEARCH_SCHEMA,
            )
            missing = [c for c in RISEARCH_COLUMNS if c not in df.columns]
            if missing:
                raise ValueError(
                    f"{path} looks like `sioff search` output but is missing "
                    f"column(s) {missing}"
                )
            return df.select(RISEARCH_COLUMNS).filter(
                pl.col("sirna_id").is_not_null()
            )
        try:
            df = pl.read_csv(
                path,
                separator="\t",
                has_header=False,
                columns=[0, 3, 4, 5, 6, 7],
                new_columns=RISEARCH_COLUMNS,
                schema_overrides=RISEARCH_SCHEMA,
                truncate_ragged_lines=True,
            )
        except Exception as e:
            raise ValueError(
                f"{path} is not a recognised predictions format. Expected "
                "either legacy RIsearch 8-column TSV (no header) or "
                "`sioff search` output (header starting with 'sirna_id')."
            ) from e
        return df.filter(pl.col("sirna_id").is_not_null())
```

Check whether `load_single_file` (per-siRNA directory mode) reads TSVs through the same legacy path — if it calls `load` or duplicates the 8-column read, give the duplicate the same sniff-or-clear-error treatment.

- [ ] **Step 4: Run tests + end-to-end check**

Run: `uv run pytest "tests/off-targets/test_risearch_parser.py" -v` — expected: PASS (new and existing legacy-format tests).
Then: `uv run sioff search examples/data/sirnas.fa /tmp/sioff_e2e.idx -t examples/data/genome.fa -o /tmp/sioff_hits.tsv` (build `/tmp/sioff_e2e.idx` first with `uv run sioff index examples/data/genome.fa -o /tmp/sioff_e2e.idx`), then `uv run sioff off-targets -r /tmp/sioff_hits.tsv -t examples/data/annotation.gtf -o /tmp/sioff_rt.tsv` — expected: exit 0, non-empty table.

- [ ] **Step 5: Full suite, then commit**

Run: `uv run pytest -q` — expected: all pass.

```bash
git add src/sioff/services/risearch_parser.py "tests/off-targets/test_risearch_parser.py"
git commit -m "fix(parser): accept sioff search TSV in off-targets -r"
```

---

### Task 5: Warn on dropped alpha/gamma pairs; document the constraint

**Finding:** `_build_alpha_gamma_pairs` (`src/sioff/core/off_targets.py:203-222`) silently drops any combination with alpha > gamma. `--alpha '0.8;1.0' --gamma '0.9;1.0'` yields 2 sweep columns instead of the 4 the README's "All combinations are computed" promises.

**Files:**
- Modify: `src/sioff/core/off_targets.py:203-222`
- Modify: `README.md` (Parameter sweeps paragraph, ~line 440)
- Test: `tests/off-targets/test_probability.py` (or wherever `_build_alpha_gamma_pairs` is currently unit-tested — check with `rg _build_alpha_gamma_pairs tests/`)

**Interfaces:**
- Produces: unchanged signature; emits one `logger.warning` listing all dropped pairs.

- [ ] **Step 1: Write the failing test**

```python
from loguru import logger

from sioff.core.off_targets import _build_alpha_gamma_pairs


def test_alpha_gt_gamma_pairs_dropped_with_warning():
    messages: list[str] = []
    handler_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        pairs = _build_alpha_gamma_pairs("0.8;1.0", "0.9;1.0")
    finally:
        logger.remove(handler_id)
    assert pairs == [(1.0, 1.0), (0.8, 0.9), (0.8, 1.0)]
    assert any("alpha" in m and "0.9" in m for m in messages)


def test_alpha_le_gamma_no_warning():
    messages: list[str] = []
    handler_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        pairs = _build_alpha_gamma_pairs("0.8", "0.9")
    finally:
        logger.remove(handler_id)
    assert pairs == [(1.0, 1.0), (0.8, 0.9)]
    assert not messages
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/off-targets -k "alpha_gt_gamma or alpha_le_gamma" -v`
Expected: first test FAILS on the warning assertion (pairs content already matches).

- [ ] **Step 3: Implement**

In `_build_alpha_gamma_pairs`, collect skipped pairs and warn once (`logger` is already imported in the module — verify, add `from loguru import logger` if not):

```python
    pairs: list[tuple[float, float]] = [(1.0, 1.0)]
    dropped: list[tuple[float, float]] = []
    for a in alpha_vals:
        for g in gamma_vals:
            if a == 1.0 and g == 1.0:
                continue
            if a <= g:
                pairs.append((a, g))
            else:
                dropped.append((a, g))
    if dropped:
        logger.warning(
            f"Skipping alpha/gamma combination(s) {dropped}: the clamp "
            "semantics require alpha <= gamma (energies below alpha*E_min "
            "are clamped up to gamma*E_min)."
        )
    return list(dict.fromkeys(pairs))
```

In `README.md`, Parameter sweeps section, replace the sentence `All combinations are computed in a single Polars `group_by` pass, producing separate `P_off_target:alpha=X,gamma=Y` columns.` with: `All combinations with alpha ≤ gamma are computed in a single Polars `group_by` pass, producing separate `P_off_target:alpha=X,gamma=Y` columns; pairs with alpha > gamma are skipped with a warning (the clamp semantics require alpha ≤ gamma).`

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/off-targets -k "alpha_gt_gamma or alpha_le_gamma" -v` — expected: PASS.

- [ ] **Step 5: Full suite, then commit**

Run: `uv run pytest -q` — expected: all pass.

```bash
git add src/sioff/core/off_targets.py README.md tests/off-targets
git commit -m "fix(sweep): warn when alpha>gamma combinations are skipped"
```

---

### Task 6: Regenerate bundled example data with risearch 3.0.0a4

**Finding:** `examples/data/predictions.out` and `examples/data/accessibility/` were generated under an older risearch alpha. With the pinned `risearch==3.0.0a4`, recomputation changes 3 binding-site boundaries (e.g. siRNA_1 `5265-5278` → `5265-5287`) and drifts energies ≤0.03 kcal/mol, so the README's claim that the full-from-FASTA example "produces the **same table as the quick start**" is false.

**Files:**
- Modify (regenerated): `examples/data/predictions.out`, `examples/data/accessibility/*.parquet`, possibly `examples/data/genome.fa`/`sirnas.fa`/`annotation.gtf` if the generator rewrites them (it is seeded — content should be identical; verify with `git diff --stat`)
- Modify: `README.md` "What you get" sample table (~line 167) if the printed values changed
- Test: existing suite + manual equality check (this task adds no new test code; the generator script is the fixture authority)

**Interfaces:**
- Consumes: Tasks 1-5 merged (run last — regeneration must happen on the fixed code).

- [ ] **Step 1: Regenerate**

Run: `uv run python examples/generate_example_data.py`
Expected: exit 0. Then `git status --short examples/` to see what changed; `git diff --stat examples/` for scope. `genome.fa`, `sirnas.fa`, `annotation.gtf` should be byte-identical (seeded generator); `predictions.out` and `accessibility/*.parquet` are expected to change.

- [ ] **Step 2: Verify the README equality claim now holds**

```bash
uv run sioff off-targets -r examples/data/predictions.out -t examples/data/annotation.gtf -a examples/data/accessibility -o /tmp/sioff_q/off_targets.tsv
uv run sioff accessibility -f examples/data/genome.fa -o /tmp/sioff_f/acc
uv run sioff off-targets -s examples/data/sirnas.fa --target-fasta examples/data/genome.fa -t examples/data/annotation.gtf -a /tmp/sioff_f/acc -o /tmp/sioff_f/off_targets.tsv
diff <(sort /tmp/sioff_q/off_targets.tsv) <(sort /tmp/sioff_f/off_targets.tsv)
```

Expected: no diff output (tables identical). If they differ, the generator and the pipeline disagree — stop and report BLOCKED with the diff; do not hand-edit data files.

- [ ] **Step 3: Full suite**

Run: `uv run pytest -q`
Expected: all pass. If golden/regression tests fail because they pin the old example data, inspect each failure: fixtures under `tests/off-targets/data/` that merely mirror `examples/data/` may be regenerated by whatever script created them (check `tests/scripts/`); hand-written expected values must NOT be edited to make tests pass without understanding the energy change — report BLOCKED with details instead.

- [ ] **Step 4: Update the README sample values if needed**

Compare the README "What you get" block (`siRNA_4   gene_6   -34.15 …` rows and the key-column examples) against the fresh `/tmp/sioff_q/off_targets.tsv`; update the printed numbers to match reality (keep 2-decimal rounding style). If values match, no edit.

- [ ] **Step 5: Commit**

```bash
git add examples/data README.md
git commit -m "chore(examples): regenerate bundled data with risearch 3.0.0a4"
```
