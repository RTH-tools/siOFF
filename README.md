# siOFF

[![CI](https://github.com/lorenzo-22/siOFF/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/lorenzo-22/siOFF/actions/workflows/ci.yml?query=branch%3Amain)
[![Release](https://img.shields.io/github/v/release/lorenzo-22/siOFF?include_prereleases&sort=semver)](https://github.com/lorenzo-22/siOFF/releases)
[![PyPI](https://img.shields.io/pypi/v/sioff)](https://pypi.org/project/sioff/)
[![License](https://img.shields.io/badge/license-BUSL--1.1-blue)](https://github.com/lorenzo-22/siOFF/blob/main/LICENSE)
[![Python](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Florenzo-22%2FsiOFF%2Fmain%2Fpyproject.toml)](https://github.com/lorenzo-22/siOFF/blob/main/pyproject.toml)

siOFF — siRNA off-target discovery pipeline.

Documentation (usage guide, changelog and API reference): <https://lorenzo-22.github.io/siOFF/>

A bioinformatics pipeline for **siRNA off-target discovery and probability quantification**. Integrates RNA-RNA interaction predictions with transcriptome annotations, RNA accessibility profiling, and thermodynamic modeling to rank off-target binding sites.

siOFF is a from-scratch re-implementation of the siRNA off-target discovery
pipeline originally distributed with
[RIsearch2](https://rth.dk/resources/risearch)
([Alkan *et al.*, Nucleic Acids Research 2017](https://doi.org/10.1093/nar/gkw1325)).
It keeps the same thermodynamic model and still reads legacy RIsearch2
prediction files, but replaces the original C toolchain with a single
Python package built on Polars, adds in-process bindings for
[RIsearch 3](https://github.com/saiden89/risearch) (Rust/PyO3, on
[PyPI](https://pypi.org/project/risearch/) and
[crates.io](https://crates.io/crates/risearch)),
Parquet-based accessibility profiles, and a Slurm-aware orchestrator. A
compatibility mode (`--legacy-format`) reproduces the old `.results` output.

The pipeline has three stages, mapped one-to-one onto CLI commands:

```mermaid
flowchart LR
    subgraph inputs [Inputs]
        Q([siRNA FASTA])
        G([Genome / transcriptome FASTA])
        A([Annotation GTF/BED<br>with expression])
    end

    subgraph s1 [1 · Predict]
        R["sioff search<br>(RIsearch RNA-RNA interactions)"]
    end

    subgraph s2 [2 · Fold]
        F["sioff accessibility<br>(RNAplfold opening energies)"]
    end

    subgraph s3 [3 · Score]
        O["sioff off-targets<br>(annotate · weight · partition function)"]
    end

    T([Ranked off-target table<br>+ per-siRNA summaries])

    Q --> R
    G --> R
    G --> F
    R -- binding sites --> O
    F -- opening energies --> O
    A -- expression --> O
    O --> T
```

Rounded nodes are data, rectangles are pipeline stages. Stages 1 and 2 are
independent and can run in parallel; stage 3 combines their outputs. Stage 1
can be skipped entirely if you already have
[RIsearch](https://github.com/saiden89/risearch) prediction files —
`sioff off-targets` reads them directly.

---

## Installation

### Prerequisites

- **Python ≥ 3.11** (tested on 3.11–3.14)
- **[uv](https://docs.astral.sh/uv/)** (recommended) or pip
- `risearch` and ViennaRNA are installed automatically as Python dependencies
  (pinned `3.0.0a4` and `2.7.2`) — no system-level install needed. Prebuilt
  `risearch` wheels cover Linux x86_64 and Apple Silicon macOS; other platforms
  build it from source, which needs a **Rust toolchain** ([rustup](https://rustup.rs))

### Install

Install from [PyPI](https://pypi.org/project/sioff/):

```bash
pip install sioff          # or: uv pip install sioff

# Verify
sioff --help
```

On a CPU without AVX2 (roughly pre-2013 Intel, e.g. Sandy/Ivy Bridge Xeons)
the default polars runtime crashes with an illegal instruction; polars warns
"Missing required CPU features" at import. Install the compat runtime instead:

```bash
pip install "sioff[lts-cpu]"   # SSE4-only polars runtime, for siOFF and risearch alike
```

With both runtimes installed polars picks the compat one; set
`POLARS_PREFER_PKG=32` to force the AVX2 runtime on a machine that has it.

The PyPI distribution name and the import name are both **`sioff`**. One install
gives you the complete pipeline: `off-targets` / `accessibility` on pre-computed
RIsearch predictions **and** the in-process `index` / `search` commands (plus
the `--sirna-fasta` in-process mode of `off-targets`).

For development, from a clone:

```bash
git clone git@github.com:lorenzo-22/siOFF.git
cd siOFF
uv sync                    # runtime deps + dev tooling (ruff, pyrefly, pytest)
uv run sioff --help
```

`uv sync` also installs the compat polars runtime (the `lts` dependency group,
on by default because the development server has no AVX2). On a modern machine
use `uv sync --no-group lts` for the AVX2 runtime; CI does the same.

See [RIsearch](#risearch) for how the in-process engine is pinned.

---

## Quick start: run the bundled example

The repository ships a small, internally consistent example dataset in
[`examples/data/`](examples/data): a 6 kb toy genome, 5 siRNAs, an annotation
with expression values, pre-computed RIsearch predictions, and pre-computed
accessibility profiles. This run never calls `risearch` (no `risearch`
needed) and takes a few seconds:

```bash
# Pre-computed predictions file
sioff off-targets \
  -r examples/data/predictions.out \
  -t examples/data/annotation.gtf \
  -a examples/data/accessibility \
  -o example_run/off_targets.tsv
```

The same run expressed as a YAML config (the config is the complete, commented
reference for every option):

```bash
sioff -c example_yaml/off-targets.example.yaml
```

Both produce the same table; the YAML run writes to
`example_yaml/sioff_example_out/`.

### What you get

`example_run/` contains the main table `off_targets.tsv` — one row per
annotated binding site, ranked by off-target probability — plus one
`<sirna_id>.summary` file per siRNA with its partition-function statistics.
The key columns:

| Column | Description |
|--------|-------------|
| `sirna_id` | siRNA the site belongs to |
| `chrom`, `start`, `end`, `strand` | Binding site location |
| `gene_id`, `transcript_id` | Overlapping annotation feature |
| `energy` | Duplex hybridization energy from RIsearch (kcal/mol) |
| `opening_energy` | Accessibility penalty — cost to unfold the site (kcal/mol) |
| `dG_total` | `energy + opening_energy` |
| `exp_value` | Expression of the overlapping transcript (e.g. RPKM) |
| `P_off_target` | Off-target probability (per-siRNA partition function) |

For example, `siRNA_4` has two equally strong binding sites in the toy data;
the site in the higher-expressed gene absorbs most of the probability:

```
sirna_id  gene_id  energy    dG_total  exp_value  P_off_target
siRNA_4   gene_6   -34.15    -28.35    600        0.727
siRNA_4   gene_5   -34.15    -28.85    100        0.273
```

The full column list is documented in [Output](#output).

## Full example: recompute everything from FASTA

The quick start consumed pre-computed predictions and accessibility profiles.
This example rebuilds both from the raw example FASTA files — the same three
stages you will run on your own data.

```bash
# 1 · Fold: accessibility profiles, one Parquet file per chromosome
sioff accessibility \
  -f examples/data/genome.fa \
  -o example_run/accessibility

# 2 · Predict + 3 · Score in one command: RIsearch runs in-process
#     on the siRNA FASTA, then sites are annotated and scored
sioff off-targets \
  -s examples/data/sirnas.fa \
  --target-fasta examples/data/genome.fa \
  -t examples/data/annotation.gtf \
  -a example_run/accessibility \
  -o example_run/off_targets.tsv
```

This produces the **same table as the quick start** — the bundled
`predictions.out` and `accessibility/` were generated exactly this way (see
[`examples/generate_example_data.py`](examples/generate_example_data.py)).

For repeated searches against the same target, build the index once and reuse
it:

```bash
sioff index examples/data/genome.fa -o example_run/genome.idx

sioff off-targets \
  -s examples/data/sirnas.fa \
  -idx example_run/genome.idx \
  --target-fasta examples/data/genome.fa \
  -t examples/data/annotation.gtf \
  -a example_run/accessibility \
  -o example_run/off_targets.tsv
```

Standalone prediction (stage 1 by itself) is also available as
`sioff search query.fa target.idx -t target.fa` — see the
[CLI reference](#cli-reference).

---

## Running on your own data

### What you need

1. **siRNA guide strands** — FASTA, one entry per siRNA (RNA or DNA alphabet).
2. **Target sequences** — genome or transcriptome FASTA. This decides the
   analysis mode: `--type gw` (genome-wide, default) intersects predictions
   with the annotation by genomic coordinates; `--type tw`
   (transcriptome-wide) matches the prediction's target ID against
   `transcript_id` directly.
3. **Annotation with expression** — GTF, GFF3, BED6 or BED7. Expression is
   read from an attribute of your choice (`--expression-metric`, default
   `RPKM`) — annotate your GTF with RPKM/TPM values from your expression data
   first. Sites that don't overlap any feature are dropped.
4. Either **pre-computed [RIsearch](https://github.com/saiden89/risearch)
   predictions** (TSV / `.out.gz` / directory
   of per-siRNA Parquet files) **or** a siRNA FASTA plus target FASTA so siOFF
   can run RIsearch itself in-process.
5. Optional: a TSV mapping `sirna_id → transcript_id` (`--on-target-ids`) so
   each siRNA's intended target enters the partition function as the on-target
   term.

### Step 1 — RIsearch predictions

If you already have RIsearch output, skip this step and pass the file (or a
directory of per-siRNA Parquet files, which enables parallel per-siRNA
processing) to `-r`.

Otherwise, let `off-targets` run RIsearch in-process via `-s/--sirna-fasta` +
`--target-fasta` (as in the full example above). For a full genome, build the
index once with `sioff index genome.fa` (suffix-array construction is the
expensive part) and pass it with `-idx`; search parameters (seed length,
energy threshold, extension, energy matrix) are documented under
[`sioff search`](#search).

### Step 2 — Accessibility profiles

```bash
sioff accessibility -f genome.fa -o accessibility/ -j 8
```

Writes one `{chrom}.accessibility.parquet` per chromosome with RNAplfold
unpaired probabilities (window `-W 80`, span `-L 40`, unpaired length
`-u 30` by default — change them consistently if your protocol differs).
Chromosomes fold in parallel (`-j`, one worker per chromosome). This is the
most compute-intensive stage: budget hours and tens of GB of memory for a
mammalian genome — run it on a cluster (see
[Orchestrated runs](#orchestrated-multi-step-runs-local--slurm)) and reuse
the profiles across all subsequent analyses of that genome.

Accessibility is optional: without `-a`, opening energies are 0 and ranking
uses hybridization energy and expression only.

### Step 3 — Off-target analysis

```bash
sioff off-targets \
  -r predictions.out \
  -t annotation.gtf \
  -a accessibility/ \
  --expression-metric RPKM \
  --type gw \
  --on-target-ids on_target_map.tsv \
  -o results/off_targets.tsv
```

Useful knobs: `--alpha/--gamma/--theta` for parameter sweeps (semicolon-
separated values, each combination adds a `P_off_target:...` column),
`-j` for worker count, `--output-format parquet` for large runs,
`--legacy-format` for the old pipeline's `.results` output. Prefer the YAML
config for reproducibility — [`example_yaml/off-targets.example.yaml`](example_yaml/off-targets.example.yaml)
documents every option; run it with `sioff -c my-config.yaml`.

#### Accessibility lookup strategy (large genomes)

`--accessibility-strategy` selects how the opening energies are read from
`-a` in the directory (per-siRNA batch) mode. All three give identical results:

| Strategy | What it does | Memory | Writes |
|----------|--------------|--------|--------|
| `chrom-major` (`auto`) | Three passes: intersect every siRNA, stream each chromosome's Parquet **once** for the hits of all siRNAs, then score. | ~ one Parquet row group per worker | nothing (a temp dir under `TMPDIR`) |
| `memmap` | Builds a uint8 `{chrom}_{plus,minus}.access.npy` cache (+ `.access.exceptions.npz` sidecar) per chromosome and strand, once, in `--accessibility-cache-dir` (default: the accessibility dir), then every worker memory-maps it: one shared page-cache copy. | the pages the hits touch; 7.5 GB per chr1 strand while building | the cache files (the only strategy that does) |
| `eager` | The original loader: each worker reads whole chromosomes into dense float32 arrays. | 30 GB per chr1 strand **per worker** — small genomes only | nothing |

`auto` is `chrom-major` whenever `-a` is given (also with `-j 1`); a single
predictions file (`-r file`) has no chromosome-major path and uses the eager
loader unless `memmap` is requested. `memmap` suits repeated runs against the
same genome: the cache is reused across runs, may live on node-local disk
(`--accessibility-cache-dir /scratch/acc_cache`), and a cache directory that is
complete needs no Parquet at all (`-a` may then point at an empty directory).
A failed worker never passes silently: the successful siRNAs are written, then
`off-targets` reports `N/M siRNAs failed ...; first error: ...` and exits 1.

### Orchestrated multi-step runs (local + Slurm)

`scripts/run_pipeline.py` runs all stages in dependency order from one config —
`index` and `accessibility` in parallel, `off-targets` after `accessibility`.
It ships in the repo (`scripts/`), not as an installed console script.

```bash
# Dry-run (print commands without executing)
python scripts/run_pipeline.py --config example_yaml/run-pipeline.example.yaml --dry-run

# Run locally (sequential, logs to logs/<timestamp>/)
python scripts/run_pipeline.py --config example_yaml/run-pipeline.example.yaml

# Run a subset of steps
python scripts/run_pipeline.py --config example_yaml/run-pipeline.example.yaml --steps accessibility,off-targets

# Submit to Slurm (add --dry-run to preview the sbatch dependency chain)
python scripts/run_pipeline.py --config example_yaml/run-pipeline.example.yaml --slurm
```

Slurm resources come from the config's `slurm:` key (per-step overrides
allowed); the CLI flags `--partition`, `--time`, `--mem`, `--cpus-per-task` and
`--account` override it for all steps. A top-level `transcriptomes:` list fans
one launch out over several genomes/transcriptomes, one Slurm job per entry,
each with its own predictions, annotation and output, so the per-siRNA
partition functions never mix. The commented
[`example_yaml/run-pipeline.example.yaml`](example_yaml/run-pipeline.example.yaml)
is the full reference for both; paths resolve relative to the config file.

---

## CLI Reference

### Global options

Available on `sioff` itself, before any subcommand.

| Flag | Description |
|------|-------------|
| `-c / --config` | Path to a YAML config file; runs the command named in it. Top-level only — `sioff -c cfg.yaml`, not `sioff off-targets -c cfg.yaml` |
| `-v / --verbose` | Enable DEBUG-level logging |
| `--version` | Print the installed version and exit (long form only — `-v` is `--verbose`) |

### `off-targets`

| Flag | Description |
|------|-------------|
| `-r / --risearch-file` | Pre-computed predictions (TSV, `.out.gz`, or directory of Parquet files — directory triggers parallel per-siRNA mode) |
| `-s / --sirna-fasta` | siRNA FASTA — runs RIsearch in-process via PyO3 bindings |
| `--target-fasta / --genome` | Target FASTA for in-process RIsearch |
| `-idx / --target-index` | Pre-built RIsearch index (speeds up repeated runs) |
| `-t / --transcriptome` | GTF, GFF3, BED6 or BED7 annotation file |
| `-a / --accessibility-dir` | Directory of per-chromosome accessibility Parquet files (from `accessibility` command); with `--accessibility-strategy memmap` it may hold only the `.access.npy` cache |
| `--accessibility-strategy` | `auto` (default), `chrom-major`, `memmap` or `eager` — see [Accessibility lookup strategy](#accessibility-lookup-strategy-large-genomes) |
| `--accessibility-cache-dir` | `memmap` only: where the `.access.npy` cache is read from / built (default: the accessibility dir) |
| `--expression-metric` | GTF attribute for expression weighting (default: `RPKM`) |
| `--type` | `gw` (genome-wide) or `tw` (transcriptome-wide, default: `gw`) |
| `--alpha / --gamma / --theta` | Parameter sweep values (semicolon-separated, e.g. `0.8;1.0`) |
| `--on-target-ids / -oi` | TSV mapping `sirna_id → transcript_id` for on-target normalization |
| `-j / --workers` | Parallel worker processes (default: CPU count) |
| `-o / --output` | Output file path (TSV by default) |

### `accessibility`

| Flag | Description |
|------|-------------|
| `-f / --fasta` | Genome or transcriptome FASTA |
| `-o / --output` | Output directory (one `{chrom}.accessibility.parquet` per chromosome) |
| `-W / --window` | RNAplfold window size W (default: 80) |
| `-L / --span` | Max base-pair span L (default: 40) |
| `-u / --unpaired` | Unpaired probability length u (default: 30) |
| `-T / --temperature` | Folding temperature °C (default: 37.0) |
| `-j / --workers` | Parallel workers, one per chromosome (default: 1) |

### `index`

| Flag | Description |
|------|-------------|
| `TARGET` | Target FASTA file to index (positional) |
| `-o / --output` | Output index path (default: `<target>.idx`) |

### `search`

| Flag | Description |
|------|-------------|
| `QUERY INDEX` | Query siRNA FASTA and pre-built `.idx` (positional) |
| `-t / --target` | Target FASTA used to build the index |
| `-s / --seed` | Seed spec, RIsearch syntax: `l`, `n:m` or `n:m/l` (default: 6) |
| `--no-gu-seed` | Forbid G:U wobble pairs inside the seed (RIsearch `--no-seed-wobble`; legacy `--noGUseed`) |
| `-z / --matrix` | Energy parameter set: `t04` (default), `slh04`, `s95-rna-dna`, `s95-dna-rna`, or the path of a custom DSM TSV table (`q1 q2 t1 t2 delta_g_kcal_per_mol`) |
| `-e / --max-extension` | Max extension length on each side (default: 20) |
| `-E / --energy` | Energy threshold in kcal/mol (default: −10.0) |
| `-o / --output` | Output TSV file (default: stdout) |

---

## Python API

siOFF can be used as a library — `import sioff`, then call the commands as plain functions; they return their results in-memory.

```python
import sioff

# Off-target analysis on a pre-computed predictions file → polars.DataFrame
df = sioff.off_targets(risearch_file="predictions.tsv", gtf_file="annotations.gtf")

# Pre-compute per-chromosome accessibility profiles → dict[chrom -> polars.DataFrame]
profiles = sioff.accessibility(genome="genome.fa")

# Build a RIsearch index → Path (risearch bindings, in-process)
idx = sioff.index("target.fa")

# Run a RIsearch search → polars.DataFrame (risearch bindings, in-process)
hits = sioff.search("query.fa", "target.fa.idx", target="target.fa")

# ... and feed hits and profiles straight into the analysis, no intermediate files
df = sioff.off_targets(predictions=hits, accessibility=profiles, gtf_file="annotations.gtf")
```

**Notes**

- `sioff.off_targets` takes its predictions from one of `predictions=` (a `polars.DataFrame` in the `sioff.search` schema: `sirna_id, chrom, start, end, strand, energy`), `risearch_file=` (a RIsearch output file) or `sirna_fasta=` + `target_fasta=` (run RIsearch in-process), and returns a `polars.DataFrame`. With a **directory** of per-siRNA files in `risearch_file=` it returns a **generator** yielding one `polars.DataFrame` per siRNA; iterate it to consume the results. No form writes files — the API layer returns results in memory, and writing output is the CLI's job.
- `sioff.accessibility` likewise writes nothing: it returns `dict[chrom -> polars.DataFrame]`, which `sioff.off_targets(accessibility=...)` takes directly (single-frame forms only; the directory form needs `accessibility_dir=`). Use the `accessibility` CLI command (or `sioff -c <config>`) if you want `{chrom}.accessibility.parquet` files on disk.
- On bad input the API functions raise ordinary Python exceptions — `FileNotFoundError` or `ValueError` — not `typer.Exit`. `typer.Exit` is raised only by the CLI layer for its own argument validation.
- `sioff.index` / `sioff.search` call the `risearch` bindings in-process — the same dependency the CLI's `index`/`search` commands use.

---

## Thermodynamic Model

Off-target probability is computed from a Boltzmann partition function over all predicted binding sites for each siRNA:

```
W_i  = Expression_i × exp(−ΔG_total_i / RT)

       ΔG_total = ΔG_hybridization + ΔG_opening

Z_s  = Σ W_i  (all off-targets of siRNA s)  +  W_on-target

P(off-target_i | siRNA_s) = W_i / Z_s
```

- **ΔG_hybridization**: duplex interaction energy from RIsearch. The nearest-neighbour parameter set is selectable with `sioff search -z/--matrix` (default Turner 2004).
- **ΔG_opening**: Accessibility penalty — cost to unfold the target region, retrieved from pre-computed `RNA.pfl_fold_up` profiles.
- **Expression weighting**: annotation-derived RPKM/TPM values scale each site's contribution.
- **Per-siRNA normalization**: Partition functions are computed independently per siRNA; mixing them is biologically incorrect.

### Parameter sweeps

`--alpha`/`--gamma` clamp extremely favorable energies; `--theta` scales energy differences around a −10 kcal/mol reference. All combinations are computed in a single Polars `group_by` pass, producing separate `P_off_target:alpha=X,gamma=Y` columns.

---

## Output

### TSV (default)

| Column | Description |
|--------|-------------|
| `chrom` | Chromosome or transcript ID |
| `start`, `end` | Binding site coordinates |
| `strand` | Strand orientation |
| `energy` | Hybridization energy (kcal/mol) |
| `opening_energy` | Accessibility penalty (kcal/mol) |
| `dG_total` | Combined free energy |
| `exp_value` | Expression level |
| `P_off_target` | Off-target probability (baseline α=γ=θ=1) |
| `P_off_target:alpha=X,gamma=Y` | Per-parameter-set probabilities |

One `<sirna_id>.summary` file per siRNA is written next to the output with the
partition-function statistics (`P`, `Z`, `Zoff`, with and without
accessibility). `--summary-only` skips the per-prediction table — the old
pipeline's default behaviour.

`--legacy-format` additionally writes the RIsearch2 pipeline's `.results`
files, for byte-level comparison with old runs.

---

## RIsearch

The in-process `index` / `search` engine is
[`risearch`](https://github.com/saiden89/risearch)
([PyPI](https://pypi.org/project/risearch/),
[crates.io](https://crates.io/crates/risearch),
[docs](https://saiden89.github.io/risearch/)): the RIsearch core rewritten in
Rust with PyO3 bindings, called in-process with no subprocess or intermediate
files. `off-targets` and `accessibility` on pre-computed predictions never
import it.

`risearch` is pinned exactly (`risearch==3.0.0a4`) because it is an alpha
series whose API and results change between alphas; the pin is bumped
deliberately, with the test suite and a fixture comparison, not automatically.

The previous generation of this pipeline, built on RIsearch2, is at
[rth.dk/resources/risearch](https://rth.dk/resources/risearch); siOFF reads its
output files unchanged.

---

## Citation

If you use siOFF in published work, please cite:

> Roncelli S, Favaro L, Anthon C, Gorodkin J. *RIsearch and siOFF: An integrated,
> high-performance framework for RNA-RNA interaction and siRNA off-target
> prediction.* In preparation.

Machine-readable metadata is in [`CITATION.cff`](CITATION.cff).

For the original method and the previous pipeline, cite:

> Alkan F, Wenzel A, Palasca O, Kerpedjiev P, Rudebeck AF, Stadler PF,
> Hofacker IL, Gorodkin J. *RIsearch2: suffix array-based large-scale
> prediction of RNA-RNA interactions and siRNA off-target discovery.*
> Nucleic Acids Research 2017;45(8):e60.
> [doi:10.1093/nar/gkw1325](https://doi.org/10.1093/nar/gkw1325)

Note that citation is not merely requested but a **term of the BUSL-1.1
licence** for any production use of siOFF or `risearch` — see [License](#license).

---

## License

siOFF (`sioff`) is released under the **Business Source License 1.1
(BUSL-1.1)** — see [LICENSE](LICENSE) — the same licence as its `risearch`
dependency. BUSL-1.1 is *source-available, not open source* and is not
OSI-approved. In brief:

- **Research and production use are free**, including commercial use, with one
  exception below.
- **No hosted services.** You may not use it to provide SaaS, PaaS or any other
  hosted or cloud-based service to third parties — commercial *or*
  non-commercial — except where such services are provided exclusively to
  accredited academic institutions, or individuals affiliated with them, for
  non-commercial research or educational purposes.
- **Citation is a licence term.** Any production use must cite the work; see
  [Citation](#citation).
- **Converts to Apache-2.0 on 14 September 2030.**
- Licensor: RTH, University of Copenhagen. Commercial licensing enquiries:
  <software@rth.dk>.

The summary above is provided for orientation only; [LICENSE](LICENSE) is the
authoritative text.
