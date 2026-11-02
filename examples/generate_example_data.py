"""Regenerate the bundled example dataset in examples/data/.

The dataset is small but internally consistent: every file describes the same
6 kb toy genome, so the quick-start run (pre-computed predictions) and the
full-pipeline run (in-process RIsearch + fresh accessibility profiles) produce
the same off-target table.

Files written (all under examples/data/):
    genome.fa            6 kb single-chromosome genome with embedded siRNA
                         binding sites (perfect and mismatched)
    sirnas.fa            5 example siRNA guide strands (21 nt)
    annotation.gtf       7 single-exon genes with RPKM expression values
    predictions.out      RIsearch2-format predictions of sirnas.fa vs genome.fa,
                         produced by the in-process RIsearch bindings
    accessibility/       the memory-mapped npy cache from `sioff accessibility`
                         (ViennaRNA RNAplfold)

Usage:
    python examples/generate_example_data.py

Requires the full install (the `risearch` dependency group) because it runs
the in-process search to produce predictions.out.
"""

import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
GENOME_LEN = 6000
SIRNA_LEN = 21
SEED = 42

# gene_id -> (exon start, exon end, RPKM); 1-based inclusive, all on '+'
GENES = {
    "gene_1": (201, 800, 1500),
    "gene_2": (1001, 1600, 300),
    "gene_3": (1801, 2400, 800),
    "gene_4": (2601, 3200, 2000),
    "gene_5": (3401, 4000, 100),
    "gene_6": (4201, 4800, 600),
    "gene_7": (5001, 5600, 1200),
}

# siRNA -> list of (0-based genome insertion offset, number of mismatches).
# Offsets sit inside the exons above so every site is annotated.
SITES = {
    "siRNA_1": [(400, 0), (2800, 2)],
    "siRNA_2": [(1200, 0), (5200, 2)],
    "siRNA_3": [(3000, 0), (2000, 3)],
    "siRNA_4": [(3600, 0), (4400, 0)],
    "siRNA_5": [(5400, 1), (600, 2)],
}

COMPLEMENT = str.maketrans("ACGT", "TGCA")


def revcomp(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]


def main() -> None:
    rng = random.Random(SEED)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    genome = [rng.choice("ACGT") for _ in range(GENOME_LEN)]

    sirnas: dict[str, str] = {
        name: "".join(rng.choice("ACGU") for _ in range(SIRNA_LEN)) for name in SITES
    }

    for name, sites in SITES.items():
        # The genomic site is the reverse complement of the guide: the guide
        # then base-pairs perfectly with the + strand at that position.
        site = revcomp(sirnas[name].replace("U", "T"))
        for offset, n_mismatch in sites:
            seq = list(site)
            for pos in rng.sample(range(3, SIRNA_LEN - 3), n_mismatch):
                seq[pos] = rng.choice([b for b in "ACGT" if b != seq[pos]])
            genome[offset : offset + SIRNA_LEN] = seq

    genome_fa = DATA_DIR / "genome.fa"
    genome_fa.write_text(
        ">chr1\n"
        + "\n".join("".join(genome[i : i + 60]) for i in range(0, GENOME_LEN, 60))
        + "\n"
    )

    (DATA_DIR / "sirnas.fa").write_text(
        "".join(f">{name}\n{seq}\n" for name, seq in sirnas.items())
    )

    gtf_lines = []
    for gene_id, (start, end, rpkm) in GENES.items():
        n = gene_id.split("_")[1]
        attrs = f'gene_id "{gene_id}"; transcript_id "transcript_{n}"; RPKM "{rpkm}";'
        gtf_lines.append(
            f"chr1\texample\ttranscript\t{start}\t{end}\t.\t+\t.\t{attrs}"
        )
        gtf_lines.append(f"chr1\texample\texon\t{start}\t{end}\t.\t+\t0\t{attrs}")
    (DATA_DIR / "annotation.gtf").write_text("\n".join(gtf_lines) + "\n")

    # predictions.out — run the in-process search and write RIsearch2's 8-column
    # layout (query id, query start/end, target id, target start/end, strand,
    # energy) so the file feeds `sioff off-targets -r` exactly like real
    # RIsearch2 output.
    import sioff

    with tempfile.TemporaryDirectory() as tmp:
        idx = sioff.index(genome_fa, output=Path(tmp) / "genome.idx")
        hits = sioff.search(DATA_DIR / "sirnas.fa", idx, target=genome_fa)
    with (DATA_DIR / "predictions.out").open("w") as fh:
        for row in hits.iter_rows(named=True):
            fh.write(
                f"{row['sirna_id']}\t1\t{SIRNA_LEN}\t{row['chrom']}\t"
                f"{row['start']}\t{row['end']}\t{row['strand']}\t"
                f"{row['energy']:.6f}\n"
            )
    print(f"predictions.out: {hits.height} hits")

    acc_dir = DATA_DIR / "accessibility"
    if acc_dir.exists():
        shutil.rmtree(acc_dir)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "sioff.cli",
            "accessibility",
            "-f",
            str(genome_fa),
            "-o",
            str(acc_dir),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
