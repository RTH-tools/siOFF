from typing import Optional, Tuple, Dict

import polars as pl
from loguru import logger

from sioff.services.accessibility import GenomeAccessibilityService

# Gas constant in kcal/mol*K
R = 0.001987
# Default temperature in Kelvin (37 C) — kept for backward compatibility
T = 310.15
RT = R * T


def _require_opening_energy(df: pl.DataFrame) -> None:
    if "opening_energy" not in df.columns:
        raise ValueError(
            "precomputed_opening_energy=True but the frame has no 'opening_energy' "
            "column"
        )


class ProbabilityService:
    """Calculate off-target probabilities by integrating hybridization energy and accessibility."""

    def __init__(
        self,
        accessibility_service: Optional[GenomeAccessibilityService] = None,
        precomputed_accessibility: Optional[pl.DataFrame] = None,
        temperature: float = 37.0,
    ):
        self.accessibility_service = accessibility_service
        self.precomputed_accessibility = precomputed_accessibility
        self._temperature = temperature
        self._RT = R * (temperature + 273.15)

    def calculate_probabilities_per_sirna(
        self,
        df: pl.DataFrame,
        alpha_gamma_pairs: Optional[list[tuple[float, float]]] = None,
        theta_values: Optional[list[float]] = None,
        on_target_map: Optional[dict[str, str]] = None,
        precomputed_opening_energy: bool = False,
        aggregate_only: bool = False,
    ) -> Tuple[pl.DataFrame, Dict]:
        """Per-siRNA partition-function probabilities with optional alpha/gamma/theta sweeps.

        With ``aggregate_only`` only the per-siRNA sums are computed (the
        metadata ``z_per_sirna``, ``on_target_weights`` and ``n_rows``) and an
        empty frame is returned: the per-row energy, dG, weight and probability
        columns of every alpha/gamma/theta set are never materialised. That is
        all ``--summary-only`` needs, and for a large grid it is most of the
        memory (385 columns on millions of rows for the RIsearch3 benchmark).

        With an ``on_target_map`` the on-target rows are already in ``df`` and
        are tagged by ID match.

        With ``precomputed_opening_energy`` the frame must already carry an
        ``opening_energy`` column (e.g. gathered chromosome-major by the
        directory core) and the accessibility service is not consulted.
        """
        if "energy" not in df.columns:
            logger.warning("No 'energy' column found. Cannot calculate probabilities.")
            return df, {}

        if "sirna_id" not in df.columns:
            raise ValueError(
                "calculate_probabilities_per_sirna needs a 'sirna_id' column"
            )

        unique_sirnas = df["sirna_id"].unique()

        logger.info(
            f"Processing {len(unique_sirnas)} siRNAs with per-siRNA partition functions"
        )

        if precomputed_opening_energy:
            _require_opening_energy(df)
        elif self.accessibility_service:
            df = self._annotate_opening_energy(df)

        if "opening_energy" in df.columns:
            df = df.with_columns(
                (pl.col("energy") + pl.col("opening_energy")).alias("dG_total")
            )
        else:
            df = df.with_columns(
                pl.lit(0.0).alias("opening_energy"),
                pl.col("energy").alias("dG_total"),
            )

        if "exp_value" not in df.columns:
            df = df.with_columns(pl.lit(1.0).alias("exp_value"))
        else:
            # A missing expression carries no weight (a null would turn the
            # row's weight and probability into null).
            df = df.with_columns(pl.col("exp_value").fill_null(0.0))

        in_predictions_mode = (
            on_target_map is not None and "transcript_id" in df.columns
        )

        if in_predictions_mode:
            # Filter to siRNAs present here — avoids O(N_corpus) OR chain per worker call
            sirna_set = set(unique_sirnas.to_list())
            local_map = {k: v for k, v in on_target_map.items() if k in sirna_set}

            tag_conditions = []
            for sirna_id, target_id in local_map.items():
                cond = pl.col("sirna_id") == sirna_id
                id_match = pl.col("transcript_id") == target_id
                if "gene_id" in df.columns:
                    id_match = id_match | (pl.col("gene_id") == target_id)
                tag_conditions.append(cond & id_match)

            if tag_conditions:
                combined = tag_conditions[0]
                for c in tag_conditions[1:]:
                    combined = combined | c
                df = df.with_columns(combined.alias("is_on_target"))
            else:
                df = df.with_columns(pl.lit(False).alias("is_on_target"))

            n_on = df.filter(pl.col("is_on_target")).height
            logger.info(
                f"In-predictions on-target mode: tagged {n_on} rows as on-target (from {len(local_map)} mappings)"
            )

        else:
            df = df.with_columns(pl.lit(False).alias("is_on_target"))

        # Use raw_e_min (from full genome scan) when available; fall back to
        # post-intersection minimum. raw_e_min is set by the parser when loading
        # per-siRNA directory files.
        if "raw_e_min" in df.columns:
            min_energies = df.group_by("sirna_id").agg(
                pl.col("raw_e_min").min().alias("E_min")
            )
        else:
            min_energies = df.group_by("sirna_id").agg(
                pl.col("energy").min().alias("E_min")
            )
        df = df.join(min_energies, on="sirna_id", how="left")

        calc_configs = [("", "dG_total", "energy")]
        all_dG_exprs = []
        # suffix -> (dG, dG_noacc) expressions, so aggregate_only can sum the
        # weights without first adding the dG columns to the frame.
        dG_exprs: dict[str, tuple[pl.Expr, pl.Expr]] = {
            "": (pl.col("dG_total"), pl.col("energy"))
        }

        if alpha_gamma_pairs:
            for alpha, gamma in alpha_gamma_pairs:
                if alpha == 1.0 and gamma == 1.0:
                    continue  # base case already in calc_configs

                suffix = f":alpha={alpha},gamma={gamma}"
                col_name = f"dG_total{suffix}"
                col_name_noacc = f"energy{suffix}"

                # Clamp: energy < α·E_min → use γ·E_min; otherwise keep energy
                dG = (
                    pl.when(pl.col("energy") < alpha * pl.col("E_min"))
                    .then(gamma * pl.col("E_min") + pl.col("opening_energy"))
                    .otherwise(pl.col("energy") + pl.col("opening_energy"))
                )
                dG_noacc = (
                    pl.when(pl.col("energy") < alpha * pl.col("E_min"))
                    .then(gamma * pl.col("E_min"))
                    .otherwise(pl.col("energy"))
                )
                all_dG_exprs.extend(
                    [dG.alias(col_name), dG_noacc.alias(col_name_noacc)]
                )
                calc_configs.append((suffix, col_name, col_name_noacc))
                dG_exprs[suffix] = (dG, dG_noacc)

        if theta_values:
            for theta in theta_values:
                suffix = f":theta={theta}"
                col_name = f"dG_total{suffix}"
                col_name_noacc = f"energy{suffix}"

                # Theta scaling: ((θ·(E + 10)) - 10) + open_energy
                dG_noacc = (theta * (pl.col("energy") + 10.0)) - 10.0
                dG = dG_noacc + pl.col("opening_energy")
                all_dG_exprs.extend(
                    [dG.alias(col_name), dG_noacc.alias(col_name_noacc)]
                )
                calc_configs.append((suffix, col_name, col_name_noacc))
                dG_exprs[suffix] = (dG, dG_noacc)

        if all_dG_exprs and not aggregate_only:
            df = df.with_columns(all_dG_exprs)

        all_weight_exprs = []
        z_aggs = []
        on_aggs = []
        z_col_names = []

        for suffix, dG_col, dG_noacc_col in calc_configs:
            weight_col = f"boltzmann_weight{suffix}"
            weight_noacc_col = f"boltzmann_weight_noacc{suffix}"
            all_weight_exprs.extend(
                [
                    (pl.col("exp_value") * ((-pl.col(dG_col) / self._RT).exp())).alias(
                        weight_col
                    ),
                    (
                        pl.col("exp_value") * ((-pl.col(dG_noacc_col) / self._RT).exp())
                    ).alias(weight_noacc_col),
                ]
            )
            if aggregate_only:
                dG, dG_noacc = dG_exprs[suffix]
                w = pl.col("exp_value") * ((-dG / self._RT).exp())
                w_noacc = pl.col("exp_value") * ((-dG_noacc / self._RT).exp())
            else:
                w, w_noacc = pl.col(weight_col), pl.col(weight_noacc_col)
            z_col = f"Z_sirna{suffix}"
            z_noacc_col = f"Z_sirna_noacc{suffix}"
            z_aggs.extend([w.sum().alias(z_col), w_noacc.sum().alias(z_noacc_col)])
            z_col_names.extend([z_col, z_noacc_col])

            if in_predictions_mode:
                on_aggs.extend(
                    [
                        w.filter(pl.col("is_on_target")).sum().alias(f"W_on{suffix}"),
                        w_noacc.filter(pl.col("is_on_target"))
                        .sum()
                        .alias(f"W_on_noacc{suffix}"),
                    ]
                )

        if not aggregate_only:
            df = df.with_columns(all_weight_exprs)

        agg_df = df.group_by("sirna_id").agg(z_aggs + on_aggs)
        z_df = agg_df.select(["sirna_id"] + z_col_names)

        on_target_weight_metadata = {}

        if in_predictions_mode:
            on_df = agg_df.drop(z_col_names)
            for row in on_df.to_dicts():
                sid = row["sirna_id"]
                on_target_weight_metadata[sid] = {
                    k: v for k, v in row.items() if k != "sirna_id"
                }
            logger.info(
                f"In-predictions mode: computed on-target weights for {on_df.height} siRNAs"
            )

        z_stats_dict = {row["sirna_id"]: row for row in z_df.to_dicts()}
        metadata = {
            "n_sirnas": len(unique_sirnas),
            "n_rows": df.height,
            "z_per_sirna": z_stats_dict,
            "on_target_weights": on_target_weight_metadata,
            "on_target_count": len(on_target_weight_metadata),
        }
        if aggregate_only:
            return df.head(0), metadata

        df = df.join(z_df, on="sirna_id", how="left")

        probs_exprs = []
        for suffix, _, _ in calc_configs:
            weight_col = f"boltzmann_weight{suffix}"
            z_col = f"Z_sirna{suffix}"
            p_col = f"P_off_target{suffix}"
            probs_exprs.append(
                pl.when(pl.col(z_col) > 0)
                .then(pl.col(weight_col) / pl.col(z_col))
                .otherwise(0.0)
                .alias(p_col)
            )

        df = df.with_columns(probs_exprs)

        return df, metadata

    def format_legacy_summary(
        self,
        sirna_id: str,
        z_per_config: dict[str, float],
        on_target_weights: dict[str, float],
        alpha_gamma_pairs: list[tuple[float, float]],
        theta_values: list[float],
    ) -> str:
        """Format the aggregated metadata into the exact legacy .off header block."""
        lines = []
        lines.append(f"# On-target info for {sirna_id} #")

        # third element (gamma) is None in theta mode, which has no gamma
        suffixes: list[tuple[str, float, float | None]] = [("", 1.0, 1.0)]
        for theta in theta_values or []:
            suffixes.append((f":theta={theta}", theta, None))
        for alpha, gamma in alpha_gamma_pairs or []:
            if alpha == 1.0 and gamma == 1.0:
                continue
            suffixes.append((f":alpha={alpha},gamma={gamma}", alpha, gamma))

        for suffix_info in suffixes:
            suffix = suffix_info[0]

            if suffix == "":
                preamble = "# For alpha=1.0 and gamma=1.0;"
            elif suffix.startswith(":theta="):
                preamble = f"# For theta={suffix_info[1]};"
            else:
                preamble = f"# For alpha={suffix_info[1]} and gamma={suffix_info[2]};"

            z = z_per_config.get(f"Z_sirna{suffix}", 0.0)
            z_noacc = z_per_config.get(f"Z_sirna_noacc{suffix}", 0.0)
            w_on = on_target_weights.get(f"W_on{suffix}", 0.0)
            w_on_noacc = on_target_weights.get(f"W_on_noacc{suffix}", 0.0)

            z_off = z - w_on
            z_off_noacc = z_noacc - w_on_noacc

            p_on = w_on / z if z > 0 else 0.0
            p_on_noacc = w_on_noacc / z_noacc if z_noacc > 0 else 0.0

            lines.append(
                f"{preamble} P: {p_on!s}; P_noacc: {p_on_noacc!s}; "
                f"Z: {z!s}; Z_noacc: {z_noacc!s}; "
                f"Zoff: {z_off!s}; Zoff_noacc: {z_off_noacc!s}"
            )

        lines.append("## End of on-target info ##\n")
        return "\n".join(lines)

    def _annotate_opening_energy(self, df: pl.DataFrame) -> pl.DataFrame:
        if df.height == 0:
            return df

        if not all(c in df.columns for c in ["chrom", "start", "end", "strand"]):
            logger.warning(
                "Missing coordinate or strand columns. Cannot lookup accessibility."
            )
            return df

        if self.precomputed_accessibility is not None:
            join_keys = ["chrom", "start", "end", "strand"]
            result = df.join(
                self.precomputed_accessibility.select(join_keys + ["opening_energy"]),
                on=join_keys,
                how="left",
            )
            if "opening_energy" in result.columns:
                result = result.with_columns(pl.col("opening_energy").fill_null(10.0))
            else:
                result = result.with_columns(pl.lit(10.0).alias("opening_energy"))
            logger.info(
                f"Annotated {result.height} rows with precomputed accessibility"
            )
            return result

        if not self.accessibility_service:
            return df

        return self.accessibility_service.annotate_opening_energy_vectorized(df)
