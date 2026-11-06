from typing import Mapping, Optional, Tuple, Dict

import polars as pl
from loguru import logger

from sioff.services.accessibility import GenomeAccessibilityService

# Gas constant in kcal/mol*K
R = 0.001987
# Default temperature in Kelvin (37 C) — kept for backward compatibility
T = 310.15
RT = R * T


class ProbabilityService:
    """Calculate off-target probabilities by integrating hybridization energy and accessibility."""

    def __init__(
        self,
        accessibility_service: Optional[GenomeAccessibilityService] = None,
        temperature: float = 37.0,
    ):
        self.accessibility_service = accessibility_service
        self._temperature = temperature
        self._RT = R * (temperature + 273.15)

    def calculate_probabilities_per_sirna(
        self,
        df: pl.DataFrame,
        alpha_gamma_pairs: Optional[list[tuple[float, float]]] = None,
        theta_values: Optional[list[float]] = None,
        on_target_map: Optional[dict[str, str]] = None,
        e_min: Optional[Mapping[str, float]] = None,
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
        are tagged by ID match. The metadata's ``on_target_weights`` then
        holds, per siRNA and parameter set, the weight sums of the on-target
        rows (``W_on*``) and of the other rows (``W_off*``, the summary's
        Zoff), each with and without accessibility (``_noacc``). ``W_off*``
        is summed directly, not taken as Z - W_on, which cancels to 0 once
        the on-target dominates.

        With ``e_min`` (``sirna_id -> E_min``) the clamp of every alpha/gamma
        set is anchored on that value; a siRNA missing from it raises
        ``ValueError``. Without it ``E_min`` is the minimum ``energy`` of the
        siRNA's rows in ``df``. The metadata's ``n_on_target_rows`` counts each
        siRNA's rows tagged on-target (0 when none).

        With an accessibility service the opening energies are looked up;
        without one an ``opening_energy`` column already in ``df`` is used,
        and otherwise they are 0.

        The Boltzmann weights (``boltzmann_weight*``), the partition
        functions (``Z_sirna*``) and the sums ``W_on*`` and ``W_off*`` are raw,
        unshifted and Float64, so a strong duplex cannot overflow them; every
        ``P_off_target*`` column is cast to Float32 after the division. The
        energy and dG columns keep their dtypes (Float32 from RIsearch).
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

        if self.accessibility_service:
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

        if e_min is not None:
            missing = sorted(str(sid) for sid in unique_sirnas if sid not in e_min)
            if missing:
                raise ValueError(f"No E_min given for siRNA(s): {', '.join(missing)}")
            min_energies = pl.DataFrame(
                {
                    "sirna_id": unique_sirnas,
                    "E_min": pl.Series([e_min[sid] for sid in unique_sirnas]).cast(
                        df["energy"].dtype
                    ),
                }
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
        z_col_names = []

        for suffix, dG_col, dG_noacc_col in calc_configs:
            weight_col = f"boltzmann_weight{suffix}"
            weight_noacc_col = f"boltzmann_weight_noacc{suffix}"
            all_weight_exprs.extend(
                [
                    self._boltzmann_weight(pl.col(dG_col)).alias(weight_col),
                    self._boltzmann_weight(pl.col(dG_noacc_col)).alias(
                        weight_noacc_col
                    ),
                ]
            )
            if aggregate_only:
                dG, dG_noacc = dG_exprs[suffix]
                w = self._boltzmann_weight(dG)
                w_noacc = self._boltzmann_weight(dG_noacc)
            else:
                w, w_noacc = pl.col(weight_col), pl.col(weight_noacc_col)
            z_col = f"Z_sirna{suffix}"
            z_noacc_col = f"Z_sirna_noacc{suffix}"
            z_aggs.extend([w.sum().alias(z_col), w_noacc.sum().alias(z_noacc_col)])
            z_col_names.extend([z_col, z_noacc_col])

        if not aggregate_only:
            df = df.with_columns(all_weight_exprs)

        weights = None
        if in_predictions_mode:
            weights = self._on_off_weights(df, z_aggs, z_col_names, aggregate_only)
        if aggregate_only and weights is not None:
            # No P comes from these Z, so Z is W_on + W_off: summing the
            # weights again would evaluate every one a second time. The
            # .summary values can differ from a full run's in the last bit.
            z_df = weights.select(
                "sirna_id",
                *[
                    (
                        pl.col(c.replace("Z_sirna", "W_on", 1))
                        + pl.col(c.replace("Z_sirna", "W_off", 1))
                    ).alias(c)
                    for c in z_col_names
                ],
            )
        else:
            z_df = df.group_by("sirna_id").agg(z_aggs)
        n_on_target_rows = dict.fromkeys(z_df["sirna_id"].to_list(), 0)
        on_target_weight_metadata = {}

        if weights is not None:
            for row in weights.to_dicts():
                sid = row.pop("sirna_id")
                n_on_target_rows[sid] = int(row.pop("_n_on"))
                on_target_weight_metadata[sid] = row
            logger.info(
                f"In-predictions mode: computed on-target weights for {weights.height} siRNAs"
            )

        z_stats_dict = {row["sirna_id"]: row for row in z_df.to_dicts()}
        metadata = {
            "n_sirnas": len(unique_sirnas),
            "n_rows": df.height,
            "z_per_sirna": z_stats_dict,
            "on_target_weights": on_target_weight_metadata,
            "on_target_count": len(on_target_weight_metadata),
            "n_on_target_rows": n_on_target_rows,
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
                .cast(pl.Float32)
                .alias(p_col)
            )

        df = df.with_columns(probs_exprs)

        return df, metadata

    @staticmethod
    def _on_off_weights(
        df: pl.DataFrame,
        z_aggs: list[pl.Expr],
        z_col_names: list[str],
        aggregate_only: bool,
    ) -> pl.DataFrame:
        """Each siRNA's weight sums over its on-target rows and over the others.

        One sum per (siRNA, on-target flag) gives both: a filtered sum per
        column (``w.filter(on).sum()``) costs about as much as the whole
        probability stage again. The off-target sum is its own sum, not
        Z - W_on, which cancels once the on-target dominates (every digit is
        gone past W_on/Zoff ~ 1e15).

        Returns
        -------
        polars.DataFrame
            One row per siRNA: ``sirna_id``, ``W_on*`` and ``W_off*`` (each
            ``Z_sirna*`` column renamed; 0.0 for a siRNA without such rows)
            and ``_n_on``, the number of on-target rows.
        """
        on = pl.col("is_on_target").fill_null(False)
        # Measured on polars 2.0, single-threaded: summing weight columns, the
        # plain column key is ~3x faster than a computed one; summing weight
        # expressions (aggregate_only), the computed key is ~15% faster.
        key = on.alias("is_on_target") if aggregate_only else "is_on_target"
        split = df.group_by("sirna_id", key).agg(z_aggs + [pl.len().alias("_n")])

        def part(rows: pl.DataFrame, prefix: str, *extra: pl.Expr) -> pl.DataFrame:
            # One row per siRNA, except a null flag (a row without IDs),
            # which is off-target: its group is added to the False one.
            return rows.group_by("sirna_id").agg(
                *[
                    pl.col(c).sum().alias(c.replace("Z_sirna", prefix, 1))
                    for c in z_col_names
                ],
                *extra,
            )

        n_on = pl.col("_n").sum().alias("_n_on")
        return (
            split.select("sirna_id")
            .unique(maintain_order=True)
            .join(part(split.filter(on), "W_on", n_on), on="sirna_id", how="left")
            .join(part(split.filter(~on), "W_off"), on="sirna_id", how="left")
            .fill_null(0)
        )

    def _boltzmann_weight(self, dG: pl.Expr) -> pl.Expr:
        """``exp_value * exp(-dG / RT)`` in Float64.

        ``dG`` and ``exp_value`` are cast to Float64 first: in Float32,
        whose range ends near 3.4e38, ``exp(-dG / RT)`` overflows for dG
        below about -54.7 kcal/mol at 37 °C (sooner with ``exp_value`` > 1)
        and the siRNA's Z with it. Float64 holds down to about -437 kcal/mol.

        Returns
        -------
        polars.Expr
            The row's raw Boltzmann weight, Float64.
        """
        return pl.col("exp_value").cast(pl.Float64) * (
            (-dG.cast(pl.Float64) / self._RT).exp()
        )

    def format_legacy_summary(
        self,
        sirna_id: str,
        z_per_config: dict[str, float],
        on_target_weights: dict[str, float],
        alpha_gamma_pairs: list[tuple[float, float]],
        theta_values: list[float],
    ) -> str:
        """Format the aggregated metadata into the exact legacy .off header block.

        Zoff is ``W_off*`` from ``on_target_weights``; without an on-target
        weight it is Z.

        Returns
        -------
        str
            The ``# On-target info`` block, one ``# For ...`` line per
            parameter set.
        """
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

            # The off-target rows' own sum: Z - W_on cancels when the
            # on-target dominates. With no on-target weight Zoff is Z itself.
            z_off = on_target_weights.get(f"W_off{suffix}", z - w_on) if w_on else z
            z_off_noacc = (
                on_target_weights.get(f"W_off_noacc{suffix}", z_noacc - w_on_noacc)
                if w_on_noacc
                else z_noacc
            )

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

        if not self.accessibility_service:
            return df

        return self.accessibility_service.annotate_opening_energy_vectorized(df)
