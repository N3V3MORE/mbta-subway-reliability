"""Track B: cluster stations by reliability and by ridership demand profile.

Two unsupervised analyses over the same station universe:

1. **Reliability clusters** -- a station x hour-of-day matrix of median delay,
   plus summary reliability statistics. Answers "which stations are chronically
   late, and is lateness a peak-only phenomenon or an all-day one?"
2. **Demand clusters** -- a station x 48-half-hour *shape* profile from gated
   entries, normalised per station so the clustering captures when people travel
   rather than how many do. Answers "what kinds of stations does this network
   have?", which is the classic transit-planning question.

Both matrices are standardised, reduced with PCA (which is exactly the SVD of the
centred matrix -- the course's SVD/LSA lecture in practice), and clustered with
k-means++ chosen by silhouette. Results are cross-checked against Ward
hierarchical clustering, Gaussian mixtures and DBSCAN, and the agreement between
methods is reported as an adjusted Rand index so the clusters are not an artefact
of one algorithm's assumptions.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from . import clean, collect_ridership, config
from .progress import Progress

log = logging.getLogger(__name__)

CLUSTERS_PATH = config.PROCESSED_DIR / "station_clusters.parquet"
METRICS_PATH = config.PROCESSED_DIR / "cluster_metrics.json"
PROFILES_PATH = config.PROCESSED_DIR / "cluster_profiles.csv"

SEED = 506
K_RANGE = range(2, 9)
#: Cap the PCA components so the clustering stays interpretable.
MAX_COMPONENTS = 6
#: Minimum stations for a cluster solution to be meaningful.
MIN_STATIONS = 8

#: A station's demand *shape* is only meaningful if it is actually observed for a
#: reasonable part of the day and carries a reasonable volume. `Longwood` in the
#: 2026-04-02..06-30 window has 7 rows in total -- one faregate entry per day --
#: so its normalised profile is six spikes at 1/6 each, which is an artefact of
#: sparsity rather than a travel pattern. Every other station has 40-48 distinct
#: half-hour periods, so these thresholds separate cleanly.
MIN_PERIODS_OBSERVED = 24   # at least half of the 48 half-hour periods
MIN_TOTAL_ENTRIES = 500.0


@dataclass
class ClusterSolution:
    name: str
    labels: np.ndarray
    k: int
    silhouette: float
    davies_bouldin: float
    inertia: float | None = None


# ---------------------------------------------------------------------------
# Matrix construction
# ---------------------------------------------------------------------------
def reliability_matrix(frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build the station x hour matrix of median delay plus reliability stats.

    Median (not mean) delay per cell, because the delay distribution has a long
    right tail and a mean would let a handful of meltdowns dominate a station's
    profile.
    """
    frame = clean.load() if frame is None else frame

    pivoted = (
        frame.pivot_table(
            index="station_name",
            columns="scheduled_hour",
            values="delay_seconds",
            aggfunc="median",
        )
        .reindex(columns=range(24))
    )
    # A station that has no service in an hour gets the network-wide value for
    # that hour, so the profile reflects "no data" rather than a fake zero delay.
    # An hour with no service anywhere (e.g. an overnight shutdown) has no
    # network value either; it carries no information, so it is dropped.
    pivoted = pivoted.dropna(axis=1, how="all")
    pivoted = pivoted.fillna(pivoted.median(axis=0))

    stats_frame = frame.groupby("station_name").agg(
        mean_delay=("delay_seconds", "mean"),
        median_delay=("delay_seconds", "median"),
        p90_delay=("delay_seconds", lambda s: s.quantile(0.90)),
        on_time_rate=("late", lambda s: 1.0 - float(s.mean())),
        n_observations=("delay_seconds", "size"),
    )
    matrix = pivoted.join(stats_frame)
    matrix.columns = [f"h{int(c):02d}" if isinstance(c, (int, np.integer)) else c
                      for c in matrix.columns]
    return matrix


def demand_matrix(station_names: pd.Series | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the station x 48-half-hour entry-shape matrix.

    ``station_names`` maps a stop id (``place-state``) to the name the delay data
    uses. The two sources name some stations differently ("State Street" vs
    "State", "Mattapan Line" vs "Mattapan"), so without it those stations never
    join to their reliability cluster. The shared stop id is the reliable key.

    Each row is normalised to sum to 1, so the clustering compares the *shape* of
    demand over the day. A busy terminus and a quiet suburban stop with the same
    commute pattern then land in the same cluster, which is the intended reading.

    Stations with too little data to have a meaningful shape are excluded, and the
    coverage table is returned alongside so the exclusion can be reported rather
    than silently applied. Normalising is what makes this necessary: dividing by a
    near-zero daily total turns measurement noise into an extreme-looking profile.
    """
    raw = collect_ridership.load_raw().copy()
    if station_names is not None and "stop_id" in raw.columns:
        raw["station_name"] = raw["stop_id"].map(station_names).fillna(raw["station_name"])
    raw["period_index"] = (
        raw["period_hour"].astype(int) * 2 + (raw["period_minute"].astype(int) // 30)
    )

    coverage = raw.groupby("station_name").agg(
        periods_observed=("period_index", "nunique"),
        days_observed=("service_date", "nunique"),
        total_entries=("gated_entries", "sum"),
    )
    well_covered = coverage[
        (coverage["periods_observed"] >= MIN_PERIODS_OBSERVED)
        & (coverage["total_entries"] >= MIN_TOTAL_ENTRIES)
    ]
    excluded = coverage.drop(index=well_covered.index)
    if not excluded.empty:
        log.warning(
            "excluding %d station(s) from demand clustering for insufficient "
            "coverage: %s", len(excluded), excluded.round(1).to_dict("index"),
        )

    raw = raw[raw["station_name"].isin(well_covered.index)]

    # Mean entries per day in each half-hour. Two corrections to a plain mean:
    # the feed splits interchange stations across lines, so lines are *summed*
    # (averaging them halved Park Street, Downtown Crossing and five others); and a
    # half-hour with no row had no taps, so sums are divided by every day the
    # station reported, not only the days that period appeared.
    per_period = (
        raw.groupby(["station_name", "period_index"])["gated_entries"].sum()
        .unstack(fill_value=0.0).reindex(columns=range(48), fill_value=0.0)
    )
    pivoted = per_period.div(coverage["days_observed"].reindex(per_period.index), axis=0)

    totals = pivoted.sum(axis=1)  # mean daily entries
    shape = pivoted.div(totals.replace(0, np.nan), axis=0).dropna(how="all")
    shape.columns = [f"p{int(c):02d}" for c in shape.columns]

    shape["total_entries"] = totals.reindex(shape.index)

    coverage = coverage.copy()
    coverage["clustered"] = coverage.index.isin(shape.index)
    return shape, coverage


# ---------------------------------------------------------------------------
# Clustering helpers
# ---------------------------------------------------------------------------
def reduce_dimensions(matrix: np.ndarray, *, name: str) -> tuple[np.ndarray, PCA]:
    """Standardise and reduce with PCA/SVD, keeping enough components for 90% variance."""
    scaled = StandardScaler().fit_transform(matrix)
    max_components = min(MAX_COMPONENTS, scaled.shape[0] - 1, scaled.shape[1])
    pca = PCA(n_components=max_components, random_state=SEED)
    reduced = pca.fit_transform(scaled)
    keep = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.90) + 1)
    keep = max(1, min(keep, max_components))
    log.info("  %s: PCA %d components explain %.1f%% of variance; using %d",
             name, max_components, pca.explained_variance_ratio_.sum() * 100, keep)
    return reduced[:, :keep], pca


def _score(matrix: np.ndarray, labels: np.ndarray, name: str, k: int,
           inertia: float | None = None) -> ClusterSolution:
    unique = np.unique(labels)
    valid = len(unique) > 1 and len(unique) < len(labels)
    return ClusterSolution(
        name=name,
        labels=labels,
        k=k,
        silhouette=float(silhouette_score(matrix, labels)) if valid else float("nan"),
        davies_bouldin=float(davies_bouldin_score(matrix, labels)) if valid else float("nan"),
        inertia=inertia,
    )


def kmeans_sweep(matrix: np.ndarray, *, name: str) -> tuple[ClusterSolution, list[dict]]:
    """Run k-means++ across k and pick the best by silhouette score."""
    diagnostics: list[dict] = []
    best: ClusterSolution | None = None

    candidates = [k for k in K_RANGE if k < len(matrix)]
    with Progress(len(candidates), f"{name} k sweep") as progress:
        for k in candidates:
            model = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=SEED)
            labels = model.fit_predict(matrix)
            solution = _score(matrix, labels, f"kmeans_k{k}", k, model.inertia_)
            diagnostics.append({
                "k": k,
                "inertia": float(model.inertia_),
                "silhouette": solution.silhouette,
                "davies_bouldin": solution.davies_bouldin,
            })
            progress.step(f"k={k}", extra=f"silhouette {solution.silhouette:.3f}")
            if best is None or (np.isfinite(solution.silhouette)
                                and solution.silhouette > best.silhouette):
                best = solution

    if best is None:
        raise ValueError(f"could not cluster {name}: only {len(matrix)} stations")
    log.info("  %s: chose k=%d (silhouette %.3f)", name, best.k, best.silhouette)
    return best, diagnostics


def compare_algorithms(matrix: np.ndarray, reference: ClusterSolution) -> dict:
    """Cross-check the chosen k-means solution against other algorithms."""
    results: dict[str, dict] = {}

    ward = AgglomerativeClustering(n_clusters=reference.k, linkage="ward")
    ward_labels = ward.fit_predict(matrix)
    results["ward"] = {
        "silhouette": float(silhouette_score(matrix, ward_labels)),
        "adjusted_rand_vs_kmeans": float(adjusted_rand_score(reference.labels, ward_labels)),
    }

    gmm = GaussianMixture(n_components=reference.k, covariance_type="diag",
                          n_init=5, random_state=SEED)
    gmm_labels = gmm.fit_predict(matrix)
    results["gaussian_mixture"] = {
        "silhouette": float(silhouette_score(matrix, gmm_labels)),
        "adjusted_rand_vs_kmeans": float(adjusted_rand_score(reference.labels, gmm_labels)),
    }

    # DBSCAN assumes density-separated clusters, which this smooth station data
    # may not have. It is reported because finding no clusters is informative.
    dbscan_labels = DBSCAN(eps=1.5, min_samples=3).fit_predict(matrix)
    n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    noise = int((dbscan_labels == -1).sum())
    dbscan_entry: dict = {"clusters_found": n_clusters, "noise_points": noise}
    if n_clusters > 1:
        mask = dbscan_labels != -1
        dbscan_entry["silhouette"] = float(silhouette_score(matrix[mask], dbscan_labels[mask]))
        dbscan_entry["adjusted_rand_vs_kmeans"] = float(
            adjusted_rand_score(reference.labels[mask], dbscan_labels[mask])
        )
    results["dbscan"] = dbscan_entry

    log.info("  agreement with k-means: ward ARI %.2f, GMM ARI %.2f, DBSCAN %s clusters",
             results["ward"]["adjusted_rand_vs_kmeans"],
             results["gaussian_mixture"]["adjusted_rand_vs_kmeans"],
             n_clusters)
    return results


# ---------------------------------------------------------------------------
# Cluster naming
# ---------------------------------------------------------------------------
RELIABILITY_LABELS = (
    "most reliable", "reliable", "moderately delayed", "delay-prone", "most delay-prone",
)

#: Share of a station's daily entries falling in each time band, used to name
#: demand clusters. Boundaries are in the 48 half-hour period indices.
DEMAND_BANDS: tuple[tuple[str, int, int], ...] = (
    ("night", 0, 12),                # 00:00-06:00
    ("morning-peaked", 12, 20),      # 06:00-10:00
    ("midday-heavy", 20, 32),        # 10:00-16:00
    ("afternoon-peaked", 32, 40),    # 16:00-20:00
    ("evening-heavy", 40, 48),       # 20:00-24:00
)

#: A band is treated as "peaked" when it carries at least this much more than its
#: proportional share of the clock. Comparing a band's share of the day against
#: the band's share of the clock is what makes the comparison fair: the night band
#: spans six hours and the morning peak only four, so raw shares would always make
#: the night look busy.
MIN_BAND_INTENSITY = 1.15


def demand_band_shares(profile: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Return each cluster's share of daily entries within each time band.

    A cluster's band share is the average over its stations of the summed share of
    the periods in that band. Because each station profile sums to 1, these are
    directly comparable percentages of a station's day.
    """
    table = profile.assign(cluster=labels).groupby("cluster")
    shares = {
        name: table[[f"p{i:02d}" for i in range(low, high)]].mean().sum(axis=1)
        for name, low, high in DEMAND_BANDS
    }
    return pd.DataFrame(shares)


def demand_band_intensity(shares: pd.DataFrame) -> pd.DataFrame:
    """Rescale band shares into ``1.0 == travels in proportion to the clock``.

    A station whose entries were spread perfectly evenly across the day scores
    exactly 1.0 in every band; a band scoring 1.5 carries half again as much of the
    day's travel as its duration alone would predict.
    """
    clock_share = pd.Series(
        {name: (high - low) / 48 for name, low, high in DEMAND_BANDS}
    )
    return shares.div(clock_share, axis=1)


def name_demand_clusters(profile: pd.DataFrame, labels: np.ndarray) -> dict[int, str]:
    """Label demand clusters by the time band where they are most over-represented.

    Labels are assigned in ascending cluster order so the result is reproducible,
    and names are guaranteed unique: a band already claimed by an earlier cluster
    is skipped, and the rare genuinely flat cluster falls back to a qualified
    "all-day steady" label. Uniqueness matters because the names are later used as
    group keys, and a duplicate would silently merge two distinct clusters.
    """
    intensity = demand_band_intensity(demand_band_shares(profile, labels))

    names: dict[int, str] = {}
    claimed: set[str] = set()

    for cluster in sorted(intensity.index):
        row = intensity.loc[cluster].sort_values(ascending=False)
        chosen: str | None = None

        for band, value in row.items():
            if value < MIN_BAND_INTENSITY:
                break
            if band not in claimed:
                chosen = band
                break

        if chosen is None:
            # Nothing is over-represented: the cluster is genuinely all-day.
            top = row.index[0]
            chosen = f"all-day steady, {top.split('-')[0]} lean"
        if chosen in claimed:
            chosen = f"{chosen} ({cluster})"

        claimed.add(chosen)
        names[int(cluster)] = chosen

    log.info("  demand cluster names: %s", names)
    return names


def name_reliability_clusters(profile: pd.DataFrame, labels: np.ndarray) -> dict[int, str]:
    """Rank clusters by mean delay and give them ordered descriptive names.

    A cluster whose trains arrive *before* the timetable on average is named
    "runs early" rather than "reliable": for a rider an early train is a missed
    one. (Green-E to Heath Street, the Green Line Extension and the Mattapan line
    form such a cluster.) Names stay unique because they are used as group keys.
    """
    ranking = profile.assign(cluster=labels).groupby("cluster")["mean_delay"].mean().sort_values()
    names: dict[int, str] = {}
    for position, (cluster, delay) in enumerate(ranking.items()):
        # Spread the label list across however many clusters were found.
        index = round(position * (len(RELIABILITY_LABELS) - 1) / max(len(ranking) - 1, 1))
        name = "runs early" if delay < 0 else RELIABILITY_LABELS[index]
        names[int(cluster)] = f"{name} ({cluster})" if name in names.values() else name
    return names


# ---------------------------------------------------------------------------
# Cross-track analysis
# ---------------------------------------------------------------------------
def cross_track_test(joined: pd.DataFrame) -> dict:
    """Test whether reliability and demand clusters are associated.

    A chi-square test on the contingency table answers "are the two typologies
    independent?", and a one-way ANOVA answers the more practically interesting
    "does mean delay differ across demand clusters?".
    """
    contingency = pd.crosstab(joined["reliability_cluster"], joined["demand_cluster"])
    out: dict = {"contingency_shape": list(contingency.shape)}

    if contingency.shape[0] > 1 and contingency.shape[1] > 1:
        chi2, p_value, dof, _ = stats.chi2_contingency(contingency)
        out["chi2"] = float(chi2)
        out["chi2_p_value"] = float(p_value)
        out["chi2_dof"] = int(dof)
    else:
        out["chi2_p_value"] = None

    groups = [
        group["mean_delay"].to_numpy()
        for _, group in joined.groupby("demand_cluster")
        if len(group) > 1
    ]
    if len(groups) > 1:
        f_stat, p_value = stats.f_oneway(*groups)
        out["anova_f"] = float(f_stat)
        out["anova_p_value"] = float(p_value)
    else:
        out["anova_p_value"] = None

    out["mean_delay_by_demand_cluster"] = (
        joined.groupby("demand_cluster")["mean_delay"].mean().round(1).to_dict()
    )
    return out


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------
def run() -> dict:
    """Cluster stations on both axes, cross-check methods, and persist results."""
    config.ensure_dirs()
    metrics: dict = {}
    profiles: list[pd.DataFrame] = []

    # --- reliability -----------------------------------------------------
    log.info("--- reliability clustering ---")
    clean_frame = clean.load()
    # Arrivals only: origin rows are platform waits, and including them made
    # terminals (a sixth of the "reliable" cluster's rows) look punctual.
    rel_matrix = reliability_matrix(clean_frame[clean.is_arrival(clean_frame)])
    # stop id -> the station name used by the delay data, so demand stations are
    # joined to reliability stations by id rather than by free-text name.
    station_names = (
        clean_frame[["parent_station", "station_name"]]
        .drop_duplicates(subset=["parent_station"])
        .set_index("parent_station")["station_name"]
    )
    del clean_frame
    if len(rel_matrix) < MIN_STATIONS:
        raise ValueError(f"only {len(rel_matrix)} stations; need {MIN_STATIONS}")
    rel_stats = rel_matrix[["mean_delay", "median_delay", "p90_delay",
                            "on_time_rate", "n_observations"]]
    rel_features = rel_matrix.drop(columns=rel_stats.columns)
    rel_reduced, rel_pca = reduce_dimensions(rel_features.to_numpy(), name="reliability")

    rel_solution, rel_diag = kmeans_sweep(rel_reduced, name="reliability")
    rel_compare = compare_algorithms(rel_reduced, rel_solution)
    rel_names = name_reliability_clusters(rel_stats, rel_solution.labels)

    rel_out = rel_stats.copy()
    rel_out["reliability_cluster_id"] = rel_solution.labels
    rel_out["reliability_cluster"] = [rel_names[int(c)] for c in rel_solution.labels]
    metrics["reliability"] = {
        "k": rel_solution.k,
        "silhouette": rel_solution.silhouette,
        "davies_bouldin": rel_solution.davies_bouldin,
        "pca_explained_variance": [round(float(v), 4)
                                   for v in rel_pca.explained_variance_ratio_],
        "k_selection": rel_diag,
        "algorithm_comparison": rel_compare,
        "cluster_names": {str(k): v for k, v in rel_names.items()},
        "cluster_sizes": rel_out["reliability_cluster"].value_counts().to_dict(),
    }
    profiles.append(
        rel_out.groupby("reliability_cluster")[
            ["mean_delay", "p90_delay", "on_time_rate"]
        ].mean().round(2).add_prefix("reliability_").reset_index()
    )

    # --- demand ----------------------------------------------------------
    log.info("--- demand clustering ---")
    demand: pd.DataFrame | None
    coverage = pd.DataFrame()
    try:
        demand, coverage = demand_matrix(station_names)
    except FileNotFoundError as exc:
        log.warning("skipping demand clustering: %s", exc)
        demand = None

    joined = rel_out.copy()
    if demand is not None and len(demand) >= MIN_STATIONS:
        demand_stats = demand[["total_entries"]]
        demand_features = demand.drop(columns=["total_entries"])
        demand_reduced, demand_pca = reduce_dimensions(
            demand_features.to_numpy(), name="demand"
        )
        dem_solution, dem_diag = kmeans_sweep(demand_reduced, name="demand")
        dem_compare = compare_algorithms(demand_reduced, dem_solution)
        dem_names = name_demand_clusters(demand_features, dem_solution.labels)

        dem_out = demand_stats.copy()
        dem_out["demand_cluster_id"] = dem_solution.labels
        dem_out["demand_cluster"] = [dem_names[int(c)] for c in dem_solution.labels]
        metrics["demand"] = {
            "k": dem_solution.k,
            "silhouette": dem_solution.silhouette,
            "davies_bouldin": dem_solution.davies_bouldin,
            "pca_explained_variance": [round(float(v), 4)
                                       for v in demand_pca.explained_variance_ratio_],
            "k_selection": dem_diag,
            "algorithm_comparison": dem_compare,
            "cluster_names": {str(k): v for k, v in dem_names.items()},
            "cluster_sizes": dem_out["demand_cluster"].value_counts().to_dict(),
            "stations_without_demand_data": int(
                len(set(rel_out.index) - set(dem_out.index))
            ),
            "stations_excluded_for_coverage": (
                coverage.loc[~coverage["clustered"]].round(1).to_dict("index")
                if not coverage.empty else {}
            ),
            "coverage_floor": {
                "min_periods_observed": MIN_PERIODS_OBSERVED,
                "min_total_entries": MIN_TOTAL_ENTRIES,
            },
        }
        profiles.append(
            dem_out.groupby("demand_cluster")[["total_entries"]]
            .agg(["mean", "size"]).round(0).reset_index()
        )
        joined = rel_out.join(
            dem_out[["demand_cluster", "demand_cluster_id", "total_entries"]], how="left"
        )
        metrics["cross_track"] = cross_track_test(joined.dropna(subset=["demand_cluster"]))

        # Persist the demand profiles so the figures can plot the cluster shapes.
        # The numeric cluster id is stored alongside the name because names can
        # legitimately repeat, and grouping on a repeated name would silently merge
        # distinct clusters.
        demand.assign(
            demand_cluster=dem_out["demand_cluster"],
            demand_cluster_id=dem_out["demand_cluster_id"],
        ).to_parquet(config.PROCESSED_DIR / "demand_profiles.parquet")

    joined = joined.reset_index().rename(columns={"index": "station_name"})
    if "station_name" not in joined.columns:
        joined = joined.rename(columns={joined.columns[0]: "station_name"})
    joined.to_parquet(CLUSTERS_PATH, index=False)

    try:
        pd.concat(profiles, axis=1).to_csv(PROFILES_PATH, index=False)
    except Exception:  # noqa: BLE001 - profiles are a convenience artefact
        log.debug("could not concatenate cluster profiles")
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")

    log.info("clustered %d stations -> %s", len(joined), CLUSTERS_PATH.name)
    return {
        "stations": int(len(joined)),
        "reliability_k": metrics["reliability"]["k"],
        "reliability_silhouette": round(metrics["reliability"]["silhouette"], 3),
        "demand_k": metrics.get("demand", {}).get("k"),
        "demand_silhouette": (round(metrics["demand"]["silhouette"], 3)
                              if "demand" in metrics else None),
    }


def load_clusters() -> pd.DataFrame:
    """Load the per-station cluster assignments."""
    if not CLUSTERS_PATH.exists():
        raise FileNotFoundError(f"{CLUSTERS_PATH} missing; run the `cluster` stage first")
    return pd.read_parquet(CLUSTERS_PATH)


def load_metrics() -> dict:
    """Load the persisted clustering diagnostics."""
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"{METRICS_PATH} missing; run the `cluster` stage first")
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
