"""
MO (modus operandi) discovery for fraud - single-file version.

Pipeline
  1. Polars lazy load + hash sampling of goods (1M of 7.8M) + all fraud
  2. Column screening: null %, null-rate leakage between files, single-feature AUC leakage,
     near-constant, Spearman correlation, LightGBM gain ranking -> top features (float32)
  3. Stage A  : LightGBM fraud vs goods, K-fold OOF scores + OOF SHAP for every fraud row
     Missed pool: temporal (train on early months, score later) / existing score / OOF
     Stage B  : LightGBM missed fraud vs goods, OOF SHAP for the missed pool
  4. Clustering: UMAP -> HDBSCAN -> merge into MO families, seed-stability check
     spaces: all_shap, missed_shap, missed_quantile (goods-quantile -> normal, fixes skew)
  5. Profiling: tree rule, population-scale lift, stability, emergence by month, drivers, demographics
  6. Rule peeling (subgroup discovery + sequential covering) as an independent cross-check
  7. report.md + CSV/parquet/PNG outputs in OUT_DIR; intermediate results cached in OUT_DIR/cache

Install: pip install polars umap-learn "scikit-learn>=1.3" scipy matplotlib pyarrow
         (no LightGBM / XGBoost needed - uses scikit-learn HistGradientBoostingClassifier;
          umap-learn is optional, PCA is used if it is missing)
Usage : edit the CONFIG section at the bottom (inside `if __name__ == "__main__":`) and run
        python mo_discovery.py      -- or import it in a notebook and call main(Config(...)).
"""
import glob
import json
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
try:
    import umap
except ImportError:          # optional - falls back to PCA
    umap = None
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.stats import rankdata
from inspect import signature

from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import adjusted_rand_score, average_precision_score, roc_auc_score
from sklearn.preprocessing import QuantileTransformer
from sklearn.tree import DecisionTreeClassifier



# ==================================================================================================
# CONFIG
# ==================================================================================================
@dataclass
class Config:
    # ------------------------------------------------------------------ paths
    goods_glob: str = "data_goods/*.parquet"          # the 7 snappy parquet files
    fraud_path: str = "data_fraud/fraud.parquet"      # .parquet or .csv
    out_dir: str = "mo_output"

    # ------------------------------------------------------------------ columns
    tran_date_col: str = "tran_dt"                     # anchor date of the transaction
    other_date_cols: List[str] = field(default_factory=lambda: ["first_credit_dt"])
    # Column names differ between the two extracts? Rename BOTH sides to a common name, applied at scan time.
    # e.g. the fraud file's transaction date is "earliest_fraud_tran_date" but goods call it "tran_dt":
    #   tran_date_col="tran_dt", rename_fraud={"earliest_fraud_tran_date": "tran_dt"}
    rename_goods: dict = field(default_factory=dict)
    rename_fraud: dict = field(default_factory=dict)
    all_date_gaps: bool = True         # gaps between EVERY pair of date columns (dormancy length,
                                       # reactivation -> transaction, ...), not just tran minus each date
    max_date_gaps: int = 60            # cap on the number of pairwise gaps built
    id_cols: List[str] = field(default_factory=list)   # e.g. ["tran_id"] - never modelled
    account_col: Optional[str] = None       # account number column (never modelled); used to remove goods rows of
                                            # fraud accounts, for account-grouped CV and account counts per cluster
    normalize_account: bool = True          # strip spaces and leading zeros before matching account numbers
    dedup_cols: List[str] = field(default_factory=list)  # duplicate key after sampling; empty = all raw columns
    group_cv_by_account: bool = True        # keep each account in a single CV fold
    max_top_account_share: float = 0.2      # cluster where one account supplies more rows than this is not an MO
    demographic_cols: List[str] = field(default_factory=list)  # held out of modelling, used only for profiling
    exclude_cols: List[str] = field(default_factory=list)      # anything label-like / known leakage
    force_keep_cols: List[str] = field(default_factory=list)   # raw columns to keep even if a filter would drop them

    # ------------------------------------------------------------------ sampling
    n_goods_sample: int = 1_000_000
    n_screen_goods: int = 150_000      # smaller goods sample used only for column screening
    seed: int = 42

    # ------------------------------------------------------------------ column screening
    max_null_pct: float = 0.95         # drop if null share exceeds this in BOTH goods and fraud
    null_flag_min_diff: float = 0.05   # add an is_null indicator if fraud/goods null rates differ by this much
    leak_null_diff: float = 0.90       # null rate differs this much between the two files -> likely extraction artefact
    leak_auc: float = 0.995            # single-feature AUC this extreme -> likely leakage, excluded and reported
    near_const_share: float = 0.9995   # most common value share above this ...
    near_const_min_auc_gap: float = 0.01  # ... and |AUC-0.5| below this -> dropped
    drop_class_asymmetric_features: bool = True   # built feature null-rate differs by leak_null_diff -> drop
    corr_threshold: float = 0.95       # |Spearman| above this -> keep only the more discriminative feature
    corr_sample_rows: int = 60_000
    low_card_max: int = 50             # string columns with <= this many levels -> categorical, else frequency-encoded
    max_model_features: int = 400      # cap on features carried into the 1M-row model

    # ------------------------------------------------------------------ modelling
    n_folds: int = 4
    alert_rate: float = 0.01           # goods alert rate that defines "caught" vs "missed" fraud
    # How "missed" fraud is defined (the pool where an undiscovered MO should live):
    #   "existing_score" - your production score/rule column (score_col, higher = riskier). Best if you have it.
    #   "temporal"       - model trained on months before temporal_cutoff scores later months; later fraud it
    #                      misses = patterns the past didn't contain. Default.
    #   "oof"            - out-of-fold on all data. Only finds fraud that looks like goods on these features;
    #                      a GBM trained on the same period will usually LEARN any real MO, so use with care.
    missed_definition: str = "temporal"
    score_col: Optional[str] = None
    temporal_cutoff: Optional[str] = None   # "YYYY-MM"; None -> month where ~70% of fraud has occurred
    hgb_params: dict = field(default_factory=lambda: dict(
        learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=100, l2_regularization=5.0, max_bins=255,
    ))
    hgb_max_features: float = 0.5      # feature subsampling per split (used if your scikit-learn supports it)
    max_iter: int = 1000               # max boosting rounds
    n_iter_no_change: int = 50         # early-stopping patience

    # ------------------------------------------------------------------ clustering
    n_shap_features: int = 50          # top features (by mean |SHAP|) used as clustering space
    umap_dims: int = 10
    umap_neighbors: int = 30
    min_cluster_frac: float = 0.005    # HDBSCAN min_cluster_size as a share of rows ...
    min_cluster_abs: int = 30          # ... but never below this
    hdbscan_min_samples: int = 10
    family_cos_dist: float = 0.2       # merge HDBSCAN clusters whose centroids are this close (cosine); 0 = off
    max_family_frac: float = 1.0       # 1.0 = merge purely by direction (recommended); < 1 stops a family
                                       # from exceeding that share of rows, at the cost of splitting one MO
    subcluster_min_frac: float = 0.25  # a family holding more than this share of rows is re-clustered internally
    max_noise_frac: float = 0.4        # above this share of noise, HDBSCAN is retried with looser settings
    subcluster_max_depth: int = 2      # how many times that re-clustering may recurse
    n_stability_runs: int = 3          # re-runs with different seeds for the stability check
    impute: str = "median"             # "median" or "zero" - only used by the goods-quantile representation

    # ------------------------------------------------------------------ subgroup peeling
    n_item_features: int = 40          # top features turned into rule conditions
    item_quantiles: tuple = (0.01, 0.05, 0.25, 0.75, 0.95, 0.99)
    max_rule_depth: int = 3
    beam_width: int = 30
    min_support: int = 30              # min fraud rows a segment must cover
    quality_a: float = 0.5             # score = n_covered**a * (precision - base_rate)
    n_peels: int = 12
    signal_top_n: int = 12             # conditions reported per cluster in the signal-rate table
    signal_min_fraud_share: float = 0.15   # a condition must cover this share of the cluster to be reported
    min_segment_lift: float = 3.0      # stop peeling once the best remaining segment is this weak

    # ------------------------------------------------------------------ MO verdict thresholds
    verdict_min_rule_recall: float = 0.5
    verdict_min_lift: float = 10.0
    verdict_min_stability: float = 0.5
    emerging_ratio: float = 2.0        # share of fraud in last third of months / first third

    # goods rows in the full population (filled automatically from the scan)
    n_goods_population: Optional[int] = None


# ==================================================================================================
# DATA_PREP
# ==================================================================================================
DATE_FORMATS = ["%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y", "%Y%m%d", "%d%b%Y", "%d-%b-%Y", "%Y/%m/%d"]


# ----------------------------------------------------------------------------- io helpers
def _collect(lf: pl.LazyFrame) -> pl.DataFrame:
    try:
        return lf.collect(engine="streaming")
    except Exception:
        return lf.collect()


def scan_goods(cfg) -> pl.LazyFrame:
    files = sorted(glob.glob(cfg.goods_glob))
    if not files:
        raise FileNotFoundError(f"No goods files match {cfg.goods_glob}")
    lf = pl.concat([pl.scan_parquet(f) for f in files], how="diagonal_relaxed")
    return lf.rename(cfg.rename_goods) if cfg.rename_goods else lf


def scan_fraud(cfg) -> pl.LazyFrame:
    lf = (pl.scan_csv(cfg.fraud_path, infer_schema_length=50_000) if cfg.fraud_path.lower().endswith(".csv")
          else pl.scan_parquet(cfg.fraud_path))
    return lf.rename(cfg.rename_fraud) if cfg.rename_fraud else lf


def hash_sample(lf: pl.LazyFrame, n_total: int, n_target: int, seed: int) -> pl.LazyFrame:
    """Row-hash sampling inside the lazy plan, so the full goods table is never materialised."""
    frac = min(1.0, n_target * 1.03 / max(n_total, 1))
    m = 10_000_000
    return (lf.with_row_index("__rid")
              .filter((pl.col("__rid").hash(seed) % m) < int(frac * m))
              .drop("__rid"))


def _trim(df: pl.DataFrame, n: int, seed: int) -> pl.DataFrame:
    return df.sample(n, seed=seed) if df.height > n else df


# ----------------------------------------------------------------------------- dtype helpers
def _is_numeric(dt) -> bool:
    return dt.is_numeric() or dt == pl.Boolean


def _is_date(dt) -> bool:
    return dt in (pl.Date,) or isinstance(dt, pl.Datetime) or dt == pl.Datetime


def _is_string(dt) -> bool:
    return dt in (pl.String, pl.Categorical) or isinstance(dt, (pl.Categorical, pl.Enum))


def date_expr(col: str, dtype_str: str) -> pl.Expr:
    c = pl.col(col)
    if dtype_str.startswith("Datetime"):
        return c.dt.date()
    if dtype_str == "Date":
        return c
    if dtype_str.startswith("Int") or dtype_str.startswith("UInt"):
        return c.cast(pl.String).str.to_date("%Y%m%d", strict=False)
    s = c.cast(pl.String).str.strip_chars()
    return pl.coalesce([s.str.to_date(f, strict=False) for f in DATE_FORMATS])


def harmonise_fraud(lf_fraud: pl.LazyFrame, goods_schema: dict, cols: list) -> pl.LazyFrame:
    """Cast fraud columns to the goods dtype where they disagree (common with CSV input)."""
    fs = lf_fraud.collect_schema()
    exprs = []
    for c in cols:
        g, f = goods_schema[c], fs[c]
        if g == f:
            exprs.append(pl.col(c))
        elif _is_numeric(g):
            exprs.append(pl.col(c).cast(pl.Float64, strict=False))
        elif _is_date(g):
            exprs.append(date_expr(c, str(f)).alias(c))
        else:
            exprs.append(pl.col(c).cast(pl.String))
    return lf_fraud.select(exprs)


# ----------------------------------------------------------------------------- feature spec -> expressions
def gap_name(later: str, earlier: str) -> str:
    return f"days__{later}_minus_{earlier}"


def date_pairs(tran, others, cfg):
    """(later, earlier) date pairs. tran minus each other date first, then the other dates among themselves
    (dormancy_start -> reactivation length, reactivation -> first credit, ...)."""
    pairs = [(tran, o) for o in others]
    if cfg.all_date_gaps:
        pairs += [(others[j], others[i]) for i in range(len(others)) for j in range(i + 1, len(others))]
    return pairs[:cfg.max_date_gaps]


def feature_exprs(spec: dict):
    """Return (list of polars expressions, list of feature names, list of categorical feature names)."""
    exprs, names, cats = [], [], []
    for c in spec["numeric"]:
        exprs.append(pl.col(c).cast(pl.Float32).alias(c)); names.append(c)
    for c in spec["numeric_str"]:
        exprs.append(pl.col(c).cast(pl.String).str.strip_chars().cast(pl.Float64, strict=False)
                     .cast(pl.Float32).alias(c)); names.append(c)
    for c, mapping in spec["lowcard"].items():
        n = f"cat__{c}"
        e = (pl.when(pl.col(c).is_null()).then(None)
             .otherwise(pl.col(c).cast(pl.String).replace_strict(mapping, default=float(len(mapping)),
                                                                   return_dtype=pl.Float32)))
        exprs.append(e.cast(pl.Float32).alias(n)); names.append(n); cats.append(n)
    for c, mapping in spec["highcard"].items():
        n = f"freq__{c}"
        e = (pl.when(pl.col(c).is_null()).then(None)
             .otherwise(pl.col(c).cast(pl.String).replace_strict(mapping, default=0.0, return_dtype=pl.Float32)))
        exprs.append(e.cast(pl.Float32).alias(n)); names.append(n)
    d = spec["dates"]
    if d.get("tran"):
        td = date_expr(d["tran"], d["dtypes"][d["tran"]])
        for later, earlier in d.get("gaps", []):
            n = gap_name(later, earlier)
            e_l = td if later == d["tran"] else date_expr(later, d["dtypes"][later])
            e_e = td if earlier == d["tran"] else date_expr(earlier, d["dtypes"][earlier])
            exprs.append((e_l - e_e).dt.total_days().cast(pl.Float32).alias(n))
            names.append(n)
        for c in d.get("null_flags", []):        # e.g. dormancy date missing = account never went dormant
            n = f"isnull__{c}"
            exprs.append(pl.col(c).is_null().cast(pl.Float32).alias(n)); names.append(n)
        if d.get("calendar", True):
            exprs.append(td.dt.weekday().cast(pl.Float32).alias("tran_weekday")); names.append("tran_weekday")
            exprs.append(td.dt.day().cast(pl.Float32).alias("tran_day_of_month")); names.append("tran_day_of_month")
    for c in spec["null_flags"]:
        n = f"isnull__{c}"
        exprs.append(pl.col(c).is_null().cast(pl.Float32).alias(n)); names.append(n)
    if spec.get("n_null_cols"):
        exprs.append(pl.sum_horizontal([pl.col(c).is_null() for c in spec["n_null_cols"]])
                     .cast(pl.Float32).alias("n_null_raw"))
        names.append("n_null_raw")
    return exprs, names, cats


def account_expr(cfg) -> pl.Expr:
    s = pl.col(cfg.account_col).cast(pl.String).str.strip_chars()
    if cfg.normalize_account:
        s = s.str.strip_chars_start("0")
        s = pl.when(s == "").then(pl.lit("0")).otherwise(s)
    return s.alias("__account")


def meta_exprs(cfg, spec: dict):
    extra = [cfg.score_col] if cfg.score_col else []
    cols = list(dict.fromkeys(cfg.id_cols + cfg.demographic_cols + extra))
    exprs = [pl.col(c) for c in cols]
    if cfg.account_col:
        exprs.append(account_expr(cfg))
    d = spec["dates"]
    if d.get("tran"):
        exprs.append(date_expr(d["tran"], d["dtypes"][d["tran"]]).dt.strftime("%Y-%m").alias("__month"))
    else:
        exprs.append(pl.lit(None, dtype=pl.String).alias("__month"))
    return exprs


def source_col(feat: str) -> str:
    for p in ("cat__", "freq__", "isnull__"):
        if feat.startswith(p):
            return feat[len(p):]
    return feat


def prune_spec(spec: dict, keep: set) -> dict:
    out = {
        "numeric": [c for c in spec["numeric"] if c in keep],
        "numeric_str": [c for c in spec["numeric_str"] if c in keep],
        "lowcard": {c: m for c, m in spec["lowcard"].items() if f"cat__{c}" in keep},
        "highcard": {c: m for c, m in spec["highcard"].items() if f"freq__{c}" in keep},
        "null_flags": [c for c in spec["null_flags"] if f"isnull__{c}" in keep],
        "n_null_cols": spec["n_null_cols"] if "n_null_raw" in keep else [],
        "dates": dict(spec["dates"]),
    }
    d = out["dates"]
    if d.get("tran"):
        d["gaps"] = [g for g in d.get("gaps", []) if gap_name(*g) in keep]
        d["null_flags"] = [c for c in d.get("null_flags", []) if f"isnull__{c}" in keep]
        d["calendar"] = ("tran_weekday" in keep) or ("tran_day_of_month" in keep)
    return out


# ----------------------------------------------------------------------------- screening statistics
def univariate_auc(X: np.ndarray, y: np.ndarray, batch: int = 100) -> np.ndarray:
    """Rank-based AUC per column with median imputation."""
    n1, n0 = y.sum(), (1 - y).sum()
    out = np.full(X.shape[1], 0.5)
    for s in range(0, X.shape[1], batch):
        B = X[:, s:s + batch].astype(np.float64, copy=True)
        med = np.nanmedian(B, axis=0)
        med = np.where(np.isnan(med), 0.0, med)
        idx = np.where(np.isnan(B))
        B[idx] = np.take(med, idx[1])
        R = rankdata(B, axis=0)
        out[s:s + batch] = (R[y == 1].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)
    return out


def top_value_share(X: np.ndarray) -> np.ndarray:
    out = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        v = np.nan_to_num(X[:, j], nan=-9.87e30)
        _, cnt = np.unique(v, return_counts=True)
        out[j] = cnt.max() / len(v)
    return out


def correlation_filter(X: np.ndarray, relevance: np.ndarray, thr: float, rng, n_rows: int) -> np.ndarray:
    """Greedy Spearman filter: walk features by relevance, drop any too correlated with a kept one."""
    rows = rng.choice(X.shape[0], size=min(n_rows, X.shape[0]), replace=False)
    Z = X[rows].astype(np.float64)
    med = np.nanmedian(Z, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    idx = np.where(np.isnan(Z))
    Z[idx] = np.take(med, idx[1])
    Z = rankdata(Z, axis=0).astype(np.float32)
    Z -= Z.mean(0)
    sd = Z.std(0)
    sd[sd == 0] = 1.0
    Z /= sd
    C = np.abs((Z.T @ Z) / Z.shape[0])
    order = np.argsort(-relevance)
    kept = []
    for j in order:
        if not kept or C[j, kept].max() < thr:
            kept.append(j)
    mask = np.zeros(X.shape[1], bool)
    mask[kept] = True
    return mask


# ----------------------------------------------------------------------------- main entry
def prepare(cfg, log=print):
    cache = os.path.join(cfg.out_dir, "cache")
    os.makedirs(cache, exist_ok=True)
    if os.path.exists(os.path.join(cache, "X.npy")):
        log("[prep] loading cached matrices")
        X = np.load(os.path.join(cache, "X.npy"), mmap_mode="r")
        y = np.load(os.path.join(cache, "y.npy"))
        meta = pl.read_parquet(os.path.join(cache, "meta.parquet"))
        info = json.load(open(os.path.join(cache, "features.json")))
        cfg.n_goods_population = info["n_goods_population"]
        return np.asarray(X), y, meta, info

    rng = np.random.default_rng(cfg.seed)
    lf_g, lf_f = scan_goods(cfg), scan_fraud(cfg)
    gs, fs = lf_g.collect_schema(), lf_f.collect_schema()
    common = [c for c in gs.names() if c in fs]
    only_g = [c for c in gs.names() if c not in fs]
    only_f = [c for c in fs.names() if c not in gs]
    if only_g or only_f:
        log(f"[prep] WARNING {len(only_g)} cols only in goods, {len(only_f)} only in fraud - ignored")
    lf_f = harmonise_fraud(lf_f, dict(gs), common)
    lf_g = lf_g.select(common)

    n_goods = _collect(lf_g.select(pl.len())).item()
    n_fraud = _collect(lf_f.select(pl.len())).item()
    cfg.n_goods_population = n_goods
    log(f"[prep] goods population {n_goods:,} rows | fraud {n_fraud:,} rows | {len(common)} common columns")

    # ---------------- screening sample with every column
    scr_g = _trim(_collect(hash_sample(lf_g, n_goods, cfg.n_screen_goods, cfg.seed + 1)), cfg.n_screen_goods, cfg.seed)
    scr_f = _collect(lf_f)
    key_cols = cfg.dedup_cols or common
    scr_f = scr_f.unique(subset=key_cols, keep="first", maintain_order=True)
    scr_g = scr_g.unique(subset=key_cols, keep="first", maintain_order=True)
    if cfg.account_col:
        fa = scr_f.select(account_expr(cfg)).to_series().drop_nulls().unique()
        scr_g = scr_g.filter(~scr_g.select(account_expr(cfg)).to_series().is_in(fa).fill_null(False))
    scr = pl.concat([scr_g.with_columns(pl.lit(0).alias("__is_fraud")),
                     scr_f.with_columns(pl.lit(1).alias("__is_fraud"))], how="diagonal_relaxed")
    del scr_g, scr_f
    schema = dict(scr.schema)

    reserved = set(cfg.id_cols) | set(cfg.demographic_cols) | set(cfg.exclude_cols) | {"__is_fraud"}
    if cfg.score_col:
        reserved.add(cfg.score_col)
    if cfg.account_col:
        if cfg.account_col not in common:
            raise KeyError(f"account_col '{cfg.account_col}' is not in both files")
        reserved.add(cfg.account_col)
    date_cols = [c for c in common if c not in reserved and _is_date(schema[c])]
    for c in [cfg.tran_date_col] + list(cfg.other_date_cols):
        if c in common and c not in date_cols:
            date_cols.append(c)
    cand = [c for c in common if c not in reserved and c not in date_cols]

    # null rates by class
    nr = scr.group_by("__is_fraud").agg([pl.col(c).is_null().mean().alias(c) for c in cand]).sort("__is_fraud")
    null_g, null_f = nr.row(0), nr.row(1)
    null_g = dict(zip(nr.columns, null_g)); null_f = dict(zip(nr.columns, null_f))

    report = {"dropped_null": [], "leak_null": [], "leak_auc": [], "near_constant": [], "correlated": 0}
    keep_raw, null_flags = [], []
    fk = set(cfg.force_keep_cols)
    for c in cand:
        ng, nf = null_g[c], null_f[c]
        if abs(ng - nf) >= cfg.leak_null_diff and c not in fk:
            report["leak_null"].append((c, round(ng, 3), round(nf, 3)))
            continue
        if ng >= cfg.max_null_pct and nf >= cfg.max_null_pct and c not in fk:
            report["dropped_null"].append(c)
            continue
        keep_raw.append(c)
        if 0.01 <= max(ng, nf) <= 0.99 and abs(ng - nf) >= cfg.null_flag_min_diff:
            null_flags.append(c)

    # type detection
    numeric, numeric_str, str_cols = [], [], []
    for c in keep_raw:
        dt = schema[c]
        if _is_numeric(dt):
            numeric.append(c)
        elif _is_string(dt):
            str_cols.append(c)
    if str_cols:
        st = scr.select(
            [pl.col(c).cast(pl.String).is_not_null().sum().alias(f"nn__{c}") for c in str_cols] +
            [pl.col(c).cast(pl.String).str.strip_chars().cast(pl.Float64, strict=False).is_not_null().sum()
             .alias(f"np__{c}") for c in str_cols] +
            [pl.col(c).n_unique().alias(f"nu__{c}") for c in str_cols]).row(0, named=True)
    lowcard, highcard = {}, {}
    goods_mask = pl.col("__is_fraud") == 0
    for c in str_cols:
        nn, npar, nu = st[f"nn__{c}"], st[f"np__{c}"], st[f"nu__{c}"]
        if nn > 0 and npar >= 0.95 * nn:
            numeric_str.append(c)
        elif nu <= cfg.low_card_max:
            levels = scr.select(pl.col(c).cast(pl.String).drop_nulls().unique().sort()).to_series().to_list()
            lowcard[c] = {lv: float(i) for i, lv in enumerate(levels)}
        else:
            vc = scr.filter(goods_mask).select(pl.col(c).cast(pl.String).drop_nulls().value_counts(normalize=True))
            vc = vc.unnest(vc.columns[0])
            highcard[c] = dict(zip(vc[c].to_list(), [float(v) for v in vc[vc.columns[1]].to_list()]))

    tran = cfg.tran_date_col if cfg.tran_date_col in date_cols else (date_cols[0] if date_cols else None)
    others = [d for d in date_cols if d != tran]
    if tran:
        dcov = scr.select([pl.col("__is_fraud")] + [date_expr(c, str(schema[c])).is_not_null().alias(c)
                                                    for c in date_cols]) \
                  .group_by("__is_fraud").mean().sort("__is_fraud")
        log(f"[prep] date coverage by class (0=goods, 1=fraud):\n{dcov}")
        report["date_coverage"] = dcov.to_dicts()
    spec = {
        "numeric": numeric, "numeric_str": numeric_str, "lowcard": lowcard, "highcard": highcard,
        "null_flags": null_flags, "n_null_cols": cand,
        "dates": {"tran": tran, "others": others, "gaps": date_pairs(tran, others, cfg) if tran else [],
                  "null_flags": others, "dtypes": {d: str(schema[d]) for d in date_cols}, "calendar": True},
    }

    # date-range sanity check (leakage between the two files)
    if tran:
        mm = scr.select(pl.col("__is_fraud"), date_expr(tran, spec["dates"]["dtypes"][tran]).alias("__d")) \
                .group_by("__is_fraud").agg(pl.col("__d").min().alias("min"), pl.col("__d").max().alias("max"),
                                            pl.col("__d").is_null().mean().alias("null_share")).sort("__is_fraud")
        log(f"[prep] {tran} range by class (0=goods,1=fraud):\n{mm}")
        report["date_ranges"] = mm.with_columns(pl.col("min", "max").cast(pl.String)).to_dicts()

    # ---------------- build screening features
    exprs, names, cats = feature_exprs(spec)
    F = scr.lazy().select(exprs).collect()
    ys = scr["__is_fraud"].to_numpy().astype(np.int8)
    del scr
    Xs = F.to_numpy().astype(np.float32)
    del F
    names = np.array(names)
    is_cat = np.isin(names, cats)
    log(f"[prep] screening matrix {Xs.shape} ({is_cat.sum()} categorical)")

    nan_g = np.isnan(Xs[ys == 0]).mean(0)
    nan_f = np.isnan(Xs[ys == 1]).mean(0)
    asym = (np.abs(nan_g - nan_f) >= cfg.leak_null_diff) if cfg.drop_class_asymmetric_features \
        else np.zeros(Xs.shape[1], bool)
    auc = univariate_auc(Xs, ys)
    gap = np.abs(auc - 0.5)
    share = top_value_share(Xs)
    fk_feat = np.array([source_col(n) in fk for n in names])
    leak = (((auc >= cfg.leak_auc) | (auc <= 1 - cfg.leak_auc)) & ~is_cat | asym) & ~fk_feat
    const = (share >= cfg.near_const_share) & (gap < cfg.near_const_min_auc_gap) & ~fk_feat
    report["leak_auc"] = [(n, round(float(a), 4)) for n, a in zip(names[leak], auc[leak])]
    report["leak_class_asymmetric"] = [(n, round(float(g), 3), round(float(f), 3))
                                       for n, g, f in zip(names[asym], nan_g[asym], nan_f[asym])]
    if report["leak_class_asymmetric"]:
        log(f"[prep] {asym.sum()} built features are present in one class only (name, goods-null, fraud-null): "
            f"{report['leak_class_asymmetric'][:10]}")
    report["near_constant"] = names[const].tolist()
    keep = ~(leak | const)

    num_idx = np.where(keep & ~is_cat)[0]
    cmask = correlation_filter(Xs[:, num_idx], gap[num_idx], cfg.corr_threshold, rng, cfg.corr_sample_rows)
    drop_corr = num_idx[~cmask & ~fk_feat[num_idx]]
    keep[drop_corr] = False
    report["correlated"] = int(len(drop_corr))
    log(f"[prep] leak(AUC)={leak.sum()} near-const={const.sum()} correlated={len(drop_corr)} -> {keep.sum()} left")

    # quick gradient-boosting model to rank by gain
    kidx = np.where(keep)[0]
    kw = dict(learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=50, max_iter=200,
              early_stopping=False, random_state=cfg.seed)
    if "max_features" in signature(HistGradientBoostingClassifier.__init__).parameters:
        kw["max_features"] = 0.5
    booster = HistGradientBoostingClassifier(**kw).fit(Xs[:, kidx], ys)
    gain = hgb_gain(booster, len(kidx))
    order = np.argsort(-gain)
    chosen = [kidx[i] for i in order if gain[i] > 0][: cfg.max_model_features]
    chosen += [j for j in kidx if fk_feat[j] and j not in chosen]
    final = set(names[chosen].tolist())
    log(f"[prep] {len(final)} features carried forward (gain>0, cap {cfg.max_model_features})")
    del Xs, booster

    spec = prune_spec(spec, final)
    exprs, fnames, fcats = feature_exprs(spec)
    mexprs = meta_exprs(cfg, spec)

    # ---------------- full sample with final features only
    row_hash = pl.struct([pl.col(c) for c in key_cols]).hash(cfg.seed).alias("__row_hash")
    g = _collect(hash_sample(lf_g, n_goods, cfg.n_goods_sample, cfg.seed).select(mexprs + [row_hash] + exprs))
    g = _trim(g, cfg.n_goods_sample, cfg.seed)
    f = _collect(lf_f.select(mexprs + [row_hash] + exprs))
    dd = {"goods_sampled": g.height, "fraud_rows": f.height, "dedup_key": "all columns" if not cfg.dedup_cols
          else cfg.dedup_cols}
    g = g.unique(subset="__row_hash", keep="first", maintain_order=True)
    f = f.unique(subset="__row_hash", keep="first", maintain_order=True)
    dd["goods_duplicates_removed"] = dd["goods_sampled"] - g.height
    dd["fraud_duplicates_removed"] = dd["fraud_rows"] - f.height
    if cfg.account_col:
        fraud_acc = f["__account"].drop_nulls().unique()
        before = g.height
        g = g.filter(~pl.col("__account").is_in(fraud_acc).fill_null(False))
        dd["fraud_accounts"] = len(fraud_acc)
        dd["goods_rows_of_fraud_accounts_removed"] = before - g.height
        dd["goods_accounts_left"] = g["__account"].n_unique()
    else:
        log("[prep] WARNING account_col not set - goods rows of fraud accounts are NOT removed")
    dd["goods_final"], dd["fraud_final"] = g.height, f.height
    report["dedup"] = dd
    log(f"[prep] dedup/account filter: {dd}")
    g, f = g.drop("__row_hash"), f.drop("__row_hash")
    full = pl.concat([f.with_columns(pl.lit(1, pl.Int8).alias("__is_fraud")),
                      g.with_columns(pl.lit(0, pl.Int8).alias("__is_fraud"))], how="vertical_relaxed")
    del g, f
    X = full.select(fnames).to_numpy().astype(np.float32)
    y = full["__is_fraud"].to_numpy()
    meta = full.select([c for c in full.columns if c not in fnames])
    del full

    info = {"features": fnames, "categorical": fcats, "spec": spec, "screening_report": report,
            "n_goods_population": n_goods, "n_goods_sample": int((y == 0).sum()), "n_fraud": int(y.sum())}
    np.save(os.path.join(cache, "X.npy"), X)
    np.save(os.path.join(cache, "y.npy"), y)
    meta.write_parquet(os.path.join(cache, "meta.parquet"))
    json.dump(info, open(os.path.join(cache, "features.json"), "w"), indent=1, default=str)
    log(f"[prep] final matrix {X.shape}: {info['n_fraud']:,} fraud + {info['n_goods_sample']:,} goods")
    return X, y, meta, info


# ==================================================================================================
# MODELING
# ==================================================================================================
_HGB_INIT = signature(HistGradientBoostingClassifier.__init__).parameters
_HGB_XVAL = "X_val" in signature(HistGradientBoostingClassifier.fit).parameters


def fit_hgb(X, y, fit_rows, es_rows, cfg):
    """HistGradientBoosting with early stopping on an account-grouped validation split."""
    kw = dict(cfg.hgb_params, max_iter=cfg.max_iter, early_stopping=True, n_iter_no_change=cfg.n_iter_no_change,
              scoring="loss", random_state=cfg.seed)
    if "max_features" in _HGB_INIT:
        kw["max_features"] = cfg.hgb_max_features
    if _HGB_XVAL:        # newer scikit-learn: early stopping on our own grouped validation rows
        m = HistGradientBoostingClassifier(**kw)
        m.fit(X[fit_rows], y[fit_rows], X_val=X[es_rows], y_val=y[es_rows])
    else:                # older versions: internal random 10% validation split
        rows = np.sort(np.r_[fit_rows, es_rows])
        m = HistGradientBoostingClassifier(validation_fraction=0.1, **kw)
        m.fit(X[rows], y[rows])
    return m


def _predict_chunks(model, X, idx, chunk=200_000):
    return np.concatenate([model.predict_proba(X[idx[s:s + chunk]])[:, 1] for s in range(0, len(idx), chunk)])


def hgb_gain(model, p):
    """Total split gain per feature (like LightGBM 'gain' importance)."""
    g = np.zeros(p)
    for it in model._predictors:
        nd = it[0].nodes
        split = nd["is_leaf"] == 0
        g += np.bincount(nd["feature_idx"][split], weights=nd["gain"][split], minlength=p)
    return g


def _node_expectations(nodes):
    """Expected (training-count weighted) output of each node's subtree; leaves keep their value."""
    E = nodes["value"].astype(np.float64).copy()
    cnt = nodes["count"].astype(np.float64)
    for i in range(len(nodes) - 1, -1, -1):
        if not nodes["is_leaf"][i]:
            l, r = nodes["left"][i], nodes["right"][i]
            E[i] = (cnt[l] * E[l] + cnt[r] * E[r]) / max(cnt[l] + cnt[r], 1.0)
    return E


def hgb_contrib(model, Xs):
    """
    Per-row, per-feature contributions in log-odds (path attribution, a.k.a. Saabas):
    walking each tree, the change in expected output at every split is credited to the split feature.
    Rows sum exactly to the model's log-odds (checked in cv_model). Replaces LightGBM's pred_contrib;
    an approximation of SHAP that works well for grouping fraud rows by *why* they are flagged.
    """
    Xs = np.asarray(Xs, dtype=np.float64)
    n, p = Xs.shape
    out = np.zeros((n, p))
    bias = float(np.ravel(model._baseline_prediction)[0])
    rows_all = np.arange(n)
    for it in model._predictors:
        nd = it[0].nodes
        E = _node_expectations(nd)
        bias += E[0]
        leaf = nd["is_leaf"].astype(bool)
        cur = np.zeros(n, dtype=np.int64)
        act = rows_all[~leaf[cur]]
        while len(act):
            node = cur[act]
            f = nd["feature_idx"][node].astype(np.int64)
            x = Xs[act, f]
            left = np.where(np.isnan(x), nd["missing_go_to_left"][node].astype(bool), x <= nd["num_threshold"][node])
            child = np.where(left, nd["left"][node], nd["right"][node]).astype(np.int64)
            out[act, f] += E[child] - E[node]
            cur[act] = child
            act = act[~leaf[child]]
    return out, bias


def group_folds(rows, y, groups, n_splits, seed):
    """Fold id per row: every group (account) in exactly one fold, groups containing fraud spread evenly."""
    g = groups[rows]
    ug, inv = np.unique(g, return_inverse=True)
    has_pos = np.bincount(inv, weights=(y[rows] == 1)) > 0
    rng = np.random.default_rng(seed)
    fold_of_group = np.empty(len(ug), int)
    for flag in (True, False):
        idx = np.where(has_pos == flag)[0]
        rng.shuffle(idx)
        fold_of_group[idx] = np.arange(len(idx)) % n_splits
    return fold_of_group[inv]


def _es_split(tr_rows, y, groups, seed):
    f = group_folds(tr_rows, y, groups, 10, seed)
    return tr_rows[f != 0], tr_rows[f == 0]


def cv_model(X, y, rows, shap_rows, cfg, tag, groups, log=print):
    """
    rows      : indices of X used in this model (goods + the fraud rows of interest)
    shap_rows : subset of rows (the fraud of interest) that get OOF SHAP values
    returns dict(oof=scores aligned to `rows`, shap=(len(shap_rows), p), gain=(p,), metrics)
    """
    cache = os.path.join(cfg.out_dir, "cache", f"model_{tag}.npz")
    if os.path.exists(cache):
        log(f"[model {tag}] loading cached results")
        z = np.load(cache, allow_pickle=True)
        return {k: z[k] for k in z.files}

    p = X.shape[1]

    rows = np.asarray(rows)
    pos_in_shap = {r: i for i, r in enumerate(shap_rows)}
    oof = np.zeros(len(rows))
    shap = np.zeros((len(shap_rows), p), np.float32)
    gain = np.zeros(p)
    folds = group_folds(rows, y, groups, cfg.n_folds, cfg.seed)
    for k in range(cfg.n_folds):
        tr, va = np.where(folds != k)[0], np.where(folds == k)[0]
        tr_rows, va_rows = rows[tr], rows[va]
        fit_rows, es_rows = _es_split(tr_rows, y, groups, cfg.seed + k)
        model = fit_hgb(X, y, np.sort(fit_rows), np.sort(es_rows), cfg)
        it = model.n_iter_
        oof[va] = _predict_chunks(model, X, va_rows)
        s_rows = np.array([r for r in va_rows if r in pos_in_shap])
        if len(s_rows):
            contrib, bias = hgb_contrib(model, X[s_rows])
            if k == 0:   # sanity check: contributions must add up to the model's log-odds
                chk = s_rows[:200]
                err = np.abs(contrib[:200].sum(1) + bias - model.decision_function(X[chk])).max()
                log(f"[model {tag}] contribution check: max |sum(contrib)+bias - logit| = {err:.2e}")
            shap[[pos_in_shap[r] for r in s_rows]] = contrib.astype(np.float32)
        gain += hgb_gain(model, p)
        log(f"[model {tag}] fold {k + 1}/{cfg.n_folds}: {it} trees, "
            f"fold PR-AUC {average_precision_score(y[va_rows], oof[va]):.4f}")

    yr = y[rows]
    thr = np.quantile(oof[yr == 0], 1 - cfg.alert_rate)
    metrics = np.array([roc_auc_score(yr, oof), average_precision_score(yr, oof),
                        float((oof[yr == 1] >= thr).mean()), float(thr)])
    log(f"[model {tag}] OOF ROC-AUC {metrics[0]:.4f} | PR-AUC {metrics[1]:.4f} | "
        f"fraud recall at {cfg.alert_rate:.1%} goods alert rate {metrics[2]:.3f}")
    res = dict(oof=oof, rows=rows, shap=shap, shap_rows=np.asarray(shap_rows), gain=gain, metrics=metrics)
    np.savez(cache, **res)
    return res


def temporal_missed(X, y, months, cfg, groups, log=print):
    """Train on months < cutoff, score months >= cutoff; later fraud scored below the alert threshold = missed."""
    fm = np.sort(months[(y == 1) & (months != "")])
    if len(fm) == 0:
        raise ValueError("temporal mode needs a parseable transaction date on the fraud rows - "
                         "check tran_date_col, or use missed_definition='oof'")
    cutoff = cfg.temporal_cutoff or fm[int(0.7 * (len(fm) - 1))]
    cache = os.path.join(cfg.out_dir, "cache", f"temporal_{cutoff}.npz")
    pre = np.where(months < cutoff)[0]
    post = np.where(months >= cutoff)[0]
    if os.path.exists(cache):
        z = np.load(cache)
        score = z["score"]
    else:
        if y[pre].sum() < 20 or (y[post] == 0).sum() == 0:
            raise ValueError(f"temporal cutoff {cutoff} leaves too little data on one side - set temporal_cutoff")
        fit_rows, es_rows = _es_split(pre, y, groups, cfg.seed)
        model = fit_hgb(X, y, np.sort(fit_rows), np.sort(es_rows), cfg)
        score = _predict_chunks(model, X, post)
        np.savez(cache, score=score)
    yp = y[post]
    thr = np.quantile(score[yp == 0], 1 - cfg.alert_rate)
    missed = post[(yp == 1) & (score < thr)]
    log(f"[temporal] cutoff {cutoff}: trained on {len(pre):,} earlier rows; "
        f"later fraud recall {(score[yp == 1] >= thr).mean():.3f} -> {len(missed):,} missed later fraud")
    return missed, cutoff


def run_models(X, y, info, meta, cfg, log=print):
    if cfg.group_cv_by_account and "__account" in meta.columns:
        codes = meta["__account"].cast(pl.Categorical).to_physical().cast(pl.Int64).to_numpy()
        codes = np.asarray(codes, dtype=float)
        nulls = np.isnan(codes)
        groups = np.where(nulls, np.nanmax(np.r_[codes[~nulls], 0]) + 1 + np.arange(len(codes)), codes).astype(np.int64)
        log(f"[models] account-grouped CV: {len(np.unique(groups)):,} groups")
    else:
        groups = np.arange(len(y))
    all_rows = np.arange(len(y))
    fraud_rows = np.where(y == 1)[0]
    goods_rows = np.where(y == 0)[0]

    A = cv_model(X, y, all_rows, fraud_rows, cfg, "A_fraud_vs_goods", groups, log)
    mode = cfg.missed_definition
    if mode == "existing_score":
        s = meta[cfg.score_col].cast(float).fill_null(-np.inf).to_numpy()
        thr = np.quantile(s[goods_rows], 1 - cfg.alert_rate)
        missed = fraud_rows[s[fraud_rows] < thr]
        desc = f"below the {cfg.score_col} alert threshold"
        key = f"score_{cfg.score_col}"
    elif mode == "temporal":
        months = meta["__month"].fill_null("").to_numpy().astype(str)
        missed, cutoff = temporal_missed(X, y, months, cfg, groups, log)
        desc = f"from {cutoff} onward, missed by a model trained on earlier months"
        key = f"temporal_{cutoff}"
    else:
        missed = fraud_rows[A["oof"][fraud_rows] < A["metrics"][3]]
        desc = "missed out-of-fold by stage A"
        key = "oof"
    log(f"[models] missed-fraud pool ({mode}): {len(missed):,} of {len(fraud_rows):,} fraud rows - {desc}")

    B_rows = np.concatenate([missed, goods_rows])
    B = cv_model(X, y, B_rows, missed, cfg, f"B_missed_vs_goods_{key}", groups, log) \
        if len(missed) >= 3 * cfg.min_cluster_abs else None
    return A, B, missed, desc


# ==================================================================================================
# CLUSTERING
# ==================================================================================================
def shap_space(shap, feat_names, k):
    imp = np.abs(shap).mean(0)
    top = np.argsort(-imp)[:k]
    return shap[:, top], [feat_names[i] for i in top], top


def goods_quantile_space(X, goods_rows, target_rows, feat_idx, feat_names, cfg):
    G = X[np.ix_(goods_rows, feat_idx)].astype(np.float64)
    T = X[np.ix_(target_rows, feat_idx)].astype(np.float64)
    if cfg.impute == "zero":
        fill = np.zeros(len(feat_idx))
    else:
        fill = np.nanmedian(G, axis=0)
        fill = np.where(np.isnan(fill), 0.0, fill)
    for M in (G, T):
        idx = np.where(np.isnan(M))
        M[idx] = np.take(fill, idx[1])
    qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal",
                             subsample=200_000, random_state=cfg.seed).fit(G)
    Z = np.clip(qt.transform(T), -5, 5).astype(np.float32)
    return Z, [feat_names[i] for i in feat_idx]


def _reduce(Z, n_comp, cfg, seed, min_dist):
    n_comp = min(n_comp, Z.shape[1])
    if umap is not None:
        return umap.UMAP(n_components=n_comp, n_neighbors=min(cfg.umap_neighbors, Z.shape[0] - 1),
                         min_dist=min_dist, random_state=seed).fit_transform(Z)
    return PCA(n_components=n_comp, random_state=seed).fit_transform(Z)


def _embed_cluster(Z, cfg, seed):
    """UMAP/PCA then HDBSCAN. If most rows come back as noise the space is diffuse, so retry once with a
    smaller minimum cluster size and keep whichever labelling leaves less noise."""
    n = Z.shape[0]
    E = _reduce(Z, cfg.umap_dims, cfg, seed, 0.0)
    mcs = max(cfg.min_cluster_abs, int(cfg.min_cluster_frac * n))
    labels = HDBSCAN(min_cluster_size=mcs, min_samples=cfg.hdbscan_min_samples).fit_predict(E)
    if np.mean(labels == -1) > cfg.max_noise_frac:
        alt = HDBSCAN(min_cluster_size=max(10, mcs // 2), min_samples=2,
                      cluster_selection_method="leaf").fit_predict(E)
        if np.mean(alt == -1) < np.mean(labels == -1):
            labels = alt
    return labels


def _stability(base, others):
    """For each base cluster: mean over re-runs of the best Jaccard match."""
    out = {}
    for c in sorted(set(base) - {-1}):
        a = base == c
        scores = []
        for o in others:
            best = 0.0
            for d in set(o) - {-1}:
                b = o == d
                best = max(best, (a & b).sum() / (a | b).sum())
            scores.append(best)
        out[c] = float(np.mean(scores)) if scores else np.nan
    return out


def _relabel_by_size(labels):
    sizes = {c: (labels == c).sum() for c in set(labels) - {-1}}
    order = {c: i for i, c in enumerate(sorted(sizes, key=lambda c: -sizes[c]))}
    return np.array([order.get(x, -1) for x in labels])


def merge_families(Z, labels, cfg, cap_frac=None):
    """
    HDBSCAN splits one MO by intensity, so clusters whose centroids point the same way are merged.
    A merge that would swallow more than max_family_frac of the rows is undone, which is what stops a
    single family from becoming a dumping ground for most of the fraud.
    """
    cs = sorted(set(labels) - {-1})
    if len(cs) < 2 or not cfg.family_cos_dist:
        return _relabel_by_size(labels)
    C = np.array([Z[labels == c].mean(0) for c in cs])
    fam = dict(zip(cs, fcluster(linkage(C, "average", metric="cosine"), cfg.family_cos_dist, "distance")))
    cap = (cap_frac if cap_frac is not None else cfg.max_family_frac) * len(labels)
    sizes = {}
    for c in cs:
        sizes[fam[c]] = sizes.get(fam[c], 0) + (labels == c).sum()
    too_big = {f for f, n in sizes.items() if n > cap}
    out = np.array([-1 if x == -1 else (f"f{fam[x]}" if fam[x] not in too_big else f"c{x}") for x in labels],
                   dtype=object)
    keys = {k: i for i, k in enumerate(sorted(set(out) - {-1}))}
    return _relabel_by_size(np.array([-1 if x == -1 else keys[x] for x in out]))


def _parts_distinct(Z, sub, parts, min_cos):
    """True when every pair of sub-parts points in a genuinely different direction."""
    C = np.array([Z[sub == p].mean(0) for p in parts])
    C = C / np.clip(np.linalg.norm(C, axis=1, keepdims=True), 1e-9, None)
    D = 1 - C @ C.T
    np.fill_diagonal(D, np.inf)
    return bool(D.min() >= min_cos)


def subcluster(Z, labels, cfg, log, depth=1):
    """Re-run UMAP+HDBSCAN inside any family that still holds too much of the fraud, so a dominant MO
    is broken into its sub-patterns instead of hiding them."""
    if depth > cfg.subcluster_max_depth or not cfg.subcluster_min_frac:
        return labels
    out = labels.copy()
    nxt = max(set(labels) - {-1}, default=-1) + 1
    for c in sorted(set(labels) - {-1}):
        idx = np.where(labels == c)[0]
        if len(idx) < cfg.subcluster_min_frac * len(labels) or len(idx) < 4 * cfg.min_cluster_abs:
            continue
        # no size cap inside a family: parts that point the same way must merge back, otherwise one MO
        # would be split into several clusters carrying the same rule
        sub = merge_families(Z[idx], _embed_cluster(Z[idx], cfg, cfg.seed + 7), cfg, cap_frac=1.01)
        parts = sorted(set(sub) - {-1})
        if len(parts) < 2 or not _parts_distinct(Z[idx], sub, parts, cfg.family_cos_dist):
            continue
        log(f"[cluster] family {c} holds {len(idx) / len(labels):.0%} of rows - split into {len(parts)} parts")
        for i, ppart in enumerate(parts):
            out[idx[sub == ppart]] = c if i == 0 else nxt + i - 1
        out[idx[sub == -1]] = c
        nxt += len(parts) - 1
        out = subcluster(Z, out, cfg, log, depth + 1) if depth < cfg.subcluster_max_depth else out
        break
    return out


def cluster(Z, cfg, log=print, tag=""):
    fine = _embed_cluster(Z, cfg, cfg.seed)
    labels = _relabel_by_size(subcluster(Z, merge_families(Z, fine, cfg), cfg, log))
    others = [_relabel_by_size(subcluster(Z, merge_families(Z, _embed_cluster(Z, cfg, cfg.seed + 100 * (i + 1)),
                                                            cfg), cfg, lambda *_: None))
              for i in range(cfg.n_stability_runs - 1)]
    stab = _stability(labels, others)
    ari = [adjusted_rand_score(labels, o) for o in others]
    if umap is None:
        log("[cluster] umap-learn not installed - using PCA (clusters will be coarser)")
    emb2 = _reduce(Z, 2, cfg, cfg.seed, 0.1)
    n_cl = len(set(labels) - {-1})
    log(f"[cluster {tag}] {len(set(fine) - {-1})} fine clusters -> {n_cl} MO families, "
        f"{np.mean(labels == -1):.1%} noise, ARI across seeds {np.round(ari, 3).tolist()}")
    return labels, stab, emb2, fine


# ==================================================================================================
# SUBGROUPS
# ==================================================================================================
_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)


def _fmt(v):
    return f"{v:.4g}"


def build_items(X, target_rows, goods_rows, feat_idx, feat_names, cat_levels, cfg):
    nf, ng = len(target_rows), len(goods_rows)
    nf_pad = (nf + 7) // 8 * 8
    items = []  # (feature_index, kind, value, text, packed_bits)

    def pack(mask_f, mask_g):
        m = np.zeros(nf_pad + ng, bool)
        m[:nf] = mask_f
        m[nf_pad:] = mask_g
        return np.packbits(m)

    for j in feat_idx:
        name = feat_names[j]
        vf, vg = X[target_rows, j], X[goods_rows, j]
        if name in cat_levels:
            inv = cat_levels[name]
            for code in np.unique(vf[~np.isnan(vf)]):
                mf = vf == code
                if mf.sum() >= cfg.min_support:
                    items.append((j, "eq", code, f"{name} == '{inv.get(code, 'OTHER')}'", pack(mf, vg == code)))
        else:
            qs = np.unique(np.nanquantile(vg, cfg.item_quantiles)) if np.isfinite(vg).any() else []
            for q in qs:
                for kind, mf, mg, txt in (("le", vf <= q, vg <= q, f"{name} <= {_fmt(q)}"),
                                          ("gt", vf > q, vg > q, f"{name} > {_fmt(q)}")):
                    if mf.sum() >= cfg.min_support:
                        items.append((j, kind, q, txt, pack(mf, mg)))
        nan_f = np.isnan(vf)
        if nan_f.sum() >= cfg.min_support and 0.01 < nan_f.mean() < 0.99:
            items.append((j, "null", None, f"{name} is null", pack(nan_f, np.isnan(vg))))
    return items, nf_pad // 8


def _counts(bits, fbytes, active):
    return int(_POP[bits[:fbytes] & active].sum()), int(_POP[bits[fbytes:]].sum())


def _compatible(rule_items, cand, items):
    j, kind = items[cand][0], items[cand][1]
    for r in rule_items:
        rj, rk = items[r][0], items[r][1]
        if r == cand or (rj == j and (rk == kind or "eq" in (rk, kind) or "null" in (rk, kind))):
            return False
    return True


def condition_masks(X, feat_idx, feat_names, cat_levels, goods_sub, quantiles):
    """Yield (text, fraud-side test, goods-side mask) for simple one-feature conditions."""
    out = []
    for j in feat_idx:
        name = feat_names[j]
        vg = X[goods_sub, j]
        if name in cat_levels:
            for code in np.unique(vg[~np.isnan(vg)]):
                lab = cat_levels[name].get(code, "OTHER")
                out.append((f"{name} == '{lab}'", (lambda v, c=code: v == c), vg == code))
        else:
            if not np.isfinite(vg).any():
                continue
            for q in np.unique(np.nanquantile(vg, quantiles)):
                out.append((f"{name} <= {q:.4g}", (lambda v, t=q: v <= t), vg <= q))
                out.append((f"{name} > {q:.4g}", (lambda v, t=q: v > t), vg > q))
        out.append((f"{name} is null", (lambda v: np.isnan(v)), np.isnan(vg)))
    return out


def signal_rates(X, c_rows, run_rows, goods_sub, goods_w, feat_idx, feat_names, cat_levels,
                 n_goods_pop, n_fraud_total, cfg, tag, cluster_id):
    """
    For every single condition: how many accounts in the cluster follow it (fraud %), how many goods
    follow it (good %), the hit rate (fraud share among all accounts following it) and the lift.
    """
    rows = []
    base = n_fraud_total / (n_fraud_total + n_goods_pop)
    for text, test, gmask in condition_masks(X, feat_idx, feat_names, cat_levels, goods_sub,
                                             cfg.item_quantiles):
        j = feat_names.index(text.split(" ")[0])
        in_c = int(np.sum(test(X[c_rows, j])))
        if in_c < max(cfg.min_support, cfg.signal_min_fraud_share * len(c_rows)):
            continue
        fr_all = int(np.sum(test(X[run_rows, j])))
        gd = float(gmask.sum()) * goods_w
        hit = fr_all / (fr_all + gd) if fr_all + gd > 0 else 0.0
        rows.append(dict(run=tag, cluster=cluster_id, condition=text,
                         fraud_in_cluster=in_c,
                         fraud_pct_of_cluster=round(100 * in_c / len(c_rows), 2),
                         fraud_pct_of_all_fraud=round(100 * fr_all / n_fraud_total, 3),
                         est_goods_matching=int(round(gd)),
                         good_pct_of_all_goods=round(100 * gd / n_goods_pop, 4),
                         hit_rate_pct=round(100 * hit, 4), lift=round(hit / base, 1)))
    rows.sort(key=lambda r: (-r["lift"], -r["fraud_pct_of_cluster"]))
    return rows[:cfg.signal_top_n]


def peel(X, target_rows, goods_rows, feat_idx, feat_names, cat_levels, goods_weight, cfg, log=print, tag=""):
    items, fbytes = build_items(X, target_rows, goods_rows, feat_idx, feat_names, cat_levels, cfg)
    log(f"[peel {tag}] {len(items)} candidate conditions on {len(feat_idx)} features")
    nf = len(target_rows)
    act = np.zeros(fbytes * 8, bool); act[:nf] = True
    active = np.packbits(act)
    N_w = len(goods_rows) * goods_weight
    base_global = nf / (nf + N_w)
    segments, member = [], np.full(nf, -1)

    for rnd in range(cfg.n_peels):
        P = int(_POP[active].sum())
        if P < cfg.min_support:
            break
        base = P / (P + N_w)

        def score(tp, fp):
            if tp < cfg.min_support:
                return -np.inf
            n = tp + fp * goods_weight
            return n ** cfg.quality_a * (tp / n - base)

        lvl = []
        for i, it in enumerate(items):
            tp, fp = _counts(it[4], fbytes, active)
            s = score(tp, fp)
            if np.isfinite(s):
                lvl.append((s, (i,), it[4]))
        if not lvl:
            break
        lvl.sort(key=lambda t: -t[0])
        beam = lvl[:cfg.beam_width]
        best = beam[0]
        usable = [t[1][0] for t in lvl]
        seen = {frozenset(t[1]) for t in beam}
        for _ in range(cfg.max_rule_depth - 1):
            new = []
            for s0, rule, bits in beam:
                for c in usable:
                    key = frozenset(rule + (c,))
                    if key in seen or not _compatible(rule, c, items):
                        continue
                    seen.add(key)
                    nb = bits & items[c][4]
                    tp, fp = _counts(nb, fbytes, active)
                    s = score(tp, fp)
                    if s > s0:
                        new.append((s, rule + (c,), nb))
            if not new:
                break
            beam = sorted(beam + new, key=lambda t: -t[0])[:cfg.beam_width]
            if beam[0][0] > best[0]:
                best = beam[0]

        s, rule, bits = best
        tp, fp = _counts(bits, fbytes, active)
        if tp / (tp + fp * goods_weight) / base_global < cfg.min_segment_lift:
            log(f"[peel {tag}] best remaining segment has lift < {cfg.min_segment_lift} - stopping")
            break
        tp_all, _ = _counts(bits, fbytes, np.packbits(np.r_[np.ones(nf, bool), np.zeros(fbytes * 8 - nf, bool)]))
        prec = tp_all / (tp_all + fp * goods_weight)
        covered = np.unpackbits(bits[:fbytes] & active)[:nf].astype(bool)
        member[covered & (member == -1)] = rnd
        segments.append(dict(
            segment=rnd, rule=" AND ".join(items[i][3] for i in rule),
            new_fraud_covered=tp, total_fraud_matching=tp_all,
            share_of_target_fraud=round(tp_all / nf, 4),
            est_goods_matching_population=int(round(fp * goods_weight)),
            precision_population=round(prec, 5), lift=round(prec / base_global, 1),
        ))
        log(f"[peel {tag}] #{rnd}: {segments[-1]['rule']} | new fraud {tp} | lift {segments[-1]['lift']}")
        active = active & ~bits[:fbytes]
    return segments, member


# ==================================================================================================
# PROFILING
# ==================================================================================================


def tree_rule(tree, leaf, feat_names, cat_levels):
    t = tree.tree_
    parent = {}
    for n in range(t.node_count):
        for ch, side in ((t.children_left[n], "L"), (t.children_right[n], "R")):
            if ch != -1:
                parent[ch] = (n, side)
    miss_left = getattr(t, "missing_go_to_left", None)
    bounds = {}
    node = leaf
    while node in parent:
        p, side = parent[node]
        f, thr = feat_names[t.feature[p]], t.threshold[p]
        b = bounds.setdefault(f, {"lo": -np.inf, "hi": np.inf, "null": None})
        if side == "L":
            b["hi"] = min(b["hi"], thr)
        else:
            b["lo"] = max(b["lo"], thr)
        if miss_left is not None:
            goes_here = bool(miss_left[p]) == (side == "L")
            b["null"] = goes_here if b["null"] is None else (b["null"] and goes_here)
        node = p
    parts = []
    for f, b in bounds.items():
        if f in cat_levels:
            lv = [v for c, v in cat_levels[f].items() if b["lo"] < c <= b["hi"]]
            txt = f"{f} in {{{', '.join(map(str, lv[:6]))}{', ...' if len(lv) > 6 else ''}}}"
        elif np.isfinite(b["lo"]) and np.isfinite(b["hi"]):
            txt = f"{_fmt(b['lo'])} < {f} <= {_fmt(b['hi'])}"
        elif np.isfinite(b["hi"]):
            txt = f"{f} <= {_fmt(b['hi'])}"
        else:
            txt = f"{f} > {_fmt(b['lo'])}"
        if b["null"]:
            txt = f"({txt} or null)"
        parts.append(txt)
    return " AND ".join(parts)


def describe_cluster(X, c_rows, run_rows, goods_sub, goods_w, feat_idx, feat_names, cat_levels, n_goods_pop, seed):
    Xc = X[np.ix_(c_rows, feat_idx)]
    Xg = X[np.ix_(goods_sub, feat_idx)]
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=max(10, int(0.02 * len(c_rows))),
                                  class_weight="balanced", random_state=seed)
    tree.fit(np.vstack([Xc, Xg]), np.r_[np.ones(len(c_rows)), np.zeros(len(goods_sub))])
    lc, lg = tree.apply(Xc), tree.apply(Xg)
    lr = tree.apply(X[np.ix_(run_rows, feat_idx)])
    base = len(run_rows) / (len(run_rows) + n_goods_pop)
    best = None
    for leaf in np.unique(lc):
        tp = (lc == leaf).sum()
        rec = tp / len(c_rows)
        if rec < 0.05:
            continue
        fr = (lr == leaf).sum()
        fp = (lg == leaf).sum() * goods_w
        prec = fr / (fr + fp) if fr + fp > 0 else 0.0
        if best is None or rec * prec > best[0]:
            best = (rec * prec, leaf, rec, prec, tp / max(fr, 1), fp)
    if best is None:
        return dict(rule="", rule_recall=0.0, rule_precision_pop=0.0, rule_lift=0.0, rule_purity=0.0,
                    rule_est_goods_pop=0, rule_fraud_matching=0)
    _, leaf, rec, prec, purity, fp = best
    fr = int((lr == leaf).sum())
    sub_names = [feat_names[j] for j in feat_idx]
    return dict(rule=tree_rule(tree, leaf, sub_names, cat_levels), rule_recall=round(rec, 3),
                rule_precision_pop=round(prec, 5), rule_lift=round(prec / base, 1),
                rule_purity=round(purity, 3), rule_est_goods_pop=int(round(fp)), rule_fraud_matching=fr)


def drivers(space, space_names, labels, c, X, c_rows, goods_sub, feat_names, cat_levels, k=6):
    m = space[labels == c].mean(0)
    top = np.argsort(-np.abs(m))[:k]
    name_to_idx = {n: i for i, n in enumerate(feat_names)}
    out = []
    for t in top:
        n = space_names[t]
        j = name_to_idx.get(n)
        if j is None:
            out.append(f"{n} ({m[t]:+.2f})")
            continue
        if n in cat_levels:
            vc, vg = X[c_rows, j], X[goods_sub, j]
            vals, cnt = np.unique(vc[~np.isnan(vc)], return_counts=True)
            if len(vals):
                top_code = vals[cnt.argmax()]
                out.append(f"{n} {m[t]:+.2f} [{cat_levels[n].get(top_code, 'OTHER')}: "
                           f"{cnt.max() / len(vc):.0%} vs goods {np.mean(vg == top_code):.0%}]")
                continue
        mc = np.nanmedian(X[c_rows, j]) if np.isfinite(X[c_rows, j]).any() else np.nan
        mg = np.nanmedian(X[goods_sub, j]) if np.isfinite(X[goods_sub, j]).any() else np.nan
        out.append(f"{n} {m[t]:+.2f} [median {_fmt(mc)} vs goods {_fmt(mg)}]")
    return "; ".join(out)


def monthly_share(months, labels):
    df = pl.DataFrame({"month": months, "cluster": labels})
    tot = df.group_by("month").len().rename({"len": "total"})
    return (df.group_by(["month", "cluster"]).len().join(tot, on="month")
              .with_columns((pl.col("len") / pl.col("total")).alias("share")).sort(["cluster", "month"]))


def emergence(ms, c, min_last=20):
    d = ms.filter(pl.col("cluster") == c).sort("month")
    months = sorted(ms["month"].drop_nulls().unique().to_list())
    if len(months) < 3:
        return np.nan, False
    k = max(1, len(months) // 3)
    first, last = set(months[:k]), set(months[-k:])
    s_first = d.filter(pl.col("month").is_in(first))["share"].sum() / k
    s_last = d.filter(pl.col("month").is_in(last))["share"].sum() / k
    n_last = d.filter(pl.col("month").is_in(last))["len"].sum()
    ratio = s_last / (s_first + 1e-3)
    return round(float(ratio), 2), bool(n_last >= min_last)


def demographics(meta_run, labels, c, demo_cols, k=3):
    out = []
    for col in demo_cols:
        s = meta_run[col]
        in_c = s.filter(pl.Series(labels == c))
        if s.dtype.is_numeric():
            out.append((0.0, f"{col} median {_fmt(in_c.median() or np.nan)} vs {_fmt(s.median() or np.nan)}"))
            continue
        allv = s.cast(pl.String).value_counts(normalize=True)
        cv = in_c.cast(pl.String).value_counts(normalize=True)
        j = cv.join(allv, on=col, suffix="_all")
        for row in j.iter_rows(named=True):
            if row["proportion"] >= 0.1 and row["proportion_all"] > 0:
                r = row["proportion"] / row["proportion_all"]
                if r >= 1.2:
                    out.append((r, f"{col}={row[col]} {row['proportion']:.0%} vs {row['proportion_all']:.0%}"))
    out.sort(key=lambda t: -t[0])
    return "; ".join(t[1] for t in out[:k])


def profile_run(tag, labels, stab, space, space_names, run_rows, X, meta, feat_names, cat_levels,
                gain, goods_sub, goods_w, n_goods_pop, in_pool, n_fraud_total, cfg, log=print):
    months = meta["__month"].to_numpy()[run_rows]
    ms = monthly_share(months, labels)
    name_to_idx = {n: i for i, n in enumerate(feat_names)}
    rule_feats = [name_to_idx[n] for n in space_names if n in name_to_idx]
    rule_feats += [j for j in np.argsort(-gain)[:30] if j not in rule_feats]
    meta_run = meta[run_rows]
    base_rate = n_fraud_total / (n_fraud_total + n_goods_pop)
    rows, signals = [], []
    for c in sorted(set(labels) - {-1}):
        c_rows = run_rows[labels == c]
        d = dict(run=tag, cluster=int(c), n=int(len(c_rows)), share_of_run=round(len(c_rows) / len(run_rows), 4),
                 stability=round(stab.get(c, np.nan), 3),
                 in_missed_pool=round(float(in_pool[labels == c].mean()), 3))
        if "__account" in meta.columns:
            acc = meta["__account"][c_rows].drop_nulls()
            d["n_accounts"] = int(acc.n_unique())
            d["top_account_share"] = round(float(acc.value_counts()["count"].max() / len(c_rows)), 3) \
                if len(acc) else 0.0
        else:
            d["n_accounts"], d["top_account_share"] = None, None
        d.update(describe_cluster(X, c_rows, run_rows, goods_sub, goods_w, rule_feats, feat_names,
                                  cat_levels, n_goods_pop, cfg.seed))
        er, enough = emergence(ms, c)
        d["emergence_ratio"] = er
        d["emerging"] = bool(enough and np.isfinite(er) and er >= cfg.emerging_ratio)
        d["drivers"] = drivers(space, space_names, labels, c, X, c_rows, goods_sub, feat_names, cat_levels)
        d["demographics"] = demographics(meta_run, labels, c, cfg.demographic_cols) if cfg.demographic_cols else ""
        d["describable"] = bool(d["rule_recall"] >= cfg.verdict_min_rule_recall)
        d["enriched"] = bool(d["rule_lift"] >= cfg.verdict_min_lift)
        d["stable"] = bool(d["stability"] >= cfg.verdict_min_stability)
        d["account_concentrated"] = bool(d["top_account_share"] is not None
                                         and d["top_account_share"] > cfg.max_top_account_share)
        d["MO_candidate"] = bool(d["describable"] and d["enriched"] and d["stable"] and not d["account_concentrated"])
        # ---- headline segment metrics, all computed from the cluster's rule at population scale
        d["fraud_n"] = d["n"]
        d["fraud_pct_of_all_fraud"] = round(100 * d["n"] / n_fraud_total, 2)
        d["rule_fraud_pct_of_all_fraud"] = round(100 * d["rule_fraud_matching"] / n_fraud_total, 2)
        d["good_pct_of_all_goods"] = round(100 * d["rule_est_goods_pop"] / n_goods_pop, 4)
        d["hit_rate_pct"] = round(100 * d["rule_precision_pop"], 4)
        d["lift"] = d["rule_lift"]
        d["description"] = (f"{d['rule']} | {d['drivers'][:160]}"
                            + (f" | {d['demographics']}" if d["demographics"] else ""))
        signals += signal_rates(X, c_rows, run_rows, goods_sub, goods_w, rule_feats[:cfg.n_item_features],
                                feat_names, cat_levels, n_goods_pop, n_fraud_total, cfg, tag, int(c))
        rows.append(d)
        log(f"[profile {tag}] cluster {c}: n={d['n']} stab={d['stability']} lift={d['rule_lift']} "
            f"recall={d['rule_recall']} emerging={d['emerging']} -> {'CANDIDATE' if d['MO_candidate'] else '-'}")
    log(f"[profile {tag}] base fraud rate {100 * base_rate:.4f}%")
    return (pl.DataFrame(rows) if rows else pl.DataFrame(),
            pl.DataFrame(signals) if signals else pl.DataFrame(), ms)


# ==================================================================================================
# RUN
# ==================================================================================================
def _log_factory(path):
    t0 = time.time()
    f = open(path, "a")

    def log(msg):
        line = f"[{time.time() - t0:7.0f}s] {msg}"
        print(line, flush=True)
        f.write(line + "\n"); f.flush()
    return log


def _plot(emb, labels, title, path):
    fig, ax = plt.subplots(figsize=(9, 7))
    noise = labels == -1
    ax.scatter(emb[noise, 0], emb[noise, 1], s=2, c="lightgrey", label="noise")
    for c in sorted(set(labels) - {-1}):
        m = labels == c
        ax.scatter(emb[m, 0], emb[m, 1], s=3, label=f"{c} (n={m.sum()})")
        ax.annotate(str(c), emb[m].mean(0), fontsize=11, weight="bold")
    ax.set_title(title); ax.legend(markerscale=4, fontsize=7, ncol=2); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _overlap(segment_member, labels):
    out = []
    for s in sorted(set(segment_member) - {-1}):
        a = segment_member == s
        best, bj = -1, 0.0
        for c in set(labels) - {-1}:
            b = labels == c
            j = (a & b).sum() / (a | b).sum()
            if j > bj:
                best, bj = c, j
        out.append((int(s), int(best), round(float(bj), 3)))
    return out


def _md_table(df: pl.DataFrame, cols):
    cols = [c for c in cols if c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.select(cols).iter_rows():
        lines.append("| " + " | ".join(str(v).replace("|", "/") for v in r) + " |")
    return "\n".join(lines)


def main(cfg: Config = None):
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    cfg = cfg or Config()
    os.makedirs(cfg.out_dir, exist_ok=True)
    log = _log_factory(os.path.join(cfg.out_dir, "run.log"))

    X, y, meta, info = prepare(cfg, log)
    feats, cats = info["features"], info["categorical"]
    spec = info["spec"]
    cat_levels = {f"cat__{c}": {v: k for k, v in m.items()} for c, m in spec["lowcard"].items()}
    n_goods_pop = cfg.n_goods_population

    A, B, missed, missed_desc = run_models(X, y, info, meta, cfg, log)
    fraud_rows = np.where(y == 1)[0]
    goods_rows = np.where(y == 0)[0]
    goods_w = n_goods_pop / len(goods_rows)
    rng = np.random.default_rng(cfg.seed)
    goods_sub = np.sort(rng.choice(goods_rows, size=min(200_000, len(goods_rows)), replace=False))
    goods_sub_w = n_goods_pop / len(goods_sub)

    scoreA_fraud = A["oof"][fraud_rows]           # A rows == all rows, fraud first
    scoreA_by_row = dict(zip(fraud_rows.tolist(), scoreA_fraud.tolist()))

    runs = {}
    # 1) all fraud, SHAP of stage A
    Z, zn, _ = shap_space(A["shap"], feats, cfg.n_shap_features)
    runs["all_shap"] = (Z, zn, fraud_rows, A["gain"])
    # 2) missed fraud, SHAP of stage B (what separates the missed fraud from goods)
    if B is not None:
        Z, zn, _ = shap_space(B["shap"], feats, cfg.n_shap_features)
        runs["missed_shap"] = (Z, zn, B["shap_rows"], B["gain"])
        # 3) missed fraud, goods-quantile (normalised) space on stage-B top features
        top = np.argsort(-B["gain"])[:cfg.n_shap_features]
        top = np.array([j for j in top if feats[j] not in set(cats)])
        Zq, zqn = goods_quantile_space(X, goods_sub, B["shap_rows"], top, feats, cfg)
        runs["missed_quantile"] = (Zq, zqn, B["shap_rows"], B["gain"])
    else:
        log(f"[run] only {len(missed)} missed fraud rows - skipping missed-fraud runs")

    all_profiles, all_signals, report = [], [], ["# MO discovery report\n"]
    m = A["metrics"]
    report.append(f"Stage A (fraud vs goods): OOF ROC-AUC {m[0]:.4f}, PR-AUC {m[1]:.4f}, "
                  f"fraud recall {m[2]:.1%} at {cfg.alert_rate:.1%} goods alert rate. "
                  f"Missed-fraud pool: {len(missed):,} of {len(fraud_rows):,} ({missed_desc}).\n")
    if B is not None:
        m = B["metrics"]
        report.append(f"Stage B (missed fraud vs goods): OOF ROC-AUC {m[0]:.4f}, PR-AUC {m[1]:.4f}.\n")
    sr = info["screening_report"]
    report.append("## Screening / leakage warnings\n")
    report.append(f"- Columns dropped as likely extraction artefacts (null rate differs between files): "
                  f"{len(sr['leak_null'])} {sr['leak_null'][:15]}")
    report.append(f"- Features dropped for suspiciously perfect single-feature AUC: {sr['leak_auc'][:15]}")
    report.append(f"- Dropped mostly-null: {len(sr['dropped_null'])}, near-constant: {len(sr['near_constant'])}, "
                  f"correlated: {sr['correlated']}. Features modelled: {len(feats)}")
    if sr.get("leak_class_asymmetric"):
        report.append(f"- Built features present in one class only (name, goods-null rate, fraud-null rate) - "
                      f"usually a column one extract does not fill: {sr['leak_class_asymmetric'][:15]}")
    if "date_coverage" in sr:
        report.append(f"- Date column coverage by class (1 = always populated): {sr['date_coverage']}")
    if "dedup" in sr:
        report.append(f"- Deduplication / account filter (after sampling): {sr['dedup']}")
    if "date_ranges" in sr:
        report.append(f"- Transaction date range by class: {sr['date_ranges']}")
    report.append("")

    for tag, (Z, zn, run_rows, gain) in runs.items():
        labels, stab, emb2, fine = cluster(Z, cfg, log, tag)
        _plot(emb2, labels, tag, os.path.join(cfg.out_dir, f"umap_{tag}.png"))
        sA = np.array([scoreA_by_row[r] for r in run_rows])
        in_pool = np.isin(run_rows, missed)
        prof, sig, ms = profile_run(tag, labels, stab, Z, zn, run_rows, X, meta, feats, cat_levels, gain,
                                    goods_sub, goods_sub_w, n_goods_pop, in_pool, len(fraud_rows), cfg, log)
        ms.write_csv(os.path.join(cfg.out_dir, f"monthly_share_{tag}.csv"))
        out = meta[run_rows].with_columns(pl.Series("cluster", labels), pl.Series("fine_cluster", fine), pl.Series("stageA_score", sA),
                                          pl.Series("in_missed_pool", in_pool))
        out.write_parquet(os.path.join(cfg.out_dir, f"clusters_{tag}.parquet"))
        if prof.height:
            prof.write_csv(os.path.join(cfg.out_dir, f"cluster_profiles_{tag}.csv"))
            all_profiles.append(prof)
        if sig.height:
            sig.write_csv(os.path.join(cfg.out_dir, f"signal_rates_{tag}.csv"))
            all_signals.append(sig)

        # rule peeling on the same target set, on the same model's top features
        feat_idx = np.argsort(-gain)[:cfg.n_item_features]
        segs, member = peel(X, run_rows, goods_rows, feat_idx, feats, cat_levels, goods_w, cfg, log, tag)
        seg_df = pl.DataFrame(segs) if segs else pl.DataFrame()
        ov = _overlap(member, labels)
        if seg_df.height:
            seg_df = seg_df.join(pl.DataFrame(ov, schema=["segment", "best_cluster", "jaccard"], orient="row"),
                                 on="segment", how="left")
            seg_df.write_csv(os.path.join(cfg.out_dir, f"segments_{tag}.csv"))

        report.append(f"## Run `{tag}` - {len(run_rows):,} fraud rows, "
                      f"{len(set(labels) - {-1})} clusters, {np.mean(labels == -1):.0%} noise\n")
        report.append(f"![umap](umap_{tag}.png)\n")
        if prof.height:
            report.append("### Clusters (MO_candidate = describable AND enriched AND stable AND not account-concentrated)\n")
            report.append(_md_table(prof.sort(["MO_candidate", "n"], descending=True),
                                    ["cluster", "fraud_n", "fraud_pct_of_all_fraud", "n_accounts",
                                     "top_account_share", "good_pct_of_all_goods", "hit_rate_pct", "lift",
                                     "rule_recall", "rule_purity", "stability", "MO_candidate", "emerging",
                                     "emergence_ratio", "in_missed_pool", "rule", "drivers", "demographics"]))
        if sig.height:
            report.append("\n### Signal rates - per condition, share of the cluster following it and "
                          "share of all goods following it\n")
            report.append(_md_table(sig, ["cluster", "condition", "fraud_in_cluster", "fraud_pct_of_cluster",
                                          "fraud_pct_of_all_fraud", "est_goods_matching",
                                          "good_pct_of_all_goods", "hit_rate_pct", "lift"]))
        if seg_df.height:
            report.append("\n### Peeled rule segments (cross-check; best_cluster/jaccard = agreement)\n")
            report.append(_md_table(seg_df, ["segment", "rule", "new_fraud_covered", "lift",
                                             "est_goods_matching_population", "best_cluster", "jaccard"]))
        report.append("")

    summary_cols = ["run", "cluster", "description", "fraud_n", "fraud_pct_of_all_fraud",
                    "rule_fraud_matching", "rule_fraud_pct_of_all_fraud", "rule_est_goods_pop",
                    "good_pct_of_all_goods", "hit_rate_pct", "lift", "rule_recall", "stability",
                    "n_accounts", "emerging", "MO_candidate"]
    summary = None
    if all_profiles:
        summary = pl.concat(all_profiles, how="diagonal_relaxed").select(
            [c for c in summary_cols if c in all_profiles[0].columns]).sort(["lift"], descending=True)
        summary.write_csv(os.path.join(cfg.out_dir, "mo_summary.csv"))
        report.append("\n## Summary of all clusters\n")
        report.append(f"Base fraud rate: {len(fraud_rows):,} fraud vs {n_goods_pop:,} goods "
                      f"= {100 * len(fraud_rows) / (len(fraud_rows) + n_goods_pop):.4f}%. "
                      "fraud_n / fraud_pct_of_all_fraud describe the cluster itself; the goods, hit-rate and "
                      "lift columns describe the cluster's RULE applied to the whole population.\n")
        report.append(_md_table(summary, summary_cols))
    if all_signals:
        sig_all = pl.concat(all_signals, how="diagonal_relaxed")
        sig_all.write_csv(os.path.join(cfg.out_dir, "signal_rates.csv"))
    with open(os.path.join(cfg.out_dir, "report.md"), "w") as f:
        f.write("\n".join(report))
    log(f"[run] done - see {os.path.join(cfg.out_dir, 'report.md')}")
    return dict(profiles=pl.concat(all_profiles, how="diagonal_relaxed") if all_profiles else None,
                summary=summary,
                signals=pl.concat(all_signals, how="diagonal_relaxed") if all_signals else None,
                stageA=A, stageB=B, missed=missed)



# ==================================================================================================
# RUN
# ==================================================================================================
if __name__ == "__main__":
    cfg = Config(
        goods_glob="data_goods/*.parquet",          # the 7 snappy parquet files
        fraud_path="data_fraud/fraud.parquet",      # .parquet or .csv
        out_dir="mo_output",
        # Rename both extracts to a shared vocabulary first, then name the shared columns below.
        rename_fraud={"earliest_fraud_tran_date": "tran_dt"},   # fraud file's transaction-date column
        rename_goods={},                                        # e.g. {"txn_date": "tran_dt"}
        tran_date_col="tran_dt",
        other_date_cols=["first_credit_dt", "dormancy_start_date", "reactivation_date"],
        all_date_gaps=True,     # builds dormancy length, reactivation -> transaction, etc.
        account_col="account_no",                   # your account number column
        id_cols=[],                                 # e.g. ["tran_id"] - never modelled
        dedup_cols=[],                              # duplicate key; empty = exact duplicate on all columns
        demographic_cols=[],                        # held out, used only to profile clusters
        exclude_cols=[],                            # label-like columns (fraud type, chargeback flags ...)
        n_goods_sample=1_000_000,
        missed_definition="temporal",               # "temporal" | "existing_score" (set score_col) | "oof"
        temporal_cutoff=None,                       # e.g. "2025-06"; None -> month where ~70% of fraud occurred
    )
    results = main(cfg)

