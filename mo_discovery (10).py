#!/usr/bin/env python3
"""
mo_discovery.py  -  Fraud MO segmentation that is aware of your EXISTING MOs
==========================================================================

End to end, one file:

 1. Load fraud + goods (Polars), dedupe, drop goods accounts that also appear in fraud.
 2. Parse pipe-delimited multi-label string columns  (cat_{i}_sub_label and any look-alike
    column, auto-detected)  ->  multi-hot token features + n_tokens + cross-column
    "anycat__<token>" features.
 3. Tag every row with your KNOWN MOs. MOs are independent labels: a row can match several and
    counts in every one it matches (no priority order).
 4. Mix goods + fraud and split TRAIN / VALIDATION (stratified by label x MO combination,
    grouped by account so an account never sits on both sides; or out-of-time).
 5. TRAIN only: HGB (all fraud vs goods, MO flags kept out) -> SHAP -> UMAP(10D) -> HDBSCAN
    (+ subclustering of oversized families) on ALL fraud.   runs: all_shap | all_quantile
 6. The MOs are then overlaid on the behaviour clusters, each MO on its own:
        ALIGNED:<mo>  the cluster is that MO (independent confirmation of the flag logic)
        SUBTYPE:<mo>  the cluster is one of several behaviours inside that MO  -> sub-MO
        CROSS-MO      flagged fraud from several MOs that behaves the same    -> new behavioural MO
        PARTIAL:<mo>  partly covered, no MO dominates
        GAP / NEW     mostly unflagged fraud (only if some fraud matches no MO)
    + depth-3 rule, profile, signal rates, token-combo mining, rule peeling.
 7. VALIDATION rows go through the frozen pipeline; every cluster / rule / segment gets
    unseen-data metrics (size stability, rule lift retention, re-cluster Jaccard, ...).
 8. All fitted objects are cached so new data can be mapped later:
        python mo_discovery.py                       # full discovery run
        python mo_discovery.py --map-new new.parquet # map a new dataset onto the clusters
"""

import argparse
import glob
import os
import pickle
import re
import time
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import polars as pl
import shap
import umap
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, average_precision_score, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

try:
    import hdbscan
    HAVE_HDBSCAN = True
except ImportError:  # fall back to sklearn's HDBSCAN + kNN assignment for unseen rows
    from sklearn.cluster import HDBSCAN as SkHDBSCAN
    HAVE_HDBSCAN = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# =====================================================================================
# CONFIG
# =====================================================================================
# ---- input --------------------------------------------------------------------------
FRAUD_PATH = "data_fraud/"          # file or folder of .parquet / .csv
GOODS_PATH = "data_goods/"          # file or folder of .parquet / .csv
COMBINED_PATH = None                # OR one file/folder with both; then LABEL_COL is used
LABEL_COL = "is_fraud"              # only used with COMBINED_PATH (1 = fraud)
ACCOUNT_COL = "account_number"      # grouping key: dedupe, goods-vs-fraud overlap, grouped split; never a feature
                                    # (use "crn" here instead if one customer can hold several accounts)
DATE_COL = "tran_dt"                # base date; used for date diffs, emergence, time split
OTHER_DATE_COLS = ["first_credit_dt", "dormancy_start_date", "reactivation_date"]
DROP_COLS = ["crn"]                 # ids / leakage columns to never use as features
GOODS_SAMPLE_FRAC = 1.0             # 37k goods fits in memory -> keep all
OUT_DIR = "mo_output"
CACHE_DIR = "cache"

# ---- pipe-delimited multi-label string columns --------------------------------------
MULTILABEL_PATTERNS = [r"^cat_\d+_sub_label$"]   # always treated as multi-label
MULTILABEL_FAMILY_PATTERNS = [r"^cat_\d+_sub_label$"]  # columns that share a token vocab
MULTILABEL_AUTODETECT = True        # also catch other columns with "a | b | c" values
MULTILABEL_DELIM = "|"
MULTILABEL_MIN_SHARE = 0.01         # autodetect if >=1% of non-null values contain DELIM
TOKEN_MIN_COUNT = 30                # token kept only if seen in >= this many TRAIN rows
MAX_TOKENS_PER_COL = 200
TOKEN_LOWERCASE = False
NULL_TOKENS = {"", "nan", "none", "null", "na", "n/a"}
MAX_ONEHOT_CARD = 30                # plain string cols: <= this many levels -> one-hot
MAX_MISSING = 0.99                  # drop numeric cols missing in > 99% of TRAIN rows
LEAKAGE_AUC = 0.995                 # a single feature separating fraud/goods this well on its own is
                                    # almost always an extraction artefact -> dropped (see leakage_check.csv)
DROP_LEAKY_FEATURES = True

# ---- known MOs ----------------------------------------------------------------------
# Short names used inside the MO expressions -> real column names in the data.
# The script reads these columns itself after loading; nothing to compute here (no `df` exists yet).
MO_FLAG_VARS = {                    # converted to 0/1 (1/0, Y/N, True/False all work; missing -> 0)
    "c1": "category_1_flag",
    "c2": "category_2_flag",
    "c3": "category_3_flag",
    "c4": "category_4_flag",
    "c5": "category_5_flag",
}
MO_NUMERIC_VARS = {                 # converted to numbers (missing / non-numeric -> 0)
    "scam_cr": "winscam_cr_cnt",
    "scam_dr": "winscam_dr_cnt",
}
FLAG_COLS = list(MO_FLAG_VARS.values()) + list(MO_NUMERIC_VARS.values())   # source columns (derived)

# Paste your full MO names back in as the keys. MOs are INDEPENDENT labels: a row can match several
# and is counted in every one it matches, so the order here does not matter.
KNOWN_MOS = {
    "MO-3": "c3 == 1 and c4 == 1 and c5 == 1",
    "MO-5": "c1 == 1 and c3 == 1 and c4 == 1",
    "MO-1": "c3 == 1 and c4 == 1 and c5 == 0 and c2 == 0 and c1 == 0",
    "MO-6": "c5 == 1 and (c4 == 1 or c3 == 1)",
    "MO-4": "c2 == 1 and (c3 == 1 or c4 == 1)",
    "MO-2": "c3 == 1",
    "MO-7": "c1 == 0 and c2 == 0 and c3 == 0 and c4 == 0 and c5 == 0 and scam_cr >= 10",
    "LAST_MO_rename_me": "c1 == 0 and c2 == 0 and c3 == 0 and c4 == 0 and c5 == 0 and scam_cr == 0 and scam_dr == 0",
}
# Catch-all buckets (e.g. "nothing flagged at all") that are not a behaviour. They are still tagged
# and reported, but they don't count towards a cluster's known-MO coverage or MO signature.
UNCOVERED_MOS = ["LAST_MO_rename_me"]

# Keep the flags OUT of the model features. Then if an unsupervised cluster lines up with
# a known MO without ever seeing its flags, that is independent confirmation of the MO -
# and uncovered fraud that lands next to it is a definition GAP, not a new MO.
EXCLUDE_FLAGS_FROM_MODEL = True     # True/False only - keeps every column in MO_FLAG_VARS and
                                    # MO_NUMERIC_VARS (incl. the winscam counts) out of the model

# ---- split --------------------------------------------------------------------------
SPLIT_MODE = "random"               # "random" (stratified, account-grouped) or "time"
VAL_SIZE = 0.30
RANDOM_STATE = 42

# ---- operating budget ---------------------------------------------------------------
GOODS_PER_FRAUD_BUDGET = 50         # ~1-in-51 alert precision
# Goods in this file are a SAMPLE of production goods. Every goods count is multiplied by
# this before comparing with the budget: production_goods_volume / goods_rows_in_file
# (counted over the same time window as the fraud). Leave 1.0 only if the file is the
# full goods population - otherwise every "within budget" verdict is too optimistic.
GOODS_WEIGHT = 1.0

# ---- modelling / clustering ---------------------------------------------------------
MAX_MODEL_FEATURES = 250            # SHAP-ranked feature selection per stage
TOP_SHAP_FOR_CLUSTERING = 50       # upper cap on SHAP columns fed to UMAP
TOP_SHAP_MIN = 10
SHAP_CUM_SHARE = 0.95               # keep the fewest features explaining 95% of the run's total |SHAP|;
                                    # with ~100 features a fixed 50 adds near-zero noise columns to UMAP
SHAP_SELECTION_SAMPLE = 4000
QUANTILE_GOODS_SAMPLE = 10000
UMAP_DIM = 10
UMAP_NEIGHBORS = 30
HDB_MIN_CLUSTER_FRAC = 0.005
HDB_MIN_CLUSTER_ABS = 40
HDB_MIN_SAMPLES = 10
SUBCLUSTER_FRAC = 0.25              # (multi-run mode only; the single all_shap run never sub-splits)
MIN_RESIDUAL_FRAUD = 300            # skip residual runs if fewer uncovered train fraud

# ---- rules / labelling --------------------------------------------------------------
RULE_DEPTH = 3
RULE_MIN_SUPPORT = 20
PEEL_MAX_SEGMENTS = 12
# Cluster labels (all fraud is clustered; MOs are overlaid afterwards, each one independently):
#   ALIGNED:<mo>  >= MO_PRECISION of the cluster matches <mo> and the cluster holds >= MO_ALIGN_RECALL
#                 of that MO's fraud -> the behaviour cluster IS the MO (independent confirmation)
#   SUBTYPE:<mo>  same precision, but the MO's fraud is spread over several clusters -> sub-MO
#   CROSS-MO      covered by known MOs, but no single MO dominates -> one behaviour across MOs
#   PARTIAL:<mo>  partly covered, no MO dominates
#   GAP:<mo> / NEW  mostly not covered by any MO (only possible if some fraud is unflagged)
MO_PRECISION = 0.70
MO_ALIGN_RECALL = 0.50
KNOWN_COVERAGE_HI = 0.70
NEW_COVERAGE_LO = 0.30
CANDIDATE_LABELS = ("SUBTYPE:", "CROSS-MO", "GAP:", "NEW")   # labels that can become MO candidates
GAP_SIMILARITY = 0.80               # centred mean-SHAP cosine to a known MO -> GAP
MIN_VAL_RECALL_RETENTION = 0.60  # val rule recall on the cluster / train rule recall
VAL_SIZE_RATIO_RANGE = (0.5, 2.0)
MIN_RECLUSTER_JACCARD = 0.30       # cluster must re-appear when VAL fraud is clustered on its own
FINAL_RUN = "all_shap"              # run whose candidate clusters feed final_segment

SENTINEL = -1e9                     # NaN stand-in for rule learning ("IS NULL")
RUNS = [("all_shap", "A", "all", False)]   # single run: all fraud, Stage-A SHAP space

# ---- segmentation (all_shap) ------------------------------------------------------
# Few, large, defensible segments instead of many density bumps. Every fraud row ends up in a
# SIGNIFICANT segment or is flagged UNIDENTIFIED.
MIN_SEGMENT_FRAC = 0.02             # a segment must hold >= 2% of train fraud (HDBSCAN min size too)
# Clustering engine: PCA of the SHAP vectors -> Gaussian mixture, number of segments chosen by
# how reproducible they are on validation. (UMAP+HDBSCAN broke apart on this data: tree-model SHAP
# vectors contain many exact duplicates, UMAP's neighbour graph splits into hundreds of disconnected
# islands, and their positions - and the clusters built on them - carry no meaning.)
PCA_VAR = 0.90                      # keep PCA components explaining 90% of SHAP variance (max PCA_MAX_DIM)
PCA_MAX_DIM = 15
K_RANGE = range(2, 13)              # candidate numbers of segments
GMM_REG = 1e-3                      # covariance regularisation (needed with duplicated rows)
K_MIN_STABILITY = 0.80              # k must reproduce on validation (ARI >= this); among those the
                                    # lowest BIC wins (detail where the data supports it, not more)
MIN_MEMBERSHIP = 0.70               # posterior below this = row sits between segments -> UNIDENTIFIED
MIN_LOGLIK_Q = 0.01                 # rows less typical than the least-typical 1% of train mules (GMM
                                    # log-likelihood) belong to no segment -> UNIDENTIFIED. Without this,
                                    # goods far from every segment would still be forced into one.
CONDITION_ON_MOS = False            # True: remove what the known MOs explain from the SHAP vectors
                                    # before clustering (linear "conditional" representation), so the
                                    # segments describe behaviour BEYOND the MOs
# A segment is SIGNIFICANT only if all of these hold:
SIG_MIN_SEPARABILITY_AUC = 0.80     # a depth-3 tree on raw features separates it from OTHER fraud (val AUC)
SIG_MIN_JACCARD = 0.50              # it re-appears when validation fraud is clustered on its own
SIG_MIN_VAL_LIFT = None             # e.g. 1.5 -> also require val lift >= 1.5 (alertable segments only).
                                    # None keeps low-lift segments: fraud that looks like goods is still an MO
                                    # (plus VAL_SIZE_RATIO_RANGE: similar share of fraud in train and val)
RUN_PEELING = False                 # optional detection-rule peeling (segments_all_shap.csv)

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# =====================================================================================
# LOADING
# =====================================================================================
def _files(path):
    if os.path.isdir(path):
        fs = sorted(glob.glob(os.path.join(path, "*.parquet")) + glob.glob(os.path.join(path, "*.csv")))
    else:
        fs = [path]
    if not fs:
        raise FileNotFoundError(f"No parquet/csv files at {path}")
    return fs


def read_any(path) -> pl.DataFrame:
    frames = []
    for f in _files(path):
        if f.lower().endswith(".csv"):
            frames.append(pl.read_csv(f, infer_schema_length=20000, try_parse_dates=True))
        else:
            frames.append(pl.read_parquet(f))
    return pl.concat(frames, how="diagonal_relaxed") if len(frames) > 1 else frames[0]


def load_data() -> pd.DataFrame:
    if COMBINED_PATH:
        df = read_any(COMBINED_PATH)
        df = df.with_columns(pl.col(LABEL_COL).cast(pl.Int8).alias("_label"))
    else:
        fr = read_any(FRAUD_PATH).with_columns(pl.lit(1, pl.Int8).alias("_label"))
        gd = read_any(GOODS_PATH).with_columns(pl.lit(0, pl.Int8).alias("_label"))
        # a column that exists in only one file is NULL for the whole other class -> perfect leak
        only = sorted((set(fr.columns) ^ set(gd.columns)) - set(FLAG_COLS))
        if only:
            log(f"WARNING dropping {len(only)} columns present in only one of fraud/goods files: {only[:20]}"
                f"{' ...' if len(only) > 20 else ''}")
            fr, gd = fr.drop([c for c in only if c in fr.columns]), gd.drop([c for c in only if c in gd.columns])
        miss_flags = [c for c in FLAG_COLS if (c in fr.columns) != (c in gd.columns)]
        if miss_flags:
            log(f"NOTE flags {miss_flags} exist in only one file -> they count as 0 for the other class")
        if GOODS_SAMPLE_FRAC < 1.0:
            gd = gd.sample(fraction=GOODS_SAMPLE_FRAC, seed=RANDOM_STATE)
        df = pl.concat([fr, gd], how="diagonal_relaxed")
    n0 = df.height
    df = df.unique(maintain_order=True)
    log(f"loaded {n0:,} rows, {df.height:,} after exact-duplicate removal")
    if ACCOUNT_COL in df.columns:
        fraud_acc = df.filter(pl.col("_label") == 1)[ACCOUNT_COL].unique()
        before = df.height
        df = df.filter(~((pl.col("_label") == 0) & pl.col(ACCOUNT_COL).is_in(fraud_acc)))
        log(f"removed {before - df.height:,} goods rows whose account also appears in fraud")
    pdf = df.to_pandas().reset_index(drop=True)
    log(f"fraud={int((pdf["_label"] == 1).sum()):,}  goods={int((pdf["_label"] == 0).sum()):,}  cols={pdf.shape[1]:,}")
    return pdf


# =====================================================================================
# KNOWN MOs
# =====================================================================================
def to01(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(np.int8)
    if pd.api.types.is_numeric_dtype(s):
        return (pd.to_numeric(s, errors="coerce").fillna(0) > 0).astype(np.int8)
    t = s.astype("string").str.strip().str.upper()
    return t.isin(["1", "1.0", "Y", "YES", "TRUE", "T"]).fillna(False).astype(np.int8)


def tag_known_mos(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in FLAG_COLS if c not in df.columns]
    if missing:
        log(f"WARNING MO columns missing {missing} -> treated as 0; MO tags will be wrong")
    bad = [m for m in UNCOVERED_MOS if m not in KNOWN_MOS]
    if bad:
        raise ValueError(f"UNCOVERED_MOS has names not in KNOWN_MOS: {bad}")
    V = {}
    for a, c in MO_FLAG_VARS.items():
        V[a] = to01(df[c]) if c in df.columns else pd.Series(np.zeros(len(df), np.int8), index=df.index)
    for a, c in MO_NUMERIC_VARS.items():
        V[a] = (pd.to_numeric(df[c], errors="coerce").fillna(0) if c in df.columns
                else pd.Series(np.zeros(len(df)), index=df.index))
    F = pd.DataFrame(V, index=df.index)
    hits = {}
    for name, expr in KNOWN_MOS.items():
        try:
            hits[name] = F.eval(expr, engine="python").astype(bool).to_numpy()
        except Exception as e:
            raise ValueError(f"Cannot evaluate KNOWN_MOS['{name}'] = '{expr}': {e} "
                             f"(usable names: {list(V)})")
    new = {f"_flag_{a}": F[a].to_numpy() for a in MO_FLAG_VARS}
    names = list(KNOWN_MOS)
    H = np.column_stack([hits[n] for n in names]) if names else np.zeros((len(df), 0), bool)
    for i, n in enumerate(names):
        new[f"_mo__{n}"] = H[:, i].astype(np.int8)
    new["_mo_set"] = np.array(["+".join(n for n, h in zip(names, row) if h) or "NONE" for row in H], dtype=object)
    new["_mo_count"] = H.sum(1).astype(np.int8)
    real = [i for i, n in enumerate(names) if n not in UNCOVERED_MOS]
    new["_covered"] = H[:, real].any(1) if real else np.zeros(len(df), bool)
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


def report_mo_overlap(df):
    """Fraud rows per MO and how often MOs co-occur on the same row (they are independent labels)."""
    fr = (df["_label"] == 1).to_numpy()
    names = list(KNOWN_MOS)
    M = df.loc[fr, [f"_mo__{n}" for n in names]].to_numpy().astype(int)
    G = df.loc[~fr, [f"_mo__{n}" for n in names]].to_numpy().astype(int)
    co = pd.DataFrame(M.T @ M, index=names, columns=names)
    co.to_csv(f"{OUT_DIR}/known_mo_cooccurrence.csv")
    out = pd.DataFrame({"known_mo": names, "fraud_matched": M.sum(0), "goods_matched": G.sum(0),
                        "fraud_also_in_other_mo": [int(((M[:, i] == 1) & (M.sum(1) > 1)).sum()) for i in range(len(names))],
                        "counts_as_covered": [n not in UNCOVERED_MOS for n in names]})
    out.to_csv(f"{OUT_DIR}/known_mo_overlap.csv", index=False)
    log("known MOs on fraud rows:\n" + out.to_string(index=False))
    log(f"fraud rows matching 0 / 1 / 2+ MOs: {(M.sum(1) == 0).sum():,} / {(M.sum(1) == 1).sum():,} / "
        f"{(M.sum(1) >= 2).sum():,}")


# =====================================================================================
# SPLIT
# =====================================================================================
def _merge_rare(s, min_n=10):
    s = np.asarray(s, dtype=object)
    vc = Counter(s)
    s = np.array([x if vc[x] >= min_n else ("F_OTHER" if str(x).startswith("F") else "G") for x in s], dtype=object)
    vc = Counter(s)
    return np.array([x if vc[x] >= 2 else ("F_NONE" if str(x).startswith("F") else "G") for x in s], dtype=object)


def make_split(df: pd.DataFrame) -> np.ndarray:
    y = df["_label"].to_numpy()
    strata = np.where(y == 1, "F_" + df["_mo_set"].astype(str).to_numpy(), "G")
    if SPLIT_MODE == "time":
        if DATE_COL not in df.columns:
            raise ValueError("SPLIT_MODE='time' needs DATE_COL")
        d = pd.to_datetime(df[DATE_COL], errors="coerce", utc=True)
        cut = d.quantile(1 - VAL_SIZE)
        is_val = (d > cut).fillna(False).to_numpy()
        for lab, nm in [(1, "fraud"), (0, "goods")]:
            dd = d[y == lab]
            log(f"{nm} dates {dd.min()} -> {dd.max()}  | val share {is_val[y == lab].mean():.1%}")
        log(f"time split cut-off {cut}  (check both classes have rows on both sides!)")
        return is_val
    if ACCOUNT_COL in df.columns and df[ACCOUNT_COL].notna().all():
        acc = pd.DataFrame({"acc": df[ACCOUNT_COL].to_numpy(), "s": strata})
        first = acc.groupby("acc", sort=False)["s"].first()
        _, va_acc = train_test_split(first.index.to_numpy(), test_size=VAL_SIZE,
                                     stratify=_merge_rare(first.to_numpy()), random_state=RANDOM_STATE)
        return df[ACCOUNT_COL].isin(set(va_acc)).to_numpy()
    idx = np.arange(len(df))
    _, va = train_test_split(idx, test_size=VAL_SIZE, stratify=_merge_rare(strata), random_state=RANDOM_STATE)
    is_val = np.zeros(len(df), bool)
    is_val[va] = True
    return is_val


# =====================================================================================
# PREPROCESSOR  (fit on TRAIN only, applied identically to VAL / new data)
# =====================================================================================
def _to_dt(s):
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s, utc=True)
    try:
        return pd.to_datetime(s, errors="coerce", utc=True, format="mixed")
    except (TypeError, ValueError):          # pandas < 2.0 has no format="mixed"
        return pd.to_datetime(s, errors="coerce", utc=True)


def _tokenize(s: pd.Series) -> pd.Series:
    def clean(lst):
        out = []
        for x in lst:
            x = x.strip()
            if TOKEN_LOWERCASE:
                x = x.lower()
            if x.lower() not in NULL_TOKENS:
                out.append(x)
        return out
    return s.astype("string").fillna("").str.split(MULTILABEL_DELIM, regex=False).map(clean)


class Preprocessor:
    def fit(self, df: pd.DataFrame, reserved: set):
        self.numeric, self.multilabel, self.onehot, self.freq, self.dates = [], {}, {}, {}, []
        self.date_base = DATE_COL if DATE_COL in df.columns else None
        date_cols = set(OTHER_DATE_COLS) | ({DATE_COL} if DATE_COL else set())
        n = len(df)
        for c in df.columns:
            if c in reserved or c.startswith("_"):
                continue
            s = df[c]
            if c in date_cols or pd.api.types.is_datetime64_any_dtype(s):
                if c != self.date_base:
                    self.dates.append(c)
                continue
            if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
                v = pd.to_numeric(s, errors="coerce")
                if v.notna().mean() >= 1 - MAX_MISSING and v.nunique(dropna=True) > 1:
                    self.numeric.append(c)
                continue
            nn = s.dropna().astype("string")
            if len(nn) == 0:
                continue
            if self._is_multilabel(c, nn):
                cnt = Counter(t for lst in _tokenize(nn) for t in lst)
                vocab = [t for t, k in cnt.most_common(MAX_TOKENS_PER_COL)
                         if k >= TOKEN_MIN_COUNT and t not in ("other", "n_tokens")]
                if vocab:
                    self.multilabel[c] = vocab
                continue
            num = pd.to_numeric(nn, errors="coerce")
            if num.notna().mean() > 0.95:           # numbers stored as strings
                if num.nunique() > 1:
                    self.numeric.append(c)
                continue
            smp = nn.sample(min(500, len(nn)), random_state=RANDOM_STATE)
            if smp.str.contains(r"\d{4}-\d{2}-\d{2}|\d{2}[/-]\d{2}[/-]\d{4}", regex=True).mean() > 0.9 \
                    and _to_dt(smp).notna().mean() > 0.9:   # dates stored as strings -> diffs, never raw
                if c != self.date_base:
                    self.dates.append(c)
                continue
            vc = nn.str.strip().value_counts()
            if len(vc) <= MAX_ONEHOT_CARD:
                lv = [x for x, k in vc.items() if k >= TOKEN_MIN_COUNT]
                if lv:
                    self.onehot[c] = lv
            else:
                self.freq[c] = (vc / n).to_dict()
        # tokens that appear in >= 2 columns of the cat_i family -> anycat__<token>
        fam = [c for c in self.multilabel if any(re.search(p, c) for p in MULTILABEL_FAMILY_PATTERNS)]
        tok_cols = Counter(t for c in fam for t in self.multilabel[c])
        self.family = fam
        self.union_tokens = sorted(t for t, k in tok_cols.items() if k >= 2)
        X = self._raw(df)
        nun = X.nunique(dropna=True)
        keep = nun > 1
        idlike = [c for c in self.numeric if nun.get(c, 0) > 0.9 * n
                  and np.nanmax(np.abs(X[c].to_numpy() - np.round(X[c].to_numpy())), initial=0) == 0]
        if idlike:
            log(f"NOTE integer near-unique columns (IDs? add to DROP_COLS if so): {idlike[:15]}")
        self.feature_names = X.columns[keep].tolist()
        tok_all = {f"{c}__{t}" for c, v in self.multilabel.items() for t in v} | \
                  {f"anycat__{t}" for t in self.union_tokens}
        self.token_features = [f for f in self.feature_names if f in tok_all]
        self.token_key = {f: f.split("__", 1)[1] if f.startswith("anycat__")
                          else next(t for c, v in self.multilabel.items() for t in v if f == f"{c}__{t}")
                          for f in self.token_features}
        log(f"features: {len(self.numeric)} numeric | {len(self.multilabel)} multi-label cols "
            f"({sum(len(v) for v in self.multilabel.values())} tokens, {len(self.union_tokens)} anycat) | "
            f"{len(self.onehot)} one-hot | {len(self.freq)} freq-encoded | {len(self.dates)} date diffs "
            f"-> {len(self.feature_names)} total")
        log(f"multi-label columns: {sorted(self.multilabel)}")
        return self

    @staticmethod
    def _is_multilabel(c, nn):
        if any(re.search(p, c) for p in MULTILABEL_PATTERNS):
            return True
        if not MULTILABEL_AUTODETECT:
            return False
        return nn.str.contains(MULTILABEL_DELIM, regex=False).mean() >= MULTILABEL_MIN_SHARE

    def _raw(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.reset_index(drop=True)
        n = len(df)
        blocks = {}
        for c in self.numeric:
            blocks[c] = (pd.to_numeric(df[c], errors="coerce").astype(np.float32).to_numpy()
                         if c in df.columns else np.full(n, np.nan, np.float32))
        fam_hot = {}
        for c, vocab in self.multilabel.items():
            if c in df.columns:
                toks = _tokenize(df[c])
                vset = set(vocab)
                ex = toks.explode().dropna()
                ex_in = ex[ex.isin(vset)]
                if len(ex_in):
                    hot = pd.crosstab(ex_in.index, ex_in.values).reindex(index=range(n), columns=vocab, fill_value=0)
                    hot = (hot.to_numpy() > 0).astype(np.float32)
                else:
                    hot = np.zeros((n, len(vocab)), np.float32)
                other = ex[~ex.isin(vset)]
                oth = np.zeros(n, np.float32)
                oth[other.index.unique()] = 1
                ntok = toks.map(len).to_numpy().astype(np.float32)
                ntok[df[c].isna().to_numpy()] = np.nan
            else:
                hot = np.zeros((n, len(vocab)), np.float32)
                oth = np.zeros(n, np.float32)
                ntok = np.full(n, np.nan, np.float32)
            for j, t in enumerate(vocab):
                blocks[f"{c}__{t}"] = hot[:, j]
            blocks[f"{c}__other"] = oth
            blocks[f"{c}__n_tokens"] = ntok
            if c in self.family:
                fam_hot[c] = dict(zip(vocab, hot.T))
        for t in self.union_tokens:
            arrs = [fam_hot[c][t] for c in self.family if t in fam_hot.get(c, {})]
            blocks[f"anycat__{t}"] = np.max(np.vstack(arrs), axis=0)
        for c, levels in self.onehot.items():
            v = df[c].astype("string").str.strip() if c in df.columns else pd.Series([pd.NA] * n)
            for lv in levels:
                blocks[f"{c}=={lv}"] = (v == lv).fillna(False).to_numpy().astype(np.float32)
        for c, fmap in self.freq.items():
            v = df[c].astype("string").str.strip() if c in df.columns else pd.Series([pd.NA] * n)
            blocks[f"{c}__freq"] = v.map(fmap).astype(float).fillna(0).to_numpy().astype(np.float32)
        if self.date_base and self.date_base in df.columns:
            base = _to_dt(df[self.date_base])
            for c in self.dates:
                if c in df.columns:
                    blocks[f"days_{self.date_base}_minus_{c}"] = ((base - _to_dt(df[c])).dt.days
                                                                   .astype(float).to_numpy().astype(np.float32))
                else:
                    blocks[f"days_{self.date_base}_minus_{c}"] = np.full(n, np.nan, np.float32)
        return pd.DataFrame(blocks)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        X = self._raw(df)
        X = X.reindex(columns=self.feature_names)
        return X.to_numpy(np.float32)


# =====================================================================================
# MODELS + SHAP
# =====================================================================================
def new_hgb():
    return HistGradientBoostingClassifier(
        learning_rate=0.06, max_iter=400, max_leaf_nodes=31, min_samples_leaf=40,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=30, random_state=RANDOM_STATE)


def shap_matrix(explainer, X, chunk=5000):
    out = []
    for i in range(0, len(X), chunk):
        sv = explainer.shap_values(X[i:i + chunk], check_additivity=False)
        if isinstance(sv, list):
            sv = sv[1]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[..., 1]
        out.append(sv.astype(np.float32))
    return np.vstack(out) if out else np.zeros((0, X.shape[1]), np.float32)


def fit_stage(name, X, y, names):
    rng = np.random.default_rng(RANDOM_STATE)
    m0 = new_hgb().fit(X, y)
    idx = rng.choice(len(X), size=min(SHAP_SELECTION_SAMPLE, len(X)), replace=False)
    imp = np.abs(shap_matrix(shap.TreeExplainer(m0), X[idx])).mean(0)
    k = int(max(1, min(MAX_MODEL_FEATURES, (imp > 0).sum())))
    sel = np.sort(np.argsort(-imp)[:k])
    m = new_hgb().fit(X[:, sel], y)
    log(f"Stage {name}: {k} features kept (of {X.shape[1]}), trained on {len(y):,} rows "
        f"({int(y.sum()):,} fraud), {m.n_iter_} iterations")
    return {"name": name, "model": m, "feat_idx": sel, "explainer": shap.TreeExplainer(m),
            "feat_names": [names[i] for i in sel]}


def stage_scores(stage, X):
    return stage["model"].predict_proba(X[:, stage["feat_idx"]])[:, 1]


def recall_at_budget(score, y):
    order = np.argsort(-score)
    yy = y[order]
    cf, cg = np.cumsum(yy == 1), np.cumsum(yy == 0)
    gpf = cg * GOODS_WEIGHT / np.maximum(cf, 1)
    ok = np.where((gpf <= GOODS_PER_FRAUD_BUDGET) & (cf > 0))[0]
    if len(ok) == 0:
        return 0.0, np.nan
    i = ok[-1]
    return cf[i] / max((y == 1).sum(), 1), score[order][i]


# =====================================================================================
# CLUSTERING
# =====================================================================================
class Clusterer:
    def __init__(self, mcs, ms, method="eom"):
        self.mcs, self.ms, self.method = int(mcs), int(ms), method

    def fit(self, emb):
        self.use_hdbscan = HAVE_HDBSCAN
        if self.use_hdbscan:
            try:
                self.m = hdbscan.HDBSCAN(min_cluster_size=self.mcs, min_samples=self.ms,
                                         cluster_selection_method=self.method, prediction_data=True).fit(emb)
            except TypeError as e:   # old hdbscan + new sklearn (force_all_finite) -> use sklearn's
                log(f"    hdbscan package failed ({e}); falling back to sklearn HDBSCAN")
                self.use_hdbscan = False
        if not self.use_hdbscan:
            from sklearn.cluster import HDBSCAN as _SkHDBSCAN
            self.m = _SkHDBSCAN(min_cluster_size=self.mcs, min_samples=self.ms,
                                cluster_selection_method=self.method).fit(emb)
            self.knn = KNeighborsClassifier(15).fit(emb, self.m.labels_)
        self.labels_ = self.m.labels_
        return self

    def predict(self, emb):
        if len(emb) == 0:
            return np.zeros(0, int)
        if self.use_hdbscan:
            return hdbscan.approximate_predict(self.m, emb)[0]
        return self.knn.predict(emb)


class RunClusterer:
    """Top-level HDBSCAN + re-clustering of oversized families; string labels C03 / C03.1 / noise."""

    def fit(self, emb):
        n = len(emb)
        self.mcs = max(HDB_MIN_CLUSTER_ABS, int(HDB_MIN_CLUSTER_FRAC * n))
        self.top = Clusterer(self.mcs, HDB_MIN_SAMPLES).fit(emb)
        lab = self.top.labels_
        self.subs = {}
        for k in sorted(set(lab) - {-1}):
            m = lab == k
            if m.mean() > SUBCLUSTER_FRAC and m.sum() >= 4 * self.mcs:
                sc = Clusterer(max(self.mcs // 2, 20), HDB_MIN_SAMPLES, "leaf").fit(emb[m])
                if len(set(sc.labels_) - {-1}) >= 2:
                    self.subs[k] = sc
                    log(f"    family C{k:02d} ({m.mean():.0%} of rows) split into "
                        f"{len(set(sc.labels_) - {-1})} subclusters")
        self.labels_ = self._names(lab, emb, fitted=True)
        return self

    def _names(self, lab, emb, fitted=False):
        out = np.array(["noise" if k == -1 else f"C{k:02d}" for k in lab], dtype=object)
        for k, sc in self.subs.items():
            m = lab == k
            if m.any():
                sl = sc.labels_ if fitted else sc.predict(emb[m])
                out[m] = [f"C{k:02d}.{s}" if s != -1 else f"C{k:02d}.x" for s in sl]
        return out

    def predict(self, emb):
        return self._names(self.top.predict(emb), emb)

    def top_level(self, names):
        return np.array([n.split(".")[0] for n in names], dtype=object)


def quantile_norm(Z, ecdf):
    out = np.empty_like(Z)
    for j in range(Z.shape[1]):
        out[:, j] = np.searchsorted(ecdf[:, j], Z[:, j], side="right") / len(ecdf)
    return out


# =====================================================================================
# RULES
# =====================================================================================
def leaf_bounds(tree):
    t = tree.tree_
    out = {}

    def rec(node, b):
        if t.children_left[node] == -1:
            out[node] = b
            return
        f, thr = int(t.feature[node]), float(t.threshold[node])
        lo, hi = b.get(f, (-np.inf, np.inf))
        bl = dict(b); bl[f] = (lo, min(hi, thr)); rec(t.children_left[node], bl)
        br = dict(b); br[f] = (max(lo, thr), hi); rec(t.children_right[node], br)

    rec(0, {})
    return out


def apply_rule(bounds, Xr):
    m = np.ones(len(Xr), bool)
    for f, (lo, hi) in bounds.items():
        m &= (Xr[:, f] > lo) & (Xr[:, f] <= hi)
    return m


def render_rule(bounds, names, is_bin, has_nan):
    """Bounds are (lo, hi] on the SENTINEL-filled matrix, so NaN sits at -1e9 (below every value)."""
    if bounds is None:
        return ""
    parts = []
    for f, (lo, hi) in sorted(bounds.items(), key=lambda kv: names[kv[0]]):
        n = names[f]
        null_in = lo == -np.inf and has_nan[f]          # NULL rows satisfy this condition
        if hi < SENTINEL / 2:
            parts.append(f"{n} IS NULL")
        elif is_bin[f]:
            if lo >= 0:
                parts.append(f"{n} = 1")
            elif hi < 1:
                parts.append(f"({n} = 0 OR {n} IS NULL)" if null_in else f"{n} = 0")
            else:
                parts.append(f"{n} IS NOT NULL")
        elif lo == -np.inf:
            parts.append(f"({n} <= {hi:.4g} OR {n} IS NULL)" if null_in else f"{n} <= {hi:.4g}")
        elif lo < SENTINEL / 2:
            parts.append(f"{n} IS NOT NULL" if hi == np.inf else f"{n} <= {hi:.4g}")
        else:
            parts.append(f"{n} > {lo:.4g}" + ("" if hi == np.inf else f" AND {n} <= {hi:.4g}"))
    return " AND ".join(parts)


def best_leaf(Xr_fit, target, min_support, mode):
    """mode 'cluster': max member recall within budget; 'peel': max fraud within budget.
    If no leaf of a class-balanced tree fits the goods budget, retry with trees that penalise goods
    harder (matters once GOODS_WEIGHT reflects production volumes)."""
    P, N = max(int(target.sum()), 1), max(int((~target).sum()), 1)
    fallback = []
    for k in (1, 4, 16, 64):
        cw = {0: k * P / N, 1: 1.0}
        tree = DecisionTreeClassifier(max_depth=RULE_DEPTH, min_samples_leaf=max(5, min_support // 2),
                                      class_weight=cw, random_state=RANDOM_STATE).fit(Xr_fit, target)
        leaf_id = tree.apply(Xr_fit)
        within = []
        for node, b in leaf_bounds(tree).items():
            m = leaf_id == node
            pos, neg = int((target & m).sum()), int((~target & m).sum())
            if pos < min_support or not b:
                continue
            gpf = neg * GOODS_WEIGHT / pos
            prec = pos / (pos + neg * GOODS_WEIGHT)
            (within if gpf <= GOODS_PER_FRAUD_BUDGET else fallback).append((pos, prec, b))
        if within:
            return max(within, key=lambda r: r[0])[2], True
    if mode == "peel" or not fallback:
        return None, False
    return max(fallback, key=lambda r: (r[0] / P) * r[1])[2], False


def hit_metrics(mule_no, good_no, mule_total, good_total, prefix=""):
    """Mule / good counts, shares, hit rate and lift for one group of rows.
    hit_rate  = mules / (mules + goods) in this sample
    lift      = hit_rate / overall hit rate of the sample (1 = no better than random)
    *_w       = same with goods scaled by GOODS_WEIGHT (production view)"""
    base = mule_total / max(mule_total + good_total, 1)
    hr = mule_no / max(mule_no + good_no, 1)
    hr_w = mule_no / max(mule_no + good_no * GOODS_WEIGHT, 1e-9)
    base_w = mule_total / max(mule_total + good_total * GOODS_WEIGHT, 1e-9)
    return {f"{prefix}mule_no": int(mule_no), f"{prefix}mule_pct": round(mule_no / max(mule_total, 1), 4),
            f"{prefix}good_no": int(good_no), f"{prefix}good_pct": round(good_no / max(good_total, 1), 4),
            f"{prefix}hit_rate": round(hr, 4), f"{prefix}lift": round(hr / max(base, 1e-9), 2),
            f"{prefix}goods_per_mule_w": round(good_no * GOODS_WEIGHT / max(mule_no, 1), 2),
            f"{prefix}hit_rate_w": round(hr_w, 4), f"{prefix}lift_w": round(hr_w / max(base_w, 1e-9), 2)}


def metrics_table(groups, seg_lab_f, seg_lab_g, prefix):
    """One row per group (segment / MO) + TOTAL, from per-row labels of mules and goods."""
    F, G = len(seg_lab_f), len(seg_lab_g)
    rows = {}
    for gname in groups:
        rows[gname] = hit_metrics(int((seg_lab_f == gname).sum()), int((seg_lab_g == gname).sum()), F, G, prefix)
    rows["TOTAL"] = hit_metrics(F, G, F, G, prefix)
    return rows


def rule_metrics(mask, y, members=None, prefix=""):
    f, g = int((mask & (y == 1)).sum()), int((mask & (y == 0)).sum())
    F, G = max(int((y == 1).sum()), 1), int((y == 0).sum())
    d = {f"{prefix}fraud_hit": f, f"{prefix}goods_hit": g,
         f"{prefix}goods_per_fraud": round(g * GOODS_WEIGHT / max(f, 1), 2),
         f"{prefix}lift": round(((f + 0.5) / (F + 1)) / ((g + 0.5) / (G + 1)), 2)}
    if members is not None:
        d[f"{prefix}cluster_recall"] = round((mask & members).sum() / max(members.sum(), 1), 3)
    return d


# =====================================================================================
# DISCOVERY
# =====================================================================================
def mo_label(cov, sig, top_mo, top_recall, near_mo, near_sim):
    if cov < NEW_COVERAGE_LO:
        return f"GAP:{near_mo}" if near_mo and near_sim >= GAP_SIMILARITY else "NEW"
    if sig:
        return ("ALIGNED:" if top_recall >= MO_ALIGN_RECALL else "SUBTYPE:") + "+".join(sig)
    if cov >= KNOWN_COVERAGE_HI:
        return "CROSS-MO"
    return f"PARTIAL:{top_mo}"


def known_mo_profiles(SA, M, fraud_mask):
    """Centred mean Stage-A SHAP vector per known MO (train fraud)."""
    mu = SA[fraud_mask].mean(0)
    prof = {}
    for mo in KNOWN_MOS:
        if mo in UNCOVERED_MOS:
            continue
        m = fraud_mask & (M[:, list(KNOWN_MOS).index(mo)] == 1)
        if m.sum() >= 20:
            prof[mo] = SA[m].mean(0) - mu
    return mu, prof


def nearest_mo(vec, prof):
    best, sim = "", 0.0
    for mo, v in prof.items():
        s = float(vec @ v / (np.linalg.norm(vec) * np.linalg.norm(v) + 1e-12))
        if s > sim:
            best, sim = mo, s
    return best, round(sim, 3)


def run_pipeline():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    if GOODS_WEIGHT == 1.0:
        log("WARNING GOODS_WEIGHT=1.0: goods are treated as the full population, so every "
            "goods-per-fraud / budget check is optimistic if the 37k goods are a sample")

    # ---------------- data, MOs, split ----------------
    df = tag_known_mos(load_data())
    is_val = make_split(df)
    df["_split"] = np.where(is_val, "val", "train")
    summ = (df.assign(group=np.where(df["_label"] == 1, "fraud:" + df["_mo_set"].astype(str), "goods"))
              .groupby(["group", "_split"]).size().unstack(fill_value=0))
    summ.to_csv(f"{OUT_DIR}/split_summary.csv")
    log("split summary (rows):\n" + summ.to_string())
    report_mo_overlap(df)
    fr = df["_label"] == 1
    log(f"known-MO coverage of fraud: {df.loc[fr, '_covered'].mean():.1%}  | "
        f"rows matching >1 MO: {(df.loc[fr, '_mo_count'] > 1).mean():.1%}")

    reserved = set(DROP_COLS) | {ACCOUNT_COL, LABEL_COL}
    if EXCLUDE_FLAGS_FROM_MODEL:
        reserved |= set(FLAG_COLS)
    tr = df[~is_val].reset_index(drop=True)
    va = df[is_val].reset_index(drop=True)
    pre = Preprocessor().fit(tr, reserved)
    X_tr, X_va = pre.transform(tr), pre.transform(va)
    y_tr, y_va = tr["_label"].to_numpy(), va["_label"].to_numpy()

    # ---------------- leakage check: single features that separate fraud/goods on their own ----------
    Xr_tr = np.nan_to_num(X_tr, nan=SENTINEL)
    auc = np.array([roc_auc_score(y_tr, Xr_tr[:, j]) for j in range(Xr_tr.shape[1])])
    sep = np.abs(auc - 0.5)
    leak_df = pd.DataFrame({"feature": pre.feature_names, "univariate_auc": auc.round(4),
                            "null_rate_fraud": np.isnan(X_tr[y_tr == 1]).mean(0).round(3),
                            "null_rate_goods": np.isnan(X_tr[y_tr == 0]).mean(0).round(3)})
    leak_df.iloc[np.argsort(-sep)[:60]].to_csv(f"{OUT_DIR}/leakage_check.csv", index=False)
    leaky = sep >= LEAKAGE_AUC - 0.5
    if leaky.any():
        bad = [pre.feature_names[j] for j in np.where(leaky)[0]]
        log(f"WARNING {len(bad)} features separate fraud/goods almost perfectly on their own: {bad[:15]}"
            + (" -> DROPPED" if DROP_LEAKY_FEATURES else " -> kept (DROP_LEAKY_FEATURES=False)"))
        if DROP_LEAKY_FEATURES:
            keep = ~leaky
            X_tr, X_va, Xr_tr = X_tr[:, keep], X_va[:, keep], Xr_tr[:, keep]
            pre.feature_names = [f for f, k in zip(pre.feature_names, keep) if k]
            kept = set(pre.feature_names)
            pre.token_features = [f for f in pre.token_features if f in kept]
    names = pre.feature_names
    Xr_va = np.nan_to_num(X_va, nan=SENTINEL)
    is_bin = np.array([np.isin(X_tr[~np.isnan(X_tr[:, j]), j], (0.0, 1.0)).all() for j in range(X_tr.shape[1])])
    has_nan = np.isnan(X_tr).any(0) | np.isnan(X_va).any(0)
    tok_idx_all = np.array([names.index(f) for f in pre.token_features], dtype=int)
    cov_tr, cov_va = tr["_covered"].to_numpy(bool), va["_covered"].to_numpy(bool)
    mo_cols = [f"_mo__{n}" for n in KNOWN_MOS]
    M_tr, M_va = tr[mo_cols].to_numpy().astype(int), va[mo_cols].to_numpy().astype(int)
    real_mo = [i for i, n in enumerate(KNOWN_MOS) if n not in UNCOVERED_MOS]

    # ---------------- stages ----------------
    stages = {"A": fit_stage("A (all fraud vs goods)", X_tr, y_tr, names)}
    resid_tr = (y_tr == 1) & ~cov_tr
    run_residual = any(r[1] == "B" for r in RUNS) and resid_tr.sum() >= MIN_RESIDUAL_FRAUD
    if run_residual:
        mB = (y_tr == 0) | resid_tr
        stages["B"] = fit_stage("B (uncovered fraud vs goods)", X_tr[mB], y_tr[mB], names)
    elif any(r[1] == "B" for r in RUNS):
        log(f"only {resid_tr.sum()} uncovered train fraud (< {MIN_RESIDUAL_FRAUD}) -> residual runs skipped")

    metrics = []
    for key, st in stages.items():
        mtr = np.ones(len(y_tr), bool) if key == "A" else ((y_tr == 0) | resid_tr)
        mva = np.ones(len(y_va), bool) if key == "A" else ((y_va == 0) | ((y_va == 1) & ~cov_va))
        s = stage_scores(st, X_va[mva])
        rec, thr = recall_at_budget(s, y_va[mva])
        metrics.append({"stage": key, "val_roc_auc": roc_auc_score(y_va[mva], s),
                        "val_pr_auc": average_precision_score(y_va[mva], s),
                        "val_recall_at_budget": rec, "val_threshold_at_budget": thr,
                        "train_rows": int(mtr.sum()), "val_rows": int(mva.sum())})
        log(f"Stage {key} VAL: ROC-AUC {metrics[-1]['val_roc_auc']:.4f}  PR-AUC {metrics[-1]['val_pr_auc']:.4f}  "
            f"recall@budget {rec:.1%}")
        if metrics[-1]["val_roc_auc"] > 0.995:
            log(f"WARNING Stage {key} is near-perfect on validation - check leakage_check.csv and the top "
                f"features ({st['feat_names'][:5]}...) for columns that only exist / differ by extraction")
    pd.DataFrame(metrics).to_csv(f"{OUT_DIR}/model_metrics.csv", index=False)

    # Stage-A SHAP for every train fraud row: used for known-MO similarity (GAP detection)
    SA = np.zeros((len(y_tr), len(stages["A"]["feat_idx"])), np.float32)
    SA[y_tr == 1] = shap_matrix(stages["A"]["explainer"], X_tr[y_tr == 1][:, stages["A"]["feat_idx"]])
    mu_A, mo_prof = known_mo_profiles(SA, M_tr, y_tr == 1)
    log(f"known-MO SHAP profiles built for {list(mo_prof)}")

    dates_tr = _to_dt(tr[DATE_COL]) if DATE_COL in tr.columns else None
    state = {"pre": pre, "stages": stages, "runs": {}, "names": names, "is_bin": is_bin,
             "config": {k: v for k, v in globals().items() if k.isupper()}}
    assignments_tr, assignments_va = {}, {}

    run = "all_shap"
    seg = segment_all_fraud(run, stages["A"], X_tr, X_va, Xr_tr, Xr_va, y_tr, y_va, SA, M_tr, M_va,
                            cov_tr, cov_va, mu_A, mo_prof, real_mo, names, is_bin, has_nan, tok_idx_all,
                            tr, dates_tr)
    state["runs"][run] = seg["state"]
    assignments_tr[run] = (seg["rows_tr"], seg["final_tr"])
    assignments_va[run] = (seg["rows_va"], seg["final_va"])
    seg["profiles"].to_csv(f"{OUT_DIR}/segment_profiles.csv", index=False)

    known_mo_report(tr, va, y_tr, y_va, assignments_tr)
    mine_token_combos(pre, X_tr, X_va, y_tr, y_va, y_tr == 1, y_va == 1, names, M_tr)
    write_assignments(tr, va, stages, X_tr, X_va, assignments_tr, assignments_va, state)

    with open(f"{CACHE_DIR}/pipeline_state.pkl", "wb") as f:
        pickle.dump(state, f)
    log(f"done. outputs in {OUT_DIR}/, fitted objects in {CACHE_DIR}/")


# =====================================================================================
# SEGMENTATION (single all_shap run)
# =====================================================================================
def merge_clusters(Z, lab, sim_thr):
    """Greedy average-linkage merge of clusters whose centred centroids point the same way.
    Returns {raw_label: merged_label}; -1 stays -1."""
    mu = Z.mean(0)
    groups = {k: [k] for k in sorted(set(lab) - {-1})}
    while len(groups) > 1:
        keys = list(groups)
        C = np.array([Z[np.isin(lab, groups[k])].mean(0) - mu for k in keys])
        Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
        sim = Cn @ Cn.T
        np.fill_diagonal(sim, -1)
        i, j = np.unravel_index(np.argmax(sim), sim.shape)
        if sim[i, j] < sim_thr:
            break
        groups[keys[i]] += groups.pop(keys[j])
    mapping = {-1: -1}
    for g, members in groups.items():
        for k in members:
            mapping[k] = g
    return mapping


def mo_residualize(Z, M, coef=None):
    """Linear 'conditioning' on the known MOs: Z - [1, M] @ B (B fitted on train fraud)."""
    M1 = np.hstack([np.ones((len(M), 1)), M.astype(np.float32)])
    if coef is None:
        coef = np.linalg.lstsq(M1, Z, rcond=None)[0]
    return (Z - M1 @ coef).astype(np.float32), coef


def separability(Rtr, Rva, t_tr, t_va, names, is_bin, has_nan):
    """Can a depth-3 tree on raw features tell this segment from the OTHER fraud?
    Returns val AUC and the best single-leaf rule (by F1) with its train/val precision & recall."""
    out = {"sep_val_auc": np.nan, "rule_vs_fraud": ""}
    if t_tr.sum() < RULE_MIN_SUPPORT or (~t_tr).sum() < RULE_MIN_SUPPORT:
        return out
    tree = DecisionTreeClassifier(max_depth=RULE_DEPTH, min_samples_leaf=20, class_weight="balanced",
                                  random_state=RANDOM_STATE).fit(Rtr, t_tr)
    if t_va.any() and (~t_va).any():
        out["sep_val_auc"] = round(roc_auc_score(t_va, tree.predict_proba(Rva)[:, 1]), 3)
    leaf = tree.apply(Rtr)
    best, best_f1 = None, -1
    for node, b in leaf_bounds(tree).items():
        m = leaf == node
        tp = (m & t_tr).sum()
        if tp == 0 or not b:
            continue
        f1 = 2 * tp / (m.sum() + t_tr.sum())
        if f1 > best_f1:
            best, best_f1 = b, f1
    if best is not None:
        out["rule_vs_fraud"] = render_rule(best, names, is_bin, has_nan)
        for nm, R, t in [("train", Rtr, t_tr), ("val", Rva, t_va)]:
            m = apply_rule(best, R)
            out[f"{nm}_rule_vs_fraud_precision"] = round((m & t).sum() / max(m.sum(), 1), 3)
            out[f"{nm}_rule_vs_fraud_recall"] = round((m & t).sum() / max(t.sum(), 1), 3)
    return out


def gmm_assign(gmm, P, ll_cut):
    post = gmm.predict_proba(P)
    ok = (post.max(1) >= MIN_MEMBERSHIP) & (gmm.score_samples(P) >= ll_cut)
    return np.where(ok, post.argmax(1), -1)


def segment_all_fraud(run, st, X_tr, X_va, Xr_tr, Xr_va, y_tr, y_va, SA, M_tr, M_va, cov_tr, cov_va,
                      mu_A, mo_prof, real_mo, names, is_bin, has_nan, tok_idx_all, tr, dates_tr):
    log(f"===== {run}: segmenting ALL fraud =====")
    pool_tr, pool_va = y_tr == 1, y_va == 1
    rows_tr, rows_va = np.where(pool_tr)[0], np.where(pool_va)[0]
    S = SA[pool_tr]
    imp = np.abs(S).mean(0)
    order = np.argsort(-imp)
    n_top = int(np.searchsorted(np.cumsum(imp[order]) / max(imp.sum(), 1e-12), SHAP_CUM_SHARE) + 1)
    top = order[:int(np.clip(n_top, TOP_SHAP_MIN, TOP_SHAP_FOR_CLUSTERING))]
    Z = S[:, top]
    Zv = shap_matrix(st["explainer"], X_va[pool_va][:, st["feat_idx"]])[:, top]
    coef = None
    if CONDITION_ON_MOS:
        Z, coef = mo_residualize(Z, M_tr[pool_tr][:, real_mo])
        Zv, _ = mo_residualize(Zv, M_va[pool_va][:, real_mo], coef)
        log("  SHAP vectors conditioned on the known MOs (MO effects removed)")
    log(f"  clustering on {len(top)} SHAP features ({SHAP_CUM_SHARE:.0%} of total |SHAP|)")

    dup = 1 - len(np.unique(np.round(Z, 6), axis=0)) / len(Z)
    log(f"  {dup:.0%} of train fraud rows share an identical SHAP profile with another row")
    pca = PCA(random_state=RANDOM_STATE).fit(Z)
    dim = int(np.clip(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), PCA_VAR) + 1, 2, PCA_MAX_DIM))
    pca = PCA(n_components=dim, random_state=RANDOM_STATE).fit(Z)
    P, Pv = pca.transform(Z), pca.transform(Zv)
    log(f"  PCA: {dim} components ({pca.explained_variance_ratio_.sum():.0%} of SHAP variance)")

    # choose k by reproducibility: fit on train, assign val, re-fit on val alone, compare (ARI)
    rows_k = []
    for k in K_RANGE:
        g = GaussianMixture(k, covariance_type="full", reg_covar=GMM_REG, n_init=3,
                            random_state=RANDOM_STATE).fit(P)
        shares = np.bincount(g.predict(P), minlength=k) / len(P)
        gv = GaussianMixture(k, covariance_type="full", reg_covar=GMM_REG, n_init=3,
                             random_state=RANDOM_STATE).fit(Pv)
        ari = adjusted_rand_score(g.predict(Pv), gv.predict(Pv))
        rows_k.append({"k": k, "val_stability_ari": round(ari, 3), "bic": round(g.bic(P), 1),
                       "min_segment_share": round(shares.min(), 4),
                       "eligible": bool(shares.min() >= MIN_SEGMENT_FRAC)})
    ksel = pd.DataFrame(rows_k)
    elig = ksel[ksel["eligible"]] if ksel["eligible"].any() else ksel
    stable = elig[elig["val_stability_ari"] >= K_MIN_STABILITY]
    if len(stable):
        k_best = int(stable.sort_values("bic").iloc[0]["k"])
    else:
        k_best = int(elig.sort_values("val_stability_ari", ascending=False).iloc[0]["k"])
        log(f"  WARNING no k reproduces with ARI >= {K_MIN_STABILITY}: the fraud has no stable grouping in "
            f"SHAP space; using the most stable k={k_best} - treat its segments with caution")
    ksel["chosen"] = ksel["k"] == k_best
    ksel.to_csv(f"{OUT_DIR}/k_selection.csv", index=False)
    log("  k selection:\n" + ksel.to_string(index=False))
    gmm = GaussianMixture(k_best, covariance_type="full", reg_covar=GMM_REG, n_init=5,
                          random_state=RANDOM_STATE).fit(P)
    ll_cut = float(np.quantile(gmm.score_samples(P), MIN_LOGLIK_Q))
    raw, raw_v = gmm_assign(gmm, P, ll_cut), gmm_assign(gmm, Pv, ll_cut)
    sizes = Counter(raw[raw != -1])
    seg_name = {-1: "UNIDENTIFIED"}
    for i, (k, _) in enumerate(sizes.most_common()):
        seg_name[k] = f"S{i + 1:02d}"
    lab = np.array([seg_name.get(k, "UNIDENTIFIED") for k in raw], dtype=object)
    lab_v = np.array([seg_name.get(k, "UNIDENTIFIED") for k in raw_v], dtype=object)
    log(f"  k={k_best} segments; rows below membership {MIN_MEMBERSHIP}: train {np.mean(raw == -1):.1%}, "
        f"val {np.mean(raw_v == -1):.1%}")
    # independent re-fit on VAL with the same k -> per-segment reproducibility (Jaccard)
    refit = GaussianMixture(k_best, covariance_type="full", reg_covar=GMM_REG, n_init=5,
                            random_state=RANDOM_STATE).fit(Pv).predict(Pv)

    # raw-feature space for rules: stage features + token features
    rcols = np.union1d(st["feat_idx"], tok_idx_all).astype(int)
    Rtr, Rva = Xr_tr[:, rcols], Xr_va[:, rcols]
    rnames, rbin, rnan = [names[i] for i in rcols], is_bin[rcols], has_nan[rcols]
    Rtr_f, Rva_f = Rtr[pool_tr], Rva[pool_va]
    sc_tr = stage_scores(st, X_tr)
    if dates_tr is not None:
        recent_cut = dates_tr[pool_tr].quantile(0.75)
        overall_recent = (dates_tr[pool_tr] > recent_cut).mean()

    mo_names = list(KNOWN_MOS)
    profiles = []
    for c in sorted(set(lab) - {"UNIDENTIFIED"}):
        t_tr, t_va = lab == c, lab_v == c
        mem = np.zeros(len(y_tr), bool); mem[rows_tr[t_tr]] = True
        mem_v = np.zeros(len(y_va), bool); mem_v[rows_va[t_va]] = True
        n, nv = int(mem.sum()), int(mem_v.sum())
        p = {"segment": c, "n_train": n, "n_val": nv,
             "share_train": round(n / pool_tr.sum(), 4), "share_val": round(nv / max(pool_va.sum(), 1), 4)}
        p["val_size_ratio"] = round(p["share_val"] / max(p["share_train"], 1e-9), 3)
        # --- significance ---
        p.update(separability(Rtr_f, Rva_f, t_tr, t_va, rnames, rbin, rnan))
        if refit is not None:
            best = 0.0
            for k in set(refit) - {-1}:
                B = refit == k
                best = max(best, (t_va & B).sum() / max((t_va | B).sum(), 1))
            p["val_recluster_jaccard"] = round(best, 3)
        lo, hi = VAL_SIZE_RATIO_RANGE
        checks = {"size": p["share_train"] >= MIN_SEGMENT_FRAC,
                  "separable": (p["sep_val_auc"] or 0) >= SIG_MIN_SEPARABILITY_AUC,
                  "reproducible": p.get("val_recluster_jaccard", 1.0) >= SIG_MIN_JACCARD,
                  "stable_size": lo <= p["val_size_ratio"] <= hi}
        p["significant"] = all(checks.values())
        p["failed_checks"] = ",".join(k for k, v in checks.items() if not v)
        # --- known-MO overlay (each MO on its own) ---
        cov = float(cov_tr[mem].mean())
        share = M_tr[mem].mean(0)
        recall = M_tr[mem].sum(0) / np.maximum(M_tr[pool_tr].sum(0), 1)
        base = M_tr[pool_tr].mean(0)
        sig = [mo_names[i] for i in real_mo if share[i] >= MO_PRECISION]
        ti = max(real_mo, key=lambda i: share[i]) if real_mo else None
        top_mo = mo_names[ti] if ti is not None else ""
        top_rec = float(recall[ti]) if ti is not None else 0.0
        near, sim = nearest_mo(SA[mem].mean(0) - mu_A, mo_prof)
        p.update({"mo_label": mo_label(cov, sig, top_mo, top_rec, near, sim),
                  "known_mo_coverage": round(cov, 3), "mo_signature": "+".join(sig), "top_mo": top_mo,
                  "top_mo_share": round(float(share[ti]), 3) if ti is not None else 0.0,
                  "top_mo_recall": round(top_rec, 3),
                  "mos_enriched": "; ".join(f"{mo_names[i]}:{share[i]:.0%}(x{share[i] / max(base[i], 1e-9):.1f})"
                                            for i in np.argsort(-share) if share[i] >= 0.2),
                  "nearest_known_mo": near, "nearest_mo_similarity": sim,
                  "stage_score_mean": round(float(sc_tr[mem].mean()), 4)})
        p["new_mo"] = bool(p["significant"] and p["mo_label"].startswith(CANDIDATE_LABELS))
        if dates_tr is not None:
            p["emergence"] = round(float((dates_tr[mem] > recent_cut).mean() / max(overall_recent, 1e-9)), 2)
        # --- detection rule vs goods (how alertable the segment is) ---
        fit_rows = mem | (y_tr == 0)
        rb, within = best_leaf(Rtr[fit_rows], mem[fit_rows], max(RULE_MIN_SUPPORT, int(0.05 * n)), "cluster")
        p["rule_vs_goods"] = render_rule(rb, rnames, rbin, rnan)
        p["rule_vs_goods_within_budget"] = within
        if rb is not None:
            p.update(rule_metrics(apply_rule(rb, Rtr), y_tr, mem, "train_rule_"))
            p.update(rule_metrics(apply_rule(rb, Rva), y_va, mem_v, "val_rule_"))
        for i, mo in enumerate(mo_names):
            p[f"share_{mo}"] = round(float(share[i]), 3)
            p[f"recall_{mo}"] = round(float(recall[i]), 3)
        profiles.append(p)

    prof = pd.DataFrame(profiles)
    sig_set = set(prof.loc[prof["significant"], "segment"]) if len(prof) else set()

    # ---- goods through the same frozen pipeline -> mule / good metrics per segment, train & val ----
    def goods_labels(X, M, y):
        g = y == 0
        Zg = shap_matrix(st["explainer"], X[g][:, st["feat_idx"]])[:, top]
        if coef is not None:
            Zg, _ = mo_residualize(Zg, M[g][:, real_mo], coef)
        return np.array([seg_name.get(k, "UNIDENTIFIED") for k in gmm_assign(gmm, pca.transform(Zg), ll_cut)],
                        dtype=object)
    glab_tr, glab_va = goods_labels(X_tr, M_tr, y_tr), goods_labels(X_va, M_va, y_va)
    order = sorted(set(lab) - {"UNIDENTIFIED"}) + ["UNIDENTIFIED"]
    mt = metrics_table(order, lab, glab_tr, "train_")
    mv = metrics_table(order, lab_v, glab_va, "val_")
    met = pd.DataFrame([{"segment": g, **mt[g], **mv[g]} for g in order + ["TOTAL"]])
    if len(prof):
        met = met.merge(prof[["segment", "significant", "mo_label", "failed_checks"]], on="segment", how="left")
    met["val_lift_retention"] = (met["val_lift"] / met["train_lift"].replace(0, np.nan)).round(3)
    met.to_csv(f"{OUT_DIR}/segment_metrics.csv", index=False)
    log("  segment metrics (all segments, before significance filtering):\n" + met[[
        "segment", "train_mule_no", "train_mule_pct", "train_good_no", "train_good_pct", "train_hit_rate",
        "train_lift", "val_mule_no", "val_mule_pct", "val_good_no", "val_good_pct", "val_hit_rate",
        "val_lift"]].to_string(index=False))
    if len(prof):
        prof = prof.merge(met.drop(columns=["significant", "mo_label", "failed_checks"], errors="ignore"),
                          on="segment", how="left")
        if SIG_MIN_VAL_LIFT is not None:
            low = prof["val_lift"] < SIG_MIN_VAL_LIFT
            prof.loc[low, "failed_checks"] = (prof.loc[low, "failed_checks"].fillna("") + ",lift").str.strip(",")
            prof.loc[low, "significant"] = False
            prof["new_mo"] = prof["new_mo"] & prof["significant"]
            sig_set = set(prof.loc[prof["significant"], "segment"])
    final_tr = np.array([c if c in sig_set else "UNIDENTIFIED" for c in lab], dtype=object)
    final_va = np.array([c if c in sig_set else "UNIDENTIFIED" for c in lab_v], dtype=object)
    if len(prof):
        lead = ["segment", "significant", "new_mo", "mo_label", "failed_checks", "n_train", "n_val",
                "share_train", "share_val", "val_size_ratio", "sep_val_auc", "val_recluster_jaccard",
                "rule_vs_fraud", "train_rule_vs_fraud_precision", "train_rule_vs_fraud_recall",
                "val_rule_vs_fraud_precision", "val_rule_vs_fraud_recall", "mos_enriched", "known_mo_coverage",
                "mo_signature", "top_mo_recall", "nearest_known_mo", "nearest_mo_similarity", "rule_vs_goods",
                "rule_vs_goods_within_budget", "train_rule_goods_per_fraud", "val_rule_goods_per_fraud"]
        lead = [c for c in lead if c in prof.columns]
        prof = prof[lead + [c for c in prof.columns if c not in lead]].sort_values(
            ["significant", "share_train"], ascending=False)
    log(f"  {len(prof)} segments, {len(sig_set)} significant | UNIDENTIFIED: "
        f"train {np.mean(final_tr == 'UNIDENTIFIED'):.1%}, val {np.mean(final_va == 'UNIDENTIFIED'):.1%}")
    if len(prof):
        log("\n" + prof[[c for c in ["segment", "significant", "mo_label", "failed_checks", "share_train",
                                      "sep_val_auc", "val_recluster_jaccard"] if c in prof.columns]].to_string(index=False))

    plot_run(run, P[:, :2], final_tr, tr["_mo_set"].to_numpy()[pool_tr], axis_note="PCA 1 / PCA 2 of SHAP")
    signal_rates(run, np.where(final_tr == "UNIDENTIFIED", "noise", final_tr), rows_tr, X_tr, tr, y_tr,
                 pool_tr, names, is_bin, S, top, st)
    if RUN_PEELING:
        peel_segments(run, pool_tr, pool_va, Rtr, Rva, y_tr, y_va, cov_tr, M_tr, rnames, rbin, rnan)
    for fname, obj in [(f"pca_{run}.pkl", pca), (f"gmm_{run}.pkl", gmm),
                       (f"top_shap_features_{run}.pkl", top)]:
        with open(f"{CACHE_DIR}/{fname}", "wb") as f:
            pickle.dump(obj, f)
    with open(f"{CACHE_DIR}/hgb_model.pkl", "wb") as f:
        pickle.dump(st["model"], f)
    seg_state = {"stage": "A", "top": top, "coef": coef, "pca": pca, "gmm": gmm, "k": k_best, "ll_cut": ll_cut,
                 "seg_name": seg_name, "significant": sig_set,
                 "labels": dict(zip(prof["segment"], prof["mo_label"])) if len(prof) else {},
                 "new_mo": set(prof.loc[prof["new_mo"], "segment"]) if len(prof) else set()}
    return {"profiles": prof, "rows_tr": rows_tr, "rows_va": rows_va, "final_tr": final_tr,
            "final_va": final_va, "state": seg_state}


# =====================================================================================
# REPORTS
# =====================================================================================
def signal_rates(run, lab, rows_idx, X_tr, tr, y_tr, pool_tr, names, is_bin, S, top, st):
    bin_idx = np.where(is_bin)[0]
    B = np.nan_to_num(X_tr[:, bin_idx])
    flags = np.column_stack([tr[f"_flag_{a}"].to_numpy() for a in MO_FLAG_VARS]).astype(np.float32)
    B = np.hstack([B, flags])
    bnames = [names[i] for i in bin_idx] + [f"flag:{a}={c}" for a, c in MO_FLAG_VARS.items()]
    g_rate, f_rate = B[y_tr == 0].mean(0), B[pool_tr].mean(0)
    Bp = B[pool_tr]
    num_names = st["feat_names"]
    Xs_pool = X_tr[pool_tr][:, st["feat_idx"]]
    Xs_goods = X_tr[y_tr == 0][:, st["feat_idx"]]
    rows = []
    for c in sorted(set(lab) - {"noise"}):
        m = lab == c
        r = Bp[m].mean(0)
        lift = (r + 1e-3) / (g_rate + 1e-3)
        for j in np.argsort(-lift)[:25]:
            if r[j] >= 0.10:
                rows.append({"run": run, "cluster": c, "type": "binary", "signal": bnames[j],
                             "rate_cluster": round(r[j], 3), "rate_pool_fraud": round(f_rate[j], 3),
                             "rate_goods": round(g_rate[j], 4), "lift_vs_goods": round(lift[j], 2),
                             "lift_vs_fraud": round((r[j] + 1e-3) / (f_rate[j] + 1e-3), 2)})
        imp = np.abs(S[m][:, top]).mean(0)
        for j in top[np.argsort(-imp)[:8]]:
            rows.append({"run": run, "cluster": c, "type": "numeric", "signal": num_names[j],
                         "median_cluster": float(np.nanmedian(Xs_pool[m][:, j])),
                         "median_pool_fraud": float(np.nanmedian(Xs_pool[:, j])),
                         "median_goods": float(np.nanmedian(Xs_goods[:, j])),
                         "mean_abs_shap_cluster": float(np.abs(S[m][:, j]).mean())})
    pd.DataFrame(rows).to_csv(f"{OUT_DIR}/signal_rates_{run}.csv", index=False)


def peel_segments(run, pool_tr, pool_va, Xr_tr, Xr_va, y_tr, y_va, cov_tr, M_tr, names, is_bin, has_nan):
    rem_tr, rem_va = pool_tr.copy(), pool_va.copy()
    goods_tr = y_tr == 0
    segs, cum_tr, cum_va = [], np.zeros(len(y_tr), bool), np.zeros(len(y_va), bool)
    for s in range(PEEL_MAX_SEGMENTS):
        if rem_tr.sum() < RULE_MIN_SUPPORT:
            break
        rows = rem_tr | goods_tr
        b, _ = best_leaf(Xr_tr[rows], rem_tr[rows], RULE_MIN_SUPPORT, "peel")
        if b is None:
            break
        m_tr, m_va = apply_rule(b, Xr_tr), apply_rule(b, Xr_va)
        new_tr, new_va = m_tr & rem_tr, m_va & rem_va
        cum_tr |= m_tr; cum_va |= m_va
        seg = {"run": run, "segment": s + 1, "rule": render_rule(b, names, is_bin, has_nan),
               "train_new_fraud": int(new_tr.sum()),
               "train_new_fraud_uncovered_share": round(float((~cov_tr[new_tr]).mean()), 3),
               "train_new_fraud_mos": "; ".join(f"{list(KNOWN_MOS)[i]}:{v:.0%}" for i, v in
                                                sorted(enumerate(M_tr[new_tr].mean(0)), key=lambda t: -t[1])[:3]
                                                if v >= 0.1) if new_tr.any() else "",
               "val_new_fraud": int(new_va.sum())}
        seg.update(rule_metrics(m_tr, y_tr, prefix="train_"))
        seg.update(rule_metrics(m_va, y_va, prefix="val_"))
        seg["train_cum_pool_recall"] = round((cum_tr & pool_tr).sum() / max(pool_tr.sum(), 1), 3)
        seg["val_cum_pool_recall"] = round((cum_va & pool_va).sum() / max(pool_va.sum(), 1), 3)
        seg["train_cum_goods_per_fraud"] = round((cum_tr & goods_tr).sum() * GOODS_WEIGHT / max((cum_tr & pool_tr).sum(), 1), 2)
        seg["val_cum_goods_per_fraud"] = round((cum_va & (y_va == 0)).sum() * GOODS_WEIGHT / max((cum_va & pool_va).sum(), 1), 2)
        segs.append(seg)
        rem_tr &= ~m_tr; rem_va &= ~m_va
    pd.DataFrame(segs).to_csv(f"{OUT_DIR}/segments_{run}.csv", index=False)
    if segs:
        log(f"  peeled {len(segs)} segments: train recall {segs[-1]['train_cum_pool_recall']:.1%} / "
            f"val recall {segs[-1]['val_cum_pool_recall']:.1%} of pool")


def plot_run(run, emb, lab, prim, axis_note=None):
    top_sets = {k for k, _ in Counter(prim).most_common(12)}
    prim = np.array([x if x in top_sets else "other" for x in prim], dtype=object)
    try:
        e2 = emb if emb.shape[1] == 2 else umap.UMAP(n_components=2, n_neighbors=UMAP_NEIGHBORS, min_dist=0.1,
                                                      random_state=RANDOM_STATE).fit_transform(emb)
        fig, ax = plt.subplots(1, 2, figsize=(18, 8))
        for a, col, title in [(ax[0], lab, "HDBSCAN clusters"), (ax[1], prim, "Known MO combination (flags)")]:
            cats = sorted(set(col), key=lambda s: (s in ("noise", "NONE", "UNIDENTIFIED", "other"), s))
            cmap = plt.get_cmap("tab20")
            for i, k in enumerate(cats):
                m = col == k
                grey = k in ("noise", "NONE", "UNIDENTIFIED", "other")
                a.scatter(e2[m, 0], e2[m, 1], s=3, alpha=0.25 if grey else 0.6,
                          c="lightgrey" if grey else [cmap(i % 20)], label=k)
                if not grey and col is lab:
                    a.annotate(k, np.median(e2[m], 0), fontsize=8, weight="bold")
            a.set_title(f"{run}: {title}")
            if axis_note:
                a.set_xlabel(axis_note)
            a.legend(markerscale=4, fontsize=7, ncol=2, loc="best")
        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/umap_{run}.png", dpi=110)
        plt.close()
    except Exception as e:
        log(f"  plot failed: {e}")


def known_mo_report(tr, va, y_tr, y_va, assignments_tr):
    rows = []
    for mo in KNOWN_MOS:
        r = {"known_mo": mo, "expression": KNOWN_MOS[mo]}
        for nm, d, y in [("train", tr, y_tr), ("val", va, y_va)]:
            h = d[f"_mo__{mo}"].to_numpy().astype(bool)
            r.update(hit_metrics((h & (y == 1)).sum(), (h & (y == 0)).sum(), (y == 1).sum(), (y == 0).sum(), f"{nm}_"))
            r[f"{nm}_fraud_overlapping_other_mo"] = int((h & (y == 1) & (d["_mo_count"].to_numpy() > 1)).sum())
        if "all_shap" in assignments_tr:
            idx, lab = assignments_tr["all_shap"]
            m = tr[f"_mo__{mo}"].to_numpy()[idx] == 1
            if m.any():
                cnt = Counter(lab[m])
                tot = sum(cnt.values())
                r["all_shap_top_clusters"] = "; ".join(f"{k}:{v / tot:.0%}" for k, v in cnt.most_common(4))
                p = np.array(list(cnt.values())) / tot
                r["all_shap_concentration"] = round(float((p ** 2).sum()), 3)  # 1 = one cluster
                r["all_shap_clusters_with_10pct"] = int((p >= 0.10).sum())         # >1 -> MO has sub-types
        rows.append(r)
    pd.DataFrame(rows).to_csv(f"{OUT_DIR}/known_mo_report.csv", index=False)


def mine_token_combos(pre, X_tr, X_va, y_tr, y_va, pool_tr, pool_va, names, M_tr):
    """Token itemsets (size 1-3) enriched in fraud vs goods - a cheap, model-free second opinion
    (e.g. prescamSpike + Oddhours). mos_among_hits shows whether a combo sits inside one known MO
    or cuts across several (= candidate behaviour your flags don't describe)."""
    tok_idx = [names.index(f) for f in pre.token_features]
    if not tok_idx:
        return
    T = np.nan_to_num(X_tr[:, tok_idx]) > 0.5
    F, G = T[pool_tr].astype(np.float32), T[y_tr == 0].astype(np.float32)
    nF, nG = len(F), len(G)
    minsup = max(30, int(0.005 * nF))
    sup = F.sum(0)
    keep = np.argsort(-sup)[:80]
    keep = keep[sup[keep] >= minsup]
    key = [pre.token_key[f] for f in pre.token_features]
    res = []

    def add(items, fvec, gvec):
        f, g = int(fvec.sum()), int(gvec.sum())
        lift = (f / nF) / ((g + 1) / (nG + 1))
        if f >= minsup and lift >= 3:
            res.append({"items": tuple(items), "train_fraud_hit": f, "train_goods_hit": g,
                        "train_goods_per_fraud": round(g * GOODS_WEIGHT / f, 2), "train_lift": round(lift, 2)})

    for i in keep:
        add([i], F[:, i], G[:, i])
    pairs = []
    for a_i, i in enumerate(keep):
        for j in keep[a_i + 1:]:
            if key[i] == key[j]:
                continue
            fv, gv = F[:, i] * F[:, j], G[:, i] * G[:, j]
            if fv.sum() >= minsup:
                add([i, j], fv, gv)
                pairs.append((fv.sum(), i, j, fv, gv))
    for _, i, j, fv, gv in sorted(pairs, key=lambda t: -t[0])[:200]:
        for k in keep:
            if k in (i, j) or key[k] in (key[i], key[j]) or k < max(i, j):
                continue
            add([i, j, k], fv * F[:, k], gv * G[:, k])
    if not res:
        log("token combos: nothing enriched enough")
        return
    Tv = np.nan_to_num(X_va[:, tok_idx]) > 0.5
    out = []
    for r in res:
        m_tr = T[:, list(r["items"])].all(1)
        m_va = Tv[:, list(r["items"])].all(1)
        o = {"tokens": " + ".join(pre.token_features[i] for i in r["items"]), "size": len(r["items"])}
        o.update({k: v for k, v in r.items() if k != "items"})
        o["train_share_of_fraud"] = round(r["train_fraud_hit"] / nF, 4)
        fh = m_tr & pool_tr
        sh = M_tr[fh].mean(0) if fh.any() else np.zeros(M_tr.shape[1])
        o["mos_among_hits"] = "; ".join(f"{list(KNOWN_MOS)[i]}:{sh[i]:.0%}" for i in np.argsort(-sh)[:3] if sh[i] >= 0.1)
        o["hits_in_no_mo"] = round(float((M_tr[fh].sum(1) == 0).mean()), 3) if fh.any() else 0.0
        vf, vg = int((m_va & pool_va).sum()), int((m_va & (y_va == 0)).sum())
        o.update({"val_fraud_hit": vf, "val_goods_hit": vg,
                  "val_goods_per_fraud": round(vg * GOODS_WEIGHT / max(vf, 1), 2),
                  "val_lift": round((vf / max(pool_va.sum(), 1)) / ((vg + 1) / ((y_va == 0).sum() + 1)), 2)})
        o["within_budget_train_and_val"] = (o["train_goods_per_fraud"] <= GOODS_PER_FRAUD_BUDGET
                                            and o["val_goods_per_fraud"] <= GOODS_PER_FRAUD_BUDGET)
        out.append(o)
    out = pd.DataFrame(out).sort_values(["within_budget_train_and_val", "train_fraud_hit"], ascending=False)
    out.to_csv(f"{OUT_DIR}/token_combos.csv", index=False)
    log(f"token combos: {len(out)} enriched itemsets, {int(out["within_budget_train_and_val"].sum())} within budget")


def write_assignments(tr, va, stages, X_tr, X_va, assignments_tr, assignments_va, state):
    run = "all_shap"
    r = state["runs"][run]
    frames = []
    for nm, d, X, asg in [("train", tr, X_tr, assignments_tr), ("val", va, X_va, assignments_va)]:
        idx, lab = asg[run]
        out = pd.DataFrame({"split": nm, "known_mos": d["_mo_set"].to_numpy()[idx],
                            "known_mo_count": d["_mo_count"].to_numpy()[idx],
                            "stageA_score": stage_scores(stages["A"], X[idx]),
                            "segment": lab})
        if ACCOUNT_COL in d.columns:
            out.insert(0, ACCOUNT_COL, d[ACCOUNT_COL].to_numpy()[idx])
        out["segment_label"] = [r["labels"].get(c, "") for c in lab]
        out["new_mo_flag"] = out["segment"].isin(r["new_mo"]).astype(int)
        out["unidentified_flag"] = (out["segment"] == "UNIDENTIFIED").astype(int)
        frames.append(out)
    res = pd.concat(frames, ignore_index=True)
    res.to_parquet(f"{OUT_DIR}/fraud_assignments.parquet", index=False)
    counts = res.groupby(["segment", "split"]).size().unstack(fill_value=0)
    counts.insert(0, "segment_label", [r["labels"].get(c, "") for c in counts.index])
    counts.to_csv(f"{OUT_DIR}/final_segment_counts.csv")
    log("final segments (fraud rows):\n" + counts.to_string())


def assign_segments(r, st, X, M):
    """Push rows through the frozen all_shap pipeline -> segment name or UNIDENTIFIED."""
    if len(X) == 0:
        return np.zeros(0, dtype=object)
    Z = shap_matrix(st["explainer"], X[:, st["feat_idx"]])[:, r["top"]]
    if r["coef"] is not None:
        real_mo = [i for i, n in enumerate(KNOWN_MOS) if n not in UNCOVERED_MOS]
        Z, _ = mo_residualize(Z, M[:, real_mo], r["coef"])
    raw = gmm_assign(r["gmm"], r["pca"].transform(Z), r["ll_cut"])
    names = [r["seg_name"].get(k, "UNIDENTIFIED") for k in raw]
    return np.array([c if c in r["significant"] else "UNIDENTIFIED" for c in names], dtype=object)


def save_compat_cache(run, reducer, rc, st, top):
    """Individual files, as the old map_new_data_to_clusters.py expected."""
    for fname, obj in [(f"umap_reducer_{run}.pkl", reducer), (f"hdbscan_model_{run}.pkl", rc.top.m),
                       (f"top_shap_features_{run}.pkl", top)]:
        with open(f"{CACHE_DIR}/{fname}", "wb") as f:
            pickle.dump(obj, f)
    with open(f"{CACHE_DIR}/hgb_model{'' if st['name'].startswith('A') else '_stageB'}.pkl", "wb") as f:
        pickle.dump(st["model"], f)


# =====================================================================================
# MAP NEW DATA
# =====================================================================================
def map_new(path):
    """Map a new dataset (fraud, or fraud + goods with LABEL_COL) onto the cached segments.
    Goods are mapped too: the share of goods landing in a segment shows how specific it is."""
    with open(f"{CACHE_DIR}/pipeline_state.pkl", "rb") as f:
        state = pickle.load(f)
    pre, stages = state["pre"], state["stages"]
    r = state["runs"]["all_shap"]
    d = read_any(path).to_pandas().reset_index(drop=True)
    has_label = LABEL_COL in d.columns
    y = (pd.to_numeric(d[LABEL_COL], errors="coerce").fillna(-1).astype(int).to_numpy()
         if has_label else np.full(len(d), -1))
    d = tag_known_mos(d)
    X = pre.transform(d)
    M = d[[f"_mo__{n}" for n in KNOWN_MOS]].to_numpy().astype(int)
    seg = assign_segments(r, stages["A"], X, M)
    out = pd.DataFrame({"known_mos": d["_mo_set"], "label": y, "stageA_score": stage_scores(stages["A"], X),
                        "segment": seg, "segment_label": [r["labels"].get(c, "") for c in seg],
                        "new_mo_flag": np.isin(seg, list(r["new_mo"])).astype(int),
                        "unidentified_flag": (seg == "UNIDENTIFIED").astype(int)})
    if ACCOUNT_COL in d.columns:
        out.insert(0, ACCOUNT_COL, d[ACCOUNT_COL].to_numpy())
    summ = []
    for c, n in Counter(seg).items():
        m = seg == c
        row = {"segment": c, "segment_label": r["labels"].get(c, ""), "rows": int(n)}
        if has_label:
            row.update(hit_metrics((m & (y == 1)).sum(), (m & (y == 0)).sum(), (y == 1).sum(), (y == 0).sum()))
        summ.append(row)
    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_parquet(f"{OUT_DIR}/mapped_rows.parquet", index=False)
    pd.DataFrame(summ).sort_values("segment").to_csv(f"{OUT_DIR}/mapped_summary.csv", index=False)
    log(f"mapped {len(d):,} rows -> {OUT_DIR}/mapped_rows.parquet, mapped_summary.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--map-new", default=None, help="parquet/csv file or folder to map onto cached clusters")
    a = ap.parse_args()
    if a.map_new:
        map_new(a.map_new)
    else:
        run_pipeline()
