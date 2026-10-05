#!/usr/bin/env python3
"""
mo_discovery.py  -  Fraud MO segmentation that is aware of your EXISTING MOs
==========================================================================

End to end, one file:

 1. Load fraud + goods (Polars), dedupe, drop goods accounts that also appear in fraud.
 2. Parse pipe-delimited multi-label string columns  (cat_{i}_sub_label and any look-alike
    column, auto-detected)  ->  multi-hot token features + n_tokens + cross-column
    "anycat__<token>" features.
 3. Tag every row with your KNOWN MOs (boolean expressions over the c1..c5 flags).
 4. Mix goods + fraud and split TRAIN / VALIDATION (stratified by label x known MO,
    grouped by account so an account never sits on both sides; or out-of-time).
 5. TRAIN only:
        Stage A : HGB  all fraud  vs goods
        Stage B : HGB  RESIDUAL fraud (covered by NO known MO) vs goods
      -> SHAP -> UMAP(10D) -> HDBSCAN (+ subclustering of oversized families)
      runs:  all_shap | residual_shap | residual_quantile
 6. Every cluster is labelled against the known MOs:
        KNOWN:<mo>   cluster is mostly fraud already covered by that MO
        KNOWN:MIXED  mostly covered, but by several MOs (they behave alike)
        GAP:<mo>     mostly UNcovered fraud that behaves like <mo>  -> flag logic misses it
        PARTIAL:<mo> half covered, half not, and not similar enough to call a gap
        NEW          uncovered and unlike any known MO  -> new MO candidate
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
from sklearn.metrics import adjusted_mutual_info_score, average_precision_score, roc_auc_score
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
ACCOUNT_COL = "account_id"          # used for dedupe + grouped split; ignored if absent
DATE_COL = "tran_dt"                # base date; used for date diffs, emergence, time split
OTHER_DATE_COLS = ["first_credit_dt", "dormancy_start_date", "reactivation_date"]
DROP_COLS = []                      # ids / leakage columns to never use as features
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

# ---- known MOs ----------------------------------------------------------------------
FLAG_COLS = ["c1", "c2", "c3", "c4", "c5"]
# Flags are normalised to 0/1 first (1/0, Y/N, True/False all work). Write each MO as a
# pandas expression over the flags; dict order = priority when a row matches several.
# >>> THESE FIVE ARE PLACEHOLDERS - replace with your real definitions <<<
KNOWN_MOS = {
    "MO_1": "c1 == 1 and c2 == 1",
    "MO_2": "c3 == 1 and c4 == 1 and c1 == 0",
    "MO_3": "c5 == 1 and c2 == 0",
    "MO_4": "c1 == 1 and c3 == 1 and c2 == 0",
    "MO_5": "c2 == 1 and c4 == 1 and c1 == 0",
}
# Keep the flags OUT of the model features. Then if an unsupervised cluster lines up with
# a known MO without ever seeing its flags, that is independent confirmation of the MO -
# and uncovered fraud that lands next to it is a definition GAP, not a new MO.
EXCLUDE_FLAGS_FROM_MODEL = True

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
TOP_SHAP_FOR_CLUSTERING = 50
SHAP_SELECTION_SAMPLE = 4000
QUANTILE_GOODS_SAMPLE = 10000
UMAP_DIM = 10
UMAP_NEIGHBORS = 30
HDB_MIN_CLUSTER_FRAC = 0.005
HDB_MIN_CLUSTER_ABS = 40
HDB_MIN_SAMPLES = 10
SUBCLUSTER_FRAC = 0.25              # re-cluster any family holding >25% of a run's rows
MIN_RESIDUAL_FRAUD = 300            # skip residual runs if fewer uncovered train fraud

# ---- rules / labelling --------------------------------------------------------------
RULE_DEPTH = 3
RULE_MIN_SUPPORT = 20
PEEL_MAX_SEGMENTS = 12
KNOWN_COVERAGE_HI = 0.70            # >= this share covered by known MOs -> KNOWN
NEW_COVERAGE_LO = 0.30              # < this share covered (and not similar) -> NEW
DOMINANT_SHARE = 0.60
GAP_SIMILARITY = 0.80               # centred mean-SHAP cosine to a known MO -> GAP
MIN_VAL_RECALL_RETENTION = 0.60  # val rule recall on the cluster / train rule recall
VAL_SIZE_RATIO_RANGE = (0.5, 2.0)
MIN_RECLUSTER_JACCARD = 0.30       # cluster must re-appear when VAL fraud is clustered on its own
FINAL_RESIDUAL_RUN = "residual_shap"

SENTINEL = -1e9                     # NaN stand-in for rule learning ("IS NULL")
RUNS = [  # name, stage, pool, quantile-normalised
    ("all_shap", "A", "all", False),
    ("residual_shap", "B", "residual", False),
    ("residual_quantile", "B", "residual", True),
]

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
    log(f"fraud={int((pdf._label == 1).sum()):,}  goods={int((pdf._label == 0).sum()):,}  cols={pdf.shape[1]:,}")
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
        log(f"WARNING flag columns missing {missing} -> treated as 0; MO tags will be wrong")
    F = pd.DataFrame({c: (to01(df[c]) if c in df.columns else np.zeros(len(df), np.int8))
                      for c in FLAG_COLS}, index=df.index)
    hits = {}
    for name, expr in KNOWN_MOS.items():
        try:
            hits[name] = F.eval(expr, engine="python").astype(bool).to_numpy()
        except Exception as e:
            raise ValueError(f"Cannot evaluate KNOWN_MOS['{name}'] = '{expr}': {e}")
    new = {f"_flag_{c}": F[c].to_numpy() for c in FLAG_COLS}
    names = list(KNOWN_MOS)
    H = np.column_stack([hits[n] for n in names]) if names else np.zeros((len(df), 0), bool)
    for i, n in enumerate(names):
        new[f"_mo__{n}"] = H[:, i].astype(np.int8)
    primary = np.full(len(df), "NONE", dtype=object)
    for i in range(len(names) - 1, -1, -1):          # reverse so first in dict wins
        primary[H[:, i]] = names[i]
    new["_mo_primary"] = primary
    new["_mo_count"] = H.sum(1).astype(np.int8)
    new["_covered"] = H.any(1)
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


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
    strata = np.where(y == 1, "F_" + df["_mo_primary"].astype(str).to_numpy(), "G")
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
    return pd.to_datetime(s, errors="coerce", utc=True, format="mixed")


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
                vocab = [t for t, k in cnt.most_common(MAX_TOKENS_PER_COL) if k >= TOKEN_MIN_COUNT]
                if vocab:
                    self.multilabel[c] = vocab
                continue
            num = pd.to_numeric(nn, errors="coerce")
            if num.notna().mean() > 0.95:           # numbers stored as strings
                if num.nunique() > 1:
                    self.numeric.append(c)
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
        keep = X.nunique(dropna=True) > 1
        self.feature_names = X.columns[keep].tolist()
        self.token_features = [f for f in self.feature_names if "__" in f
                               and not f.endswith("__n_tokens") and not f.endswith("__other")
                               and (f.split("__", 1)[0] in self.multilabel or f.startswith("anycat__"))]
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
    k = int(min(MAX_MODEL_FEATURES, (imp > 0).sum()))
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
        if HAVE_HDBSCAN:
            self.m = hdbscan.HDBSCAN(min_cluster_size=self.mcs, min_samples=self.ms,
                                     cluster_selection_method=self.method, prediction_data=True).fit(emb)
        else:
            self.m = SkHDBSCAN(min_cluster_size=self.mcs, min_samples=self.ms,
                               cluster_selection_method=self.method).fit(emb)
            self.knn = KNeighborsClassifier(15).fit(emb, self.m.labels_)
        self.labels_ = self.m.labels_
        return self

    def predict(self, emb):
        if len(emb) == 0:
            return np.zeros(0, int)
        if HAVE_HDBSCAN:
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


def render_rule(bounds, names, is_bin):
    if bounds is None:
        return ""
    parts = []
    for f, (lo, hi) in sorted(bounds.items(), key=lambda kv: names[kv[0]]):
        n = names[f]
        if hi < SENTINEL / 2:
            parts.append(f"{n} IS NULL")
        elif is_bin[f]:
            parts.append(f"{n} = 1" if lo >= 0 else f"{n} = 0")
        elif lo == -np.inf:
            parts.append(f"({n} <= {hi:.4g} OR {n} IS NULL)")
        elif lo < SENTINEL / 2:
            parts.append(f"{n} IS NOT NULL" if hi == np.inf else f"{n} <= {hi:.4g}")
        else:
            parts.append(f"{n} > {lo:.4g}" + ("" if hi == np.inf else f" AND {n} <= {hi:.4g}"))
    return " AND ".join(parts)


def best_leaf(Xr_fit, target, min_support, mode):
    """mode 'cluster': max member recall within budget; 'peel': max fraud within budget."""
    tree = DecisionTreeClassifier(max_depth=RULE_DEPTH, min_samples_leaf=max(5, min_support // 2),
                                  class_weight="balanced", random_state=RANDOM_STATE).fit(Xr_fit, target)
    leaf_id = tree.apply(Xr_fit)
    P = max(target.sum(), 1)
    within, fallback = [], []
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
def mo_label(coverage, dom_mo, dom_share, near_mo, near_sim):
    if coverage >= KNOWN_COVERAGE_HI:
        return f"KNOWN:{dom_mo}" if dom_share >= DOMINANT_SHARE else "KNOWN:MIXED"
    if near_mo and near_sim >= GAP_SIMILARITY:
        return f"GAP:{near_mo}"
    if coverage >= NEW_COVERAGE_LO:
        return f"PARTIAL:{dom_mo}"
    return "NEW"


def known_mo_profiles(SA, primary, fraud_mask):
    """Centred mean Stage-A SHAP vector per known MO (train fraud)."""
    mu = SA[fraud_mask].mean(0)
    prof = {}
    for mo in KNOWN_MOS:
        m = fraud_mask & (primary == mo)
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

    # ---------------- data, MOs, split ----------------
    df = tag_known_mos(load_data())
    is_val = make_split(df)
    df["_split"] = np.where(is_val, "val", "train")
    summ = (df.assign(group=np.where(df._label == 1, "fraud:" + df._mo_primary.astype(str), "goods"))
              .groupby(["group", "_split"]).size().unstack(fill_value=0))
    summ.to_csv(f"{OUT_DIR}/split_summary.csv")
    log("split summary (rows):\n" + summ.to_string())
    fr = df._label == 1
    log(f"known-MO coverage of fraud: {df.loc[fr, '_covered'].mean():.1%}  | "
        f"rows matching >1 MO: {(df.loc[fr, '_mo_count'] > 1).mean():.1%}")

    reserved = set(DROP_COLS) | {ACCOUNT_COL, LABEL_COL}
    if EXCLUDE_FLAGS_FROM_MODEL:
        reserved |= set(FLAG_COLS)
    tr = df[~is_val].reset_index(drop=True)
    va = df[is_val].reset_index(drop=True)
    pre = Preprocessor().fit(tr, reserved)
    names = pre.feature_names
    X_tr, X_va = pre.transform(tr), pre.transform(va)
    Xr_tr, Xr_va = np.nan_to_num(X_tr, nan=SENTINEL), np.nan_to_num(X_va, nan=SENTINEL)
    is_bin = ((np.nan_to_num(X_tr) == 0) | (np.nan_to_num(X_tr) == 1)).all(0)
    y_tr, y_va = tr._label.to_numpy(), va._label.to_numpy()
    cov_tr, cov_va = tr._covered.to_numpy(bool), va._covered.to_numpy(bool)
    prim_tr, prim_va = tr._mo_primary.to_numpy(), va._mo_primary.to_numpy()

    # ---------------- stages ----------------
    stages = {"A": fit_stage("A (all fraud vs goods)", X_tr, y_tr, names)}
    resid_tr = (y_tr == 1) & ~cov_tr
    run_residual = resid_tr.sum() >= MIN_RESIDUAL_FRAUD
    if run_residual:
        mB = (y_tr == 0) | resid_tr
        stages["B"] = fit_stage("B (uncovered fraud vs goods)", X_tr[mB], y_tr[mB], names)
    else:
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
    pd.DataFrame(metrics).to_csv(f"{OUT_DIR}/model_metrics.csv", index=False)

    # Stage-A SHAP for every train fraud row: used for known-MO similarity (GAP detection)
    SA = np.zeros((len(y_tr), len(stages["A"]["feat_idx"])), np.float32)
    SA[y_tr == 1] = shap_matrix(stages["A"]["explainer"], X_tr[y_tr == 1][:, stages["A"]["feat_idx"]])
    mu_A, mo_prof = known_mo_profiles(SA, prim_tr, y_tr == 1)
    log(f"known-MO SHAP profiles built for {list(mo_prof)}")

    dates_tr = _to_dt(tr[DATE_COL]) if DATE_COL in tr.columns else None
    state = {"pre": pre, "stages": stages, "runs": {}, "names": names, "is_bin": is_bin,
             "config": {k: v for k, v in globals().items() if k.isupper()}}
    all_profiles, assignments_tr, assignments_va = [], {}, {}

    for run, skey, pool_kind, quant in RUNS:
        if skey not in stages:
            continue
        log(f"===== run {run} =====")
        st = stages[skey]
        pool_tr = (y_tr == 1) if pool_kind == "all" else resid_tr
        pool_va = (y_va == 1) if pool_kind == "all" else ((y_va == 1) & ~cov_va)
        Xs_tr = X_tr[:, st["feat_idx"]]
        S = SA[pool_tr] if skey == "A" else shap_matrix(st["explainer"], Xs_tr[pool_tr])
        top = np.argsort(-np.abs(S).mean(0))[:TOP_SHAP_FOR_CLUSTERING]
        Z = S[:, top]
        ecdf = None
        if quant:
            g_idx = np.where(y_tr == 0)[0]
            g_idx = np.random.default_rng(RANDOM_STATE).choice(g_idx, min(QUANTILE_GOODS_SAMPLE, len(g_idx)), replace=False)
            ecdf = np.sort(shap_matrix(st["explainer"], Xs_tr[g_idx])[:, top], axis=0)
            Z = quantile_norm(Z, ecdf)
        reducer = umap.UMAP(n_components=UMAP_DIM, n_neighbors=UMAP_NEIGHBORS, min_dist=0.0,
                            random_state=RANDOM_STATE).fit(Z)
        emb = reducer.embedding_
        rc = RunClusterer().fit(emb)
        lab = rc.labels_
        log(f"  {pool_tr.sum():,} train fraud -> {len(set(lab) - {'noise'})} clusters, "
            f"noise {np.mean(lab == 'noise'):.1%}")

        # ---- validation assignment ----
        Xs_va = X_va[:, st["feat_idx"]]
        Zv = shap_matrix(st["explainer"], Xs_va[pool_va])[:, top]
        Zg = shap_matrix(st["explainer"], Xs_va[y_va == 0])[:, top]
        if quant:
            Zv, Zg = quantile_norm(Zv, ecdf), quantile_norm(Zg, ecdf)
        emb_v = reducer.transform(Zv) if len(Zv) else np.zeros((0, UMAP_DIM))
        emb_g = reducer.transform(Zg) if len(Zg) else np.zeros((0, UMAP_DIM))
        lab_v, lab_g = rc.predict(emb_v), rc.predict(emb_g)
        # independent re-clustering of VAL fraud -> reproducibility
        ami, refit = np.nan, None
        if len(emb_v) >= 3 * HDB_MIN_CLUSTER_ABS:
            refit = Clusterer(max(20, int(rc.mcs * len(emb_v) / len(emb))), HDB_MIN_SAMPLES).fit(emb_v).labels_
            ami = adjusted_mutual_info_score(rc.top_level(lab_v), refit)
        log(f"  val: {pool_va.sum():,} fraud assigned, noise {np.mean(lab_v == 'noise'):.1%}, "
            f"re-cluster AMI {ami:.3f}")

        # ---- per-cluster profile ----
        sc_tr = stage_scores(st, X_tr)
        rows_idx_tr, rows_idx_va = np.where(pool_tr)[0], np.where(pool_va)[0]
        recent_cut = dates_tr[pool_tr].quantile(0.75) if dates_tr is not None else None
        overall_recent = (dates_tr[pool_tr] > recent_cut).mean() if dates_tr is not None else np.nan
        profiles = []
        top_tr, top_v = rc.top_level(lab), rc.top_level(lab_v)
        refit_sub = np.full(len(lab_v), -1)            # same test one level down, inside each split family
        for k, sc in rc.subs.items():
            fv, ft = top_v == f"C{k:02d}", top_tr == f"C{k:02d}"
            if fv.sum() >= 3 * 20:
                l2 = Clusterer(max(10, int(sc.mcs * fv.sum() / ft.sum())), HDB_MIN_SAMPLES, "leaf").fit(emb_v[fv]).labels_
                refit_sub[fv] = np.where(l2 == -1, -1, 1000 * (k + 1) + l2)
        units = [(c, "sub" if "." in c else "cluster", lab == c, lab_v == c)
                 for c in sorted(set(lab), key=lambda s: (s == "noise", s))]
        units += [(f"C{k:02d}", "family", top_tr == f"C{k:02d}", top_v == f"C{k:02d}") for k in sorted(rc.subs)]
        for c, level, mem_local, mem_local_v in units:
            mem = np.zeros(len(y_tr), bool); mem[rows_idx_tr[mem_local]] = True
            mem_v = np.zeros(len(y_va), bool); mem_v[rows_idx_va[mem_local_v]] = True
            n = int(mem.sum())
            cov = float(cov_tr[mem].mean())
            pc = Counter(prim_tr[mem & cov_tr])
            dom_mo, dom_n = (pc.most_common(1)[0] if pc else ("", 0))
            dom_share = dom_n / max(sum(pc.values()), 1)
            near, sim = nearest_mo(SA[mem].mean(0) - mu_A, mo_prof)
            p = {"run": run, "cluster": c, "level": level, "n_train": n, "share_train": round(n / pool_tr.sum(), 4),
                 "known_mo_coverage": round(cov, 3), "dominant_known_mo": dom_mo,
                 "dominant_share": round(dom_share, 3), "nearest_known_mo": near, "nearest_mo_similarity": sim,
                 "stage_score_mean": round(float(sc_tr[mem].mean()), 4)}
            p["mo_label"] = "noise" if c == "noise" else mo_label(cov, dom_mo, dom_share, near, sim)
            for mo in KNOWN_MOS:
                p[f"share_{mo}"] = round(float((prim_tr[mem] == mo).mean()), 3)
            if dates_tr is not None:
                p["emergence"] = round(float((dates_tr[mem] > recent_cut).mean() / max(overall_recent, 1e-9)), 2)
            # rule: cluster members vs train goods
            rule_b = None
            if c != "noise" and n >= RULE_MIN_SUPPORT:
                fit_rows = mem | (y_tr == 0)
                rule_b, within = best_leaf(Xr_tr[fit_rows], mem[fit_rows], max(RULE_MIN_SUPPORT, int(0.05 * n)), "cluster")
                p["rule_within_budget"] = within
            p["rule"] = render_rule(rule_b, names, is_bin)
            if rule_b is not None:
                p.update(rule_metrics(apply_rule(rule_b, Xr_tr), y_tr, mem, "train_rule_"))
                p.update(rule_metrics(apply_rule(rule_b, Xr_va), y_va, mem_v, "val_rule_"))
                p["val_recall_retention"] = round(p["val_rule_cluster_recall"] / max(p["train_rule_cluster_recall"], 1e-9), 3)
                p["val_rule_within_budget"] = p["val_rule_goods_per_fraud"] <= GOODS_PER_FRAUD_BUDGET
            # validation stability
            nv = int(mem_v.sum())
            p["n_val"] = nv
            p["share_val"] = round(nv / max(pool_va.sum(), 1), 4)
            p["val_size_ratio"] = round(p["share_val"] / max(p["share_train"], 1e-9), 3)
            p["val_known_mo_coverage"] = round(float(cov_va[mem_v].mean()), 3) if nv else np.nan
            ng = int((lab_g == c).sum())
            p["val_goods_landing_in_cluster"] = ng
            p["val_emb_goods_per_fraud"] = round(ng * GOODS_WEIGHT / max(nv, 1), 2)
            if refit is not None and c != "noise":
                A = mem_local_v       # subclusters are compared with the re-clustered subclusters
                ref = refit_sub if level == "sub" else refit
                best = 0.0
                for k in set(ref) - {-1}:
                    B = ref == k
                    best = max(best, (A & B).sum() / max((A | B).sum(), 1))
                p["val_recluster_jaccard"] = round(best, 3)
            lo, hi = VAL_SIZE_RATIO_RANGE
            p["MO_candidate"] = bool(
                c != "noise" and (p["mo_label"] == "NEW" or p["mo_label"].startswith("GAP:"))
                and n >= HDB_MIN_CLUSTER_ABS and p.get("rule_within_budget", False)
                and p.get("val_rule_within_budget", False)
                and p.get("val_recall_retention", 0) >= MIN_VAL_RECALL_RETENTION and lo <= p["val_size_ratio"] <= hi
                and p.get("val_recluster_jaccard", 1.0) >= MIN_RECLUSTER_JACCARD)
            profiles.append(p)
        prof_df = pd.DataFrame(profiles)
        prof_df.to_csv(f"{OUT_DIR}/cluster_profiles_{run}.csv", index=False)
        all_profiles.append(prof_df)
        log(f"  labels: {dict(Counter(prof_df.mo_label))}  | MO candidates: {int(prof_df.MO_candidate.sum())}")

        # ---- signal rates ----
        signal_rates(run, lab, rows_idx_tr, X_tr, tr, y_tr, pool_tr, names, is_bin, S, top, st)

        # ---- peeling ----
        peel_segments(run, pool_tr, pool_va, Xr_tr, Xr_va, y_tr, y_va, cov_tr, prim_tr, names, is_bin)

        # ---- plot ----
        plot_run(run, emb, lab, prim_tr[pool_tr])

        assignments_tr[run] = (rows_idx_tr, lab)
        assignments_va[run] = (rows_idx_va, lab_v)
        state["runs"][run] = {"stage": skey, "pool": pool_kind, "top": top, "ecdf": ecdf,
                              "reducer": reducer, "clusterer": rc,
                              "labels": dict(zip(prof_df.cluster, prof_df.mo_label)),
                              "candidates": prof_df.loc[prof_df.MO_candidate, "cluster"].tolist()}
        save_compat_cache(run, reducer, rc, st, top)

    # ---------------- summaries ----------------
    summary = pd.concat(all_profiles, ignore_index=True)
    lead = ["run", "cluster", "level", "mo_label", "MO_candidate", "n_train", "n_val", "share_train", "share_val",
            "known_mo_coverage", "dominant_known_mo", "nearest_known_mo", "nearest_mo_similarity", "rule",
            "train_rule_cluster_recall", "train_rule_goods_per_fraud", "train_rule_lift",
            "val_rule_cluster_recall", "val_rule_goods_per_fraud", "val_rule_lift", "val_recall_retention",
            "val_size_ratio", "val_recluster_jaccard", "emergence"]
    lead = [c for c in lead if c in summary.columns]
    summary = summary[lead + [c for c in summary.columns if c not in lead]]
    summary.sort_values(["MO_candidate", "run", "n_train"], ascending=[False, True, False]).to_csv(
        f"{OUT_DIR}/mo_summary.csv", index=False)

    known_mo_report(tr, va, y_tr, y_va, assignments_tr)
    if run_residual:
        mine_token_combos(pre, X_tr, X_va, y_tr, y_va, resid_tr, (y_va == 1) & ~cov_va, names)
    write_assignments(tr, va, stages, X_tr, X_va, assignments_tr, assignments_va, state)

    with open(f"{CACHE_DIR}/pipeline_state.pkl", "wb") as f:
        pickle.dump(state, f)
    log(f"done. outputs in {OUT_DIR}/, fitted objects in {CACHE_DIR}/")


# =====================================================================================
# REPORTS
# =====================================================================================
def signal_rates(run, lab, rows_idx, X_tr, tr, y_tr, pool_tr, names, is_bin, S, top, st):
    bin_idx = np.where(is_bin)[0]
    B = np.nan_to_num(X_tr[:, bin_idx])
    flags = np.column_stack([tr[f"_flag_{c}"].to_numpy() for c in FLAG_COLS]).astype(np.float32)
    B = np.hstack([B, flags])
    bnames = [names[i] for i in bin_idx] + [f"flag:{c}" for c in FLAG_COLS]
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


def peel_segments(run, pool_tr, pool_va, Xr_tr, Xr_va, y_tr, y_va, cov_tr, prim_tr, names, is_bin):
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
        seg = {"run": run, "segment": s + 1, "rule": render_rule(b, names, is_bin),
               "train_new_fraud": int(new_tr.sum()),
               "train_new_fraud_uncovered_share": round(float((~cov_tr[new_tr]).mean()), 3),
               "train_new_fraud_top_known_mo": Counter(prim_tr[new_tr]).most_common(1)[0][0] if new_tr.any() else "",
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


def plot_run(run, emb, lab, prim):
    try:
        e2 = umap.UMAP(n_components=2, n_neighbors=UMAP_NEIGHBORS, min_dist=0.1,
                       random_state=RANDOM_STATE).fit_transform(emb)
        fig, ax = plt.subplots(1, 2, figsize=(18, 8))
        for a, col, title in [(ax[0], lab, "HDBSCAN clusters"), (ax[1], prim, "Known MO (flags)")]:
            cats = sorted(set(col), key=lambda s: (s in ("noise", "NONE"), s))
            cmap = plt.get_cmap("tab20")
            for i, k in enumerate(cats):
                m = col == k
                grey = k in ("noise", "NONE")
                a.scatter(e2[m, 0], e2[m, 1], s=3, alpha=0.25 if grey else 0.6,
                          c="lightgrey" if grey else [cmap(i % 20)], label=k)
                if not grey and col is lab:
                    a.annotate(k, np.median(e2[m], 0), fontsize=8, weight="bold")
            a.set_title(f"{run}: {title}")
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
            r.update(rule_metrics(h, y, prefix=f"{nm}_"))
            r[f"{nm}_fraud_overlapping_other_mo"] = int((h & (y == 1) & (d._mo_count.to_numpy() > 1)).sum())
        if "all_shap" in assignments_tr:
            idx, lab = assignments_tr["all_shap"]
            m = tr._mo_primary.to_numpy()[idx] == mo
            if m.any():
                cnt = Counter(lab[m])
                tot = sum(cnt.values())
                r["all_shap_top_clusters"] = "; ".join(f"{k}:{v / tot:.0%}" for k, v in cnt.most_common(4))
                p = np.array(list(cnt.values())) / tot
                r["all_shap_concentration"] = round(float((p ** 2).sum()), 3)  # 1 = one cluster
        rows.append(r)
    pd.DataFrame(rows).to_csv(f"{OUT_DIR}/known_mo_report.csv", index=False)


def mine_token_combos(pre, X_tr, X_va, y_tr, y_va, pool_tr, pool_va, names):
    """Token itemsets (size 1-3) enriched in UNCOVERED fraud vs goods - a cheap, model-free
    second opinion for new MOs (e.g. prescamSpike + Oddhours)."""
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
    key = [pre.token_features[i].split("__", 1)[1] for i in range(len(tok_idx))]
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
        o["train_share_of_uncovered_fraud"] = round(r["train_fraud_hit"] / nF, 4)
        o["train_all_fraud_hit_incl_covered"] = int((m_tr & (y_tr == 1)).sum())
        vf, vg = int((m_va & pool_va).sum()), int((m_va & (y_va == 0)).sum())
        o.update({"val_fraud_hit": vf, "val_goods_hit": vg,
                  "val_goods_per_fraud": round(vg * GOODS_WEIGHT / max(vf, 1), 2),
                  "val_lift": round((vf / max(pool_va.sum(), 1)) / ((vg + 1) / ((y_va == 0).sum() + 1)), 2)})
        o["within_budget_train_and_val"] = (o["train_goods_per_fraud"] <= GOODS_PER_FRAUD_BUDGET
                                            and o["val_goods_per_fraud"] <= GOODS_PER_FRAUD_BUDGET)
        out.append(o)
    out = pd.DataFrame(out).sort_values(["within_budget_train_and_val", "train_fraud_hit"], ascending=False)
    out.to_csv(f"{OUT_DIR}/token_combos_uncovered.csv", index=False)
    log(f"token combos: {len(out)} enriched itemsets, {int(out.within_budget_train_and_val.sum())} within budget")


def write_assignments(tr, va, stages, X_tr, X_va, assignments_tr, assignments_va, state):
    frames = []
    for nm, d, X, asg in [("train", tr, X_tr, assignments_tr), ("val", va, X_va, assignments_va)]:
        fr = np.where(d._label.to_numpy() == 1)[0]
        out = pd.DataFrame({"split": nm, "known_mo_primary": d._mo_primary.to_numpy()[fr],
                            "known_mo_count": d._mo_count.to_numpy()[fr],
                            "stageA_score": stage_scores(stages["A"], X[fr])})
        if ACCOUNT_COL in d.columns:
            out.insert(0, ACCOUNT_COL, d[ACCOUNT_COL].to_numpy()[fr])
        pos = {r: i for i, r in enumerate(fr)}
        for run, (idx, lab) in asg.items():
            col = np.full(len(fr), "", dtype=object)
            for r, l in zip(idx, lab):
                col[pos[r]] = l
            out[f"cluster_{run}"] = col
            out[f"label_{run}"] = [state["runs"][run]["labels"].get(c, "") if c else "" for c in col]
        final = out.known_mo_primary.astype(object).copy()
        if f"cluster_{FINAL_RESIDUAL_RUN}" in out:
            cands = set(state["runs"][FINAL_RESIDUAL_RUN]["candidates"])
            unc = final == "NONE"
            cl = out[f"cluster_{FINAL_RESIDUAL_RUN}"]
            lbl = out[f"label_{FINAL_RESIDUAL_RUN}"]
            fam = cl.str.split(".").str[0]
            labels = state["runs"][FINAL_RESIDUAL_RUN]["labels"]
            use = np.where(cl.isin(cands), cl, np.where(fam.isin(cands), fam, ""))
            use_lbl = pd.Series([labels.get(u, "") for u in use], index=out.index)
            hit = unc & (use != "")
            final[hit] = (use_lbl + " [" + FINAL_RESIDUAL_RUN + ":" + pd.Series(use, index=out.index) + "]")[hit]
            final[unc & ~hit] = "UNASSIGNED"
        out["final_segment"] = final
        frames.append(out)
    res = pd.concat(frames, ignore_index=True)
    res.to_parquet(f"{OUT_DIR}/fraud_assignments.parquet", index=False)
    seg = res.groupby(["final_segment", "split"]).size().unstack(fill_value=0)
    seg.to_csv(f"{OUT_DIR}/final_segment_counts.csv")
    log("final segments (fraud rows):\n" + seg.to_string())


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
    with open(f"{CACHE_DIR}/pipeline_state.pkl", "rb") as f:
        state = pickle.load(f)
    pre, stages = state["pre"], state["stages"]
    d = read_any(path).to_pandas().reset_index(drop=True)
    has_label = LABEL_COL in d.columns
    y = d[LABEL_COL].astype(int).to_numpy() if has_label else np.full(len(d), -1)
    d = tag_known_mos(d)
    X = pre.transform(d)
    cov = d._covered.to_numpy(bool)
    out = pd.DataFrame({"known_mo_primary": d._mo_primary, "label": y,
                        "stageA_score": stage_scores(stages["A"], X)})
    if ACCOUNT_COL in d.columns:
        out.insert(0, ACCOUNT_COL, d[ACCOUNT_COL].to_numpy())
    summ = []
    for run, r in state["runs"].items():
        st = stages[r["stage"]]
        pool = np.ones(len(d), bool) if r["pool"] == "all" else ~cov
        Z = shap_matrix(st["explainer"], X[pool][:, st["feat_idx"]])[:, r["top"]]
        if r["ecdf"] is not None:
            Z = quantile_norm(Z, r["ecdf"])
        lab = r["clusterer"].predict(r["reducer"].transform(Z))
        col = np.full(len(d), "", dtype=object)
        col[pool] = lab
        out[f"cluster_{run}"] = col
        out[f"label_{run}"] = [r["labels"].get(c, "unseen") if c else "" for c in col]
        for c, n in Counter(lab).items():
            m = col == c
            row = {"run": run, "cluster": c, "mo_label": r["labels"].get(c, ""), "rows": int(n)}
            if has_label:
                row["fraud"] = int((m & (y == 1)).sum()); row["goods"] = int((m & (y == 0)).sum())
                row["fraud_pct"] = round(row["fraud"] / max(((y == 1) & pool).sum(), 1), 4)
                row["goods_pct"] = round(row["goods"] / max(((y == 0) & pool).sum(), 1), 4)
            summ.append(row)
    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_parquet(f"{OUT_DIR}/mapped_rows.parquet", index=False)
    pd.DataFrame(summ).sort_values(["run", "cluster"]).to_csv(f"{OUT_DIR}/mapped_summary.csv", index=False)
    log(f"mapped {len(d):,} rows -> {OUT_DIR}/mapped_rows.parquet, mapped_summary.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--map-new", default=None, help="parquet/csv file or folder to map onto cached clusters")
    a = ap.parse_args()
    if a.map_new:
        map_new(a.map_new)
    else:
        run_pipeline()
