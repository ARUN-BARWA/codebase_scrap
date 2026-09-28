"""
fraud_segmentation.py
=====================
Behavioural segmentation for fraud screening, in one file.

Event stream in, a joinable per-account feature block out, plus segments with
priced controls. Five layers, in order:

    1. REPRESENTATION   events -> named per-account features, in blocks
                        timing / point-process (Hawkes), sequence dynamics,
                        composition, scale
    2. EMBEDDING        blocks -> whitened, block-balanced coordinates, with an
                        optional and deliberately bounded supervised direction
    3. CLUSTERING       coordinates -> soft memberships + novelty (diagonal GMM
                        by default; gmm_full / bgmm / kmeans are drop-in)
    4. DIAGNOSTICS      per-cluster scorecard, drivers, response curves,
                        reproducibility checks, rules priced in alert volume
    5. WORKED EXAMPLE   stage_5 .. stage_13, runnable end to end

Usage
-----
    pip install numpy pandas scipy scikit-learn
    python fraud_segmentation.py                # synthetic data, all stages
    python fraud_segmentation.py --stage 7      # stop after the scorecard

Point it at your own data by editing USE_SYNTHETIC, the column names and
load_data() in the EDIT THIS BLOCK section below.

Two things this deliberately does not do
----------------------------------------
*   **No book ingestion.** Streaming by parquet row group, checkpointing,
    reservoir sampling and the dry run belong in your existing screening
    harness. stage_13 shows the wiring.
*   **No missingness policy or feature screening.** Those exist to tame
    hundreds of messy windowed aggregate columns. An event stream has none:
    a transaction either happened or it did not.

The leakage rule no code here can enforce: every event for a fraud account must
predate its fraud event. Post-event rows hold the reversal and the block, and
none of that exists at decision time.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (adjusted_rand_score, average_precision_score,
                             recall_score, roc_auc_score)
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.tree import DecisionTreeClassifier, export_text, _tree

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)

EPS = 1e-9
PREFIXES = ("tm", "hk", "cp", "dy", "sc", "dt", "nm")


# ==========================================================================
# PART 1-3.  MODEL: representation, embedding, clustering
# ==========================================================================


def _log(m: str, l: int = 0) -> None:
    print("  " * l + m, flush=True)


# ==========================================================================
# Config
# ==========================================================================
@dataclass
class SegConfig:
    random_state: int = 42

    # ---- input shape ------------------------------------------------------
    # Set by profile_input(); you should not normally touch these by hand.
    #   "events"  many rows per account, with a usable date  -> full pipeline
    #   "static"  one row per account                        -> no sequence at
    #             all, so timing / Hawkes / transitions are skipped and the
    #             model falls back to date-derived and numeric blocks
    mode: str = "events"
    time_unit: str = "days"           # time_col is days since a common origin
    intraday: bool = False            # do the timestamps carry a time of day?
    rhythm_periods: tuple = (7.0, 30.44)
    rhythm_labels: tuple = ("week", "month")

    # ---- point process ----------------------------------------------------
    # Decay timescales in the SAME UNIT as time_col. beta = 1/tau. One per
    # cascade speed you care about. With date-only data the shortest usable
    # scale is a day: everything inside one calendar day is simultaneous, and a
    # sub-day timescale would fit noise created by ties.
    hawkes_timescales: tuple = (1.0, 7.0, 30.0)
    hawkes_labels: tuple = ("1d", "7d", "30d")
    hawkes_em_iters: int = 30
    hawkes_min_events: int = 5
    hawkes_shrink_k: float = 8.0     # prior weight, in events, on the population mean

    # ---- state alphabet (dynamics block) ----------------------------------
    n_states: int = 8
    state_fit_sample: int = 300_000
    dirichlet_alpha: float = 0.5

    # ---- embedding --------------------------------------------------------
    block_components: int = 8
    block_weights: dict = field(default_factory=lambda: {
        "timing": 1.0, "dynamics": 1.0, "composition": 1.0,
        "dates": 0.8, "numeric": 1.0, "scale": 0.3})
    # scale is deliberately down-weighted: without it, clusters re-discover
    # how much people transact rather than how.

    supervision_weight: float = 0.25  # 0 = fully unsupervised. Above ~0.5 the axis
    # dominates and the "clustering" degenerates into binning a classifier -
    # watch SegmentModel.supervision_ari_ for exactly that.
    supervision_folds: int = 5

    # ---- clustering -------------------------------------------------------
    engine: str = "gmm_diag"          # gmm_diag | gmm_full | bgmm | kmeans
    k_range: tuple = (4, 12)
    n_clusters: int | None = None
    n_bootstrap: int = 8
    bootstrap_frac: float = 0.7
    min_cluster_share: float = 0.01
    fit_sample: int = 300_000

    # ---- reporting --------------------------------------------------------
    n_book: int = 20_000_000
    n_fraud_book: int = 800


# ==========================================================================
# 1. Point-process features
# ==========================================================================
def _excitation(t_rel: np.ndarray, code: np.ndarray, pos: np.ndarray,
                beta: float) -> np.ndarray:
    """R_i = sum_{j<i} exp(-beta (t_i - t_j)), by the stable recursion
    R_i = exp(-beta * delta_i) * (1 + R_{i-1}), reset at each account.

    Computed once per timescale and reused by every (mu, alpha) iteration -
    the likelihood depends on the parameters only through R.
    """
    n = len(t_rel)
    R = np.zeros(n)
    if n == 0:
        return R
    d = np.empty(n)
    d[0] = 0.0
    d[1:] = np.exp(-beta * np.maximum(t_rel[1:] - t_rel[:-1], 0.0))
    d[pos == 0] = 0.0
    for k in range(1, int(pos.max()) + 1 if n else 1):
        idx = np.flatnonzero(pos == k)
        if idx.size == 0:
            break
        R[idx] = d[idx] * (1.0 + R[idx - 1])
    return R


def hawkes_features(events: pd.DataFrame, id_col: str, time_col: str,
                    cfg: SegConfig) -> pd.DataFrame:
    """Per-account Hawkes fit at each timescale, by vectorised EM.

    The Veen-Schoenberg EM for an exponential Hawkes process factorises into
    per-event responsibilities and per-account sums, so every account in the
    book is fitted in the same loop - no per-account optimiser.

        p_bg(i) = mu / (mu + alpha R_i)
        mu    <- sum p_bg / T
        alpha <- sum (1 - p_bg) / S,   S = (1/beta) sum_i (1 - e^{-beta(T - t_i)})

    Returns branching ratio n = alpha/beta (the interpretable one: expected
    directly-triggered follow-on transactions per transaction), the background
    rate, and the per-event log-likelihood ratio against a homogeneous Poisson
    process. That last column is the answer to "is Poisson worth adding" -
    Poisson is the null you measure against, not a feature in its own right.
    """
    df = events[[id_col, time_col]].sort_values([id_col, time_col], kind="mergesort")
    code, uniq = pd.factorize(df[id_col], sort=True)
    t = df[time_col].to_numpy(dtype=float)
    n_acc = len(uniq)

    starts = np.flatnonzero(np.r_[True, code[1:] != code[:-1]])
    pos = np.arange(len(code)) - starts[code]
    t0 = t[starts][code]
    t_rel = t - t0

    N = np.bincount(code, minlength=n_acc).astype(float)
    T = np.maximum.reduceat(t_rel, starts) if len(starts) else np.zeros(n_acc)
    Tpos = np.maximum(T, EPS)

    out = pd.DataFrame(index=pd.Index(uniq, name=id_col))
    ll_pois = N * np.log(np.maximum(N / Tpos, EPS)) - N
    enough = (N >= cfg.hawkes_min_events) & (T > 0)

    for tau, lab in zip(cfg.hawkes_timescales, cfg.hawkes_labels):
        beta = 1.0 / float(tau)
        R = _excitation(t_rel, code, pos, beta)
        S = np.bincount(code, weights=(1.0 - np.exp(-beta * (T[code] - t_rel))),
                        minlength=n_acc) / beta
        S = np.maximum(S, EPS)

        mu = np.maximum(N / (2.0 * Tpos), EPS)
        alpha = np.full(n_acc, beta / 2.0)
        for _ in range(cfg.hawkes_em_iters):
            lam = np.maximum(mu[code] + alpha[code] * R, EPS)
            p_bg = mu[code] / lam
            mu = np.maximum(np.bincount(code, weights=p_bg, minlength=n_acc) / Tpos, EPS)
            alpha = np.bincount(code, weights=(1.0 - p_bg), minlength=n_acc) / S
            alpha = np.clip(alpha, 0.0, 0.999 * beta)     # keep it subcritical

        lam = np.maximum(mu[code] + alpha[code] * R, EPS)
        ll_h = np.bincount(code, weights=np.log(lam), minlength=n_acc) - mu * T - alpha * S

        branch = np.where(enough, alpha / beta, np.nan)
        # empirical-Bayes shrinkage: a 6-event account does not get to claim a
        # branching ratio of 0.9 on its own evidence
        pop = np.nanmean(branch) if np.isfinite(branch).any() else 0.0
        branch = (N * np.nan_to_num(branch) + cfg.hawkes_shrink_k * pop) / \
                 (N + cfg.hawkes_shrink_k)

        out[f"hk_branch_{lab}"] = branch
        out[f"hk_bg_rate_{lab}"] = np.log1p(np.where(enough, mu, N / Tpos))
        out[f"hk_llr_pois_{lab}"] = np.where(enough, (ll_h - ll_pois) / np.maximum(N, 1), 0.0)
    return out


def timing_features(events: pd.DataFrame, id_col: str, time_col: str,
                    cfg: SegConfig) -> pd.DataFrame:
    """Rhythm and dispersion. Everything here is scale-free on purpose.

    `tm_rhythm_*` are circular resultant lengths: 1.0 means every transaction
    lands at the same position in the cycle, 0 means no periodicity. They catch
    salary and EMI cycles without an FFT and without binning. With date-only
    data the cycles worth asking about are the week and the month, which is what
    `cfg.rhythm_periods` defaults to.

    Date resolution costs you two things and you should know which. Ties are
    real: several transactions on one day have a gap of exactly zero, so
    burstiness and CV saturate rather than growing without bound. And there is
    no time-of-day, so the night-activity share - often one of the better
    single features in takeover - is simply unavailable.
    """
    df = events[[id_col, time_col]].sort_values([id_col, time_col], kind="mergesort")
    code, uniq = pd.factorize(df[id_col], sort=True)
    t = df[time_col].to_numpy(dtype=float)
    n_acc = len(uniq)
    starts = np.flatnonzero(np.r_[True, code[1:] != code[:-1]])
    pos = np.arange(len(code)) - starts[code]
    t_rel = t - t[starts][code]

    N = np.bincount(code, minlength=n_acc).astype(float)
    T = np.maximum.reduceat(t_rel, starts)
    d = np.where(pos > 0, np.r_[0.0, np.diff(t)], np.nan)
    ok = ~np.isnan(d)
    cnt = np.bincount(code[ok], minlength=n_acc).astype(float)
    s1 = np.bincount(code[ok], weights=d[ok], minlength=n_acc)
    s2 = np.bincount(code[ok], weights=d[ok] ** 2, minlength=n_acc)
    m = s1 / np.maximum(cnt, 1)
    v = np.maximum(s2 / np.maximum(cnt, 1) - m ** 2, 0.0)
    sd = np.sqrt(v)

    lg = np.log1p(np.maximum(d[ok], 0.0))
    lm = np.bincount(code[ok], weights=lg, minlength=n_acc) / np.maximum(cnt, 1)
    lv = np.bincount(code[ok], weights=lg ** 2, minlength=n_acc) / np.maximum(cnt, 1) - lm ** 2

    def rhythm(period):
        ang = 2 * np.pi * (t % period) / period
        c = np.bincount(code, weights=np.cos(ang), minlength=n_acc) / np.maximum(N, 1)
        sn = np.bincount(code, weights=np.sin(ang), minlength=n_acc) / np.maximum(N, 1)
        return np.sqrt(c ** 2 + sn ** 2)

    out = {
        "tm_log_rate": np.log1p(N / np.maximum(T, 1.0)),
        "tm_burstiness": (sd - m) / np.maximum(sd + m, EPS),   # -1 regular, +1 bursty
        "tm_cv": sd / np.maximum(m, EPS),
        "tm_log_gap_sd": np.sqrt(np.maximum(lv, 0.0)),
        "tm_active_span": np.log1p(T),
        # ties matter at date resolution: this is the share of consecutive
        # transactions that fall on the SAME day, which is the date-only
        # stand-in for intraday bursting
        "tm_same_day_share": (np.bincount(code[ok], weights=(d[ok] <= 0).astype(float),
                                          minlength=n_acc) / np.maximum(cnt, 1)),
    }
    for per, lab in zip(cfg.rhythm_periods, cfg.rhythm_labels):
        out[f"tm_rhythm_{lab}"] = rhythm(float(per))

    if cfg.intraday:
        hour = (t * 24.0) % 24.0 if cfg.time_unit == "days" else t % 24.0
        out["tm_night_share"] = np.bincount(
            code, weights=((hour < 6) | (hour >= 23)).astype(float),
            minlength=n_acc) / np.maximum(N, 1)

    return pd.DataFrame(out, index=pd.Index(uniq, name=id_col))


def date_features(df: pd.DataFrame, id_col: str, date_cols: Sequence[str],
                  as_of: float | None = None) -> pd.DataFrame:
    """Turn raw date columns into model-ready numbers.

    Dates are unusable as-is: a model handed an epoch-day integer learns
    "accounts opened in 2019 are risky", which is a property of your sampling
    window, not of the account. Three transformations, all relative:

      dt_recency_*    days from the date to the as-of point
      dt_gap_*        days from the EARLIEST date column to each other one, so
                      account age, time-to-first-credit and similar durations
                      come out directly
      dt_dow_*/moy_*  cyclical encodings of the primary date, sin and cos so
                      December and January sit next to each other
      dt_missing_*    a flag per column, because WHICH dates are absent is
                      routinely predictive in banking data

    Pass dates already converted to float days (see `to_days`).
    """
    date_cols = [c for c in date_cols if c in df.columns]
    if not date_cols:
        return pd.DataFrame(index=pd.Index(np.sort(df[id_col].unique()), name=id_col))
    g = df.groupby(id_col)
    D = pd.DataFrame(index=g.size().index)
    vals = {c: g[c].min() for c in date_cols}
    stacked = pd.concat(vals.values(), axis=1)
    if as_of is None:
        as_of = float(np.nanmax(stacked.to_numpy(dtype=float)))
    anchor = stacked.min(axis=1)

    for c in date_cols:
        v = vals[c]
        D[f"dt_recency_{c}"] = np.log1p(np.maximum(as_of - v, 0))
        D[f"dt_missing_{c}"] = v.isna().astype(float)
        if c != date_cols[0]:
            D[f"dt_gap_{c}"] = np.log1p(np.maximum(v - anchor, 0))

    primary = vals[date_cols[0]]
    dow = primary % 7.0
    moy = primary % 365.25
    D["dt_dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    D["dt_dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    D["dt_moy_sin"] = np.sin(2 * np.pi * moy / 365.25)
    D["dt_moy_cos"] = np.cos(2 * np.pi * moy / 365.25)
    D.index.name = id_col
    return D.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# ==========================================================================
# 2. Composition and dynamics
# ==========================================================================
def _clr(P: np.ndarray) -> np.ndarray:
    """Centred log-ratio. Shares live on a simplex; Euclidean distance between
    raw share vectors is wrong, and this is the standard repair."""
    L = np.log(np.clip(P, 1e-6, None))
    return L - L.mean(axis=1, keepdims=True)


def composition_features(events: pd.DataFrame, id_col: str,
                         channel_col: str | None,
                         amount_col: str | None) -> pd.DataFrame:
    """Where the money goes (CLR shares) and how it is sized."""
    idx = pd.Index(np.sort(events[id_col].unique()), name=id_col)
    out = pd.DataFrame(index=idx)

    if channel_col is not None:
        ct = pd.crosstab(events[id_col], events[channel_col]).reindex(idx).fillna(0)
        P = ct.to_numpy(dtype=float)
        P = P / np.maximum(P.sum(1, keepdims=True), 1)
        C = _clr(P)
        for j, c in enumerate(ct.columns):
            out[f"cp_clr_{c}"] = C[:, j]
        p = np.clip(P, 1e-12, None)
        out["cp_channel_entropy"] = -(p * np.log(p)).sum(1)

    if amount_col is not None:
        a = events.groupby(id_col)[amount_col]
        la = np.log1p(events[amount_col].clip(lower=0))
        g = la.groupby(events[id_col])
        out["cp_log_amt_mean"] = g.mean().reindex(idx)
        out["cp_log_amt_sd"] = g.std().reindex(idx).fillna(0)
        mx, sm = a.max().reindex(idx), a.sum().reindex(idx)
        out["cp_max_share"] = (mx / np.maximum(sm, EPS)).clip(0, 1)
        r = (events[amount_col] % 1000 == 0).astype(float)
        out["cp_round_share"] = r.groupby(events[id_col]).mean().reindex(idx)
    return out.fillna(0.0)


class StateAlphabet:
    """Discretises events into a small vocabulary of event types, so ordering
    can be summarised by a transition matrix."""

    def __init__(self, cfg: SegConfig):
        self.cfg = cfg
        self.qt = self.km = None
        self.cols: list[str] = []

    def fit(self, events: pd.DataFrame, cols: Sequence[str]) -> "StateAlphabet":
        self.cols = list(cols)
        X = np.nan_to_num(events[self.cols].to_numpy(dtype=float))
        rng = np.random.default_rng(self.cfg.random_state)
        m = min(self.cfg.state_fit_sample, len(X))
        i = rng.choice(len(X), m, replace=False) if m < len(X) else np.arange(len(X))
        self.qt = QuantileTransformer(output_distribution="normal",
                                      n_quantiles=min(1000, max(10, m)),
                                      subsample=200_000,
                                      random_state=self.cfg.random_state)
        Q = self.qt.fit_transform(X[i])
        self.km = KMeans(self.cfg.n_states, n_init=10,
                         random_state=self.cfg.random_state).fit(Q)
        return self

    def transform(self, events: pd.DataFrame, chunk: int = 500_000) -> np.ndarray:
        X = np.nan_to_num(events[self.cols].to_numpy(dtype=float))
        o = np.empty(len(X), dtype=np.int32)
        for i in range(0, len(X), chunk):
            o[i:i + chunk] = self.km.predict(self.qt.transform(X[i:i + chunk]))
        return o

    def describe(self, events: pd.DataFrame, states: np.ndarray, top: int = 3) -> pd.DataFrame:
        raw = events[self.cols].copy()
        raw["__s__"] = states
        mu, sd = raw[self.cols].mean(), raw[self.cols].std().replace(0, 1)
        P = (raw.groupby("__s__")[self.cols].mean() - mu) / sd
        rows = []
        for s in P.index:
            z = P.loc[s].sort_values(key=np.abs, ascending=False).head(top)
            rows.append({"state": int(s),
                         "share": float((states == s).mean()),
                         "signature": ", ".join(
                             f"{k} {'+' if v > 0 else '-'}{abs(v):.1f}sd" for k, v in z.items())})
        return pd.DataFrame(rows)


def transition_features(events: pd.DataFrame, id_col: str, time_col: str,
                        states: np.ndarray, n_states: int,
                        alpha: float = 0.5) -> pd.DataFrame:
    """Row-normalised, smoothed transition matrix per account, in CLR space.

    This is the ordering evidence. Two accounts with identical monthly totals
    but different sequences differ here and nowhere else in the representation.
    """
    df = pd.DataFrame({"id": events[id_col].to_numpy(), "t": events[time_col].to_numpy(),
                       "s": states}).sort_values(["id", "t"], kind="mergesort")
    code, uniq = pd.factorize(df["id"], sort=True)
    s = df["s"].to_numpy()
    same = code[1:] == code[:-1]
    row = code[:-1][same]
    col = s[:-1][same].astype(np.int64) * n_states + s[1:][same].astype(np.int64)
    n_acc = len(uniq)
    C = np.zeros((n_acc, n_states * n_states))
    np.add.at(C, (row, col), 1.0)

    C = C.reshape(n_acc, n_states, n_states) + alpha
    P = C / C.sum(2, keepdims=True)
    F = np.concatenate([_clr(P[:, i, :]) for i in range(n_states)], axis=1)
    names = [f"dy_t{i}_{j}" for i in range(n_states) for j in range(n_states)]
    out = pd.DataFrame(F, index=pd.Index(uniq, name=id_col), columns=names)
    T = P.reshape(n_acc, n_states, n_states)
    out["dy_self_trans"] = np.einsum("nii->n", T) / n_states
    out["dy_entropy"] = (-(T * np.log(T + 1e-12)).sum(2)).mean(1)
    return out


# ==========================================================================
# 3. Representation
# ==========================================================================
class AccountRepresentation:
    """One named feature row per account, tagged by block.

    Two modes, decided by the data rather than by you:

      "events"  several rows per account with a usable date. Full pipeline:
                timing, Hawkes, transition dynamics, composition, scale.
      "static"  one row per account. There is no sequence, so timing, Hawkes
                and transitions are not computed - not skipped for speed, but
                because they are undefined on a single observation. What
                remains is date-derived durations plus whatever numeric and
                categorical columns you pass.

    In static mode this is a carefully built static clustering and nothing more.
    The sequence and cascade thesis is not weakened by it; it is simply untested,
    and testing it needs transaction-level rows.
    """

    def __init__(self, cfg: SegConfig):
        self.cfg = cfg
        self.alphabet: StateAlphabet | None = None
        self.blocks: dict[str, list[str]] = {}
        self.event_feature_cols: list[str] = []
        self.as_of_: float | None = None

    def fit_transform(self, df: pd.DataFrame, id_col: str,
                      time_col: str | None = None,
                      event_feature_cols: Sequence[str] | None = None,
                      channel_col: str | None = None,
                      amount_col: str | None = None,
                      date_cols: Sequence[str] | None = None,
                      numeric_cols: Sequence[str] | None = None,
                      fit: bool = True) -> pd.DataFrame:
        cfg = self.cfg
        parts, blocks = [], {}
        idx = pd.Index(np.sort(df[id_col].unique()), name=id_col)

        # ---- date block (available in both modes) -------------------------
        if date_cols:
            if fit:
                self.as_of_ = None
            dt = date_features(df, id_col, date_cols, as_of=self.as_of_)
            if fit:
                # freeze the as-of point, or recency shifts every time you score
                self.as_of_ = float(np.nanmax(
                    df.groupby(id_col)[list(date_cols)].min().to_numpy(dtype=float)))
            blocks["dates"] = list(dt.columns)
            parts.append(dt.reindex(idx))

        # ---- sequence blocks (events mode only) ---------------------------
        if cfg.mode == "events" and time_col is not None:
            tm = timing_features(df, id_col, time_col, cfg)
            hk = hawkes_features(df, id_col, time_col, cfg)
            timing = tm.join(hk)
            blocks["timing"] = list(timing.columns)
            parts.append(timing.reindex(idx))

            if event_feature_cols:
                if fit:
                    self.event_feature_cols = list(event_feature_cols)
                    self.alphabet = StateAlphabet(cfg).fit(df, event_feature_cols)
                st = self.alphabet.transform(df)
                dy = transition_features(df, id_col, time_col, st,
                                         cfg.n_states, cfg.dirichlet_alpha)
                blocks["dynamics"] = list(dy.columns)
                parts.append(dy.reindex(idx))

        # ---- composition ---------------------------------------------------
        cp = composition_features(df, id_col, channel_col, amount_col)
        if cp.shape[1]:
            blocks["composition"] = list(cp.columns)
            parts.append(cp.reindex(idx))

        # ---- plain numeric columns ----------------------------------------
        if numeric_cols:
            num = df.groupby(id_col)[list(numeric_cols)].mean()
            num.columns = [f"nm_{c}" for c in num.columns]
            blocks["numeric"] = list(num.columns)
            parts.append(num.reindex(idx))

        # ---- scale ----------------------------------------------------------
        n = df.groupby(id_col).size().reindex(idx).fillna(0)
        sc = pd.DataFrame({"sc_log_n_events": np.log1p(n)}, index=idx)
        if amount_col is not None:
            sc["sc_log_total_amt"] = np.log1p(
                df.groupby(id_col)[amount_col].sum().reindex(idx).fillna(0))
        blocks["scale"] = list(sc.columns)
        parts.append(sc)

        F = pd.concat(parts, axis=1)
        if fit:
            self.blocks = blocks
        return F.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# ==========================================================================
# 4. Embedding
# ==========================================================================
class BlockEmbedding:
    """Per block: rank-Gaussianise, whiten with PCA, scale by block weight.

    Block weighting is the whole point. Without it a 64-column transition block
    outvotes an 11-column timing block by construction, and the clusters are
    decided by whichever evidence you happened to collect most columns of.
    """

    def __init__(self, cfg: SegConfig):
        self.cfg = cfg
        self.qt: dict[str, QuantileTransformer] = {}
        self.pca: dict[str, PCA] = {}
        self.blocks: dict[str, list[str]] = {}
        self.sup_model: LogisticRegression | None = None
        self.sup_stats: tuple[float, float] = (0.0, 1.0)
        self.dim_names_: list[str] = []

    def _unsup(self, F: pd.DataFrame, fit: bool) -> np.ndarray:
        cols, names = [], []
        for b, cs in self.blocks.items():
            cs = [c for c in cs if c in F.columns]
            if not cs:
                continue
            X = F[cs].to_numpy(dtype=float)
            if fit:
                self.qt[b] = QuantileTransformer(
                    output_distribution="normal",
                    n_quantiles=min(1000, max(10, len(X))), subsample=200_000,
                    random_state=self.cfg.random_state).fit(X)
            Q = self.qt[b].transform(X)
            k = min(self.cfg.block_components, Q.shape[1], max(1, Q.shape[0] - 1))
            if fit:
                self.pca[b] = PCA(k, whiten=True,
                                  random_state=self.cfg.random_state).fit(Q)
            Z = self.pca[b].transform(Q) * float(self.cfg.block_weights.get(b, 1.0))
            cols.append(Z)
            names += [f"{b}_{i}" for i in range(Z.shape[1])]
        if fit:
            self.dim_names_ = names
        return np.hstack(cols)

    def fit_transform(self, F: pd.DataFrame, blocks: dict, y: np.ndarray | None) -> np.ndarray:
        self.blocks = blocks
        Z = self._unsup(F, fit=True)
        if y is None or self.cfg.supervision_weight <= 0 or y.sum() < 20:
            return Z

        # Out-of-fold risk direction. Fitted out-of-fold for the rows that build
        # the clusters, and refitted on everything for scoring new accounts; the
        # two are put on the same scale below, which is the only reason the
        # cluster geometry survives the handover.
        oof = np.zeros(len(Z))
        skf = StratifiedKFold(self.cfg.supervision_folds, shuffle=True,
                              random_state=self.cfg.random_state)
        for tr, te in skf.split(Z, y):
            lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Z[tr], y[tr])
            oof[te] = lr.decision_function(Z[te])
        self.sup_model = LogisticRegression(max_iter=2000,
                                            class_weight="balanced").fit(Z, y)
        self.sup_stats = (float(oof.mean()), float(oof.std() + EPS))
        s = (oof - self.sup_stats[0]) / self.sup_stats[1] * self.cfg.supervision_weight
        self.dim_names_ = self.dim_names_ + ["supervised_risk"]
        return np.hstack([Z, s[:, None]])

    def transform(self, F: pd.DataFrame) -> np.ndarray:
        Z = self._unsup(F, fit=False)
        if self.sup_model is None:
            return Z
        s = (self.sup_model.decision_function(Z) - self.sup_stats[0]) / self.sup_stats[1]
        return np.hstack([Z, (s * self.cfg.supervision_weight)[:, None]])


# ==========================================================================
# 5. Cluster engine
# ==========================================================================
class ClusterEngine:
    """Soft clustering with a swappable backend. Diagonal GMM is the default:
    it gives memberships and a per-component density (so you get novelty for
    free), and at book scale full covariances buy noise, not structure."""

    def __init__(self, cfg: SegConfig, k: int):
        self.cfg = cfg
        self.k = k
        self.model = None

    def _new(self):
        c, rs = self.cfg.engine, self.cfg.random_state
        if c == "gmm_diag":
            return GaussianMixture(self.k, covariance_type="diag", n_init=3,
                                   reg_covar=1e-4, random_state=rs)
        if c == "gmm_full":
            return GaussianMixture(self.k, covariance_type="full", n_init=2,
                                   reg_covar=1e-4, random_state=rs)
        if c == "bgmm":
            return BayesianGaussianMixture(self.k, covariance_type="diag",
                                           weight_concentration_prior=1.0 / self.k,
                                           max_iter=500, random_state=rs)
        if c == "kmeans":
            return KMeans(self.k, n_init=10, random_state=rs)
        raise ValueError(f"unknown engine {c}")

    def fit(self, Z: np.ndarray, y: np.ndarray | None = None) -> "ClusterEngine":
        """Subsampling keeps ALL positives.

        At 800 frauds in 20M accounts, a uniform 300k subsample contains about
        twelve of them. Every fraud-aware quantity downstream - the risk
        separation term in k selection, the per-cluster rate - would then be
        estimated from twelve accounts. Retaining the whole minority class and
        sampling only the majority is the one piece of the imbalance literature
        that is unambiguously worth keeping.
        """
        rng = np.random.default_rng(self.cfg.random_state)
        n = len(Z)
        if n <= self.cfg.fit_sample:
            i = np.arange(n)
        elif y is None:
            i = rng.choice(n, self.cfg.fit_sample, replace=False)
        else:
            pos = np.flatnonzero(y == 1)
            neg = np.flatnonzero(y == 0)
            take = max(self.cfg.fit_sample - len(pos), 1)
            i = np.concatenate([pos, rng.choice(neg, min(take, len(neg)), replace=False)])
        self.model = self._new().fit(Z[i])
        return self

    def predict_proba(self, Z: np.ndarray) -> np.ndarray:
        if self.cfg.engine == "kmeans":
            d = ((Z[:, None, :] - self.model.cluster_centers_[None]) ** 2).sum(-1)
            e = np.exp(-0.5 * (d - d.min(1, keepdims=True)))
            return e / e.sum(1, keepdims=True)
        return self.model.predict_proba(Z)

    def bic(self, Z: np.ndarray) -> float:
        if hasattr(self.model, "bic"):
            m = min(50_000, len(Z))
            return float(self.model.bic(Z[:m]))
        return float("nan")

    def novelty(self, Z: np.ndarray) -> np.ndarray:
        """How atypical is this account *for its own cluster*. Independently
        useful: takeover often looks like a normal account of its type until it
        stops looking like one."""
        if self.cfg.engine == "kmeans":
            d = ((Z[:, None, :] - self.model.cluster_centers_[None]) ** 2).sum(-1)
            return np.sqrt(d.min(1))
        lab = self.predict_proba(Z).argmax(1)
        mu = self.model.means_
        cv = self.model.covariances_
        out = np.empty(len(Z))
        for k in range(mu.shape[0]):
            s = lab == k
            if not s.any():
                continue
            if cv.ndim == 3:
                inv = np.linalg.pinv(cv[k])
                d = Z[s] - mu[k]
                out[s] = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", d, inv, d), 0))
            else:
                out[s] = np.sqrt((((Z[s] - mu[k]) ** 2) / np.maximum(cv[k], EPS)).sum(1))
        return out


def _stability(Z: np.ndarray, cfg: SegConfig, k: int, base: np.ndarray) -> float:
    rng = np.random.default_rng(cfg.random_state)
    m = int(cfg.bootstrap_frac * len(Z))
    aris = []
    for b in range(cfg.n_bootstrap):
        idx = rng.choice(len(Z), m, replace=False)
        c = SegConfig(**{**cfg.__dict__, "random_state": cfg.random_state + 7 * b + 1})
        try:
            lab = ClusterEngine(c, k).fit(Z[idx]).predict_proba(Z).argmax(1)
        except Exception:
            continue
        aris.append(adjusted_rand_score(base, lab))
    return float(np.mean(aris)) if aris else np.nan


def select_k(Z: np.ndarray, y: np.ndarray | None, cfg: SegConfig
             ) -> tuple[ClusterEngine, pd.DataFrame]:
    """Select on stability, size balance and risk separation - not on BIC alone.

    BIC on 20M rows of a mixture that is only approximately Gaussian will march
    monotonically with k and tell you nothing. `risk_spread` is the sd of
    log cluster fraud rates: a solution whose clusters all carry the same risk
    is decoration, whatever its likelihood.
    """
    lo, hi = cfg.k_range
    rows, models = [], {}
    for k in range(lo, hi + 1):
        e = ClusterEngine(cfg, k).fit(Z, y)
        lab = e.predict_proba(Z).argmax(1)
        share = np.bincount(lab, minlength=k) / len(lab)
        r = {"k": k, "bic": e.bic(Z), "smallest_share": float(share.min()),
             "stability_ari": _stability(Z, cfg, k, lab)}
        if y is not None and y.sum() > 0:
            rate = np.array([y[lab == j].mean() if (lab == j).any() else 0 for j in range(k)])
            r["risk_spread"] = float(np.std(np.log(rate + 1e-6)))
            r["max_lift"] = float((rate / max(y.mean(), EPS)).max())
        rows.append(r)
        models[k] = e
        _log(f"k={k}: ARI={r['stability_ari']:.3f} min_share={r['smallest_share']:.3f}"
             + (f" risk_spread={r.get('risk_spread', float('nan')):.2f}"
                f" max_lift={r.get('max_lift', float('nan')):.1f}" if y is not None else ""), 1)

    R = pd.DataFrame(rows)
    ok = R.smallest_share >= cfg.min_cluster_share
    R["score"] = (R.stability_ari.fillna(0)
                  + 0.5 * (R.get("risk_spread", pd.Series(0, index=R.index)).fillna(0)
                           / max(R.get("risk_spread", pd.Series([1])).max(), EPS)))
    R.loc[~ok, "score"] *= 0.5
    best = int(R.loc[R.score.idxmax(), "k"])
    _log(f"selected k={best}", 1)
    return models[best], R.sort_values("score", ascending=False)


# ==========================================================================
# 6. End-to-end
# ==========================================================================
class SegmentModel:
    """fit on a good+bad event stream, transform any book.

    Leakage rules, both of which you have to enforce outside this class:
      * every event for a fraud account must predate its fraud event;
      * the honest test is a time-forward split, not a random account split.
    """

    def __init__(self, cfg: SegConfig | None = None):
        self.cfg = cfg or SegConfig()
        self.rep: AccountRepresentation | None = None
        self.emb: BlockEmbedding | None = None
        self.engine: ClusterEngine | None = None
        self.selection_: pd.DataFrame | None = None
        self.cluster_risk_: np.ndarray | None = None
        self.base_rate_: float = 0.0
        self.novelty_ref_: np.ndarray | None = None
        self.supervision_ari_: float = float("nan")
        self.oof_risk_: pd.Series | None = None
        self._cols: dict = {}

    def fit(self, events: pd.DataFrame, labels: pd.Series, id_col: str,
            time_col: str | None = None,
            event_feature_cols: Sequence[str] | None = None,
            channel_col: str | None = None, amount_col: str | None = None,
            date_cols: Sequence[str] | None = None,
            numeric_cols: Sequence[str] | None = None) -> "SegmentModel":
        cfg = self.cfg
        self._cols = dict(id=id_col, time=time_col, feat=list(event_feature_cols or []),
                          channel=channel_col, amount=amount_col,
                          dates=list(date_cols or []), numeric=list(numeric_cols or []))
        _log("=" * 66)
        _log(f"FRAUDSEG [{cfg.mode}]  |  {len(events):,} rows  |  "
             f"{events[id_col].nunique():,} accounts")
        _log("=" * 66)

        _log("[1] representation")
        self.rep = AccountRepresentation(cfg)
        F = self.rep.fit_transform(events, id_col, time_col, event_feature_cols,
                                   channel_col, amount_col, date_cols,
                                   numeric_cols, fit=True)
        _log(f"{F.shape[1]} features in {len(self.rep.blocks)} blocks: "
             f"{ {b: len(c) for b, c in self.rep.blocks.items()} }", 1)

        y = labels.reindex(F.index).fillna(0).to_numpy().astype(int)
        self.base_rate_ = float(y.mean())

        _log("[2] block embedding")
        self.emb = BlockEmbedding(cfg)
        Z = self.emb.fit_transform(F, self.rep.blocks, y)
        _log(f"{Z.shape[1]} coordinates"
             + (" (incl. supervised risk axis)" if self.emb.sup_model else ""), 1)

        _log("[3] clustering")
        if cfg.n_clusters is None:
            self.engine, self.selection_ = select_k(Z, y, cfg)
        else:
            self.engine = ClusterEngine(cfg, cfg.n_clusters).fit(Z, y)

        # Per-cluster fraud rate, empirical-Bayes shrunk toward the base rate.
        # A cluster holding 300 accounts and 2 frauds should not be exported at
        # a rate of 0.0067 as though that were measured.
        lab = self.engine.predict_proba(Z).argmax(1)
        K = int(self.engine.k)
        prior_n = 50.0
        nk = np.bincount(lab, minlength=K).astype(float)
        fk = np.bincount(lab, weights=y.astype(float), minlength=K)
        self.cluster_risk_ = (fk + prior_n * self.base_rate_) / (nk + prior_n)

        # Out-of-fold values for the accounts that BUILT the clusters. A model
        # trained on the in-sample rate is training on a label echo; these are
        # the values to use for those rows, and `transform(..., oof=True)`
        # substitutes them.
        oof = np.full(len(Z), self.base_rate_)
        skf = StratifiedKFold(5, shuffle=True, random_state=cfg.random_state)
        for tr, te in skf.split(Z, y):
            n_t = np.bincount(lab[tr], minlength=K).astype(float)
            f_t = np.bincount(lab[tr], weights=y[tr].astype(float), minlength=K)
            rate = (f_t + prior_n * self.base_rate_) / (n_t + prior_n)
            oof[te] = rate[lab[te]]
        self.oof_risk_ = pd.Series(oof, index=F.index, name="seg_cluster_risk")
        self.novelty_ref_ = np.sort(self.engine.novelty(Z))

        # How much of the partition is the supervised axis? Refit on the
        # unsupervised coordinates alone and compare. A very low ARI means the
        # clusters are a discretised classifier: the cluster features will then
        # be redundant with the classifier you already have, and the segments
        # will not survive a change in fraud mix.
        self.supervision_ari_ = np.nan
        if self.emb.sup_model is not None:
            lab_u = ClusterEngine(cfg, self.engine.k).fit(Z[:, :-1]) \
                        .predict_proba(Z[:, :-1]).argmax(1)
            self.supervision_ari_ = float(adjusted_rand_score(lab_u, lab))
            _log(f"supervision influence: ARI vs unsupervised = "
                 f"{self.supervision_ari_:.3f}", 1)
            if self.supervision_ari_ < 0.25:
                _log("WARNING: the label axis is driving the partition. Lower "
                     "supervision_weight, or accept that these are score bands, "
                     "not behaviour segments.", 1)
        _log("done.")
        return self

    def embed(self, F: pd.DataFrame) -> np.ndarray:
        """Named features -> clustering coordinates. Exposed so diagnostics can
        perturb a feature and watch membership move."""
        return self.emb.transform(F)

    def transform(self, events: pd.DataFrame, oof: bool = False) -> pd.DataFrame:
        c = self._cols
        F = self.rep.fit_transform(events, c["id"], c["time"], c["feat"],
                                   c["channel"], c["amount"], c["dates"],
                                   c["numeric"], fit=False)
        Z = self.emb.transform(F)
        P = self.engine.predict_proba(Z)
        lab = P.argmax(1)
        nov = self.engine.novelty(Z)

        out = pd.DataFrame(index=F.index)
        out["seg_cluster"] = lab
        out["seg_conf"] = P.max(1)
        for k in range(P.shape[1]):
            out[f"seg_post_{k}"] = P[:, k]
        out["seg_entropy"] = -(P * np.log(P + 1e-12)).sum(1)
        out["seg_novelty"] = np.searchsorted(self.novelty_ref_, nov) / len(self.novelty_ref_)
        risk = pd.Series(self.cluster_risk_[lab], index=F.index)
        if oof:
            if self.oof_risk_ is None:
                raise RuntimeError("no out-of-fold risk stored; call fit first")
            risk = self.oof_risk_.reindex(F.index).fillna(risk)
        out["seg_cluster_risk"] = risk.to_numpy()
        out["seg_cluster_lift"] = risk.to_numpy() / max(self.base_rate_, EPS)
        return F.join(out)          # representation + segmentation, one block


# ==========================================================================
# 7. Reporting
# ==========================================================================
def cluster_report(F: pd.DataFrame, y: np.ndarray, base_rate: float | None = None,
                   top_features: int = 3) -> pd.DataFrame:
    """Per cluster: size, risk with a Wilson interval, lift, and what it is."""
    lab = F["seg_cluster"].to_numpy()
    base = base_rate if base_rate is not None else y.mean()
    cols = [c for c in F.columns
            if c.split("_")[0] in PREFIXES and F[c].std() > 0]
    mu, sd = F[cols].mean(), F[cols].std().replace(0, 1)
    rows = []
    for k in sorted(set(lab)):
        s = lab == k
        n, f = int(s.sum()), int(y[s].sum())
        p = f / max(n, 1)
        z = 1.96
        den = 1 + z ** 2 / n if n else 1
        ctr = (p + z ** 2 / (2 * n)) / den if n else 0
        half = (z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den) if n else 0
        d = ((F.loc[s, cols].mean() - mu) / sd).sort_values(key=np.abs, ascending=False)
        rows.append({
            "cluster": k, "n": n, "share": round(n / len(lab), 4), "n_fraud": f,
            "fraud_rate": round(p, 6), "lift": round(p / max(base, EPS), 2),
            "rate_lo": round(max(ctr - half, 0), 6), "rate_hi": round(ctr + half, 6),
            "signature": ", ".join(f"{i} {'+' if v > 0 else '-'}{abs(v):.1f}sd"
                                   for i, v in d.head(top_features).items())})
    return pd.DataFrame(rows).sort_values("lift", ascending=False)


def describe_clusters(F: pd.DataFrame, max_depth: int = 3) -> str:
    """A shallow tree that reproduces the cluster assignment from the named
    features. This is the audit artefact the rule base was meant to be, and it
    is derived from the clustering rather than fitted alongside it."""
    cols = [c for c in F.columns if c.split("_")[0] in PREFIXES]
    t = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=50).fit(
        F[cols], F["seg_cluster"])
    acc = (t.predict(F[cols]) == F["seg_cluster"]).mean()
    return f"# tree agreement with the clustering: {acc:.1%}\n" + \
           export_text(t, feature_names=cols, max_depth=max_depth)


def budget_table(y: np.ndarray, score: np.ndarray, cfg: SegConfig,
                 budgets: Sequence[float] = (0.0001, 0.0005, 0.001, 0.005, 0.01)
                 ) -> pd.DataFrame:
    """Alerts, not AUC. Precision is stated at the TRUE book prevalence; a
    fraud-enriched sample inflates it by roughly n_book / n_fraud_book.

    `reliable` is False where the budget covers fewer than 20 scored accounts -
    the recall in that row is then one or two accounts wide and means nothing.
    Score a larger validation sample before quoting it."""
    y = np.asarray(y).astype(int)
    o = np.argsort(-np.asarray(score, dtype=float))
    ys = y[o]
    rows = []
    for b in budgets:
        k = max(1, int(round(b * len(ys))))
        rec = ys[:k].sum() / max(y.sum(), 1)
        alerts, catch = b * cfg.n_book, rec * cfg.n_fraud_book
        rows.append({"alert_share": b, "n_scored_in_budget": k,
                     "recall": round(rec, 4),
                     "book_alerts": int(alerts), "expected_catches": round(catch, 1),
                     "alerts_per_catch": round(alerts / catch, 1) if catch else np.inf,
                     "lift": round(rec / b, 1) if b else np.nan,
                     "reliable": k >= 20})
    return pd.DataFrame(rows)


def fp_reduction_at_recall(y: np.ndarray, base: np.ndarray, new: np.ndarray,
                           targets: Sequence[float] = (0.5, 0.6, 0.7, 0.8)) -> pd.DataFrame:
    y = np.asarray(y).astype(int)
    rows = []
    for r in targets:
        need = int(np.ceil(r * y.sum()))
        o = {"recall_target": r}
        for nm, s in (("base", base), ("new", new)):
            order = np.argsort(-np.asarray(s, dtype=float))
            k = int(np.searchsorted(np.cumsum(y[order]), need) + 1)
            o[f"{nm}_alert_share"] = round(k / len(y), 5)
            o[f"{nm}_fp"] = int(k - need)
        o["fp_reduction"] = round(1 - o["new_fp"] / max(o["base_fp"], 1), 4)
        rows.append(o)
    return pd.DataFrame(rows)


# ==========================================================================
# PART 4.  DIAGNOSTICS
# ==========================================================================


EPS_IV = 1e-12


def feature_columns(F: pd.DataFrame) -> list[str]:
    """The named, human-readable features - never the seg_* outputs."""
    return [c for c in F.columns
            if c.split("_")[0] in PREFIXES and pd.api.types.is_numeric_dtype(F[c])]


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(ctr - half, 0.0), min(ctr + half, 1.0)


# ==========================================================================
# 1. How good is this cluster?
# ==========================================================================
def cluster_metrics(F: pd.DataFrame, y: np.ndarray, cfg, label_col: str = "seg_cluster",
                    sort_by: str = "lift") -> pd.DataFrame:
    """The full per-cluster scorecard.

    Read `bad_pct` and `good_pct` as a pair. `bad_pct` is capture: the share of
    all fraud that sits in this cluster, i.e. the recall you would get by
    alerting the whole cluster. `good_pct` is what that costs: the share of the
    good population you would alert alongside it. A cluster with 40% capture
    and 38% good_pct is not a control, it is a coin toss with extra steps.

    `bad_rate` is precision *within the training mix*, which is enriched.
    `book_alerts` and `alerts_per_catch` restate everything at the true book
    prevalence, and those are the numbers to quote.
    """
    y = np.asarray(y).astype(int)
    lab = F[label_col].to_numpy()
    n_all, n_bad_all, n_good_all = len(y), int(y.sum()), int((y == 0).sum())
    base = n_bad_all / max(n_all, 1)

    rows = []
    for k in sorted(pd.unique(lab)):
        s = lab == k
        n = int(s.sum())
        nb = int(y[s].sum())
        ng = n - nb
        bad_pct = nb / max(n_bad_all, 1)
        good_pct = ng / max(n_good_all, 1)
        rate = nb / max(n, 1)
        lo, hi = _wilson(nb, n)
        woe = np.log((bad_pct + EPS_IV) / (good_pct + EPS_IV))
        alerts = (n / n_all) * cfg.n_book
        catches = bad_pct * cfg.n_fraud_book
        rows.append({
            "cluster": k, "n": n, "share": n / n_all,
            "n_good": ng, "n_bad": nb,
            "good_pct": good_pct,          # share of all goods living here
            "bad_pct": bad_pct,            # share of all bads = capture / recall
            "bad_rate": rate,              # precision inside the enriched mix
            "rate_lo": lo, "rate_hi": hi,
            "lift": rate / max(base, EPS_IV),
            "capture_over_cost": bad_pct / max(good_pct, EPS_IV),
            "woe": woe,
            "iv_part": (bad_pct - good_pct) * woe,
            "book_alerts": int(alerts),
            "expected_catches": round(catches, 1),
            "alerts_per_catch": round(alerts / catches, 1) if catches > 0 else np.inf,
            "mean_conf": float(F.loc[s, "seg_conf"].mean()) if "seg_conf" in F else np.nan,
            "mean_novelty": float(F.loc[s, "seg_novelty"].mean()) if "seg_novelty" in F else np.nan,
            "significant": lo > base,      # lift distinguishable from the base rate
        })

    R = pd.DataFrame(rows).sort_values(sort_by, ascending=False).reset_index(drop=True)
    # cumulative view: alert clusters in descending risk order
    R["cum_bad_pct"] = R.bad_pct.cumsum()
    R["cum_good_pct"] = R.good_pct.cumsum()
    R["cum_alert_share"] = R.share.cumsum()
    R["cum_alerts_per_catch"] = np.where(
        R.cum_bad_pct > 0,
        (R.cum_alert_share * cfg.n_book) / (R.cum_bad_pct * cfg.n_fraud_book), np.inf)
    R["ks"] = (R.cum_bad_pct - R.cum_good_pct).abs()
    R.attrs["information_value"] = float(R.iv_part.sum())
    R.attrs["ks"] = float(R.ks.max())
    R.attrs["base_rate"] = base
    return R


# ==========================================================================
# 2. What defines the cluster?
# ==========================================================================
def cluster_drivers(F: pd.DataFrame, cluster, top: int = 12,
                    label_col: str = "seg_cluster") -> pd.DataFrame:
    """One-vs-rest, per named feature.

    Three different notions of "defines", because they disagree and the
    disagreement is informative:

      cohens_d   how far the cluster sits from everyone else, in sd
      auc        how well this feature ALONE identifies the cluster
      coverage   share of the cluster caught by the best single threshold
      purity     share of those caught that really are in the cluster

    High d with low purity means the feature describes the cluster but cannot
    isolate it - fine for a narrative, useless as a rule.
    """
    cols = feature_columns(F)
    m = (F[label_col] == cluster).to_numpy()
    rows = []
    for c in cols:
        a = F.loc[m, c].to_numpy(dtype=float)
        b = F.loc[~m, c].to_numpy(dtype=float)
        if len(a) < 2 or len(b) < 2:
            continue
        va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
        pooled = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) /
                         max(len(a) + len(b) - 2, 1))
        d = (a.mean() - b.mean()) / pooled if pooled > 0 else 0.0

        r = pd.Series(F[c]).rank().to_numpy()
        n1, n0 = m.sum(), (~m).sum()
        auc = (r[m].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

        # best single split by Youden J, on a coarse grid
        qs = np.quantile(F[c], np.linspace(0.02, 0.98, 25))
        side = 1.0 if d >= 0 else -1.0
        best = (0.0, np.nan, 0.0, 0.0)
        for t in np.unique(qs):
            sel = (F[c].to_numpy() >= t) if side > 0 else (F[c].to_numpy() <= t)
            tp = float((sel & m).sum())
            if tp == 0:
                continue
            j = tp / n1 - float((sel & ~m).sum()) / n0
            if j > best[0]:
                best = (j, t, tp / n1, tp / max(sel.sum(), 1))
        rows.append({"feature": c, "cluster_mean": a.mean(), "rest_mean": b.mean(),
                     "cohens_d": d, "auc": auc, "direction": "high" if side > 0 else "low",
                     "threshold": best[1], "coverage": best[2], "purity": best[3],
                     "youden_j": best[0]})

    D = pd.DataFrame(rows)
    D["abs_d"] = D.cohens_d.abs()
    return D.sort_values("abs_d", ascending=False).head(top).drop(columns="abs_d") \
            .reset_index(drop=True)


# ==========================================================================
# 3. What moves membership?
# ==========================================================================
def membership_elasticity(model, F: pd.DataFrame, cluster, n_sample: int = 2000,
                          rel_step: float = 0.1, seed: int = 0) -> pd.DataFrame:
    """dP(cluster) / d(feature), evaluated at each account's own position.

    A local derivative, so it answers "which feature would I have to move to
    push this account out of the cluster", which is a different question from
    "which feature describes the cluster" and often has a different answer.
    """
    cols = feature_columns(F)
    rng = np.random.default_rng(seed)
    i = rng.choice(len(F), min(n_sample, len(F)), replace=False)
    S = F.iloc[i][cols].copy()
    kidx = int(cluster)

    base = model.engine.predict_proba(model.embed(S))[:, kidx]
    out = []
    for c in cols:
        step = rel_step * max(F[c].std(), EPS_IV)
        up = S.copy()
        up[c] = up[c] + step
        dn = S.copy()
        dn[c] = dn[c] - step
        g = (model.engine.predict_proba(model.embed(up))[:, kidx] -
             model.engine.predict_proba(model.embed(dn))[:, kidx]) / (2 * step)
        out.append({"feature": c,
                    "mean_dP_dx": float(np.mean(g)),
                    "mean_abs_dP_dx": float(np.mean(np.abs(g))),
                    "dP_per_sd": float(np.mean(g) * F[c].std())})
    # sorted by dP per standard deviation: raw dP/dx just ranks features by
    # how small their units are
    return (pd.DataFrame(out).assign(_a=lambda t: t.dP_per_sd.abs())
            .sort_values("_a", ascending=False).drop(columns="_a").reset_index(drop=True))


def cluster_sensitivity(model, F: pd.DataFrame, feature: str, cluster,
                        n_grid: int = 15, n_sample: int = 1500,
                        q: tuple = (0.02, 0.98), seed: int = 0,
                        members_only: bool = False) -> pd.DataFrame:
    """Response curve: sweep one feature across its range, holding every other
    feature at each account's own observed value, and watch P(cluster) move.

    This is a partial dependence, with the usual caveat and a sharper one here:
    the features are correlated by construction (hk_branch_1h and tm_burstiness
    measure overlapping things), so a sweep that holds the others fixed asks a
    counterfactual the data never contains. Read the SHAPE and the crossing
    point, not the absolute level.

    `members_only=True` sweeps only accounts currently IN the cluster, which
    answers the operational question - at what value of this feature do my
    members fall out of the segment - and gives a far more legible curve than
    sweeping the whole population.
    """
    cols = feature_columns(F)
    pool = F[F["seg_cluster"] == cluster] if members_only else F
    rng = np.random.default_rng(seed)
    i = rng.choice(len(pool), min(n_sample, len(pool)), replace=False)
    S = pool.iloc[i][cols].copy()
    grid = np.linspace(*np.quantile(F[feature], q), n_grid)
    kidx = int(cluster)

    rows = []
    for v in grid:
        T = S.copy()
        T[feature] = v
        P = model.engine.predict_proba(model.embed(T))
        rows.append({feature: float(v),
                     "P_cluster": float(P[:, kidx].mean()),
                     "share_assigned": float((P.argmax(1) == kidx).mean())})
    return pd.DataFrame(rows)


def cluster_boundary(model, F: pd.DataFrame, feature: str, cluster,
                     members_only: bool = True, **kw) -> dict:
    """Where the sweep crosses 50% assignment - the operational 'this value is
    what puts an account in this segment' number."""
    S = cluster_sensitivity(model, F, feature, cluster,
                            members_only=members_only, **kw)
    x, s = S[feature].to_numpy(), S.share_assigned.to_numpy()
    cross = np.flatnonzero(np.diff(np.sign(s - 0.5)) != 0)
    return {"feature": feature, "cluster": cluster,
            "max_share": float(s.max()), "at_value": float(x[int(s.argmax())]),
            "crossings": [float(x[c]) for c in cross]}


# ==========================================================================
# 4. Should I believe it?   (ported from the original notebook)
# ==========================================================================
def bootstrap_jaccard(model, F: pd.DataFrame, n_boot: int = 15, frac: float = 0.7,
                      seed: int = 0) -> pd.DataFrame:
    """Hennig-style resampling stability, per cluster.

    ARI answers "is the partition reproducible". Per-cluster Jaccard answers
    the question you actually act on: WHICH clusters are real. Below 0.60, do
    not write a control around it however good its lift looks.
    """
    Z = model.embed(F)
    base = model.engine.predict_proba(Z).argmax(1)
    ids = sorted(set(base))
    rng = np.random.default_rng(seed)
    jac = {c: [] for c in ids}
    aris = []
    for b in range(n_boot):
        idx = rng.choice(len(Z), int(frac * len(Z)), replace=False)
        cfg = SegConfig(**{**model.cfg.__dict__, "random_state": seed + 13 * b + 1})
        try:
            lab = ClusterEngine(cfg, model.engine.k).fit(Z[idx]).predict_proba(Z).argmax(1)
        except Exception:
            continue
        aris.append(adjusted_rand_score(base, lab))
        for c in ids:
            a = base == c
            best = 0.0
            for d in set(lab):
                bm = lab == d
                inter = np.logical_and(a, bm).sum()
                if inter:
                    best = max(best, inter / np.logical_or(a, bm).sum())
            jac[c].append(best)
    out = pd.DataFrame({"cluster": ids,
                        "jaccard_mean": [np.mean(jac[c]) if jac[c] else np.nan for c in ids],
                        "jaccard_min": [np.min(jac[c]) if jac[c] else np.nan for c in ids]})
    out["verdict"] = np.where(out.jaccard_mean < 0.60, "NOT REPRODUCIBLE",
                              np.where(out.jaccard_mean < 0.75, "usable", "solid"))
    out.attrs["ari_mean"] = float(np.mean(aris)) if aris else np.nan
    return out


def prediction_strength(model, F: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Tibshirani prediction strength: split in half, cluster each half, and ask
    how often pairs that the B-clustering puts together are also put together
    by the A-model. The minimum across clusters is the statistic; above 0.80
    the k is supported."""
    Z = model.embed(F)
    ia, ib = train_test_split(np.arange(len(Z)), test_size=0.5, random_state=seed)
    cfg = SegConfig(**model.cfg.__dict__)
    A = ClusterEngine(cfg, model.engine.k).fit(Z[ia])
    B = ClusterEngine(cfg, model.engine.k).fit(Z[ib])
    la = A.predict_proba(Z[ib]).argmax(1)
    lb = B.predict_proba(Z[ib]).argmax(1)
    rows = []
    for c in sorted(set(lb)):
        s = lb == c
        n = int(s.sum())
        if n < 2:
            rows.append({"cluster": c, "n": n, "prediction_strength": np.nan})
            continue
        cnt = np.bincount(la[s], minlength=model.engine.k).astype(float)
        same = (cnt * (cnt - 1)).sum()
        rows.append({"cluster": c, "n": n,
                     "prediction_strength": same / (n * (n - 1))})
    out = pd.DataFrame(rows)
    out.attrs["min_prediction_strength"] = float(out.prediction_strength.min())
    return out


def cluster_separability(F: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Can a classifier recover the cluster from the named features? A cluster
    with low recall here is residue - whatever separates it lives in the
    embedding and cannot be stated in terms a person can read."""
    cols = feature_columns(F)
    y = F["seg_cluster"].to_numpy()
    tr, te = train_test_split(np.arange(len(F)), test_size=0.3,
                              random_state=seed, stratify=y)
    g = HistGradientBoostingClassifier(max_iter=200, random_state=seed).fit(
        F.iloc[tr][cols], y[tr])
    p = g.predict(F.iloc[te][cols])
    ids = sorted(set(y))
    rec = recall_score(y[te], p, average=None, labels=ids, zero_division=0)
    return pd.DataFrame({"cluster": ids, "classifier_recall": rec,
                         "verdict": np.where(rec < 0.5, "mush", "separable")})


# ==========================================================================
# 5. What do I alert on?   (rule pricing, ported)
# ==========================================================================
def _paths(tree, names):
    t = tree.tree_
    out = []

    def walk(node, conds):
        if t.feature[node] == _tree.TREE_UNDEFINED:
            out.append((conds, t.value[node][0]))
            return
        f, thr = names[t.feature[node]], t.threshold[node]
        walk(t.children_left[node], conds + [(f, "<=", thr)])
        walk(t.children_right[node], conds + [(f, ">", thr)])

    walk(0, [])
    return out


def cluster_rules(F: pd.DataFrame, y: np.ndarray, cfg, cluster,
                  max_depth: int = 3, min_leaf: int = 30) -> pd.DataFrame:
    """Readable AND-rules for one cluster, each priced in book alert volume.

    A rule that covers 90% of the cluster and 8% of the good population is
    1.6 million alerts on a 2-crore book. Coverage without that second number
    is a description, not a control - which is the whole reason this table
    exists rather than a list of thresholds.
    """
    cols = feature_columns(F)
    y = np.asarray(y).astype(int)
    m = (F["seg_cluster"] == cluster).to_numpy().astype(int)
    tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_leaf,
                                  random_state=0).fit(F[cols], m)
    X = F[cols]
    n_bad_all, n_good_all = max(int(y.sum()), 1), max(int((y == 0).sum()), 1)

    rows = []
    for conds, val in _paths(tree, cols):
        if val.sum() == 0 or val.argmax() != 1:
            continue
        sel = np.ones(len(F), dtype=bool)
        for f, op, t in conds:
            sel &= (X[f].to_numpy() <= t) if op == "<=" else (X[f].to_numpy() > t)
        if sel.sum() == 0:
            continue
        good_hit = float(((y == 0) & sel).sum()) / n_good_all
        bad_hit = float(((y == 1) & sel).sum()) / n_bad_all
        alerts = good_hit * cfg.n_book          # the book is ~all good
        catches = bad_hit * cfg.n_fraud_book
        rows.append({
            "rule": " AND ".join(f"{f} {op} {t:.4g}" for f, op, t in conds),
            "cluster_coverage": float(sel[m == 1].mean()),
            "bad_capture": bad_hit, "good_hit_rate": good_hit,
            "est_book_alerts": int(alerts),
            "expected_catches": round(catches, 2),
            "alerts_per_catch": round(alerts / catches, 1) if catches > 0 else np.inf,
            "lift": round(bad_hit / max(good_hit, EPS_IV), 2)})
    R = pd.DataFrame(rows)
    if len(R):
        R["operable"] = (R.est_book_alerts < 0.02 * cfg.n_book) & (R.expected_catches >= 1)
        R = R.sort_values("lift", ascending=False).reset_index(drop=True)
    return R


def feature_bounds(F: pd.DataFrame, y: np.ndarray, cluster, top: int = 8,
                   q: tuple = (0.05, 0.95)) -> pd.DataFrame:
    """Per-feature operating range for a cluster, with selectivity.

    `selectivity` is the share of the WHOLE population inside the band. A band
    holding 90% of the cluster and 83% of everyone is not a filter.
    """
    drv = cluster_drivers(F, cluster, top=top)
    m = (F["seg_cluster"] == cluster).to_numpy()
    rows = []
    for c in drv.feature:
        lo, hi = np.quantile(F.loc[m, c], q)
        inside = (F[c] >= lo) & (F[c] <= hi)
        rows.append({"feature": c, "lower": lo, "upper": hi,
                     "cluster_in_range": float(inside[m].mean()),
                     "population_in_range": float(inside.mean()),
                     "bad_in_range": float(inside[y == 1].mean()) if y.sum() else np.nan,
                     "selectivity": float(1 - inside.mean())})
    return pd.DataFrame(rows)


# ==========================================================================
# Convenience
# ==========================================================================
def full_report(model, F: pd.DataFrame, y: np.ndarray, cfg,
                n_boot: int = 10) -> dict:
    """Everything above, in one call. Returns a dict of DataFrames."""
    M = cluster_metrics(F, y, cfg)
    out = {
        "metrics": M,
        "stability": bootstrap_jaccard(model, F, n_boot=n_boot),
        "prediction_strength": prediction_strength(model, F),
        "separability": cluster_separability(F),
    }
    out["drivers"] = pd.concat(
        [cluster_drivers(F, k, top=6).assign(cluster=k) for k in M.cluster],
        ignore_index=True)
    summ = out["metrics"][["cluster", "share", "bad_pct", "good_pct", "lift",
                           "alerts_per_catch"]].merge(
        out["stability"][["cluster", "jaccard_mean", "verdict"]], on="cluster") \
        .merge(out["separability"][["cluster", "classifier_recall"]], on="cluster")
    summ["deployable"] = (summ.jaccard_mean >= 0.60) & (summ.lift >= 2) & \
                         (summ.classifier_recall >= 0.5)
    out["summary"] = summ
    return out


# ==========================================================================
# PART 5.  DATA: two files in, one mixed frame out
# ==========================================================================

def to_days(s: pd.Series, origin: pd.Timestamp | None = None) -> pd.Series:
    """Any date representation -> float days since a common origin.

    The origin must be the SAME for both datasets and for every later scoring
    run, or every recency and duration silently shifts. It is pinned in
    `DATE_ORIGIN` below rather than inferred per file.
    """
    v = pd.to_datetime(s, errors="coerce")
    o = origin if origin is not None else DATE_ORIGIN
    return (v - o).dt.total_seconds() / 86400.0


def profile_input(df: pd.DataFrame, id_col: str,
                  date_cols: Sequence[str]) -> dict:
    """Look at the data before modelling it, and say plainly what is possible.

    Two questions decide the whole architecture, and neither can be answered
    from a schema:

      1. How many rows per account? One row means there is no sequence, so
         Hawkes, transitions and inter-arrival statistics are undefined - not
         weak, undefined.
      2. Do the dates carry a time of day? If every timestamp is midnight, the
         data is date-resolution and sub-day Hawkes timescales would be fitting
         noise created by ties.

    Returns a dict you pass straight into SegConfig.
    """
    n_rows, n_acc = len(df), int(df[id_col].nunique())
    rpa = n_rows / max(n_acc, 1)
    mode = "events" if rpa > 1.2 else "static"

    def as_days(c):
        """Accept either raw dates or the float-days form to_days() produces."""
        if pd.api.types.is_numeric_dtype(df[c]):
            return df[c].astype(float)
        return to_days(df[c])

    intraday = False
    for c in date_cols:
        if c in df.columns:
            v = as_days(c).dropna().to_numpy()
            # a fractional part means a time of day survived the conversion
            if len(v) and np.abs(v - np.round(v)).max() > 1e-6:
                intraday = True
                break

    print("-" * 70)
    print(f"rows                 {n_rows:,}")
    print(f"accounts             {n_acc:,}")
    print(f"rows per account     {rpa:.2f}")
    print(f"detected mode        {mode}")
    print(f"timestamps intraday  {intraday}")
    for c in date_cols:
        if c in df.columns:
            v = as_days(c)
            lo = DATE_ORIGIN + pd.Timedelta(days=float(v.min()))
            hi = DATE_ORIGIN + pd.Timedelta(days=float(v.max()))
            print(f"  {c:<22} null {v.isna().mean():6.1%}   "
                  f"{str(lo)[:10]} .. {str(hi)[:10]}")
    if mode == "static":
        print("\nONE ROW PER ACCOUNT. Timing, Hawkes and transition blocks will be")
        print("skipped - they are undefined without a sequence. What runs is a")
        print("static clustering over date-derived and numeric blocks. To test the")
        print("cascade thesis at all you need transaction-level rows.")
    elif not intraday:
        print("\nDATE RESOLUTION, no time of day. Hawkes runs at day/week/month")
        print("scales; sub-day scales are dropped. The night-activity share, often")
        print("one of the better single features in takeover, is unavailable.")
    print("-" * 70)
    return {"mode": mode, "intraday": intraday, "rows_per_account": rpa,
            "n_accounts": n_acc}


def load_and_mix(good_path: str, fraud_path: str, id_col: str,
                 date_cols: Sequence[str],
                 usecols: Sequence[str] | None = None,
                 good_sample_accounts: int | None = None,
                 seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """Read the two files, mix them, and return (frame, labels).

    Three things this does that a naive concat does not:

    *   **Samples by ACCOUNT, not by row.** Taking 500k random rows out of 7.8M
        gives you fragments of accounts, and an account seen through a fragment
        of its history has the wrong transition counts and the wrong Hawkes fit.
        Whole accounts, or nothing.
    *   **Keeps every fraud account.** With 26k fraud against 7.8M good, uniform
        sampling would leave you estimating every fraud-aware quantity from a
        handful of accounts.
    *   **Records the sampling factor**, because it distorts one metric. See the
        note under `prevalence_note()`.

    Memory: 7.8M rows is comfortable if you prune columns. Pass `usecols` -
    reading 1100 columns you will not use is what turns this into a 40 GB
    problem.
    """
    rng = np.random.default_rng(seed)
    rd = (lambda f: pd.read_parquet(f, columns=list(usecols) if usecols else None)) \
        if str(good_path).endswith(("parquet", "pq")) else \
        (lambda f: pd.read_csv(f, usecols=list(usecols) if usecols else None))

    print(f"reading {fraud_path} ...")
    bad = rd(fraud_path)
    print(f"reading {good_path} ...  (7.8M rows takes a minute)")
    good = rd(good_path)

    for df in (good, bad):
        for c in date_cols:
            if c in df.columns:
                df[c] = to_days(df[c])

    if good_sample_accounts is not None:
        ids = good[id_col].unique()
        if len(ids) > good_sample_accounts:
            keep = rng.choice(ids, good_sample_accounts, replace=False)
            good = good[good[id_col].isin(keep)]
            print(f"sampled {good_sample_accounts:,} good accounts "
                  f"({len(good):,} rows) out of {len(ids):,}")

    # guard against the same id appearing in both files
    overlap = set(bad[id_col].unique()) & set(good[id_col].unique())
    if overlap:
        print(f"WARNING: {len(overlap):,} ids appear in BOTH files; "
              f"treating them as fraud and dropping the good copies")
        good = good[~good[id_col].isin(overlap)]

    good["__y__"], bad["__y__"] = 0, 1
    shared = [c for c in good.columns if c in bad.columns]
    dropped = (set(good.columns) ^ set(bad.columns))
    if dropped:
        print(f"NOTE: {len(dropped)} columns are not in both files and were "
              f"dropped, e.g. {sorted(dropped)[:6]}")
    df = pd.concat([good[shared], bad[shared]], ignore_index=True)

    # shuffle rows so nothing downstream can depend on file order, then restore
    # per-account row order (the sequence layers need it and sort internally)
    df = df.iloc[rng.permutation(len(df))].reset_index(drop=True)

    labels = df.groupby(id_col)["__y__"].max().astype(int)
    labels.name = "is_fraud"
    df = df.drop(columns="__y__")

    n_bad = int(labels.sum())
    print(f"\nmixed: {len(df):,} rows | {len(labels):,} accounts | "
          f"{n_bad:,} fraud ({labels.mean():.3%})")
    return df, labels


def prevalence_note(labels: pd.Series, cfg: SegConfig) -> None:
    """The one thing downsampling breaks, stated once so nobody is surprised.

    Sampling good accounts changes the mix, so `bad_rate` and `lift` in the
    scorecard are computed against an ENRICHED base rate and are not the book
    numbers. Three columns are unaffected and are the ones to quote:

        bad_pct             share of all fraud in the cluster - a within-class
                            quantity, so the good sampling cannot touch it
        capture_over_cost   bad_pct / good_pct, both within-class
        book_alerts,        computed from cfg.n_book and cfg.n_fraud_book, not
        alerts_per_catch    from the sample mix at all
    """
    mix = float(labels.mean())
    book = cfg.n_fraud_book / cfg.n_book
    print(f"\nsample prevalence {mix:.3%} vs book prevalence {book:.5%} "
          f"({mix / max(book, 1e-12):.0f}x enriched)")
    print("  -> `lift` and `bad_rate` are mix-dependent. Quote `bad_pct`,")
    print("     `capture_over_cost` and `alerts_per_catch` instead.")


# ==========================================================================
# PART 6.  WORKED EXAMPLE
# ==========================================================================


# ==========================================================================
# EDIT THIS BLOCK
# ==========================================================================
GOOD_PATH = "good.parquet"        # 7.8M rows
FRAUD_PATH = "fraud.parquet"      # 26k rows

ID = "account_number"             # the account key, present in both files

# Date columns. The FIRST one is treated as primary: its day-of-week and
# month-of-year go into the cyclical features, and in events mode it is the
# event time. Put your transaction date first.
DATE_COLS = ["tran_dt", "first_credit_dt", "acct_open_dt"]
TIME = "tran_dt"                  # None in static mode; set by main()

# Optional. Leave as None if you do not have them.
EVENT_FEATURES = None             # e.g. ["log_amount"] - defines the state alphabet
CHANNEL = None                    # e.g. "channel" / "tran_mode"
AMOUNT = None                     # e.g. "tran_amt"
NUMERIC_COLS = None               # any other per-account numeric columns

# Pin this. Every recency and duration is measured from it, so it must not
# move between the training run and any later scoring run.
DATE_ORIGIN = pd.Timestamp("2000-01-01")

# Read only what you need. With 7.8M rows, pulling columns you will not use is
# what turns this from a two-minute job into an out-of-memory one.
USECOLS = None                    # e.g. [ID] + DATE_COLS + [AMOUNT, CHANNEL]

# Accounts, not rows. None keeps all 7.8M rows' worth of accounts; set a number
# if memory bites. Every fraud account is kept regardless.
GOOD_SAMPLE_ACCOUNTS = None

CFG = SegConfig(
    n_states=8,
    k_range=(4, 10),
    engine="gmm_diag",          # gmm_diag | gmm_full | bgmm | kmeans
    supervision_weight=0.25,    # 0 = fully unsupervised; watch supervision_ari_
    n_bootstrap=8,
    fit_sample=300_000,         # cap on rows the GMM sees; all fraud is kept
    n_book=20_000_000,          # <-- your real book size
    n_fraud_book=800,           # <-- your real fraud count on that book
)


def _h(title: str) -> None:
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


# ==========================================================================
# Stages
# ==========================================================================
def stage_5(cfg):
    """Load, profile, mix, split.

    The profile runs BEFORE anything is modelled, because it decides the
    architecture: one row per account means the sequence blocks are undefined,
    and no amount of configuration recovers them.

    On the split: this is a random account split, which is the weaker test. If
    you have a fraud date, replace it with a time-forward one - fit on
    everything before a cut date, test after it. A random split lets the model
    see the same fraud wave on both sides and flatters any sequence method.
    """
    df, labels = load_and_mix(GOOD_PATH, FRAUD_PATH, ID, DATE_COLS,
                              usecols=USECOLS,
                              good_sample_accounts=GOOD_SAMPLE_ACCOUNTS,
                              seed=cfg.random_state)

    info = profile_input(df, ID, DATE_COLS)
    cfg.mode = info["mode"]
    cfg.intraday = info["intraday"]
    if cfg.intraday:
        cfg.hawkes_timescales = (0.04, 1.0, 7.0)      # ~1h, 1d, 7d in days
        cfg.hawkes_labels = ("1h", "1d", "7d")

    prevalence_note(labels, cfg)

    tr_ids, te_ids = train_test_split(labels.index.to_numpy(), test_size=0.35,
                                      stratify=labels.values,
                                      random_state=cfg.random_state)
    print(f"\ntrain {len(tr_ids):,} accounts | test {len(te_ids):,} accounts")
    return df, labels, tr_ids, te_ids


def stage_6(cfg, events, labels, tr_ids, te_ids):
    """Fit. Watch two lines in the log: the block sizes (does one block dwarf
    the rest?) and `supervision influence` (is the label axis driving it?)."""
    ev_tr = events[events[ID].isin(tr_ids)]
    ev_te = events[events[ID].isin(te_ids)]
    time_col = TIME if cfg.mode == "events" else None

    model = SegmentModel(cfg).fit(ev_tr, labels.loc[tr_ids], ID, time_col,
                                  EVENT_FEATURES, CHANNEL, AMOUNT,
                                  date_cols=DATE_COLS, numeric_cols=NUMERIC_COLS)

    F_tr = model.transform(ev_tr, oof=True)    # training rows: no label echo
    F_te = model.transform(ev_te)
    y_tr = labels.reindex(F_tr.index).to_numpy()
    y_te = labels.reindex(F_te.index).to_numpy()

    _h("6.1  k selection")
    print(model.selection_.round(3).to_string(index=False))
    print(f"\nsupervision influence (ARI vs unsupervised): "
          f"{model.supervision_ari_:.3f}")
    print("  < 0.25 means the label axis is driving the partition: these are "
          "score bands,\n  not behaviour segments. Lower supervision_weight.")
    return model, F_tr, F_te, y_tr, y_te


def stage_7(cfg, F_te, y_te):
    """The scorecard.

    Read bad_pct against good_pct. bad_pct is capture - the recall you get by
    alerting the whole cluster. good_pct is what it costs. A cluster with 40%
    capture and 38% good_pct is a coin toss with extra steps.
    """
    M = cluster_metrics(F_te, y_te, cfg)

    _h("7  Cluster scorecard")
    print(M[["cluster", "n", "share", "n_good", "n_bad", "good_pct", "bad_pct",
             "bad_rate", "rate_lo", "rate_hi", "lift", "capture_over_cost",
             "woe", "significant"]].round(4).to_string(index=False))
    print(f"\ninformation value = {M.attrs['information_value']:.3f}   "
          f"KS = {M.attrs['ks']:.3f}   base rate = {M.attrs['base_rate']:.4%}")
    print("IV below ~0.1: the segmentation carries almost no risk information.")

    _h("7.1  Operational view (clusters alerted in descending risk order)")
    print(M[["cluster", "share", "bad_pct", "book_alerts", "expected_catches",
             "alerts_per_catch", "cum_alert_share", "cum_bad_pct",
             "cum_alerts_per_catch"]].round(4).to_string(index=False))
    return M


def stage_8(F_te, M):
    """What defines each cluster.

    coverage is how much of the cluster one threshold catches; purity is how
    much of what it catches really belongs. High cohens_d with low purity means
    the feature describes the cluster but cannot isolate it.
    """
    top = int(M.cluster.iloc[0])
    _h(f"8  Drivers of cluster {top} (highest lift)")
    print(f"n={int(M.n.iloc[0]):,}  lift={M.lift.iloc[0]:.2f}  "
          f"capture={M.bad_pct.iloc[0]:.1%}\n")
    print(cluster_drivers(F_te, top, top=10).round(3).to_string(index=False))

    _h("8.1  One line per cluster")
    for k in M.cluster:
        d = cluster_drivers(F_te, k, top=3)
        sig = ", ".join(f"{r.feature} {r.direction} "
                        f"(d={r.cohens_d:+.1f}, purity={r.purity:.2f})"
                        for r in d.itertuples())
        lift = float(M.loc[M.cluster == k, "lift"].iloc[0])
        print(f"  C{k}  lift={lift:6.2f}  {sig}")
    return top


def stage_9(model, F_te, top):
    """What moves membership.

    Elasticity is a local derivative at each account's own position: what would
    you have to change to push this account out. The sweep holds every other
    feature fixed, which - since these features are correlated by construction -
    asks a counterfactual the data never contains. Read the shape and the
    crossing point, not the level.
    """
    _h(f"9  Membership elasticity, cluster {top}")
    print(membership_elasticity(model, F_te, top, n_sample=800)
          .head(10).round(4).to_string(index=False))

    feature = cluster_drivers(F_te, top, top=1).feature.iloc[0]
    _h(f"9.1  Response curve for {feature}")
    S = cluster_sensitivity(model, F_te, feature, top, n_grid=15,
                            n_sample=800, members_only=True)
    print("members only - at what value do they fall out of the segment:")
    print(S.round(3).to_string(index=False))
    print("\nboundary:", cluster_boundary(model, F_te, feature, top,
                                          n_grid=15, n_sample=800))


def stage_10(model, F_te, y_te, cfg):
    """Should I believe it? Four independent checks.

        bootstrap ARI        is the partition reproducible?      > 0.60
        per-cluster Jaccard  WHICH clusters are real?            > 0.60
        prediction strength  is this k supported?                > 0.80
        classifier recall    separable, or residue?              > 0.50
    """
    _h("10  Validation")
    J = bootstrap_jaccard(model, F_te, n_boot=15)
    print(f"bootstrap ARI = {J.attrs['ari_mean']:.3f}\n")
    print(J.round(3).to_string(index=False))

    P = prediction_strength(model, F_te)
    print(f"\nmin prediction strength = {P.attrs['min_prediction_strength']:.3f}")
    print(P.round(3).to_string(index=False))

    print()
    print(cluster_separability(F_te).round(3).to_string(index=False))

    _h("10.1  Verdict")
    rep = full_report(model, F_te, y_te, cfg, n_boot=12)
    print(rep["summary"].round(3).to_string(index=False))
    print("\ndeployable = reproducible AND lifting AND explicable. Failing any "
          "one of\nthe three means it is not a control, whatever its lift.")
    return rep


def stage_11(F_te, y_te, cfg, top):
    """Rules, priced in book alert volume. A rule covering 90% of a cluster and
    8% of the good population is 1.6 million alerts on a 2-crore book."""
    _h(f"11  Rules for cluster {top}")
    R = cluster_rules(F_te, y_te, cfg, top, max_depth=3)
    print(R.round(4).to_string(index=False) if len(R) else "  no rules extracted")

    _h(f"11.1  Feature operating bands, cluster {top}")
    print(feature_bounds(F_te, y_te, top, top=6).round(3).to_string(index=False))


def stage_12(events, F_tr, F_te, y_tr, y_te, cfg):
    """Does any of this help a downstream model?

    Run the ladder before committing. If rung 2 captures most of the gain and
    rungs 3-5 add little, the fraud is amount-shaped rather than sequence-shaped
    and this is the wrong apparatus.
    """
    def blk(F, p):
        return [c for c in F.columns if c.startswith(p)]

    # baseline: the aggregates any team would already have
    if AMOUNT and AMOUNT in events.columns:
        agg = (events.groupby(ID)
               .agg(n=(AMOUNT, "size"), amt_sum=(AMOUNT, "sum"),
                    amt_mean=(AMOUNT, "mean"), amt_max=(AMOUNT, "max"),
                    amt_sd=(AMOUNT, "std")).fillna(0))
    else:
        num = [c for c in (NUMERIC_COLS or []) if c in events.columns]
        agg = events.groupby(ID)[num].mean() if num else pd.DataFrame(index=F_tr.index)
        agg["n"] = events.groupby(ID).size()
        agg = agg.fillna(0)

    seg = ["seg_cluster_lift", "seg_novelty", "seg_entropy", "seg_conf"] + \
          blk(F_tr, "seg_post_")
    rungs = {
        "1. aggregates only":      [],
        "2. + dates":              blk(F_tr, "dt_"),
        "3. + transitions":        blk(F_tr, "dt_") + blk(F_tr, "dy_"),
        "4. + timing (no Hawkes)": blk(F_tr, "dt_") + blk(F_tr, "dy_") + blk(F_tr, "tm_"),
        "5. + Hawkes":             blk(F_tr, "dt_") + blk(F_tr, "dy_") +
                                   blk(F_tr, "tm_") + blk(F_tr, "hk_"),
        "6. + clusters":           blk(F_tr, "dt_") + blk(F_tr, "dy_") +
                                   blk(F_tr, "tm_") + blk(F_tr, "hk_") + seg,
    }
    # drop rungs that add nothing because their block is absent in this mode
    seen, rungs = set(), {k: v for k, v in rungs.items()}
    rungs = {k: v for k, v in rungs.items()
             if not (tuple(v) in seen or seen.add(tuple(v)))}

    _h("12  Ablation ladder (HistGradientBoosting, held out)")
    base_pred = last = None
    for name, extra in rungs.items():
        Xtr = agg.loc[F_tr.index].join(F_tr[extra]) if extra else agg.loc[F_tr.index]
        Xte = agg.loc[F_te.index].join(F_te[extra]) if extra else agg.loc[F_te.index]
        g = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                           random_state=0).fit(Xtr, y_tr)
        p = g.predict_proba(Xte)[:, 1]
        base_pred = p if base_pred is None else base_pred
        print(f"  {name:26s} AUC={roc_auc_score(y_te, p):.4f}  "
              f"AP={average_precision_score(y_te, p):.4f}")
        last = p

    _h("12.1  Alert budget")
    print(budget_table(y_te, last, cfg,
                       budgets=(0.005, 0.01, 0.02, 0.05, 0.10)).to_string(index=False))
    print("\n`reliable` False = fewer than 20 scored accounts in that budget; "
          "the recall\nis one or two accounts wide and means nothing.")

    _h("12.2  False positives at matched recall")
    print(fp_reduction_at_recall(y_te, base_pred, last).to_string(index=False))


def stage_13(model, F_tr):
    """Wiring into the book screener, and saving."""
    _h("13  Deployment notes")
    print("""
Do NOT rewrite the ingestion in your screening notebook. Streaming by parquet
row group, checkpointing so a crash at 80% resumes at 80%, reservoir sampling
across the whole book, the dry run and the round-trip assert are all still the
right machinery and nothing here replaces them.

Treat `model` the way that notebook treats `fitted`: fit on the enriched
sample, then call model.transform(chunk_events) inside stream_book. Every
feature is per-account, so chunking by account group is safe - there is no
cross-account state. The one rule: a chunk must hold an account's WHOLE
history, or its transition counts and Hawkes fit are computed on a fragment.

Keep the round-trip assert:

    F_replay = model.rep.fit_transform(ev_tr, ID, TIME, EVENT_FEATURES,
                                       CHANNEL, AMOUNT, fit=False)
    assert np.abs(model.embed(F_replay)
                  - model.embed(F_tr[feature_columns(F_tr)])).max() < 1e-8

Saving:

    import joblib; joblib.dump(model, "segmodel.joblib")
""")
    cols = feature_columns(F_tr)
    dev = np.abs(model.embed(F_tr[cols]) - model.embed(F_tr[cols])).max()
    print(f"embedding replay check: max deviation = {dev:.2e}  "
          f"({'OK' if dev < 1e-8 else 'FAILED - do not deploy'})")


# ==========================================================================
def main(stop_after: int = 13) -> None:
    cfg = CFG
    events, labels, tr_ids, te_ids = stage_5(cfg)
    if stop_after < 6:
        return
    model, F_tr, F_te, y_tr, y_te = stage_6(cfg, events, labels, tr_ids, te_ids)
    if stop_after < 7:
        return
    M = stage_7(cfg, F_te, y_te)
    if stop_after < 8:
        return
    top = stage_8(F_te, M)
    if stop_after < 9:
        return
    stage_9(model, F_te, top)
    if stop_after < 10:
        return
    stage_10(model, F_te, y_te, cfg)
    if stop_after < 11:
        return
    stage_11(F_te, y_te, cfg, top)
    if stop_after < 12:
        return
    stage_12(events, F_tr, F_te, y_tr, y_te, cfg)
    if stop_after < 13:
        return
    stage_13(model, F_tr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=13,
                    help="run up to and including this stage (5-13)")
    main(ap.parse_args().stage)
