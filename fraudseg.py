"""
fraudseg.py
===========
Behavioural segmentation for fraud screening, v2.

What changed from v1, and why
-----------------------------
*   **The belief rule base is gone.** It is a piecewise-multilinear function of
    three inputs - a strictly weaker function class than the gradient boosting
    that consumes its output. It earns its keep only as an audit artefact, and
    a depth-3 decision tree over the cluster features (`describe_clusters`)
    does that job better and in a form model risk will actually sign off.
*   **The Markov mixture is demoted from engine to feature block.** Transition
    structure is one kind of evidence, not the coordinate system. It now sits
    beside timing and composition evidence, and a plain diagonal GMM does the
    clustering. Cheaper, far more stable, and swappable.
*   **Point-process features added.** A Hawkes branching ratio at several
    timescales, and the likelihood ratio against a Poisson null. This is the
    layer that separates "spent a lot" from "spent in a cascade".
*   **Optional supervised direction in the embedding.** Purely unsupervised
    clustering of a 2-crore book finds salary / merchant / dormant. All real,
    all fraud-irrelevant. A controlled amount of label information steers the
    geometry toward splits that carry risk, without letting the labels define
    the clusters outright.

Three swappable layers, in order:

    AccountRepresentation  events -> named per-account features, in blocks
    BlockEmbedding         blocks -> whitened, block-balanced coordinates
    ClusterEngine          coordinates -> soft memberships

Each can be replaced without touching the others.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer
from sklearn.tree import DecisionTreeClassifier, export_text

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

__all__ = [
    "SegConfig", "hawkes_features", "timing_features", "composition_features",
    "transition_features", "AccountRepresentation", "BlockEmbedding",
    "ClusterEngine", "SegmentModel", "cluster_report", "describe_clusters",
    "budget_table", "fp_reduction_at_recall",
]

EPS = 1e-9


def _log(m: str, l: int = 0) -> None:
    print("  " * l + m, flush=True)


# ==========================================================================
# Config
# ==========================================================================
@dataclass
class SegConfig:
    random_state: int = 42

    # ---- point process ----------------------------------------------------
    # Decay timescales in HOURS. beta = 1/tau. One per cascade speed you care
    # about: minutes-to-hours is takeover, days is mule layering, a week is
    # ordinary rhythm. Features are produced at every scale.
    hawkes_timescales: tuple = (1.0, 24.0, 168.0)
    hawkes_labels: tuple = ("1h", "1d", "7d")
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
        "timing": 1.0, "dynamics": 1.0, "composition": 1.0, "scale": 0.3})
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


def timing_features(events: pd.DataFrame, id_col: str, time_col: str) -> pd.DataFrame:
    """Rhythm and dispersion. Everything here is scale-free on purpose.

    `rhythm_day` / `rhythm_week` are circular resultant lengths: 1.0 means every
    transaction lands at the same clock position, 0 means no periodicity. They
    catch salary and EMI cycles without an FFT and without binning.
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
        s = np.bincount(code, weights=np.sin(ang), minlength=n_acc) / np.maximum(N, 1)
        return np.sqrt(c ** 2 + s ** 2)

    hour = t % 24.0
    night = np.bincount(code, weights=((hour < 6) | (hour >= 23)).astype(float),
                        minlength=n_acc) / np.maximum(N, 1)

    return pd.DataFrame({
        "tm_log_rate": np.log1p(N / np.maximum(T, 1.0)),
        "tm_burstiness": (sd - m) / np.maximum(sd + m, EPS),   # -1 regular, +1 bursty
        "tm_cv": sd / np.maximum(m, EPS),
        "tm_log_gap_sd": np.sqrt(np.maximum(lv, 0.0)),
        "tm_rhythm_day": rhythm(24.0),
        "tm_rhythm_week": rhythm(168.0),
        "tm_night_share": night,
        "tm_active_span": np.log1p(T),
    }, index=pd.Index(uniq, name=id_col))


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
    """events -> one named feature row per account, tagged by block."""

    def __init__(self, cfg: SegConfig):
        self.cfg = cfg
        self.alphabet: StateAlphabet | None = None
        self.blocks: dict[str, list[str]] = {}
        self.event_feature_cols: list[str] = []

    def fit_transform(self, events: pd.DataFrame, id_col: str, time_col: str,
                      event_feature_cols: Sequence[str] | None = None,
                      channel_col: str | None = None,
                      amount_col: str | None = None,
                      fit: bool = True) -> pd.DataFrame:
        cfg = self.cfg
        parts, blocks = [], {}

        tm = timing_features(events, id_col, time_col)
        hk = hawkes_features(events, id_col, time_col, cfg)
        timing = tm.join(hk)
        blocks["timing"] = list(timing.columns)
        parts.append(timing)

        if event_feature_cols:
            if fit:
                self.event_feature_cols = list(event_feature_cols)
                self.alphabet = StateAlphabet(cfg).fit(events, event_feature_cols)
            st = self.alphabet.transform(events)
            dy = transition_features(events, id_col, time_col, st,
                                     cfg.n_states, cfg.dirichlet_alpha)
            blocks["dynamics"] = list(dy.columns)
            parts.append(dy)
            self._last_states = st

        cp = composition_features(events, id_col, channel_col, amount_col)
        if cp.shape[1]:
            blocks["composition"] = list(cp.columns)
            parts.append(cp)

        n = events.groupby(id_col).size()
        sc = pd.DataFrame({"sc_log_n_events": np.log1p(n)})
        if amount_col is not None:
            sc["sc_log_total_amt"] = np.log1p(events.groupby(id_col)[amount_col].sum())
        blocks["scale"] = list(sc.columns)
        parts.append(sc)

        F = pd.concat([p.reindex(parts[0].index) for p in parts], axis=1)
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

    def fit(self, Z: np.ndarray) -> "ClusterEngine":
        rng = np.random.default_rng(self.cfg.random_state)
        m = min(self.cfg.fit_sample, len(Z))
        i = rng.choice(len(Z), m, replace=False) if m < len(Z) else np.arange(len(Z))
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
        e = ClusterEngine(cfg, k).fit(Z)
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
        self._cols: dict = {}

    def fit(self, events: pd.DataFrame, labels: pd.Series, id_col: str, time_col: str,
            event_feature_cols: Sequence[str] | None = None,
            channel_col: str | None = None, amount_col: str | None = None) -> "SegmentModel":
        cfg = self.cfg
        self._cols = dict(id=id_col, time=time_col, feat=list(event_feature_cols or []),
                          channel=channel_col, amount=amount_col)
        _log("=" * 66)
        _log(f"FRAUDSEG  |  {len(events):,} events  |  "
             f"{events[id_col].nunique():,} accounts")
        _log("=" * 66)

        _log("[1] representation")
        self.rep = AccountRepresentation(cfg)
        F = self.rep.fit_transform(events, id_col, time_col, event_feature_cols,
                                   channel_col, amount_col, fit=True)
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
            self.engine = ClusterEngine(cfg, cfg.n_clusters).fit(Z)

        # out-of-fold cluster risk, so the exported rate is not a label echo
        lab = self.engine.predict_proba(Z).argmax(1)
        K = lab.max() + 1
        oof = np.zeros(K)
        skf = StratifiedKFold(5, shuffle=True, random_state=cfg.random_state)
        acc = np.zeros((K, 2))
        for tr, te in skf.split(Z, y):
            for k in range(K):
                s = (lab[tr] == k)
                acc[k] += [y[tr][s].sum(), s.sum()]
        # empirical-Bayes shrinkage toward the base rate
        prior_n = 50.0
        oof = (acc[:, 0] + prior_n * self.base_rate_) / (acc[:, 1] + prior_n)
        self.cluster_risk_ = oof
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

    def transform(self, events: pd.DataFrame) -> pd.DataFrame:
        c = self._cols
        F = self.rep.fit_transform(events, c["id"], c["time"], c["feat"],
                                   c["channel"], c["amount"], fit=False)
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
        out["seg_cluster_risk"] = self.cluster_risk_[lab]
        out["seg_cluster_lift"] = self.cluster_risk_[lab] / max(self.base_rate_, EPS)
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
            if c.split("_")[0] in ("tm", "hk", "cp", "dy", "sc") and F[c].std() > 0]
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
    cols = [c for c in F.columns if c.split("_")[0] in ("tm", "hk", "cp", "dy", "sc")]
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
