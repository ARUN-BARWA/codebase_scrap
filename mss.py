"""
MSS - Micro-Signal Similarity
=============================
An account-level similarity score for banking data where fraud and genuine accounts look
almost the same on raw fields. It compares accounts on how they deviate from their peers
and how internally coherent they are, then classifies by similarity to labelled examples.

Pipeline (matches the "Custom score: MSS" tab of the design doc):
  1. Engineer size-free micro-features per account (amount shape, timing, coherence ratios,
     merchant/payment mix, device/IP sharing).
  2. Peer groups (account type x occupation x age band, with back-off), fitted on genuine
     training accounts; every feature becomes a rank-Gauss score within the peer group.
  3. Five bounded block distances:
        N  behaviour + coherence  : attention-weighted, within-class whitened distance
        C  merchant / payment mix : Hellinger
        O  occupation             : Jensen-Shannon between occupations' behaviour
        P  static profile         : Gower (age, income, account type)
        G  network                : Jaccard distance on shared devices / IPs
  4. Block weights learned by a pairwise logistic model with hard negatives.
  5. Density-adaptive kernel  MSS(a,b) = exp(-D^2 / (sigma_a * sigma_b)),
     then score(a) = log( mean top-k MSS to fraud / mean top-k MSS to genuine ).

Usage
-----
  python mss.py --inspect --tx transactions.csv --profiles account_profiles.csv
  python mss.py --tx transactions.csv --profiles account_profiles.csv --out results
  python mss.py --tx ... --profiles ... --colmap colmap.json      # if column names differ
  python mss.py --tx ... --max-accounts 15000                      # subsample accounts (all fraud kept)

Requires: numpy, pandas, scipy, scikit-learn.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------------------------
# Configuration. Override any column name with --colmap '{"amount": "txn_amount", ...}'.
# A column that is missing from the data is skipped and the blocks that need it switch off.
# ---------------------------------------------------------------------------------------------
COLS = dict(
    account="account_id", ts="timestamp", amount="amount", merchant="merchant_category",
    method="payment_method", device="device_id", ip="ip_address", phone="phone", tx_fraud="is_fraud",
    tx_fraud_type="fraud_type",
    age="age", income="annual_income", occupation="occupation", acct_type="account_type",
    signup="signup_date", acct_fraud="is_fraud", acct_fraud_type="fraud_type",
)
# Never used as features: identifiers, a protected attribute, and a label proxy (ring_id).
NEVER_FEATURES = {"account_id", "transaction_id", "gender", "ring_id", "is_fraud", "fraud_type"}

TINY_AMOUNT = 3.0                          # currency units; "test charge" threshold
RISKY_MERCHANTS = {"gambling", "crypto", "money_transfer"}
MIN_PEER = 150                              # min genuine training accounts per peer group
K_SIGMA = 7                                 # neighbour used for the adaptive bandwidth
K_EXEMPLAR = 5                              # exemplars averaged per class when classifying
K_TYPE = 3                                  # exemplars averaged per fraud type
CHUNK = 256                                 # rows per distance chunk (lower if memory is tight)
EPS = 1e-12


# =============================================================================================
# Loading
# =============================================================================================
def inspect(tx_path, prof_path):
    for name, path in [("transactions", tx_path), ("profiles", prof_path)]:
        if not path:
            continue
        df = pd.read_csv(path, nrows=5000)
        print(f"\n== {name}: {path}")
        for c in df.columns:
            s = df[c]
            ex = ", ".join(map(str, s.dropna().unique()[:4]))
            print(f"  {c:28s} {str(s.dtype):10s} nunique~{s.nunique():6d}  e.g. {ex}")
    print("\nIf names differ from COLS in mss.py, pass --colmap with the mapping.")


def load(tx_path, prof_path, C, max_accounts=None, seed=0):
    tx = pd.read_csv(tx_path)
    prof = pd.read_csv(prof_path) if prof_path else None
    A = C["account"]
    tx[A] = tx[A].astype(str)
    if prof is not None:
        prof[A] = prof[A].astype(str)
        prof = prof.drop_duplicates(A).set_index(A)

    # account labels: profile label if present, else "any fraudulent transaction"
    g = tx.groupby(A)
    if prof is not None and C["acct_fraud"] in prof:
        y = prof[C["acct_fraud"]].reindex(g.size().index).fillna(0).astype(int)
    else:
        y = g[C["tx_fraud"]].max().astype(int)
    if prof is not None and C["acct_fraud_type"] in prof:
        ftype = prof[C["acct_fraud_type"]].reindex(y.index).astype(str)
    elif C["tx_fraud_type"] in tx:
        ft = tx[tx[C["tx_fraud"]] == 1].groupby(A)[C["tx_fraud_type"]].agg(lambda s: s.mode().iat[0])
        ftype = ft.reindex(y.index).fillna("none").astype(str)
    else:
        ftype = pd.Series(np.where(y == 1, "fraud", "none"), index=y.index)
    ftype[y == 0] = "genuine"
    ftype[(y == 1) & ftype.isin(["none", "nan", ""])] = "fraud"

    if max_accounts and len(y) > max_accounts:
        rng = np.random.default_rng(seed)
        fraud = y.index[y == 1]
        legit = y.index[y == 0]
        keep = set(fraud) | set(rng.choice(legit, max_accounts - len(fraud), replace=False))
        tx = tx[tx[A].isin(keep)]
        y, ftype = y[y.index.isin(keep)], ftype[ftype.index.isin(keep)]
    return tx, prof, y.sort_index(), ftype.sort_index()


# =============================================================================================
# Step 1 - micro-features
# =============================================================================================
def build_features(tx, prof, C):
    A, T, AMT = C["account"], C["ts"], C["amount"]
    tx = tx.copy()
    tx[T] = pd.to_datetime(tx[T])
    tx = tx.sort_values([A, T]).reset_index(drop=True)
    amt = tx[AMT].astype(float).clip(lower=0)
    tx["_amt"] = amt
    tx["_night"] = tx[T].dt.hour.between(0, 5).astype(float)
    tx["_tiny"] = (amt <= TINY_AMOUNT).astype(float)
    tx["_round"] = ((amt >= 50) & (np.abs(amt - np.round(amt, -1)) < 0.005)).astype(float)
    g = tx.groupby(A, sort=True)
    tx["_gap_h"] = g[T].diff().dt.total_seconds() / 3600
    tx["_burst"] = (tx["_gap_h"] < 1).astype(float)
    n_by = g[A].transform("size")
    tx["_pos"] = g.cumcount() / n_by
    tx["_late"] = tx["_pos"] >= 0.8

    agg = g.agg(n_tx=("_amt", "size"), avg=("_amt", "mean"), mx=("_amt", "max"), mn=("_amt", "min"),
                sd=("_amt", "std"), night=("_night", "mean"), tiny=("_tiny", "mean"),
                rnd=("_round", "mean"), burst=("_burst", "mean"), first=(T, "min"), last=(T, "max"))
    idx = agg.index
    late = tx[tx["_late"]].groupby(A)
    early = tx[~tx["_late"]].groupby(A)
    span_m = ((agg["last"] - agg["first"]).dt.total_seconds() / 86400 / 30).clip(lower=1)

    F = pd.DataFrame(index=idx)
    F["log_avg"] = np.log1p(agg["avg"])
    F["spikiness"] = np.log((agg["mx"] + 0.01) / (agg["avg"] + 0.01))
    F["floor"] = np.log((agg["mn"] + 0.01) / (agg["avg"] + 0.01))
    F["cv"] = agg["sd"].fillna(0) / (agg["avg"] + 0.01)
    F["night_share"] = agg["night"]
    F["tiny_share"] = agg["tiny"]
    F["round_share"] = agg["rnd"]
    F["burst_share"] = agg["burst"]
    F["log_tx_per_month"] = np.log(agg["n_tx"] / span_m)
    F["late_vs_early_amt"] = np.log((late["_amt"].mean() + 0.01) / (early["_amt"].mean() + 0.01)) \
        .reindex(idx).fillna(0)
    F["late_night_shift"] = (late["_night"].mean() - early["_night"].mean()).reindex(idx).fillna(0)

    mixes = {}
    if C["merchant"] in tx:
        P = pd.crosstab(tx[A], tx[C["merchant"]], normalize="index").reindex(idx).fillna(0)
        mixes["merchant"] = P
        F["risky_merchant_share"] = P[[c for c in P.columns if c in RISKY_MERCHANTS]].sum(axis=1)
        F["merchant_entropy"] = -(P * np.log(P + EPS)).sum(axis=1)
    if C["method"] in tx:
        mixes["method"] = pd.crosstab(tx[A], tx[C["method"]], normalize="index").reindex(idx).fillna(0)

    # devices / IPs: count, late new-device share, and an incidence matrix for the network block
    tokens = []
    if C["device"] in tx:
        dev_first = tx.groupby([A, C["device"]])["_pos"].transform("min")
        new_late = (tx["_late"] & (dev_first >= 0.8)).astype(float)
        F["late_new_device_share"] = new_late[tx["_late"]].groupby(tx.loc[tx["_late"], A]).mean() \
            .reindex(idx).fillna(0)
        F["n_devices"] = g[C["device"]].nunique()
        tokens.append("d:" + tx[C["device"]].astype(str))
    if C["ip"] in tx:
        F["n_ips"] = np.log1p(g[C["ip"]].nunique())
        tokens.append("i:" + tx[C["ip"]].astype(str))
    if C["phone"] in tx:   # a phone number shared across accounts is a ring signal (email domain is not)
        tokens.append("p:" + tx[C["phone"]].astype(str))
    M = None
    if tokens:
        tok = pd.concat([pd.DataFrame({A: tx[A], "t": t}) for t in tokens]).drop_duplicates()
        rows = pd.Categorical(tok[A], categories=idx).codes
        cols, _ = pd.factorize(tok["t"])
        M = sparse.csr_matrix((np.ones(len(tok), dtype=np.float32), (rows, cols)),
                              shape=(len(idx), cols.max() + 1))
        co = (M @ M.T).tocsr()
        F["share_degree"] = np.log1p(np.diff(co.indptr) - 1)   # other accounts sharing a device/IP

    # profile + economic coherence
    pf = pd.DataFrame(index=idx)
    if prof is not None:
        for key in ["age", "income", "occupation", "acct_type", "signup"]:
            if C[key] in prof:
                pf[key] = prof[C[key]].reindex(idx)
    if "income" in pf:
        m_inc = pf["income"].astype(float).clip(lower=1) / 12
        F["spend_to_income"] = np.log((agg["avg"] * agg["n_tx"] / span_m + 1) / (m_inc + 1))
        F["max_to_income"] = np.log((agg["mx"] + 1) / (m_inc + 1))
        u_inc = pf["income"].rank(pct=True)
        grp = [c for c in ["occupation", "age_band"] if c in pf or c == "age_band"]
        if "age" in pf:
            pf["age_band"] = pd.cut(pf["age"], [0, 25, 35, 45, 55, 65, 200], labels=False)
        grp = [c for c in ["occupation", "age_band"] if c in pf]
        F["income_residual"] = (u_inc - u_inc.groupby([pf[c] for c in grp]).transform("median")) \
            if grp else 0.0
    if "age" in pf and "age_band" not in pf:
        pf["age_band"] = pd.cut(pf["age"], [0, 25, 35, 45, 55, 65, 200], labels=False)
    if "acct_type" in pf and "occupation" in pf:
        p_type = pf.groupby("occupation")["acct_type"].value_counts(normalize=True)
        F["type_surprisal"] = -np.log(p_type.reindex(list(zip(pf["occupation"], pf["acct_type"]))).values + 1e-4)
    if "signup" in pf:
        F["log_account_age_days"] = np.log1p(
            (agg["last"] - pd.to_datetime(pf["signup"])).dt.days.clip(lower=0))

    raw = pd.DataFrame({"avg": agg["avg"], "max": agg["mx"], "min": agg["mn"]}, index=idx)
    for k in ["age", "income", "occupation", "acct_type"]:
        if k in pf:
            raw[k] = pf[k]
    F = F.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return F, mixes, pf, M, raw


# =============================================================================================
# Step 2 - peer groups and rank-Gauss transform
# =============================================================================================
def peer_levels(pf):
    keys = [k for k in ["acct_type", "occupation", "age_band"] if k in pf]
    levels = []
    for L in range(len(keys), -1, -1):
        levels.append(pf[keys[:L]].astype(str).agg("|".join, axis=1) if L else
                      pd.Series("all", index=pf.index))
    return levels


def peer_quantile(F, levels, ref_mask):
    """Rank-Gauss score of every account within the finest peer group that has >= MIN_PEER
    reference (genuine training) accounts. Reference distributions never include fraud."""
    X = F.to_numpy(float)
    n = len(X)
    assigned = np.full(n, -1)
    for L, key in enumerate(levels):
        counts = key[ref_mask].value_counts()
        ok = key.map(counts).fillna(0).to_numpy() >= MIN_PEER
        assigned[(assigned < 0) & ok] = L
    assigned[assigned < 0] = len(levels) - 1
    Z = np.zeros_like(X)
    for L, key in enumerate(levels):
        sel = assigned == L
        if not sel.any():
            continue
        kv = key.to_numpy()
        for gk in np.unique(kv[sel]):
            rows = np.where(sel & (kv == gk))[0]
            ref = np.sort(X[ref_mask & (kv == gk)], axis=0)
            m = len(ref)
            for j in range(X.shape[1]):
                lo = np.searchsorted(ref[:, j], X[rows, j], "left")
                hi = np.searchsorted(ref[:, j], X[rows, j], "right")
                Z[rows, j] = norm.ppf(((lo + hi) / 2 + 0.5) / (m + 1))
    return np.clip(Z, -3.5, 3.5), assigned


# =============================================================================================
# Steps 3-5 - the MSS model
# =============================================================================================
class MSS:
    def __init__(self, Z, feat_names, mixes, pf, M, cls, y, train_idx, ref_mask):
        self.feat_names = list(feat_names)
        self.cls = np.asarray(cls)
        self.y = np.asarray(y)
        self.tr = train_idx
        self.blocks = []

        # ---- Block N: attention weights (equal-class-weight Fisher score) + within-class whitening
        Zt, ct = Z[train_idx], self.cls[train_idx]
        classes = np.unique(ct)
        mus = np.array([Zt[ct == c].mean(0) for c in classes])
        within = np.mean([Zt[ct == c].var(0) for c in classes], axis=0)
        omega = mus.var(0) / (within + 1e-6)
        self.attention = np.sqrt(omega / omega.mean())
        AZ = Z * self.attention
        S = np.mean([np.cov(AZ[train_idx][ct == c], rowvar=False) for c in classes
                     if (ct == c).sum() > 2], axis=0)
        d = S.shape[0]
        S = 0.8 * S + 0.2 * np.trace(S) / d * np.eye(d)           # shrinkage
        w, V = np.linalg.eigh(S)
        self.whiten = V @ np.diag(1 / np.sqrt(np.maximum(w, 1e-6))) @ V.T
        self.EN = (AZ @ self.whiten).astype(np.float32)
        rng = np.random.default_rng(0)
        a, b = rng.choice(train_idx, 4000), rng.choice(train_idx, 4000)
        self.d0sq = float(np.median(((self.EN[a] - self.EN[b]) ** 2).sum(1)))
        self.blocks.append("N")

        # ---- Block C: Hellinger on mixes (embedding sqrt(p)/sqrt(2))
        self.EC = [np.sqrt(P.to_numpy(float) / 2).astype(np.float32) for P in mixes.values()]
        if self.EC:
            self.blocks.append("C")

        # ---- Block O: JS divergence between occupations' behaviour (genuine training accounts)
        self.occ = None
        if "occupation" in pf and "merchant" in mixes:
            occ_codes, occ_names = pd.factorize(pf["occupation"].astype(str))
            Pm = mixes["merchant"].to_numpy(float)
            gen = ref_mask
            B = np.array([Pm[gen & (occ_codes == k)].mean(0) if (gen & (occ_codes == k)).any()
                          else Pm.mean(0) for k in range(len(occ_names))]) + 1e-6
            B /= B.sum(1, keepdims=True)
            K = len(B)
            JS = np.zeros((K, K))
            for i in range(K):
                for j in range(K):
                    m = (B[i] + B[j]) / 2
                    JS[i, j] = 0.5 * (B[i] * np.log2(B[i] / m)).sum() + 0.5 * (B[j] * np.log2(B[j] / m)).sum()
            self.occ, self.JS = occ_codes, JS / max(JS.max(), EPS)   # rescale to [0,1]
            self.blocks.append("O")

        # ---- Block P: Gower on global quantiles of age, income + account-type mismatch
        cols = []
        for k in ["age", "income"]:
            if k in pf:
                cols.append(pf[k].rank(pct=True).fillna(0.5).to_numpy())
        self.UP = np.column_stack(cols).astype(np.float32) if cols else None
        self.typ = pd.factorize(pf["acct_type"].astype(str))[0] if "acct_type" in pf else None
        if self.UP is not None or self.typ is not None:
            self.blocks.append("P")

        # ---- Block G: Jaccard distance on shared devices / IPs
        self.M = M
        if M is not None:
            self.msize = np.asarray(M.sum(1)).ravel()
            self.blocks.append("G")
        self.beta = {b: 1.0 for b in self.blocks}

    # ---------------- distances ----------------
    def _p_terms(self, ia, ib, pair):
        parts = []
        if self.UP is not None:
            U = self.UP
            parts.append(np.abs(U[ia] - U[ib]).sum(1) if pair else
                         np.abs(U[ia][:, None, :] - U[ib][None, :, :]).sum(2))
        if self.typ is not None:
            parts.append((self.typ[ia] != self.typ[ib]).astype(np.float32) if pair else
                         (self.typ[ia][:, None] != self.typ[ib][None, :]).astype(np.float32))
        k = (self.UP.shape[1] if self.UP is not None else 0) + (self.typ is not None)
        return sum(parts) / k

    def pair_blocks(self, ia, ib):
        out = {}
        dn = ((self.EN[ia] - self.EN[ib]) ** 2).sum(1)
        out["N"] = dn / (dn + self.d0sq)
        if "C" in self.blocks:
            out["C"] = np.mean([np.sqrt(((E[ia] - E[ib]) ** 2).sum(1)) for E in self.EC], axis=0)
        if "O" in self.blocks:
            out["O"] = self.JS[self.occ[ia], self.occ[ib]]
        if "P" in self.blocks:
            out["P"] = self._p_terms(ia, ib, True)
        if "G" in self.blocks:
            inter = np.asarray(self.M[ia].multiply(self.M[ib]).sum(1)).ravel()
            union = self.msize[ia] + self.msize[ib] - inter
            out["G"] = 1 - inter / np.maximum(union, 1)
        return out

    def matrix_blocks(self, ia, ib):
        out = {}
        Ea, Eb = self.EN[ia], self.EN[ib]
        dn = np.maximum((Ea ** 2).sum(1)[:, None] + (Eb ** 2).sum(1)[None, :] - 2 * Ea @ Eb.T, 0)
        out["N"] = dn / (dn + self.d0sq)
        if "C" in self.blocks:
            acc = 0
            for E in self.EC:
                Xa, Xb = E[ia], E[ib]
                acc = acc + np.sqrt(np.maximum((Xa ** 2).sum(1)[:, None] + (Xb ** 2).sum(1)[None, :]
                                               - 2 * Xa @ Xb.T, 0))
            out["C"] = acc / len(self.EC)
        if "O" in self.blocks:
            out["O"] = self.JS[np.ix_(self.occ[ia], self.occ[ib])].astype(np.float32)
        if "P" in self.blocks:
            out["P"] = self._p_terms(ia, ib, False)
        if "G" in self.blocks:
            inter = (self.M[ia] @ self.M[ib].T).toarray()
            union = self.msize[ia][:, None] + self.msize[ib][None, :] - inter
            out["G"] = 1 - inter / np.maximum(union, 1)
        return out

    @staticmethod
    def combine(blocks, beta):
        tot = sum(beta.values())
        return sum(beta[k] * blocks[k] for k in beta if beta[k] > 0) / max(tot, EPS)

    # ---------------- Step 4: learn block weights ----------------
    def fit_weights(self, peer_assign_key, seed=0):
        rng = np.random.default_rng(seed)
        tr = self.tr
        c = self.cls[tr]
        pos_a, pos_b, neg_a, neg_b = [], [], [], []
        for cl in np.unique(c):
            members = tr[c == cl]
            n_pairs = min(4000 if cl == "genuine" else 800, len(members) * (len(members) - 1) // 2)
            pos_a.append(rng.choice(members, n_pairs)); pos_b.append(rng.choice(members, n_pairs))
        fraud = tr[self.y[tr] == 1]
        genuine = tr[self.y[tr] == 0]
        key = peer_assign_key.to_numpy()
        gen_by_peer = pd.Series(genuine).groupby(key[genuine]).apply(np.array).to_dict()
        for f in fraud:   # hard negatives: genuine accounts from the same peer group (same-looking)
            pool = gen_by_peer.get(key[f], genuine)
            neg_a.append(np.full(10, f)); neg_b.append(rng.choice(pool, 10))
            other = fraud[self.cls[fraud] != self.cls[f]]
            if len(other):
                neg_a.append(np.full(5, f)); neg_b.append(rng.choice(other, 5))
            neg_a.append(np.full(5, f)); neg_b.append(rng.choice(genuine, 5))
        ia = np.concatenate(pos_a + neg_a)
        ib = np.concatenate(pos_b + neg_b)
        lab = np.r_[np.ones(sum(map(len, pos_a))), np.zeros(sum(map(len, neg_a)))]
        keep = ia != ib
        ia, ib, lab = ia[keep], ib[keep], lab[keep]
        B = self.pair_blocks(ia, ib)
        X = -np.column_stack([B[k] for k in self.blocks])
        lr = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000).fit(X, lab)
        coef = np.maximum(lr.coef_.ravel(), 0)
        if coef.sum() == 0:
            coef = np.ones_like(coef)
        self.beta = {k: float(v) for k, v in zip(self.blocks, coef)}
        return self.beta

    # ---------------- Step 5: adaptive kernel + exemplar scoring ----------------
    def score(self, test_idx, variants):
        """For each variant (name -> (beta, adaptive)), return per test account:
        S_fraud, S_genuine (mean of top-k kernel similarity) and per-type similarities."""
        ref = self.tr
        y_ref = self.y[ref]
        types = sorted(set(self.cls[ref][y_ref == 1]))
        type_cols = {t: np.where(self.cls[ref] == t)[0] for t in types}
        fcol, gcol = np.where(y_ref == 1)[0], np.where(y_ref == 0)[0]

        # sigma of reference accounts (k-th NN inside the reference set)
        sig_ref = {v: np.empty(len(ref), np.float32) for v in variants}
        for s in range(0, len(ref), CHUNK):
            ia = ref[s:s + CHUNK]
            blk = self.matrix_blocks(ia, ref)
            for v, (beta, _) in variants.items():
                D = self.combine(blk, beta)
                D[np.arange(len(ia)), np.arange(s, s + len(ia))] = np.inf
                sig_ref[v][s:s + len(ia)] = np.partition(D, K_SIGMA - 1, axis=1)[:, K_SIGMA - 1]

        out = {v: dict(S_f=np.empty(len(test_idx)), S_g=np.empty(len(test_idx)),
                       S_type=np.empty((len(test_idx), len(types))), sigma=np.empty(len(test_idx)),
                       best_fraud=np.empty(len(test_idx), dtype=int)) for v in variants}
        for s in range(0, len(test_idx), CHUNK):
            ia = test_idx[s:s + CHUNK]
            blk = self.matrix_blocks(ia, ref)
            for v, (beta, adaptive) in variants.items():
                D = self.combine(blk, beta)
                sig_a = np.partition(D, K_SIGMA - 1, axis=1)[:, K_SIGMA - 1]
                if adaptive:
                    Sm = np.exp(-D ** 2 / np.maximum(sig_a[:, None] * sig_ref[v][None, :], EPS))
                else:
                    g2 = float(np.median(sig_ref[v])) ** 2
                    Sm = np.exp(-D ** 2 / max(g2, EPS))
                o = out[v]
                o["sigma"][s:s + len(ia)] = sig_a
                o["S_f"][s:s + len(ia)] = topk_mean(Sm[:, fcol], K_EXEMPLAR)
                o["S_g"][s:s + len(ia)] = topk_mean(Sm[:, gcol], K_EXEMPLAR)
                o["best_fraud"][s:s + len(ia)] = ref[fcol[np.argmax(Sm[:, fcol], 1)]]
                for j, t in enumerate(types):
                    o["S_type"][s:s + len(ia), j] = topk_mean(Sm[:, type_cols[t]], K_TYPE)
        for v in out:
            out[v]["score"] = np.log((out[v]["S_f"] + EPS) / (out[v]["S_g"] + EPS))
            out[v]["types"] = types
        self.sig_ref = sig_ref
        return out

    def pair_similarity(self, ia, ib, sig):
        D = self.combine(self.pair_blocks(ia, ib), self.beta)
        return np.exp(-D ** 2 / np.maximum(sig[ia] * sig[ib], EPS))

    def drivers(self, Z, i, top=3):
        contrib = np.abs(Z[i] * self.attention)
        order = np.argsort(-contrib)[:top]
        return "; ".join(f"{self.feat_names[j]} z={Z[i, j]:+.1f}" for j in order)


def topk_mean(S, k):
    k = min(k, S.shape[1])
    return np.partition(S, S.shape[1] - k, axis=1)[:, -k:].mean(1)


# =============================================================================================
# Baselines
# =============================================================================================
def gower_raw_scores(raw, y, train_idx, test_idx):
    """Raw-field Gower kNN: what a 'compare the columns as they are' approach sees."""
    num = [c for c in ["avg", "max", "min", "age", "income"] if c in raw]
    cat = [c for c in ["occupation", "acct_type"] if c in raw]
    Xn = raw[num].to_numpy(float)
    rng_ = Xn[train_idx].max(0) - Xn[train_idx].min(0) + EPS
    Xc = np.column_stack([pd.factorize(raw[c].astype(str))[0] for c in cat]) if cat else None
    yr = y[train_idx]
    fcol, gcol = np.where(yr == 1)[0], np.where(yr == 0)[0]
    S_f, S_g = np.empty(len(test_idx)), np.empty(len(test_idx))

    def sim(ia, ib):
        d = np.abs(Xn[ia][:, None, :] - Xn[ib][None, :, :]) / rng_
        tot = d.sum(2)
        if Xc is not None:
            tot = tot + (Xc[ia][:, None, :] != Xc[ib][None, :, :]).sum(2)
        return 1 - tot / (len(num) + len(cat))

    for s in range(0, len(test_idx), CHUNK):
        ia = test_idx[s:s + CHUNK]
        Sm = sim(ia, train_idx)
        S_f[s:s + len(ia)] = topk_mean(Sm[:, fcol], K_EXEMPLAR)
        S_g[s:s + len(ia)] = topk_mean(Sm[:, gcol], K_EXEMPLAR)
    return np.log((S_f + EPS) / (S_g + EPS)), sim


def euclid_scores(F, y, train_idx, test_idx):
    """Standardised engineered features, plain Euclidean, global kernel (no peers/weights/blocks)."""
    X = F.to_numpy(float)
    mu, sd = X[train_idx].mean(0), X[train_idx].std(0) + EPS
    X = ((X - mu) / sd).astype(np.float32)
    yr = y[train_idx]
    fcol, gcol = np.where(yr == 1)[0], np.where(yr == 0)[0]
    Xr = X[train_idx]
    S_f, S_g = np.empty(len(test_idx)), np.empty(len(test_idx))
    gam = None
    for s in range(0, len(test_idx), CHUNK):
        Xa = X[test_idx[s:s + CHUNK]]
        d2 = np.maximum((Xa ** 2).sum(1)[:, None] + (Xr ** 2).sum(1)[None, :] - 2 * Xa @ Xr.T, 0)
        gam = gam or float(np.median(d2))
        Sm = np.exp(-d2 / gam)
        S_f[s:s + len(Xa)] = topk_mean(Sm[:, fcol], K_EXEMPLAR)
        S_g[s:s + len(Xa)] = topk_mean(Sm[:, gcol], K_EXEMPLAR)
    return np.log((S_f + EPS) / (S_g + EPS))


# =============================================================================================
# Evaluation helpers
# =============================================================================================
def metrics(y_true, score):
    k = int(y_true.sum())
    top = np.argsort(-score)[:k]
    return dict(pr_auc=round(float(average_precision_score(y_true, score)), 4),
                roc_auc=round(float(roc_auc_score(y_true, score)), 4),
                precision_at_k=round(float(y_true[top].mean()), 4))


def separation(s_same, s_diff):
    j = (s_same.mean() - s_diff.mean()) ** 2 / (s_same.var() + s_diff.var() + EPS)
    auc = roc_auc_score(np.r_[np.ones(len(s_same)), np.zeros(len(s_diff))], np.r_[s_same, s_diff])
    return dict(fisher_J=round(float(j), 3), pair_auc=round(float(auc), 4),
                mean_same=round(float(s_same.mean()), 4), mean_diff=round(float(s_diff.mean()), 4))


# =============================================================================================
# Main
# =============================================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tx", required=True)
    ap.add_argument("--profiles")
    ap.add_argument("--colmap", help="JSON string or file overriding COLS")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--max-accounts", type=int)
    ap.add_argument("--test-size", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    if a.inspect:
        return inspect(a.tx, a.profiles)

    C = dict(COLS)
    if a.colmap:
        C.update(json.load(open(a.colmap)) if os.path.exists(a.colmap) else json.loads(a.colmap))
    t0 = time.time()
    tx, prof, y_s, ftype_s = load(a.tx, a.profiles, C, a.max_accounts, a.seed)
    F, mixes, pf, M, raw = build_features(tx, prof, C)
    F = F[[c for c in F.columns if c not in NEVER_FEATURES]]
    y = y_s.reindex(F.index).to_numpy()
    cls = ftype_s.reindex(F.index).to_numpy()
    print(f"accounts={len(F)}  fraud={y.sum()} ({y.mean():.2%})  features={F.shape[1]}  "
          f"classes={sorted(set(cls))}  [{time.time() - t0:.0f}s]")

    idx = np.arange(len(F))
    tr, te = train_test_split(idx, test_size=a.test_size, stratify=cls, random_state=a.seed)
    tr, te = np.sort(tr), np.sort(te)
    ref_mask = np.zeros(len(F), bool)
    ref_mask[tr] = True
    ref_mask &= (y == 0)

    levels = peer_levels(pf) if len(pf.columns) else [pd.Series("all", index=F.index)]
    Z, assigned = peer_quantile(F, levels, ref_mask)
    peer_key = pd.Series([levels[L].iat[i] for i, L in enumerate(assigned)], index=F.index)
    print(f"peer groups used: {peer_key.nunique()}  [{time.time() - t0:.0f}s]")

    model = MSS(Z, F.columns, mixes, pf, M, cls, y, tr, ref_mask)
    beta = model.fit_weights(peer_key, seed=a.seed)
    print("blocks:", model.blocks, " learned weights:", {k: round(v, 3) for k, v in beta.items()})

    variants = {"MSS (full)": (beta, True), "MSS, global bandwidth": (beta, False)}
    for b in model.blocks:
        if len(model.blocks) > 1:
            variants[f"MSS without block {b}"] = ({k: (0.0 if k == b else v) for k, v in beta.items()}, True)
    res = model.score(te, variants)
    print(f"scored {len(te)} test accounts against {len(tr)} references  [{time.time() - t0:.0f}s]")

    y_te = y[te]
    report = {"n_accounts": int(len(F)), "n_fraud": int(y.sum()), "n_test": int(len(te)),
              "blocks": model.blocks, "block_weights": beta,
              "attention": dict(zip(F.columns, np.round(model.attention, 3).tolist())),
              "results": {}}
    g_score, g_sim = gower_raw_scores(raw, y, tr, te)
    report["results"]["Raw-field Gower kNN"] = metrics(y_te, g_score)
    report["results"]["Euclidean kNN on engineered features"] = metrics(y_te, euclid_scores(F, y, tr, te))
    hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced",
                                         random_state=a.seed).fit(F.iloc[tr], y[tr])
    report["results"]["Gradient boosting (same features)"] = metrics(y_te, hgb.predict_proba(F.iloc[te])[:, 1])
    for v in variants:
        report["results"][v] = metrics(y_te, res[v]["score"])

    # fraud-type identification among test fraud accounts
    full = res["MSS (full)"]
    fmask = y_te == 1
    pred_type = np.array(full["types"])[np.argmax(full["S_type"][fmask], 1)]
    true_type = cls[te][fmask]
    report["type_accuracy"] = round(float((pred_type == true_type).mean()), 4)
    report["type_confusion"] = pd.crosstab(pd.Series(true_type, name="true"),
                                           pd.Series(pred_type, name="pred")).to_dict()

    # separation on HARD pairs (fraud vs genuine from the same peer group), test accounts only
    rng = np.random.default_rng(a.seed)
    sig = np.zeros(len(F), np.float32)
    sig[te] = full["sigma"]
    te_f, te_g = te[y_te == 1], te[y_te == 0]
    keyv = peer_key.to_numpy()
    same_a, same_b, diff_a, diff_b = [], [], [], []
    for f in te_f:
        mates = te_f[(cls[te_f] == cls[f]) & (te_f != f)]
        if len(mates):
            same_a += [f] * 3; same_b += list(rng.choice(mates, 3))
        peers = te_g[keyv[te_g] == keyv[f]]
        pool = peers if len(peers) else te_g
        diff_a += [f] * 10; diff_b += list(rng.choice(pool, 10))
    for gacc in rng.choice(te_g, 1000):
        peers = te_g[(keyv[te_g] == keyv[gacc]) & (te_g != gacc)]
        if len(peers):
            same_a.append(gacc); same_b.append(rng.choice(peers))
    same_a, same_b, diff_a, diff_b = map(np.array, (same_a, same_b, diff_a, diff_b))
    gs = lambda ia, ib: np.array([g_sim(np.array([i]), np.array([j]))[0, 0] for i, j in zip(ia, ib)])
    report["hard_pair_separation"] = {
        "Raw-field Gower": separation(gs(same_a, same_b), gs(diff_a, diff_b)),
        "MSS": separation(model.pair_similarity(same_a, same_b, sig),
                          model.pair_similarity(diff_a, diff_b, sig)),
    }

    os.makedirs(a.out, exist_ok=True)
    acc_ids = F.index.to_numpy()
    scores = pd.DataFrame({
        "account_id": acc_ids[te], "is_fraud": y_te, "true_class": cls[te],
        "mss_score": np.round(full["score"], 4),
        "predicted_type": np.array(full["types"])[np.argmax(full["S_type"], 1)],
        "most_similar_known_fraud": acc_ids[full["best_fraud"]],
        "top_drivers": [model.drivers(Z, i) for i in te],
    }).sort_values("mss_score", ascending=False)
    scores.to_csv(f"{a.out}/mss_scores.csv", index=False)
    with open(f"{a.out}/mss_report.json", "w") as fh:
        json.dump(report, fh, indent=2, default=str)

    print("\n=== Fraud detection on held-out accounts ===")
    print(pd.DataFrame(report["results"]).T.to_string())
    print(f"\nFraud-type accuracy (MSS, among fraud accounts): {report['type_accuracy']:.1%}")
    print("\n=== Separation on hard pairs (same peer group) ===")
    print(pd.DataFrame(report["hard_pair_separation"]).T.to_string())
    print(f"\nTop features by learned attention:")
    for k, v in sorted(report["attention"].items(), key=lambda kv: -kv[1])[:8]:
        print(f"  {k:26s} {v:.2f}")
    print(f"\nWrote {a.out}/mss_scores.csv and {a.out}/mss_report.json  [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
