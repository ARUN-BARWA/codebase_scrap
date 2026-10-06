def category_distribution(run, pre, tr, va, rows_tr, rows_va, seg_tr, seg_va, y_tr, prof):
    """Actual value distribution of text columns (e.g. schme_code) inside each segment, against all
    mules and all goods, with train and val shares. Adds a compact top_<col> column to the profiles."""
    cols = CATEGORY_PROFILE_COLS or sorted(set(pre.freq) | set(pre.onehot))
    cols = [c for c in cols if c in tr.columns and c not in DROP_COLS and c != ACCOUNT_COL
            and tr[c].nunique(dropna=True) <= 0.5 * len(tr)]          # skip ID-like text columns
    if not cols:
        return prof
    rows, compact = [], {}
    for col in cols:
        def vals(d):
            return d[col].astype("string").str.strip().fillna("<NULL>").to_numpy(dtype=object)
        vt, vv = vals(tr), vals(va)
        m_tr, g_tr, m_va = vt[rows_tr], vt[y_tr == 0], vv[rows_va]
        all_m, all_g = Counter(m_tr), Counter(g_tr)
        nM, nG = max(len(m_tr), 1), max(len(g_tr), 1)
        for seg in sorted(set(seg_tr)):
            ms, mv = m_tr[seg_tr == seg], m_va[seg_va == seg]
            c, cv = Counter(ms), Counter(mv)
            n, nv = max(len(ms), 1), max(len(mv), 1)
            top = c.most_common(TOP_N_CATEGORIES)
            for val, k in top:
                sh, base, gsh = k / n, all_m[val] / nM, all_g[val] / nG
                rows.append({"segment": seg, "column": col, "value": val,
                             "train_mule_no": k, "train_share_in_segment": round(sh, 4),
                             "val_mule_no": cv[val], "val_share_in_segment": round(cv[val] / nv, 4),
                             "share_all_mules": round(base, 4), "share_all_goods": round(gsh, 4),
                             "enrichment_vs_mules": round(sh / max(base, 1e-9), 2),
                             "enrichment_vs_goods": round((sh + 1e-4) / (gsh + 1e-4), 2)})
            rest = len(ms) - sum(k for _, k in top)
            if rest > 0:
                rest_v = len(mv) - sum(cv[v] for v, _ in top)
                rows.append({"segment": seg, "column": col, "value": "<OTHER>", "train_mule_no": rest,
                             "train_share_in_segment": round(rest / n, 4), "val_mule_no": rest_v,
                             "val_share_in_segment": round(rest_v / nv, 4)})
            compact[(seg, col)] = "; ".join(
                f"{v} {k / n:.0%} (x{(k / n + 1e-4) / (all_g[v] / nG + 1e-4):.1f} vs goods)" for v, k in top[:3])
    out = pd.DataFrame(rows)
    out.to_csv(f"{OUT_DIR}/category_distribution_{run}.csv", index=False)
    log(f"  category distributions per segment for {len(cols)} text columns -> category_distribution_{run}.csv")
    if len(prof):
        prof = prof.copy()
        for col in cols:
            prof[f"top_{col}"] = [compact.get((sg, col), "") for sg in prof["segment"]]
    return prof





seg["profiles"] = category_distribution(run, pre, tr, va, seg["rows_tr"], seg["rows_va"],
                                            seg["final_tr"], seg["final_va"], y_tr, seg["profiles"])
    seg["profiles"].to_csv(f"{OUT_DIR}/segment_profiles.csv", index=False)
