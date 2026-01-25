# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse, json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

from core.scoring import score_peptide
from core.peptide_features import net_charge, mean_hydropathy, amphipathicity_proxy, helix_propensity


def guess_sequence_col(df: pd.DataFrame) -> str:
    # common names first
    for c in df.columns:
        if str(c).lower() in {"sequence", "seq", "peptide", "peptide_sequence", "pep"}:
            return c
    # heuristic: most-like AA column
    aa = set("ACDEFGHIKLMNPQRSTVWY")
    best, best_score = df.columns[0], -1.0
    for c in df.columns:
        s = df[c].astype(str).fillna("")
        ok = s.apply(lambda x: len(x) >= 5 and all((ch in aa) for ch in x.strip().upper()))
        score = float(ok.mean()) if len(ok) else 0.0
        if score > best_score:
            best, best_score = c, score
    return str(best)


def load_site_vec(path: str) -> np.ndarray:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return np.array(d["site_feature_vector"], float)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--peptide_xlsx", required=True)
    ap.add_argument("--out_xlsx", required=True)

    # 新版参数（推荐）
    ap.add_argument("--site_profile_global", default=None)
    ap.add_argument("--site_profile_w136_shell", default=None)

    # 旧版兼容参数（你现在就是用的这个）
    ap.add_argument("--site_profile", default=None, help="(legacy) same as --site_profile_global")

    ap.add_argument("--alpha_w136", type=float, default=1.5)
    ap.add_argument("--beta_anchor", type=float, default=0.8)

    ap.add_argument("--min_len", type=int, default=5)
    ap.add_argument("--max_len", type=int, default=60)

    args = ap.parse_args()

    # ---- resolve profiles with backward compatibility ----
    # if legacy arg is provided, treat it as global
    if args.site_profile_global is None and args.site_profile is not None:
        args.site_profile_global = args.site_profile

    if args.site_profile_global is None:
        raise ValueError("Missing site profile. Use --site_profile_global + --site_profile_w136_shell (recommended), "
                         "or legacy --site_profile.")

    site_vec_g = load_site_vec(args.site_profile_global)

    # if shell profile missing, fall back to global (won't crash; but W136 specificity weaker)
    if args.site_profile_w136_shell is None:
        site_vec_s = site_vec_g.copy()
        print("[WARN] --site_profile_w136_shell not provided; fallback to global. W136 specificity will be weaker.")
    else:
        site_vec_s = load_site_vec(args.site_profile_w136_shell)

    # ---- load peptides ----
    df = pd.read_excel(Path(args.peptide_xlsx))
    seq_col = guess_sequence_col(df)
    seqs = df[seq_col].astype(str).str.strip().str.upper()

    aa = set("ACDEFGHIKLMNPQRSTVWY")
    rows = []
    for i, s in enumerate(seqs.tolist()):
        if not s:
            continue
        if len(s) < args.min_len or len(s) > args.max_len:
            continue
        if any(ch not in aa for ch in s):
            continue

        sc = score_peptide(
            s,
            site_vec_global=site_vec_g,
            site_vec_w136_shell=site_vec_s,
            alpha_w136=float(args.alpha_w136),
            beta_anchor=float(args.beta_anchor),
        )

        rows.append({
            "peptide_id": i + 1,
            "sequence": s,
            **sc,
            "net_charge": net_charge(s),
            "mean_hydropathy": mean_hydropathy(s),
            "amphipathicity": amphipathicity_proxy(s),
            "helix_propensity": helix_propensity(s),
            "length": len(s),
        })

    out_df = pd.DataFrame(rows).sort_values("Score_total", ascending=False).reset_index(drop=True)

    out = Path(args.out_xlsx)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_excel(out, index=False)
    print(f"[OK] Saved ranked table: {out} (n={len(out_df)})")


if __name__ == "__main__":
    main()
