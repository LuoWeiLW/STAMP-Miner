# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np

def safe_mean(x):
    x = np.asarray(x, float)
    return float(np.nanmean(x)) if x.size else float("nan")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--priors_dir", required=True)
    ap.add_argument("--ranked_xlsx", required=True)
    ap.add_argument("--top100_xlsx", required=True)
    ap.add_argument("--out_xlsx", required=True)
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()

    pri = Path(args.priors_dir)
    geom_p = pri / "w136_geometry.json"
    sum_p  = pri / "native_ifp_summary.json"

    geom = json.loads(geom_p.read_text(encoding="utf-8")) if geom_p.exists() else {}
    summ = json.loads(sum_p.read_text(encoding="utf-8")) if sum_p.exists() else {}

    ranked = pd.read_excel(args.ranked_xlsx)
    top100 = pd.read_excel(args.top100_xlsx)

    # Determine ranking column
    rank_col = "Score_total" if "Score_total" in ranked.columns else ranked.columns[0]
    r = ranked.sort_values(rank_col, ascending=False).reset_index(drop=True)
    topk = r.head(int(args.topk)).copy()

    criteria = []

    # ---- Native geometry sanity ----
    w_ok = bool(geom.get("ok", False))
    criteria.append({
        "Metric": "W136 geometry present",
        "Definition": "w136_geometry.json ok==true",
        "Threshold": "PASS if true",
        "Value": w_ok,
        "Pass": "PASS" if w_ok else "FAIL",
    })

    dmin = geom.get("w136_min_atom_dist_to_ligand", None)
    if dmin is not None:
        criteria.append({
            "Metric": "W136 min atom distance to ligand",
            "Definition": "min heavy-atom distance (Å)",
            "Threshold": "<= 4.5 Å",
            "Value": float(dmin),
            "Pass": "PASS" if float(dmin) <= 4.5 else "WARN",
        })

    # shell size
    n_shell = summ.get("n_contacts_residues_w136_shell", None)
    if n_shell is not None:
        criteria.append({
            "Metric": "W136-shell residue count",
            "Definition": "number of residues within W136-shell used in profile",
            "Threshold": ">= 4",
            "Value": int(n_shell),
            "Pass": "PASS" if int(n_shell) >= 4 else "WARN",
        })

    # ---- Pre-docking W136 specificity ----
    def add_enrich(col, name, thr="> 0"):
        if col in r.columns:
            delta = safe_mean(topk[col].values) - safe_mean(r[col].values)
            criteria.append({
                "Metric": f"Top{args.topk} {name} enrichment",
                "Definition": f"mean(TopK {col}) - mean(all {col})",
                "Threshold": thr,
                "Value": float(delta),
                "Pass": "PASS" if delta > 0 else "WARN",
            })

    add_enrich("Score_W136_local", "W136-local")
    add_enrich("Score_W136_anchor", "W136-anchor")

    # NEW: complementarity evidence
    add_enrich("Score_pocket_comp", "pocket complementarity")
    add_enrich("Score_W136_comp", "W136 complementarity")
    add_enrich("Score_pocket_elec", "pocket electrostatic complement")
    add_enrich("Score_W136_elec", "W136 electrostatic complement")
    add_enrich("Score_pocket_hbond", "pocket H-bond complement")
    add_enrich("Score_W136_hbond", "W136 H-bond complement")

    # ---- Ablation-style comparison (no external labels needed) ----
    ablation_rows = []
    if set(["Score_pocket_sim","Score_W136_sim","Score_W136_anchor"]).issubset(r.columns):
        alpha = 1.5  # use same default as screening; if you logged alpha, read from config
        beta  = 0.8
        sim_only = r["Score_pocket_sim"] + alpha*r["Score_W136_sim"] + beta*r["Score_W136_anchor"]
        sim_only_top = sim_only.sort_values(ascending=False).head(int(args.topk)).index
        ablation_rows.append({
            "Model": "sim_only",
            "TopK_mean_W136_local": safe_mean(r.loc[sim_only_top, "Score_W136_sim"].values),
            "TopK_mean_anchor": safe_mean(r.loc[sim_only_top, "Score_W136_anchor"].values),
        })

    if set(["Score_pocket_comp","Score_W136_comp","Score_W136_anchor"]).issubset(r.columns):
        alpha = 1.5
        beta  = 0.8
        comp_only = r["Score_pocket_comp"] + alpha*r["Score_W136_comp"] + beta*r["Score_W136_anchor"]
        comp_only_top = comp_only.sort_values(ascending=False).head(int(args.topk)).index
        ablation_rows.append({
            "Model": "comp_only",
            "TopK_mean_W136_comp": safe_mean(r.loc[comp_only_top, "Score_W136_comp"].values),
            "TopK_mean_anchor": safe_mean(r.loc[comp_only_top, "Score_W136_anchor"].values),
        })

    if "Score_total" in r.columns:
        idx_top = r.head(int(args.topk)).index
        ablation_rows.append({
            "Model": "V6.2(sim+comp)",
            "TopK_mean_W136_local": safe_mean(r.loc[idx_top, "Score_W136_local"].values) if "Score_W136_local" in r.columns else np.nan,
            "TopK_mean_W136_comp": safe_mean(r.loc[idx_top, "Score_W136_comp"].values) if "Score_W136_comp" in r.columns else np.nan,
            "TopK_mean_anchor": safe_mean(r.loc[idx_top, "Score_W136_anchor"].values) if "Score_W136_anchor" in r.columns else np.nan,
        })

    # ---- Cluster diversity (if column exists) ----
    for c in ["cluster", "cluster_id", "Cluster", "ClusterID"]:
        if c in top100.columns:
            ncl = int(pd.Series(top100[c]).nunique())
            criteria.append({
                "Metric": "Top100 cluster diversity",
                "Definition": "unique clusters in Top100",
                "Threshold": ">= 10",
                "Value": ncl,
                "Pass": "PASS" if ncl >= 10 else "WARN",
            })
            break

    out = Path(args.out_xlsx)
    out.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out, engine="openpyxl") as w:
        pd.DataFrame(criteria).to_excel(w, index=False, sheet_name="Criteria")
        topk.to_excel(w, index=False, sheet_name=f"Top{args.topk}")
        if ablation_rows:
            pd.DataFrame(ablation_rows).to_excel(w, index=False, sheet_name="Ablation")

    print(f"[OK] Saved evaluation report: {out}")

if __name__ == "__main__":
    main()
