# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse, re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser

from core.pi_geometry import (
    ring_centroid_and_normal,
    plane_offsets,
    normal_angle_deg,
    pi_pi_classify,
    score_pi_pi,
    cation_center,
    score_cation_pi,
)

STD_AA = {
    "ALA","CYS","ASP","GLU","PHE","GLY","HIS","ILE","LYS","LEU","MET",
    "ASN","PRO","GLN","ARG","SER","THR","VAL","TRP","TYR"
}
AROM_AA = {"PHE","TYR","TRP","HIS"}
CAT_AA  = {"LYS","ARG"}

def pick_peptide_chain(structure, receptor_chain: str, max_len=80):
    """
    Heuristic: peptide chain is non-receptor chain with 5..max_len standard residues.
    """
    model = next(structure.get_models())
    cand = []
    for ch in model:
        if ch.id == receptor_chain:
            continue
        res = [r for r in ch.get_residues() if r.get_resname().upper() in STD_AA]
        if 5 <= len(res) <= max_len:
            cand.append((len(res), ch.id))
    if not cand:
        return None
    cand.sort(key=lambda x: x[0])  # shortest first
    return cand[0][1]

def get_residue(chain, resseq: int):
    for r in chain.get_residues():
        if r.id[1] == int(resseq):
            return r
    return None

def min_heavy_atom_dist(resA, resB):
    da = []
    for a in resA.get_atoms():
        if a.element == "H":
            continue
        da.append(a.get_coord())
    db = []
    for b in resB.get_atoms():
        if b.element == "H":
            continue
        db.append(b.get_coord())
    if (not da) or (not db):
        return np.nan
    A = np.array(da, float)
    B = np.array(db, float)
    # brute-force but fast for residues
    dmin = 1e9
    for i in range(A.shape[0]):
        v = B - A[i]
        d = np.sqrt((v*v).sum(axis=1)).min()
        if d < dmin:
            dmin = d
    return float(dmin)

def map_pose_to_peptide_id(fname: str, top100: pd.DataFrame):
    """
    Try to parse peptide_id from filename; fallback to None.
    """
    # most common patterns: peptide_12, pep12, 001, id12
    m = re.search(r"(?:peptide|pep|id)[-_]?(\d{1,4})", fname, re.IGNORECASE)
    if m:
        pid = int(m.group(1))
        if "peptide_id" in top100.columns and pid in set(top100["peptide_id"].astype(int).tolist()):
            return pid
    # fallback: any number
    m2 = re.search(r"(\d{1,4})", fname)
    if m2 and "peptide_id" in top100.columns:
        pid = int(m2.group(1))
        if pid in set(top100["peptide_id"].astype(int).tolist()):
            return pid
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses_dir", required=True)
    ap.add_argument("--top100_xlsx", required=True)
    ap.add_argument("--out_csv", required=True)

    ap.add_argument("--protein_chain", default="A")
    ap.add_argument("--peptide_chain", default=None)
    ap.add_argument("--w136_chain", default="A")
    ap.add_argument("--w136_resseq", type=int, default=136)

    # distance thresholds
    ap.add_argument("--contact_cutoff", type=float, default=4.0)

    ap.add_argument("--pi_dcc", type=float, default=6.0)
    ap.add_argument("--pi_dperp", type=float, default=4.0)
    ap.add_argument("--pi_dpara", type=float, default=2.5)

    ap.add_argument("--cat_d", type=float, default=6.0)
    ap.add_argument("--cat_dperp", type=float, default=4.0)
    ap.add_argument("--cat_dpara", type=float, default=3.0)

    args = ap.parse_args()

    poses_dir = Path(args.poses_dir)
    top100 = pd.read_excel(args.top100_xlsx)
    if "sequence" not in top100.columns:
        raise ValueError("top100_xlsx must contain column: sequence")
    if "peptide_id" not in top100.columns:
        top100["peptide_id"] = np.arange(1, len(top100) + 1, dtype=int)

    parser = PDBParser(QUIET=True)
    rows = []

    pdb_files = sorted(list(poses_dir.glob("*.pdb")))
    if not pdb_files:
        raise FileNotFoundError(f"No .pdb files found in {poses_dir}")

    for fp in pdb_files:
        st = parser.get_structure(fp.stem, str(fp))
        model = next(st.get_models())

        # receptor and peptide chains
        if args.protein_chain not in model:
            # fallback: take first chain as receptor
            receptor_chain = next(model.get_chains()).id
        else:
            receptor_chain = args.protein_chain

        pep_chain = args.peptide_chain
        if pep_chain is None:
            pep_chain = pick_peptide_chain(st, receptor_chain=receptor_chain)
        if pep_chain is None or pep_chain not in model:
            # cannot analyze this pose reliably
            continue

        # W136 residue
        if args.w136_chain not in model:
            continue
        wch = model[args.w136_chain]
        wres = get_residue(wch, args.w136_resseq)
        if wres is None or wres.get_resname().upper() != "TRP":
            continue

        w_cent, w_norm = ring_centroid_and_normal(wres, "TRP")
        if w_cent is None:
            continue

        # peptide residues
        pch = model[pep_chain]
        pres = [r for r in pch.get_residues() if r.get_resname().upper() in STD_AA]

        # contact to W136 (min heavy atom distance)
        dmin_w = np.nan
        for r in pres:
            d = min_heavy_atom_dist(wres, r)
            if np.isnan(dmin_w) or d < dmin_w:
                dmin_w = d
        w_contact = int((not np.isnan(dmin_w)) and (dmin_w <= args.contact_cutoff))

        # ---- π-π scan ----
        best_pi = {"score": 0.0, "type": None, "d_cc": np.nan, "d_perp": np.nan, "d_para": np.nan, "theta": np.nan, "res": None}
        for r in pres:
            rn = r.get_resname().upper()
            if rn not in AROM_AA:
                continue
            c2, n2 = ring_centroid_and_normal(r, rn)
            if c2 is None:
                continue
            d_cc, d_perp, d_para = plane_offsets(w_cent, w_norm, c2)
            theta = normal_angle_deg(w_norm, n2)
            typ = pi_pi_classify(theta)
            sc = score_pi_pi(d_cc, d_perp, d_para, theta)
            if sc > best_pi["score"]:
                best_pi = {"score": sc, "type": typ, "d_cc": d_cc, "d_perp": d_perp, "d_para": d_para, "theta": theta, "res": f"{rn}{r.id[1]}"}

        pi_hit = int(
            (best_pi["score"] > 0.0) and
            (best_pi["d_cc"] <= args.pi_dcc) and
            (best_pi["d_perp"] <= args.pi_dperp) and
            (best_pi["d_para"] <= args.pi_dpara) and
            (best_pi["type"] in {"parallel", "Tshape"})
        )

        # ---- cation-π scan ----
        best_cat = {"score": 0.0, "d": np.nan, "d_perp": np.nan, "d_para": np.nan, "res": None}
        for r in pres:
            rn = r.get_resname().upper()
            if rn not in CAT_AA:
                continue
            cpos = cation_center(r, rn)
            if cpos is None:
                continue
            d = float(np.linalg.norm(cpos - w_cent))
            # decompose relative to W136 plane
            d_cc, d_perp, d_para = plane_offsets(w_cent, w_norm, cpos)
            sc = score_cation_pi(d, d_perp, d_para)
            if sc > best_cat["score"]:
                best_cat = {"score": sc, "d": d, "d_perp": d_perp, "d_para": d_para, "res": f"{rn}{r.id[1]}"}

        cat_hit = int(
            (best_cat["score"] > 0.0) and
            (best_cat["d"] <= args.cat_d) and
            (best_cat["d_perp"] <= args.cat_dperp) and
            (best_cat["d_para"] <= args.cat_dpara)
        )

        pid = map_pose_to_peptide_id(fp.name, top100)
        seq = None
        if pid is not None:
            seq = str(top100.loc[top100["peptide_id"].astype(int) == int(pid), "sequence"].values[0])

        rows.append({
            "pose_file": fp.name,
            "peptide_id": pid,
            "sequence": seq,

            "w136_min_heavy_atom_dist": float(dmin_w) if not np.isnan(dmin_w) else np.nan,
            "w136_contact": int(w_contact),

            "pi_pi_w136": int(pi_hit),
            "pi_pi_best_score": float(best_pi["score"]),
            "pi_pi_type": best_pi["type"],
            "pi_pi_best_res": best_pi["res"],
            "pi_pi_dcc": float(best_pi["d_cc"]) if best_pi["d_cc"] is not None else np.nan,
            "pi_pi_dperp": float(best_pi["d_perp"]) if best_pi["d_perp"] is not None else np.nan,
            "pi_pi_dpara": float(best_pi["d_para"]) if best_pi["d_para"] is not None else np.nan,
            "pi_pi_theta": float(best_pi["theta"]) if best_pi["theta"] is not None else np.nan,

            "cation_pi_w136": int(cat_hit),
            "cation_pi_best_score": float(best_cat["score"]),
            "cation_pi_best_res": best_cat["res"],
            "cation_pi_d": float(best_cat["d"]) if best_cat["d"] is not None else np.nan,
            "cation_pi_dperp": float(best_cat["d_perp"]) if best_cat["d_perp"] is not None else np.nan,
            "cation_pi_dpara": float(best_cat["d_para"]) if best_cat["d_para"] is not None else np.nan,

            "protein_chain": receptor_chain,
            "peptide_chain": pep_chain,
            "w136_chain": args.w136_chain,
            "w136_resseq": int(args.w136_resseq),
        })

    out = pd.DataFrame(rows)
    # optional: aggregate best pose per peptide_id (keep both files if you like)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved: {out_path} (rows={len(out)})")


if __name__ == "__main__":
    main()
