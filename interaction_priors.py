# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional
import json
import numpy as np
import pandas as pd

from .pdb_utils import load_structure, split_protein_ligand, group_by_residue, residue_min_distance, ring_centroid_and_normal_trp
from .peptide_features import FEATURE_KEYS, AA_FEATURES

def residue_to_feature(resname3: str) -> np.ndarray:
    aa_map = {"ALA":"A","CYS":"C","ASP":"D","GLU":"E","PHE":"F","GLY":"G","HIS":"H","ILE":"I",
              "LYS":"K","LEU":"L","MET":"M","ASN":"N","PRO":"P","GLN":"Q","ARG":"R","SER":"S",
              "THR":"T","VAL":"V","TRP":"W","TYR":"Y","MSE":"M"}
    aa1 = aa_map.get(resname3.upper(), None)
    if aa1 is None or aa1 not in AA_FEATURES:
        return np.zeros(len(FEATURE_KEYS), float)
    return np.array([AA_FEATURES[aa1][k] for k in FEATURE_KEYS], float)

def extract_native_contacts(complex_pdb: Path,
                            protein_chain: str="A",
                            ligand_chain: str="G",
                            ligand_resnames: Optional[List[str]]=None,
                            cutoff: float=4.5) -> Dict:
    """
    Interface-based extraction:
    For each protein residue, compute min distance to ANY ligand atom (across all ligand residues).
    This avoids centroid artifacts for long ligands (e.g., GlcNAc6).
    """
    structure = load_structure(complex_pdb)
    lig_set = set(ligand_resnames) if ligand_resnames else None
    prot_atoms, lig_atoms = split_protein_ligand(structure, protein_chain, ligand_chain, lig_set)
    prot_res = group_by_residue(prot_atoms)
    lig_res = group_by_residue(lig_atoms)

    contacts = []
    for pk, patoms in prot_res.items():
        dmin = 1e9
        for lk, latoms in lig_res.items():
            d = residue_min_distance(patoms, latoms)
            if d < dmin:
                dmin = d
        if dmin <= cutoff:
            chain, resseq, icode, resname = pk
            contacts.append({
                "residue": f"{chain}:{resname}{resseq}{icode if icode else ''}",
                "chain": chain,
                "resname": resname,
                "resseq": resseq,
                "icode": icode,
                "min_dist": float(dmin),
            })
    df = pd.DataFrame(contacts).sort_values("min_dist", ascending=True).reset_index(drop=True)
    return {"contacts_df": df, "n_protein_atoms": len(prot_atoms), "n_ligand_atoms": len(lig_atoms)}

def build_site_profile(contacts_df: pd.DataFrame,
                       w136_residue_id: str="A:TRP136",
                       w136_boost: float=2.0) -> Dict:
    """
    Weighted site fingerprint:
      w_i = exp(-d_i/sigma) and boosted for W136.
    """
    sigma = 2.5
    vec = np.zeros(len(FEATURE_KEYS), float)
    wsum = 0.0
    per_res = []
    for _, r in contacts_df.iterrows():
        rid = f"{r['chain']}:{r['resname']}{int(r['resseq'])}{str(r['icode']) if r['icode'] else ''}"
        d = float(r["min_dist"])
        w = float(np.exp(-d / sigma))
        if rid.startswith(w136_residue_id):
            w *= float(w136_boost)
        vec += w * residue_to_feature(str(r["resname"]))
        wsum += w
        per_res.append({"residue": rid, "weight": w, "min_dist": d})
    if wsum > 0:
        vec = vec / wsum
    return {"feature_keys": FEATURE_KEYS,
            "site_feature_vector": vec.tolist(),
            "w136_residue": w136_residue_id,
            "w136_boost": w136_boost,
            "per_residue": per_res}

def w136_geometry(complex_pdb: Path,
                  protein_chain: str="A",
                  ligand_chain: str="G",
                  w136_chain: str="A",
                  w136_resseq: int=136,
                  ligand_resnames: Optional[List[str]]=None) -> Dict:
    structure = load_structure(complex_pdb)
    lig_set = set(ligand_resnames) if ligand_resnames else None
    prot_atoms, lig_atoms = split_protein_ligand(structure, protein_chain, ligand_chain, lig_set)
    from .pdb_utils import group_by_residue, min_distance
    prot_res = group_by_residue(prot_atoms)
    lig_xyz = np.vstack([a.xyz for a in lig_atoms]) if lig_atoms else None

    w_atoms = None
    w_key = None
    for (ch, rs, ic, rn), atoms in prot_res.items():
        if ch == w136_chain and rs == int(w136_resseq) and rn.upper() == "TRP":
            w_atoms = atoms
            w_key = f"{ch}:{rn}{rs}{ic if ic else ''}"
            break
    if w_atoms is None or lig_xyz is None:
        return {"ok": False, "reason": "missing W136 or ligand"}

    w_xyz = np.vstack([a.xyz for a in w_atoms if a.element != "H"])
    dmin_atom = float(min_distance(w_xyz, lig_xyz))
    c, n = ring_centroid_and_normal_trp(w_atoms)
    dmin_centroid = float(np.min(np.linalg.norm(lig_xyz - c[None,:], axis=1)))
    return {"ok": True,
            "w136_residue": w_key,
            "w136_min_atom_dist_to_ligand": dmin_atom,
            "w136_ring_centroid": c.tolist(),
            "w136_ring_normal": n.tolist(),
            "w136_min_centroid_dist_to_ligand": dmin_centroid}

def extract_w136_shell_from_native(complex_pdb: Path,
                                   protein_chain: str="A",
                                   ligand_chain: str="G",
                                   ligand_resnames: Optional[List[str]]=None,
                                   contact_cutoff: float=4.5,
                                   w136_chain: str="A",
                                   w136_resseq: int=136,
                                   shell_radius: float=6.0):
    """
    1) 先用 ligand contact_cutoff 得到 pocket residues
    2) 再从 pocket residues 中筛选与 W136 原子最小距离 <= shell_radius 的残基
    """
    structure = load_structure(complex_pdb)
    lig_set = set(ligand_resnames) if ligand_resnames else None
    prot_atoms, lig_atoms = split_protein_ligand(structure, protein_chain, ligand_chain, lig_set)
    prot_res = group_by_residue(prot_atoms)
    lig_res  = group_by_residue(lig_atoms)

    # Find W136 atoms
    w_atoms = None
    for (ch, rs, ic, rn), atoms in prot_res.items():
        if ch == w136_chain and rs == int(w136_resseq) and rn.upper() == "TRP":
            w_atoms = atoms
            break
    if w_atoms is None:
        return pd.DataFrame([])

    # pocket residues by ligand contacts
    pocket = []
    for pk, patoms in prot_res.items():
        dmin = 1e9
        for lk, latoms in lig_res.items():
            d = residue_min_distance(patoms, latoms)
            if d < dmin:
                dmin = d
        if dmin <= contact_cutoff:
            pocket.append((pk, float(dmin)))

    # filter by W136 shell
    shell = []
    for pk, d_lig in pocket:
        patoms = prot_res[pk]
        d_w = residue_min_distance(patoms, w_atoms)
        if d_w <= shell_radius:
            chain, resseq, icode, resname = pk
            shell.append({
                "residue": f"{chain}:{resname}{resseq}{icode if icode else ''}",
                "chain": chain,
                "resname": resname,
                "resseq": resseq,
                "icode": icode,
                "min_dist_to_ligand": float(d_lig),
                "min_dist_to_W136": float(d_w),
            })

    return pd.DataFrame(shell).sort_values(
        ["min_dist_to_W136","min_dist_to_ligand"], ascending=True
    ).reset_index(drop=True)


def build_site_profile_from_df(df: pd.DataFrame,
                               w136_residue_id: str="A:TRP136",
                               w136_boost: float=1.0,
                               dist_col: str="min_dist",
                               extra_decay_col: Optional[str]=None):
    """
    Generalized profile builder.
    - dist_col: distance used for exp(-d/sigma)
    - extra_decay_col: optional second distance (e.g., to W136) for additional decay
    """
    sigma = 2.5
    vec = np.zeros(len(FEATURE_KEYS), float)
    wsum = 0.0
    per_res = []

    for _, r in df.iterrows():
        rid = str(r.get("residue", "")).strip()
        d = float(r[dist_col])
        w = float(np.exp(-d / sigma))

        if extra_decay_col is not None and extra_decay_col in df.columns:
            d2 = float(r[extra_decay_col])
            w *= float(np.exp(-d2 / 3.0))

        if rid.startswith(w136_residue_id):
            w *= float(w136_boost)

        vec += w * residue_to_feature(str(r["resname"]))
        wsum += w
        per_res.append({"residue": rid, "weight": w, "d": d})

    if wsum > 0:
        vec = vec / wsum

    return {
        "feature_keys": FEATURE_KEYS,
        "site_feature_vector": vec.tolist(),
        "w136_residue": w136_residue_id,
        "w136_boost": float(w136_boost),
        "per_residue": per_res
    }
