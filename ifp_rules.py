#关键优化点 C：对接后真实 IFP + W136_contact
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List
import numpy as np
from .pdb_utils import AtomRec, group_by_residue, min_distance, ring_centroid_and_normal_trp

AROM3 = {"PHE","TYR","TRP","HIS"}
POS3  = {"LYS","ARG","HIS"}
NEG3  = {"ASP","GLU"}
HYDRO3= {"ALA","VAL","ILE","LEU","MET","PHE","TRP","PRO","TYR"}
POLAR3= {"SER","THR","ASN","GLN","HIS","TYR","CYS","ASP","GLU","LYS","ARG"}

def _coords(atoms: List[AtomRec]) -> np.ndarray:
    pts = [a.xyz for a in atoms if a.element != "H"]
    return np.vstack(pts) if pts else np.zeros((0,3), float)

def residue_centroid(atoms: List[AtomRec]) -> np.ndarray:
    P = _coords(atoms)
    return P.mean(axis=0) if len(P) else np.zeros(3, float)

def classify_interactions(prot_atoms: List[AtomRec],
                          pep_atoms: List[AtomRec],
                          w136_chain: str="A",
                          w136_resseq: int=136,
                          contact_cutoff: float=5.0) -> Dict[str,float]:
    """
    Observed IFP (coarse but robust):
      hbond (polar within 3.5Å),
      salt_bridge (pos-neg within 4.0Å),
      pi_pi (aromatic centroid <= 6.0Å),
      cation_pi (pos centroid to aromatic centroid <= 6.0Å),
      hydrophobic (hydrophobic within 4.5Å),
      w136_contact (0/1; peptide within 4.5Å of TRP136 atoms or ring centroid).
    Returns normalized counts per residue-residue contacts for comparability.
    """
    prot_res = group_by_residue(prot_atoms)
    pep_res  = group_by_residue(pep_atoms)

    prot_arom, pep_arom = {}, {}
    for k, atoms in prot_res.items():
        rn = k[3].upper()
        if rn in AROM3:
            c = ring_centroid_and_normal_trp(atoms)[0] if rn == "TRP" else residue_centroid(atoms)
            prot_arom[k] = c
    for k, atoms in pep_res.items():
        rn = k[3].upper()
        if rn in AROM3:
            c = ring_centroid_and_normal_trp(atoms)[0] if rn == "TRP" else residue_centroid(atoms)
            pep_arom[k] = c

    w_atoms = None
    for (ch, rs, ic, rn), atoms in prot_res.items():
        if ch == w136_chain and rs == int(w136_resseq) and rn.upper() == "TRP":
            w_atoms = atoms
            break
    w_centroid = ring_centroid_and_normal_trp(w_atoms)[0] if w_atoms else None

    counts = {"hbond":0.0,"salt_bridge":0.0,"pi_pi":0.0,"cation_pi":0.0,"hydrophobic":0.0}
    n_contacts = 0.0

    for pk, pa in prot_res.items():
        P = _coords(pa); rn_p = pk[3].upper()
        if len(P)==0:
            continue
        for qk, qa in pep_res.items():
            Q = _coords(qa); rn_q = qk[3].upper()
            if len(Q)==0:
                continue
            dmin = min_distance(P,Q)
            if dmin > contact_cutoff:
                continue

            n_contacts += 1.0
            if dmin <= 4.5 and rn_p in HYDRO3 and rn_q in HYDRO3:
                counts["hydrophobic"] += 1.0
            if dmin <= 4.0 and ((rn_p in POS3 and rn_q in NEG3) or (rn_p in NEG3 and rn_q in POS3)):
                counts["salt_bridge"] += 1.0
            if rn_p in AROM3 and rn_q in AROM3:
                if np.linalg.norm(prot_arom.get(pk, residue_centroid(pa)) - pep_arom.get(qk, residue_centroid(qa))) <= 6.0:
                    counts["pi_pi"] += 1.0
            if rn_p in POS3 and rn_q in AROM3:
                if np.linalg.norm(residue_centroid(pa) - pep_arom.get(qk, residue_centroid(qa))) <= 6.0:
                    counts["cation_pi"] += 1.0
            if rn_p in AROM3 and rn_q in POS3:
                if np.linalg.norm(prot_arom.get(pk, residue_centroid(pa)) - residue_centroid(qa)) <= 6.0:
                    counts["cation_pi"] += 1.0
            if dmin <= 3.5 and (rn_p in POLAR3) and (rn_q in POLAR3):
                counts["hbond"] += 1.0

    w136_contact = 0.0
    if w_atoms is not None and pep_atoms:
        W = _coords(w_atoms)
        Q = _coords(pep_atoms)
        if len(W) and len(Q):
            d_atom = min_distance(W,Q)
            d_cent = float(np.min(np.linalg.norm(Q - w_centroid[None,:], axis=1))) if w_centroid is not None else 1e9
            w136_contact = 1.0 if (d_atom <= 4.5 or d_cent <= 4.5) else 0.0

    if n_contacts > 0:
        for k in counts:
            counts[k] /= n_contacts
    counts["w136_contact"] = float(w136_contact)
    counts["n_contacts"] = float(n_contacts)
    return counts
