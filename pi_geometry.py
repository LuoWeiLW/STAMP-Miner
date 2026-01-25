# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np

# Aromatic ring atom names (heavy atoms) for residue types
RING_ATOMS = {
    "PHE": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "TYR": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "TRP": ["CD2", "CE2", "CE3", "CD1", "CZ2", "CZ3", "CH2", "NE1"],
    "HIS": ["CG", "ND1", "CD2", "CE1", "NE2"],
}

def _coords(atoms):
    return np.array([a.get_coord() for a in atoms], float)

def ring_centroid_and_normal(residue, resname: str):
    """
    Compute centroid and normal for aromatic ring using PCA on ring atoms.
    Returns (centroid, normal_unit) or (None, None) if atoms missing.
    """
    resname = str(resname).upper()
    if resname not in RING_ATOMS:
        return None, None
    atoms = []
    for an in RING_ATOMS[resname]:
        if an in residue:
            atoms.append(residue[an])
    if len(atoms) < 4:
        return None, None

    X = _coords(atoms)
    c = X.mean(axis=0)
    Y = X - c
    # PCA: normal is eigenvector of smallest variance
    cov = Y.T @ Y
    w, v = np.linalg.eigh(cov)
    n = v[:, 0]
    n = n / (np.linalg.norm(n) + 1e-12)
    return c, n

def plane_offsets(c1, n1, c2):
    """
    Decompose vector c2-c1 into perpendicular (to plane) and parallel components.
    Return (d_cc, d_perp, d_para).
    """
    v = c2 - c1
    d_cc = float(np.linalg.norm(v))
    d_perp = float(abs(np.dot(v, n1)))
    v_para = v - np.dot(v, n1) * n1
    d_para = float(np.linalg.norm(v_para))
    return d_cc, d_perp, d_para

def normal_angle_deg(n1, n2):
    x = float(abs(np.dot(n1, n2)))
    x = max(min(x, 1.0), -1.0)
    return float(np.degrees(np.arccos(x)))

def pi_pi_classify(theta_deg: float):
    if theta_deg <= 30.0:
        return "parallel"
    if 60.0 <= theta_deg <= 120.0:
        return "Tshape"
    return "other"

def score_pi_pi(d_cc, d_perp, d_para, theta_deg):
    """
    Soft score in [0,1] for ranking best interaction (not an energy).
    """
    geom = np.exp(-d_cc/6.0) * np.exp(-d_perp/4.0) * np.exp(-d_para/2.5)
    t = pi_pi_classify(theta_deg)
    bonus = 1.0
    if t == "parallel":
        bonus = 1.15
    elif t == "Tshape":
        bonus = 1.05
    return float(max(0.0, min(1.0, geom * bonus)))

def cation_center(residue, resname: str):
    """
    Return approximate cation center for Lys/Arg.
    """
    resname = str(resname).upper()
    if resname == "LYS":
        if "NZ" in residue:
            return residue["NZ"].get_coord()
        return None
    if resname == "ARG":
        names = [n for n in ["NE", "CZ", "NH1", "NH2"] if n in residue]
        if len(names) >= 2:
            X = np.array([residue[n].get_coord() for n in names], float)
            return X.mean(axis=0)
        return None
    return None

def score_cation_pi(d, d_perp, d_para):
    geom = np.exp(-d/6.0) * np.exp(-d_perp/4.0) * np.exp(-d_para/3.0)
    return float(max(0.0, min(1.0, geom)))