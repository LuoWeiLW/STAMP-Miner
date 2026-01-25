# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np

FEATURE_KEYS = ["donor","acceptor","hydrophobic","aromatic","positive","negative"]

AA_FEATURES = {
    "A": dict(donor=0, acceptor=0, hydrophobic=1, aromatic=0, positive=0, negative=0),
    "V": dict(donor=0, acceptor=0, hydrophobic=1, aromatic=0, positive=0, negative=0),
    "I": dict(donor=0, acceptor=0, hydrophobic=1, aromatic=0, positive=0, negative=0),
    "L": dict(donor=0, acceptor=0, hydrophobic=1, aromatic=0, positive=0, negative=0),
    "M": dict(donor=0, acceptor=0, hydrophobic=1, aromatic=0, positive=0, negative=0),
    "F": dict(donor=0, acceptor=0, hydrophobic=1, aromatic=1, positive=0, negative=0),
    "W": dict(donor=0, acceptor=0, hydrophobic=1, aromatic=1, positive=0, negative=0),
    "Y": dict(donor=1, acceptor=1, hydrophobic=1, aromatic=1, positive=0, negative=0),
    "P": dict(donor=0, acceptor=0, hydrophobic=1, aromatic=0, positive=0, negative=0),
    "C": dict(donor=1, acceptor=1, hydrophobic=1, aromatic=0, positive=0, negative=0),
    "S": dict(donor=1, acceptor=1, hydrophobic=0, aromatic=0, positive=0, negative=0),
    "T": dict(donor=1, acceptor=1, hydrophobic=0, aromatic=0, positive=0, negative=0),
    "N": dict(donor=1, acceptor=1, hydrophobic=0, aromatic=0, positive=0, negative=0),
    "Q": dict(donor=1, acceptor=1, hydrophobic=0, aromatic=0, positive=0, negative=0),
    "H": dict(donor=1, acceptor=1, hydrophobic=0, aromatic=1, positive=1, negative=0),
    "K": dict(donor=1, acceptor=0, hydrophobic=0, aromatic=0, positive=1, negative=0),
    "R": dict(donor=1, acceptor=0, hydrophobic=0, aromatic=0, positive=1, negative=0),
    "D": dict(donor=0, acceptor=1, hydrophobic=0, aromatic=0, positive=0, negative=1),
    "E": dict(donor=0, acceptor=1, hydrophobic=0, aromatic=0, positive=0, negative=1),
    "G": dict(donor=0, acceptor=0, hydrophobic=0, aromatic=0, positive=0, negative=0),
}

CHARGE = {"K": 1, "R": 1, "H": 1, "D": -1, "E": -1}
HYDRO = {"I":4.5,"V":4.2,"L":3.8,"F":2.8,"C":2.5,"M":1.9,"A":1.8,"G":-0.4,"T":-0.7,"S":-0.8,
         "W":-0.9,"Y":-1.3,"P":-1.6,"H":-3.2,"E":-3.5,"Q":-3.5,"D":-3.5,"N":-3.5,"K":-3.9,"R":-4.5}
HELIX = {"A":1.45,"L":1.34,"E":1.51,"M":1.20,"Q":1.11,"K":1.07,"R":0.98,"H":1.00,"I":1.08,
         "V":1.06,"F":1.13,"W":1.14,"Y":0.61,"T":0.82,"S":0.77,"D":0.98,"N":0.73,"C":0.77,"P":0.59,"G":0.53}

def feature_vector(seq: str, normalize: bool=True) -> np.ndarray:
    seq = str(seq).strip().upper()
    v = np.zeros(len(FEATURE_KEYS), float)
    n = 0
    for aa in seq:
        if aa in AA_FEATURES:
            v += np.array([AA_FEATURES[aa][k] for k in FEATURE_KEYS], float)
            n += 1
    if normalize and n > 0:
        v /= n
    return v

def net_charge(seq: str) -> float:
    return float(sum(CHARGE.get(a, 0) for a in str(seq).strip().upper()))

def mean_hydropathy(seq: str) -> float:
    vals = [HYDRO.get(a, 0.0) for a in str(seq).strip().upper() if a in HYDRO]
    return float(np.mean(vals)) if vals else 0.0

def helix_propensity(seq: str) -> float:
    vals = [HELIX.get(a, 1.0) for a in str(seq).strip().upper() if a in HELIX]
    return float(np.mean(vals)) if vals else 1.0

def amphipathicity_proxy(seq: str) -> float:
    vals = [HYDRO.get(a, 0.0) for a in str(seq).strip().upper() if a in HYDRO]
    return float(np.std(vals)) if len(vals) >= 3 else 0.0
