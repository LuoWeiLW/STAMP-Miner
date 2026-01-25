# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np

AROM = set("FWY")      # π-π potential
CAT  = set("KRH")      # cation-π potential (H as weak cationic)
HYD  = set("AVILMFPWY")  # hydrophobic packing around aromatic pocket

def w136_anchor_score(seq: str, max_gap: int = 4) -> dict:
    """
    Sequence-level feasibility proxy for binding TRP pocket (W136):
    - aromatic density (FWY)
    - cationic density (KRH)
    - local co-occurrence of (cation, aromatic) within max_gap (cation-π synergy)
    - hydrophobic density (packing)

    Returns: score in [0, 1] (bounded by sigmoid), plus components.
    """
    s = str(seq).strip().upper()
    n = len(s)
    if n == 0:
        return {"Score_W136_anchor": 0.0, "arom_frac": 0.0, "cat_frac": 0.0, "hyd_frac": 0.0, "pair_density": 0.0}

    arom_pos = [i for i,ch in enumerate(s) if ch in AROM]
    cat_pos  = [i for i,ch in enumerate(s) if ch in CAT]
    hyd_pos  = [i for i,ch in enumerate(s) if ch in HYD]

    arom_frac = len(arom_pos) / n
    cat_frac  = len(cat_pos) / n
    hyd_frac  = len(hyd_pos) / n

    # cation-aromatic pairs within distance <= max_gap
    pair = 0
    for i in cat_pos:
        for j in arom_pos:
            if abs(i - j) <= max_gap:
                pair += 1
    # normalize by length (rough density proxy)
    pair_density = pair / max(n, 1)

    # Weighted linear combination -> sigmoid to [0,1]
    # Emphasize aromatic + pair synergy; hydrophobic supports packing; cation alone is weaker.
    z = 2.2*arom_frac + 1.8*pair_density + 0.8*hyd_frac + 0.5*cat_frac - 0.6
    score = 1.0 / (1.0 + np.exp(-z))

    return {
        "Score_W136_anchor": float(score),
        "arom_frac": float(arom_frac),
        "cat_frac": float(cat_frac),
        "hyd_frac": float(hyd_frac),
        "pair_density": float(pair_density),
    }
