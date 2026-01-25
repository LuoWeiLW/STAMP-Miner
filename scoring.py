# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
from .peptide_features import FEATURE_KEYS, feature_vector
from .w136_anchor import w136_anchor_score


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / (na * nb))


def _idx_map():
    return {k: i for i, k in enumerate(FEATURE_KEYS)}


def _find_first(idx: dict, candidates: list[str]):
    for k in candidates:
        if k in idx:
            return idx[k]
    return None


def _build_similarity_weights(idx: dict,
                              w_hydrophobic: float = 1.4,
                              w_aromatic: float = 1.8):
    """
    Similarity channel weights: emphasize hydrophobic/aromatic; mute comp-related dims (charge/hbond)
    so they don't double-count.
    """
    w = np.ones(len(FEATURE_KEYS), float)

    # emphasize
    i_hyd = _find_first(idx, ["hydrophobic"])
    i_aro = _find_first(idx, ["aromatic"])
    if i_hyd is not None:
        w[i_hyd] *= w_hydrophobic
    if i_aro is not None:
        w[i_aro] *= w_aromatic

    # mute comp dims if present
    for keyset in [
        ["positive", "pos", "basic"],
        ["negative", "neg", "acidic"],
        ["hbond_donor", "donor"],
        ["hbond_acceptor", "acceptor"],
    ]:
        i = _find_first(idx, keyset)
        if i is not None:
            w[i] = 0.0

    return w


def _complementarity_score(pep_vec: np.ndarray,
                           site_vec: np.ndarray,
                           idx: dict,
                           like_charge_penalty: float = 0.6,
                           hbond_like_penalty: float = 0.2):
    """
    Complementarity channel:
      electrostatic: pep_pos*site_neg + pep_neg*site_pos  - penalty*(pep_pos*site_pos + pep_neg*site_neg)
      hbond: pep_donor*site_acceptor + pep_acceptor*site_donor - penalty*(donor*donor + acceptor*acceptor)

    Output is squashed by tanh to keep scale stable for ranking.
    """
    p_pos = _find_first(idx, ["positive", "pos", "basic"])
    p_neg = _find_first(idx, ["negative", "neg", "acidic"])
    p_don = _find_first(idx, ["hbond_donor", "donor"])
    p_acc = _find_first(idx, ["hbond_acceptor", "acceptor"])

    elec = 0.0
    if (p_pos is not None) and (p_neg is not None):
        pep_pos = float(pep_vec[p_pos]); pep_neg = float(pep_vec[p_neg])
        sit_pos = float(site_vec[p_pos]); sit_neg = float(site_vec[p_neg])
        elec = (pep_pos * sit_neg + pep_neg * sit_pos) - like_charge_penalty * (pep_pos * sit_pos + pep_neg * sit_neg)

    hbond = 0.0
    if (p_don is not None) and (p_acc is not None):
        pep_d = float(pep_vec[p_don]); pep_a = float(pep_vec[p_acc])
        sit_d = float(site_vec[p_don]); sit_a = float(site_vec[p_acc])
        hbond = (pep_d * sit_a + pep_a * sit_d) - hbond_like_penalty * (pep_d * sit_d + pep_a * sit_a)

    # stable scaling
    elec_s = float(np.tanh(elec))
    hbond_s = float(np.tanh(hbond))
    comp = elec_s + hbond_s
    comp_s = float(np.tanh(comp))  # final squash

    return comp_s, elec_s, hbond_s


def w136_local_preference_vector(site_vec: np.ndarray) -> np.ndarray:
    """
    For W136 neighborhood: aromatic/hydrophobic packing is more important.
    """
    v = np.asarray(site_vec, float).copy()
    idx = _idx_map()
    i_aro = _find_first(idx, ["aromatic"])
    i_hyd = _find_first(idx, ["hydrophobic"])
    if i_aro is not None:
        v[i_aro] *= 1.8
    if i_hyd is not None:
        v[i_hyd] *= 1.5
    return v


def score_peptide(seq: str,
                  site_vec_global: np.ndarray,
                  site_vec_w136_shell: np.ndarray,
                  alpha_w136: float = 1.5,
                  beta_anchor: float = 0.8,
                  lambda_comp: float = 1.0,
                  like_charge_penalty: float = 0.6,
                  hbond_like_penalty: float = 0.2):
    """
    V6.2 scoring = similarity(hydrophobic/aromatic) + complementarity(charge/hbond) + W136 anchor
    """
    idx = _idx_map()
    pep_vec = feature_vector(seq, normalize=True)
    site_g = np.asarray(site_vec_global, float)
    site_s = np.asarray(site_vec_w136_shell, float)

    # -------- Similarity channel (global + W136-shell) --------
    w_sim = _build_similarity_weights(idx)
    sim_g = cosine(pep_vec * w_sim, site_g * w_sim)

    site_s_pref = w136_local_preference_vector(site_s)
    sim_s = cosine(pep_vec * w_sim, site_s_pref * w_sim)

    # -------- Complementarity channel (global + W136-shell) --------
    comp_g, elec_g, hbond_g = _complementarity_score(
        pep_vec, site_g, idx,
        like_charge_penalty=like_charge_penalty,
        hbond_like_penalty=hbond_like_penalty
    )
    comp_s, elec_s, hbond_s = _complementarity_score(
        pep_vec, site_s_pref, idx,
        like_charge_penalty=like_charge_penalty,
        hbond_like_penalty=hbond_like_penalty
    )

    # -------- Sequence anchor feasibility around W136 --------
    anc = w136_anchor_score(seq)
    score_anchor = float(anc["Score_W136_anchor"])

    # -------- Compose --------
    score_pocket = float(sim_g + lambda_comp * comp_g)
    score_w136_local = float(sim_s + lambda_comp * comp_s)
    score_total = float(score_pocket + alpha_w136 * score_w136_local + beta_anchor * score_anchor)

    return {
        # keep old names
        "Score_pocket": score_pocket,
        "Score_W136_local": score_w136_local,
        "Score_W136_anchor": score_anchor,
        "Score_total": score_total,

        # NEW: decomposed components for scientific interpretability / ablation
        "Score_pocket_sim": float(sim_g),
        "Score_pocket_comp": float(comp_g),
        "Score_pocket_elec": float(elec_g),
        "Score_pocket_hbond": float(hbond_g),

        "Score_W136_sim": float(sim_s),
        "Score_W136_comp": float(comp_s),
        "Score_W136_elec": float(elec_s),
        "Score_W136_hbond": float(hbond_s),

        # anchor components for figures
        "anchor_arom_frac": anc["arom_frac"],
        "anchor_cat_frac": anc["cat_frac"],
        "anchor_hyd_frac": anc["hyd_frac"],
        "anchor_pair_density": anc["pair_density"],
    }
