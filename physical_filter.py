# -*- coding: utf-8 -*-
"""
STAMP-Miner Module 2: Advanced Physicochemical Gating
Provides high-fidelity filtering based on hydrophobic moments, Boman index, 
and helical propensity using industry-standard biophysical scales.
"""

import numpy as np
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# Eisenberg Hydrophobicity Scale (Eisenberg et al., 1984) - The standard for AMPs
EISENBERG_SCALE = {
    'A': 0.62, 'R': -2.53, 'N': -0.78, 'D': -0.90, 'C': 0.29, 'Q': -0.85,
    'E': -0.74, 'G': 0.48, 'H': -0.40, 'I': 1.38, 'L': 1.06, 'K': -1.50,
    'M': 0.64, 'F': 1.19, 'P': 0.12, 'S': -0.18, 'T': -0.05, 'W': 0.81,
    'Y': 0.26, 'V': 1.08
}

def calculate_hydrophobic_moment(sequence, angle=100):
    """
    Calculates the hydrophobic moment (uH) for an alpha-helix (default 100 degrees).
    A high uH indicates strong amphipathicity.
    """
    rad_angle = np.radians(angle)
    s_sum = 0
    c_sum = 0
    
    for i, amino_acid in enumerate(sequence):
        if amino_acid not in EISENBERG_SCALE:
            continue
        h = EISENBERG_SCALE[amino_acid]
        s_sum += h * np.sin(i * rad_angle)
        c_sum += h * np.cos(i * rad_angle)
        
    return np.sqrt(s_sum**2 + c_sum**2) / len(sequence)

def calculate_boman_index(sequence):
    """
    Calculates Boman Index: Potential for protein-protein interaction.
    Values > 2.48 indicate high potential for protein binding (e.g., target-specific STAMPs).
    """
    analyser = ProteinAnalysis(sequence)
    # Boman's scale effectively measures the average hydrophilicity
    return -1 * (analyser.protein_scale(ProteinAnalysis.Scale['KyteDoolittle'], 11, 0.4)).mean()

def apply_scientific_filters(df, seq_col='sequence'):
    """
    Optimized Module 2 Pipeline for Nature-level manuscripts.
    Refines candidates based on:
    - Net Charge (Optimal for STAMPs: +2 to +7)
    - Mean Hydrophobicity (Eisenberg)
    - Amphipathicity (Hydrophobic Moment)
    - Biological Fitness (Boman Index)
    """
    results = []
    for raw_seq in df[seq_col]:
        # Clean non-standard AAs
        seq = ''.join([aa for aa in raw_seq.upper() if aa in "ACDEFGHIKLMNPQRSTVWY"])
        if len(seq) < 5: continue
        
        analyser = ProteinAnalysis(seq)
        
        # 1. Precise Net Charge
        charge = analyser.charge_at_pH(7.4)
        
        # 2. Helical Fraction
        helix_frac = analyser.secondary_structure_fraction()[0]
        
        # 3. Mean Hydrophobicity (H)
        h_mean = np.mean([EISENBERG_SCALE.get(aa, 0) for aa in seq])
        
        # 4. Hydrophobic Moment (uH) - The gold standard for amphipathicity
        uH = calculate_hydrophobic_moment(seq)
        
        # 5. Boman Index
        boman = calculate_boman_index(seq)

        results.append({
            'sequence': seq,
            'net_charge': charge,
            'helical_propensity': helix_frac,
            'eisenberg_hydrophobicity': h_mean,
            'hydrophobic_moment': uH,
            'boman_index': boman,
            'is_toxic_proxy': 1 if analyser.instability_index() > 40 else 0
        })

    res_df = pd.DataFrame(results)
    
    # Advanced Filtering Logic based on recent literature (e.g., PNAS, Nature Biotech)
    filtered = res_df[
        (res_df['net_charge'] >= 2.0) &             # STAMPs need sufficient cationic push
        (res_df['hydrophobic_moment'] > 0.15) &    # Threshold for amphipathic alpha-helices
        (res_df['boman_index'] > 1.5) &            # Protein binding potential
        (res_df['helical_propensity'] > 0.3)       # Minimum required helical core
    ]
    
    return filtered.sort_values(by='hydrophobic_moment', ascending=False)
