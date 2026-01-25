# -*- coding: utf-8 -*-
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis


def apply_physical_filters(df, seq_col='sequence'):
    """
    Module 2: Filters peptides based on structural and chemical descriptors.
    Thresholds: Helix > 0.5, Amphipathicity > 0, Net Charge > 0.
    """
    results = []
    for seq in df[seq_col]:
        analyser = ProteinAnalysis(seq)

        # 1. Helicity (Secondary structure fraction: [Helix, Turn, Sheet])
        helix_frac = analyser.secondary_structure_fraction()[0]

        # 2. Net Charge (at pH 7.0)
        charge = analyser.charge_at_pH(7.0)

        # 3. Amphipathicity Proxy (Hydrophobicity deviation / Moment)
        # Using standard hydropathy scales to calculate variation as a proxy
        hydro_scores = [analyser.protein_scale(ProteinAnalysis.Scale['KyteDoolittle'], 11, 0.4)]
        amphipathicity = pd.Series(hydro_scores[0]).std()

        results.append({
            'sequence': seq,
            'helix_content': helix_frac,
            'net_charge': charge,
            'amphipathicity_index': amphipathicity
        })

    res_df = pd.DataFrame(results)
    # Applying the heuristic thresholds
    filtered = res_df[
        (res_df['helix_content'] > 0.5) &
        (res_df['net_charge'] > 0) &
        (res_df['amphipathicity_index'] > 0)
        ]
    return filtered