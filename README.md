 # STAMP-Miner

**STAMP-Miner** is a structure-anchored deep learning framework designed for the precision discovery of Specifically Targeted Antimicrobial Peptides (STAMPs). 

Traditional discovery methods are often "sequence-centric," failing to account for target-specific binding mechanics. STAMP-Miner bridges this gap by integrating atomistic interaction logic with deep learning, specifically optimized for targeting the **VhChip (chitoporin)** channel of *Vibrio harveyi*.

### Key Innovations
- **Structure-First Prioritization:** Explicitly encodes the W136 "gatekeeper" hotspot interaction logic.
- **Dual-Channel Scoring:** Balances physicochemical similarity and molecular complementarity.
- **AWLSTM Architecture:** Overcomes data scarcity by reformulating sequence alignment into a binary classification task with attention mechanisms.

### Workflow
The screening funnel follows a 3-tier evolutionary logic:
1. **Structural Decoy (VhChip-based):** 3D constraint satisfaction.
2. **Physiological Fitness (Prior Knowledge):** Helicity, amphipathicity, and charge filtering.
3. **Specific Recognition (AWLSTM):** Deep learning-guided specificity validation.
