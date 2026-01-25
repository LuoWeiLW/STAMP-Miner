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

## 🚀 Installation & Reproducibility
### 1. Clone the repository
```bash
git clone https://github.com/LuoWeiLW/STAMP-Miner.git
cd STAMP-Miner

#This step will automatically refactor all the absolute paths in the source code (such as the path on drive D) into relative paths suitable for your local environment:
python setup_reproducibility.py

#npm install
pip install -r requirements.txt

#Use workflow
Module 1 (Structure Screening): Run scripts/01_screen_peptides for preliminary biophysical scoring.
Module 2 (Physicochemical Filtration): Use step2_prior_knowledge/physical_filter.py to perform biological constraints such as helicity and charge.
Module 3 (Specific Identification): Run scripts/ 04_predict_specification.py and use the AWLSTM model to lock the final candidate peptides P1-P4.
