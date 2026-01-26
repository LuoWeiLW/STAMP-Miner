 # STAMP-Miner

**STAMP-Miner** is a structure-anchored deep learning framework designed for the precision discovery of Specifically Targeted Antimicrobial Peptides (STAMPs). 

Traditional discovery methods are often "sequence-centric," failing to account for target-specific binding mechanics. STAMP-Miner bridges this gap by integrating atomistic interaction logic with deep learning, specifically optimized for targeting the **VhChip (chitoporin)** channel of *Vibrio harveyi*.

### Key Innovations
- **Structure-First Prioritization:** Explicitly encodes the W136 "gatekeeper" hotspot interaction logic.
- **Dual-Channel Scoring:** Balances physicochemical similarity and molecular complementarity.
- **AWLSTM Architecture:** Overcomes data scarcity by reformulating sequence alignment into a binary classification task with attention mechanisms.

### Workflow
#Use workflow
Module 1 (Structure Screening): Run scripts/01_screen_peptides for preliminary biophysical scoring.
Module 2 (Physicochemical Filtration): Use step2_prior_knowledge/physical_filter.py to perform biological constraints such as helicity and charge.
Module 3 (Specific Identification): Run scripts/ 04_predict_specification.py and use the AWLSTM model to lock the final candidate peptides P1-P4.

### 🚀 Installation & Reproducibility
### 1. Clone the repository
git clone https://github.com/LuoWeiLW/STAMP-Miner.git

cd STAMP-Miner

### 2. IMPORTANT: Auto-refactor hardcoded paths to your local environment
###This script will fix all absolute paths (D:\...) to work on your machine
python setup_reproducibility.py

### 3. Install dependencies
pip install -r requirements.txt

### 4. Usage Workflow
### Step 1: Structural Screening (VhChip-based)
#### This stage identifies candidates with high binding potential for the VhChip gatekeeper residue (W136).
#### 1.1 Extract Native Interaction Priors

python scripts/00_extract_native_priors.py \
  --complex_pdb data/VhChip-chitohexaose.pdb \
  --out_dir results/01_priors \
  --protein_chain A --ligand_chain G --w136_chain A --w136_resseq 136

#### 1.2 High-Throughput Peptide Screening

python scripts/01_screen_peptides.py \
  --peptide_xlsx data/TSP_sca_new.xlsx \
  --site_profile results/01_priors/site_profile.json \
  --out_xlsx results/02_screening/peptide_screening_ranked.xlsx \
  --alpha_w136 1.5

#### 1.3 Diversity-Based Selection (Top 100)

python scripts/02_cluster_select_top100.py \
  --ranked_xlsx results/02_screening/peptide_screening_ranked.xlsx \
  --out_xlsx results/03_clustering/top100_cluster_selected.xlsx \
  --top_n 100 --method ward --n_clusters 20


#### 1.4 Post-Docking Interaction Profiling
(Note: This step assumes 3D docking poses have been generated via CDOCKER or similar tools)

python scripts/05_extract_observed_ifp.py \
  --poses_dir results/03_docking/poses_complex \
  --top100_xlsx results/03_clustering/top100_cluster_selected.xlsx \
  --out_csv results/04_docking_ifp/top100_observed_ifp.csv \
  --w136_chain A --w136_resseq 136

#### 1.5 Pipeline Performance Evaluation

python scripts/06_evaluate_pipeline.py \
  --complex_pdb data/VhChip-chitohexaose.pdb \
  --priors_dir results/01_priors \
  --ranked_xlsx results/02_screening/peptide_screening_ranked.xlsx \
  --top100_xlsx results/03_clustering/top100_cluster_selected.xlsx \
  --out_xlsx results/05_eval/evaluation_report.xlsx

### Step 2: Biophysical Filtering

python step2_prior_knowledge/physical_filter.py --input results/03_clustering/top100_cluster_selected.xlsx

### Step 3: Specificity Recognition (AWLSTM)

python scripts/04_predict_specificity.py \
  --input_csv results/04_docking_ifp/top100_observed_ifp.csv \
  --model_path bin/AWLSTM.pth


