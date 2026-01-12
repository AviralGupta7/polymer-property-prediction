"""
Configuration and Constants for Polymer Property Prediction
Updated for Local Execution with Flexible Path Handling
"""

import os
import sys

# Target properties
TARGETS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

# Base set of descriptors considered fundamentally important for all targets
REQUIRED_DESCRIPTORS = {
    'MolWt', 'MolLogP', 'TPSA', 'NumRotatableBonds', 'HeavyAtomCount',
    'graph_diameter', 'num_cycles', 'avg_shortest_path'
}

# Feature selection filters for each target
FILTERS = {
    'Tg': sorted(list(set([
        'BalabanJ','BertzCT','Chi1','Chi3n','Chi4n','EState_VSA4','EState_VSA8',
        'FpDensityMorgan3','HallKierAlpha','Kappa3','MaxAbsEStateIndex','MolLogP',
        'NumAmideBonds','NumHeteroatoms','NumHeterocycles','NumRotatableBonds',
        'PEOE_VSA14','Phi','RingCount','SMR_VSA1','SPS','SlogP_VSA1','SlogP_VSA5',
        'SlogP_VSA8','TPSA','VSA_EState1','VSA_EState4','VSA_EState6','VSA_EState7',
        'VSA_EState8','fr_C_O_noCOO','fr_NH1','fr_benzene','fr_bicyclic','fr_ether',
        'fr_unbrch_alkane'
    ]).union(REQUIRED_DESCRIPTORS))),

    'FFV': sorted(list(set([
        'AvgIpc','BalabanJ','BertzCT','Chi0','Chi0n','Chi0v','Chi1','Chi1n','Chi1v',
        'Chi2n','Chi2v','Chi3n','Chi3v','Chi4n','EState_VSA10','EState_VSA5',
        'EState_VSA7','EState_VSA8','EState_VSA9','ExactMolWt','FpDensityMorgan1',
        'FpDensityMorgan2','FpDensityMorgan3','FractionCSP3','HallKierAlpha',
        'HeavyAtomMolWt','Kappa1','Kappa2','Kappa3','MaxAbsEStateIndex',
        'MaxEStateIndex','MinEStateIndex','MolLogP','MolMR','MolWt','NHOHCount',
        'NOCount','NumAromaticHeterocycles','NumHAcceptors','NumHDonors',
        'NumHeterocycles','NumRotatableBonds','PEOE_VSA14','RingCount','SMR_VSA1',
        'SMR_VSA10','SMR_VSA3','SMR_VSA5','SMR_VSA6','SMR_VSA7','SMR_VSA9','SPS',
        'SlogP_VSA1','SlogP_VSA10','SlogP_VSA11','SlogP_VSA12','SlogP_VSA2',
        'SlogP_VSA3','SlogP_VSA4','SlogP_VSA5','SlogP_VSA6','SlogP_VSA7',
        'SlogP_VSA8','TPSA','VSA_EState1','VSA_EState10','VSA_EState2',
        'VSA_EState3','VSA_EState4','VSA_EState5','VSA_EState6','VSA_EState7',
        'VSA_EState8','VSA_EState9','fr_Ar_N','fr_C_O','fr_NH0','fr_NH1',
        'fr_aniline','fr_ether','fr_halogen','fr_thiophene'
    ]).union(REQUIRED_DESCRIPTORS))),

    'Tc': sorted(list(set([
        'BalabanJ','BertzCT','Chi0','EState_VSA5','ExactMolWt','FpDensityMorgan1',
        'FpDensityMorgan2','FpDensityMorgan3','HeavyAtomMolWt','MinEStateIndex',
        'MolWt','NumAtomStereoCenters','NumRotatableBonds','NumValenceElectrons',
        'SMR_VSA10','SMR_VSA7','SPS','SlogP_VSA6','SlogP_VSA8','VSA_EState1',
        'VSA_EState7','fr_NH1','fr_ester','fr_halogen'
    ]).union(REQUIRED_DESCRIPTORS))),

    'Density': sorted(list(set([
        'BalabanJ','Chi3n','Chi3v','Chi4n','EState_VSA1','ExactMolWt',
        'FractionCSP3','HallKierAlpha','Kappa2','MinEStateIndex','MolMR','MolWt',
        'NumAliphaticCarbocycles','NumHAcceptors','NumHeteroatoms',
        'NumRotatableBonds','SMR_VSA10','SMR_VSA5','SlogP_VSA12','SlogP_VSA5',
        'TPSA','VSA_EState10','VSA_EState7','VSA_EState8'
    ]).union(REQUIRED_DESCRIPTORS))),

    'Rg': sorted(list(set([
        'AvgIpc','Chi0n','Chi1v','Chi2n','Chi3v','ExactMolWt','FpDensityMorgan1',
        'FpDensityMorgan2','FpDensityMorgan3','HallKierAlpha','HeavyAtomMolWt',
        'Kappa3','MaxAbsEStateIndex','MolWt','NOCount','NumRotatableBonds',
        'NumUnspecifiedAtomStereoCenters','NumValenceElectrons','PEOE_VSA14',
        'PEOE_VSA6','SMR_VSA1','SMR_VSA5','SPS','SlogP_VSA1','SlogP_VSA2',
        'SlogP_VSA7','SlogP_VSA8','VSA_EState1','VSA_EState8','fr_alkyl_halide',
        'fr_halogen'
    ]).union(REQUIRED_DESCRIPTORS)))
}

# Pre-tuned XGBoost hyperparameters
MODEL_PARAMS = {
    'Tg': {'n_estimators': 2173, 'learning_rate': 0.0672, 'max_depth': 6, 'reg_lambda': 5.545},
    'FFV': {'n_estimators': 2202, 'learning_rate': 0.0722, 'max_depth': 4, 'reg_lambda': 2.887},
    'Tc': {'n_estimators': 1488, 'learning_rate': 0.0104, 'max_depth': 5, 'reg_lambda': 9.970},
    'Density': {'n_estimators': 1958, 'learning_rate': 0.1095, 'max_depth': 5, 'reg_lambda': 3.074},
    'Rg': {'n_estimators': 520, 'learning_rate': 0.0732, 'max_depth': 5, 'reg_lambda': 0.971}
}

# Add common parameters to all models
for p in MODEL_PARAMS.values():
    p['random_state'] = 42
    p['n_jobs'] = -1
    p['tree_method'] = 'hist'


# ============================================================================
# FLEXIBLE DATA PATH CONFIGURATION
# ============================================================================

def get_data_paths():
    """
    Automatically detects and returns appropriate data paths.

    Priority:
    1. Local project directory structure
    2. Kaggle environment paths
    3. Custom paths if environment variable is set

    Returns:
        dict: Dictionary with data paths
    """

    # Check for custom data directory via environment variable
    custom_data_dir = os.environ.get('POLYMER_DATA_DIR')

    if custom_data_dir and os.path.exists(custom_data_dir):
        print(f"[INFO] Using custom data directory: {custom_data_dir}")
        return {
            'train': os.path.join(custom_data_dir, 'train.csv'),
            'test': os.path.join(custom_data_dir, 'test.csv'),
            'tc_smiles': os.path.join(custom_data_dir, 'Tc_SMILES.csv'),
            'tg_smiles': os.path.join(custom_data_dir, 'TgSS_enriched_cleaned.csv'),
            'jcim_bigsmiles': os.path.join(custom_data_dir, 'JCIM_sup_bigsmiles.csv'),
            'data_tg3': os.path.join(custom_data_dir, 'data_tg3.xlsx'),
            'data_dnst1': os.path.join(custom_data_dir, 'data_dnst1.xlsx'),
            'dataset4': os.path.join(custom_data_dir, 'dataset4.csv')
        }

    # Try local project directory first
    local_paths = {
        'train': r'polymer prediction\train.csv',
        'test': r'polymer prediction\test.csv',
        'tc_smiles': r'polymer prediction\Tc_SMILES.csv',
        'tg_smiles': r'polymer prediction\TgSS_enriched_cleaned.csv',
        'jcim_bigsmiles': r'polymer prediction\JCIM_sup_bigsmiles.csv',
        'data_tg3': r'polymer prediction\data_tg3.xlsx',
        'data_dnst1': r'polymer prediction\data_dnst1.xlsx',
        'dataset4': r'polymer prediction\dataset4.csv'
    }

    # Check if local paths exist
    if os.path.exists(local_paths['train']):
        print("[INFO] Using local project data paths")
        return local_paths

    # Try Kaggle environment paths (for compatibility)
    kaggle_paths = {
        'train': '/kaggle/input/neurips-open-polymer-prediction-2025/train.csv',
        'test': '/kaggle/input/neurips-open-polymer-prediction-2025/test.csv',
        'tc_smiles': '/kaggle/input/tc-smiles/Tc_SMILES.csv',
        'tg_smiles': '/kaggle/input/tg-smiles-pid-polymer-class/TgSS_enriched_cleaned.csv',
        'jcim_bigsmiles': '/kaggle/input/smiles-extra-data/JCIM_sup_bigsmiles.csv',
        'data_tg3': '/kaggle/input/smiles-extra-data/data_tg3.xlsx',
        'data_dnst1': '/kaggle/input/smiles-extra-data/data_dnst1.xlsx',
        'dataset4': '/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset4.csv'
    }

    if os.path.exists(kaggle_paths['train']):
        print("[INFO] Using Kaggle environment data paths")
        return kaggle_paths

    # If no paths found, warn user and return local paths as default
    print("[WARNING] Data files not found in standard locations!")
    print("[INFO] Returning local project paths as default")
    print("[TIP] Set POLYMER_DATA_DIR environment variable or ensure files are in 'polymer prediction' directory")
    return local_paths


# Get data paths at module import time
DATA_PATHS = get_data_paths()

# Also support setting custom path via a simple function
def set_data_paths(directory):
    """
    Set custom data directory for all data files.

    Args:
        directory (str): Path to directory containing all data files
    """
    global DATA_PATHS

    if not os.path.exists(directory):
        raise ValueError(f"Directory does not exist: {directory}")

    DATA_PATHS = {
        'train': os.path.join(directory, 'train.csv'),
        'test': os.path.join(directory, 'test.csv'),
        'tc_smiles': os.path.join(directory, 'Tc_SMILES.csv'),
        'tg_smiles': os.path.join(directory, 'TgSS_enriched_cleaned.csv'),
        'jcim_bigsmiles': os.path.join(directory, 'JCIM_sup_bigsmiles.csv'),
        'data_tg3': os.path.join(directory, 'data_tg3.xlsx'),
        'data_dnst1': os.path.join(directory, 'data_dnst1.xlsx'),
        'dataset4': os.path.join(directory, 'dataset4.csv')
    }

    print(f"[INFO] Data paths updated to: {directory}")


def check_data_files():
    """
    Check which data files are available and which are missing.

    Returns:
        dict: Status of each data file
    """
    status = {}
    for name, path in DATA_PATHS.items():
        exists = os.path.exists(path)
        status[name] = exists
        symbol = "✓" if exists else "✗"
        print(f"{symbol} {name}: {path}")

    return status
