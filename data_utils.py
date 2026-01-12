"""
Data Loading, Cleaning, and Validation Utilities
Updated for Local File Paths
"""

import gc
import os
import pandas as pd
from rdkit import Chem
from config import TARGETS, DATA_PATHS, check_data_files


def clean_and_validate_smiles(smiles):
    """
    Validates and cleans SMILES strings, specifically targeting non-standard
    notations found in external chemical datasets.

    Args:
        smiles (str): SMILES string to validate

    Returns:
        str or None: Canonical SMILES if valid, None otherwise
    """
    if not isinstance(smiles, str) or len(smiles) == 0:
        return None

    # Explicitly filter known non-standard R-group notations
    bad_patterns = ['[R]', '[R1]', '[R2]', '[R3]', '[R4]', '[R5]', "[R']", '[R"]']
    if any(pattern in smiles for pattern in bad_patterns):
        return None

    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except:
        return None


def check_required_files():
    """
    Checks if required data files exist.

    Returns:
        tuple: (bool, list) - (all_found, missing_files)
    """
    required_files = ['train', 'test']
    missing_files = []

    for file_key in required_files:
        if file_key not in DATA_PATHS:
            missing_files.append(file_key)
            continue

        path = DATA_PATHS[file_key]
        if not os.path.exists(path):
            missing_files.append(f"{file_key} ({path})")

    return len(missing_files) == 0, missing_files


def add_extra_data_clean(df_train, df_extra, target, targets_list):
    """
    Integrates external data with training data for a specific target.

    Args:
        df_train (pd.DataFrame): Training DataFrame
        df_extra (pd.DataFrame): External data to integrate
        target (str): Target property name
        targets_list (list): List of all targets

    Returns:
        pd.DataFrame: Merged DataFrame
    """
    # Ensure the target column is numeric before grouping
    df_extra[target] = pd.to_numeric(df_extra[target], errors='coerce')
    df_extra.dropna(subset=[target], inplace=True)

    df_extra['SMILES'] = df_extra['SMILES'].apply(clean_and_validate_smiles)
    df_extra.dropna(subset=['SMILES'], inplace=True)
    if df_extra.empty:
        return df_train

    df_extra = df_extra.groupby('SMILES', as_index=False)[target].mean()

    # Merge to fill NAs and add new SMILES
    df_train = pd.merge(df_train, df_extra, on='SMILES', how='outer', suffixes=('', '_new'))
    df_train[target] = df_train[target].fillna(df_train[target + '_new'])
    df_train.drop(columns=[target + '_new'], inplace=True)

    return df_train


def load_and_integrate_data(data_paths=None):
    """
    Loads competition data and integrates external datasets.

    Args:
        data_paths (dict): Dictionary of data paths (uses DATA_PATHS from config if None)

    Returns:
        tuple: (train_extended, test_df) DataFrames

    Raises:
        FileNotFoundError: If required files are not found
    """
    if data_paths is None:
        data_paths = DATA_PATHS

    print("="*80)
    print("DATA FILE STATUS CHECK")
    print("="*80)
    check_data_files()
    print()

    print("="*80)
    print("DATA LOADING AND INTEGRATION")
    print("="*80)
    print("Starting data ingestion and standardization...")

    # Check for required files
    all_found, missing = check_required_files()
    if not all_found:
        print(f"\n❌ ERROR: Missing required files:")
        for file in missing:
            print(f"   - {file}")
        print(f"\n📁 Expected files in: {os.path.abspath('polymer prediction')}")
        print("\n💡 SOLUTION:")
        print("   1. Ensure files are in 'polymer prediction' directory")
        print("   2. OR set custom path using: config.set_data_paths('/your/path')")
        print("   3. OR set environment variable: POLYMER_DATA_DIR=/your/path")
        raise FileNotFoundError(f"Missing required data files: {missing}")

    print("✓ All required files found\n")

    # Load competition data
    print(f"Loading train data from: {data_paths['train']}")
    train = pd.read_csv(data_paths['train'])
    print(f"  Loaded {len(train)} training samples")

    print(f"Loading test data from: {data_paths['test']}")
    test_df = pd.read_csv(data_paths['test'])
    print(f"  Loaded {len(test_df)} test samples\n")

    # Apply initial cleaning to base data
    train['SMILES'] = train['SMILES'].apply(clean_and_validate_smiles)
    test_df['SMILES'] = test_df['SMILES'].apply(clean_and_validate_smiles)
    train.dropna(subset=['SMILES'], inplace=True)
    test_df.dropna(subset=['SMILES'], inplace=True)

    # Load and process external datasets
    print("Loading and integrating external datasets...")
    train_extended = train.copy()

    # List of tuples: (path_key, target, processing_function)
    datasets_to_load = [
        ('tc_smiles', 'Tc', 
         lambda df: df.rename(columns={'TC_mean': 'Tc'})),
        ('tg_smiles', 'Tg', 
         lambda df: df),
        ('jcim_bigsmiles', 'Tg', 
         lambda df: df.rename(columns={'Tg (C)': 'Tg'})),
        ('data_tg3', 'Tg', 
         lambda df: df.rename(columns={'Tg [K]': 'Tg'}).assign(Tg=lambda x: x['Tg'] - 273.15)),
        ('data_dnst1', 'Density', 
         lambda df: df.rename(columns={'density(g/cm3)': 'Density'})),
        ('dataset4', 'FFV', 
         lambda df: df)
    ]

    for path_key, target, processor in datasets_to_load:
        try:
            if path_key not in data_paths:
                print(f"  ⚠️  {path_key}: Not configured (skipping)")
                continue

            path = data_paths[path_key]

            if not os.path.exists(path):
                print(f"  ⚠️  {path_key}: File not found (skipping)")
                continue

            print(f"  Loading {path_key}...", end=" ")
            ext_df = pd.read_excel(path) if path.endswith('.xlsx') else pd.read_csv(path)
            ext_df = processor(ext_df)
            train_extended = add_extra_data_clean(train_extended, ext_df, target, TARGETS)
            print(f"✓ ({len(ext_df)} samples)")

        except Exception as e:
            print(f"  ✗ {path_key}: Error - {str(e)[:50]}")

    # Drop duplicates, keeping the first instance (prioritizing original data)
    train_extended.drop_duplicates(subset=['SMILES'], keep='first', inplace=True)

    print("\n" + "="*80)
    print("DATA INTEGRATION SUMMARY")
    print("="*80)
    print(f"Total unique polymers for training: {train_extended['SMILES'].nunique():,}")
    for target in TARGETS:
        count = train_extended[target].notna().sum()
        percentage = (count / len(train_extended)) * 100
        print(f"  {target:10s}: {count:6,} samples ({percentage:5.1f}%)")

    # Clean up memory
    gc.collect()

    return train_extended, test_df


def separate_subtables(train_df):
    """
    Separates training data into subtables for each target property.

    Args:
        train_df (pd.DataFrame): Training DataFrame

    Returns:
        dict: Dictionary mapping target names to their respective subtables
    """
    subtables = {}
    for label in TARGETS:
        subtables[label] = train_df[['SMILES', label]][train_df[label].notna()]
    return subtables
