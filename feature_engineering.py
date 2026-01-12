"""
Feature Engineering for Molecular Property Prediction
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys, rdmolops
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
import networkx as nx


def generate_features(df_input):
    """
    Generates a comprehensive feature set from a DataFrame containing SMILES strings.

    This function creates three types of features:
    1. Descriptors: ~200 physicochemical properties (e.g., MolWt, LogP, TPSA)
    2. Fingerprints: Binary vectors (Morgan and MACCS keys) encoding substructural features
    3. Graph Features: Topological indices describing molecular connectivity and shape

    Args:
        df_input (pd.DataFrame): Input DataFrame which must contain a 'SMILES' column

    Returns:
        pd.DataFrame: DataFrame containing all calculated features, indexed identically to input
    """
    all_features_list = []

    # Initialize the Morgan fingerprint generator once
    morgan_gen = GetMorganGenerator(radius=2, fpSize=128)

    for smiles in df_input['SMILES']:
        mol = Chem.MolFromSmiles(smiles)

        # If a molecule is invalid, append a dictionary of NaNs and continue
        if mol is None:
            all_features_list.append({})
            continue

        # --- Feature Calculation ---
        # 1. Descriptors
        descriptors = Descriptors.CalcMolDescriptors(mol)

        # 2. Fingerprints
        maccs_fp = {f'maccs_{i}': bit for i, bit in enumerate(MACCSkeys.GenMACCSKeys(mol))}
        morgan_fp = {f'morgan_{i}': bit for i, bit in enumerate(morgan_gen.GetFingerprint(mol))}

        # 3. Graph-based features
        graph_features = {}
        try:
            adj = Chem.rdmolops.GetAdjacencyMatrix(mol)
            G = nx.from_numpy_array(adj)
            if nx.is_connected(G):
                graph_features['graph_diameter'] = nx.diameter(G)
                graph_features['avg_shortest_path'] = nx.average_shortest_path_length(G)
            else:
                graph_features['graph_diameter'] = 0
                graph_features['avg_shortest_path'] = 0
            graph_features['num_cycles'] = len(list(nx.cycle_basis(G)))
        except:
            graph_features['graph_diameter'] = np.nan
            graph_features['avg_shortest_path'] = np.nan
            graph_features['num_cycles'] = np.nan

        # Combine all features for the current molecule
        combined_features = {**descriptors, **maccs_fp, **morgan_fp, **graph_features}
        all_features_list.append(combined_features)

    # Create the final DataFrame and fill any missing values
    features_df = pd.DataFrame(all_features_list, index=df_input.index).fillna(0)

    return features_df


def augment_smiles_dataset(smiles_list, labels, num_augments=1):
    """
    Augments a list of SMILES strings by generating randomized versions.
    This increases the diversity of the training data.

    Args:
        smiles_list (list): List of SMILES strings
        labels (array-like): Corresponding labels/target values
        num_augments (int): Number of augmented versions to generate per SMILES

    Returns:
        tuple: (augmented_smiles, augmented_labels) lists
    """
    augmented_smiles = []
    augmented_labels = []

    for smiles, label in zip(smiles_list, labels):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        # Add the original SMILES and its label
        augmented_smiles.append(smiles)
        augmented_labels.append(label)

        # Add randomized versions
        for _ in range(num_augments):
            rand_smiles = Chem.MolToSmiles(mol, doRandom=True)
            augmented_smiles.append(rand_smiles)
            augmented_labels.append(label)

    return augmented_smiles, np.array(augmented_labels)


def align_feature_columns(train_features_df, test_features_df):
    """
    Aligns feature columns between training and test sets.

    Args:
        train_features_df (pd.DataFrame): Training features
        test_features_df (pd.DataFrame): Test features

    Returns:
        tuple: (aligned_train, aligned_test) DataFrames
    """
    train_cols = set(train_features_df.columns)
    test_cols = set(test_features_df.columns)

    missing_in_test = list(train_cols - test_cols)
    missing_in_train = list(test_cols - train_cols)

    for col in missing_in_test:
        test_features_df[col] = 0
    for col in missing_in_train:
        train_features_df[col] = 0

    # Reorder test columns to match training
    test_features_df = test_features_df[train_features_df.columns]

    return train_features_df, test_features_df
