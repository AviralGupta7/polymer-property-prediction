"""
Inference Script for Polymer Property Prediction
"""

import pandas as pd
import joblib
from feature_engineering import generate_features
from config import TARGETS, FILTERS


def predict_properties(smiles_list, model_dir='production_models'):
    """
    Predicts polymer properties for a list of SMILES strings.

    Args:
        smiles_list (list): List of SMILES strings
        model_dir (str): Directory containing trained models

    Returns:
        pd.DataFrame: DataFrame with predictions for all targets
    """
    # Create a DataFrame from SMILES
    df = pd.DataFrame({'SMILES': smiles_list})

    # Generate features
    print("Generating molecular features...")
    features_df = generate_features(df)

    # Make predictions for each target
    predictions = pd.DataFrame()
    predictions['SMILES'] = smiles_list

    for label in TARGETS:
        print(f"Predicting {label}...")

        # Load model and feature columns
        model = joblib.load(f'{model_dir}/{label}_model.joblib')
        kept_columns = joblib.load(f'{model_dir}/{label}_kept_columns.joblib')

        # Select and filter features
        X = features_df[FILTERS[label]]
        X = X[kept_columns]

        # Make predictions
        predictions[label] = model.predict(X)

    return predictions


def predict_from_file(input_file, output_file='predictions.csv', 
                     model_dir='production_models'):
    """
    Predicts properties for SMILES in a CSV file and saves results.

    Args:
        input_file (str): Path to input CSV file with 'SMILES' column
        output_file (str): Path to output CSV file
        model_dir (str): Directory containing trained models
    """
    # Read input file
    df = pd.read_csv(input_file)

    if 'SMILES' not in df.columns:
        raise ValueError("Input file must contain a 'SMILES' column")

    # Make predictions
    predictions = predict_properties(df['SMILES'].tolist(), model_dir)

    # Add ID column if it exists in input
    if 'id' in df.columns:
        predictions.insert(0, 'id', df['id'])

    # Save results
    predictions.to_csv(output_file, index=False)
    print(f"\n✅ Predictions saved to {output_file}")
    print(predictions.head())


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python predict.py <input_file> [output_file] [model_dir]")
        print("Example: python predict.py test.csv predictions.csv production_models")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'predictions.csv'
    model_dir = sys.argv[3] if len(sys.argv) > 3 else 'production_models'

    predict_from_file(input_file, output_file, model_dir)
