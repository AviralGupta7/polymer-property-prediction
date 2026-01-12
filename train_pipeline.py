"""
Main Training Pipeline for Polymer Property Prediction
"""

import time
import pandas as pd
from data_utils import load_and_integrate_data, separate_subtables
from feature_engineering import generate_features, align_feature_columns
from model_utils import train_all_models
from config import TARGETS, FILTERS


def run_training_pipeline(data_paths=None):
    """
    Executes the complete training pipeline.

    Args:
        data_paths (dict): Dictionary of data paths (optional)

    Returns:
        tuple: (oof_scores, final_test_predictions, wmae_score, test_df)
    """
    # Step 1: Load and integrate data
    print("="*80)
    print("STEP 1: DATA LOADING AND INTEGRATION")
    print("="*80)
    train_extended, test_df = load_and_integrate_data(data_paths)

    # Step 2: Generate features
    print("\n" + "="*80)
    print("STEP 2: FEATURE GENERATION")
    print("="*80)
    print("Starting feature pre-computation for all unique molecules...")
    start_time = time.time()

    train_features_df = generate_features(train_extended)
    test_features_df = generate_features(test_df)

    # Align feature columns
    train_features_df, test_features_df = align_feature_columns(
        train_features_df, test_features_df
    )

    end_time = time.time()
    print(f"Feature generation complete for {len(train_features_df)} train and "
          f"{len(test_features_df)} test molecules.")
    print(f"Total features: {len(train_features_df.columns)}. "
          f"Time taken: {end_time - start_time:.2f} seconds.")

    # Step 3: Prepare data subsets
    print("\n" + "="*80)
    print("STEP 3: DATA PREPARATION")
    print("="*80)
    subtables = separate_subtables(train_extended)
    print("Data separated into target-specific subtables.")

    # Step 4: Train models
    print("\n" + "="*80)
    print("STEP 4: MODEL TRAINING")
    print("="*80)
    oof_scores, final_test_predictions, wmae_score = train_all_models(
        train_extended, train_features_df, test_features_df, 
        subtables, FILTERS
    )

    return oof_scores, final_test_predictions, wmae_score, test_df


def create_submission(final_test_predictions, test_df, output_file='submission.csv'):
    """
    Creates a submission file.

    Args:
        final_test_predictions (pd.DataFrame): Predictions for test set
        test_df (pd.DataFrame): Test DataFrame containing IDs
        output_file (str): Output filename
    """
    submission_df = final_test_predictions.copy()
    submission_df.insert(0, 'id', test_df['id'].values)
    submission_df.to_csv(output_file, index=False)

    print(f"\n✅ {output_file} created successfully.")
    print(submission_df.head())


if __name__ == "__main__":
    # Run the complete pipeline
    oof_scores, predictions, wmae_score, test_df = run_training_pipeline()

    # Create submission file
    create_submission(predictions, test_df)
