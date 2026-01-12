"""
Save Production Models and Artifacts
Fixed for XGBoost Compatibility
"""

import os
import joblib
from sklearn.feature_selection import VarianceThreshold
from xgboost import XGBRegressor
from data_utils import load_and_integrate_data, separate_subtables
from feature_engineering import generate_features
from config import TARGETS, FILTERS, MODEL_PARAMS


def save_production_models(train_extended, train_features_df, subtables, 
                           output_dir='production_models'):
    """
    Trains final models on complete datasets and saves them for production use.

    Args:
        train_extended (pd.DataFrame): Extended training data
        train_features_df (pd.DataFrame): Training features
        subtables (dict): Dictionary of target-specific data subsets
        output_dir (str): Directory to save model artifacts
    """
    os.makedirs(output_dir, exist_ok=True)

    for label in TARGETS:
        print(f"--- Finalizing and Saving Model for: {label} ---")

        target_df = subtables[label]
        X = train_features_df.loc[target_df.index]
        y = target_df[label]

        # Use the sorted list of features
        X = X[FILTERS[label]]

        print(f"  Training samples: {len(X)}")
        print(f"  Features before selection: {len(X.columns)}")

        # Fit the selector to find which columns to keep
        selector = VarianceThreshold(threshold=0.01)
        selector.fit(X)

        # Save the list of column names to keep
        kept_columns = X.columns[selector.get_support()].tolist()
        joblib.dump(kept_columns, f'{output_dir}/{label}_kept_columns.joblib')
        print(f"  Features after selection: {len(kept_columns)}")

        # Filter the data using the list of kept columns
        X_selected = X[kept_columns]

        # Train the final model on this correctly filtered data
        print(f"  Training final model...", end=" ", flush=True)
        final_model = XGBRegressor(**MODEL_PARAMS[label])

        try:
            # Try newer XGBoost API (2.0+)
            final_model.fit(X_selected, y, verbose=False)
        except TypeError:
            # Fallback to older XGBoost API
            final_model.fit(X_selected, y, verbose=False)

        joblib.dump(final_model, f'{output_dir}/{label}_model.joblib')
        print("✓")
        print(f"  Model saved successfully")

    print(f"\n✅ All production artifacts have been saved to {output_dir}/")


def load_production_model(label, model_dir='production_models'):
    """
    Loads a production model and its feature columns.

    Args:
        label (str): Target property name
        model_dir (str): Directory containing model artifacts

    Returns:
        tuple: (model, kept_columns)
    """
    model = joblib.load(f'{model_dir}/{label}_model.joblib')
    kept_columns = joblib.load(f'{model_dir}/{label}_kept_columns.joblib')
    return model, kept_columns


if __name__ == "__main__":
    # Load and prepare data
    print("Loading data for model training...")
    train_extended, _ = load_and_integrate_data()

    print("\nGenerating features...")
    train_features_df = generate_features(train_extended)

    print("\nPreparing subtables...")
    subtables = separate_subtables(train_extended)

    # Save production models
    print("\n" + "="*80)
    print("SAVING PRODUCTION MODELS")
    print("="*80)
    save_production_models(train_extended, train_features_df, subtables)
