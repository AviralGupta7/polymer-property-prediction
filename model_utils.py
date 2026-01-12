"""
Model Training and Evaluation Utilities
Fixed for XGBoost Compatibility
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from config import TARGETS, MODEL_PARAMS


def calculate_wmae_weights(df, targets):
    """
    Calculates the weighted Mean Absolute Error (wMAE) weights based on the 
    competition's specific formula.

    Args:
        df (pd.DataFrame): DataFrame containing target columns
        targets (list): List of target property names

    Returns:
        dict: Dictionary mapping target names to their wMAE weights
    """
    K = len(targets)
    n_i = df[targets].notna().sum()
    r_i = df[targets].max() - df[targets].min()
    inv_sqrt_n = 1 / np.sqrt(n_i)
    normalization_factor = np.sum(inv_sqrt_n)
    weights = (1 / r_i) * (K * inv_sqrt_n / normalization_factor)
    return weights.to_dict()


def train_model_kfold(X, y, label, n_splits=5, random_state=42):
    """
    Trains a model using K-Fold cross-validation.

    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target values
        label (str): Target property name
        n_splits (int): Number of folds for cross-validation
        random_state (int): Random seed

    Returns:
        tuple: (oof_predictions, oof_mae)
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_preds = np.zeros(len(X))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Initialize the model with pre-defined parameters
        model = XGBRegressor(**MODEL_PARAMS[label])

        # Fit model - using compatible parameters for different XGBoost versions
        try:
            # Try newer XGBoost API (2.0+)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='mae',
                verbose=False,
                early_stopping_rounds=50
            )
        except TypeError:
            # Fallback to older XGBoost API (< 2.0)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
                early_stopping_rounds=50
            )

        # Store out-of-fold predictions
        oof_preds[val_idx] = model.predict(X_val)

    # Calculate out-of-fold MAE
    oof_mae = mean_absolute_error(y, oof_preds)

    return oof_preds, oof_mae


def train_all_models(train_extended, train_features_df, test_features_df, 
                     subtables, filters):
    """
    Trains models for all target properties using K-Fold cross-validation.

    Args:
        train_extended (pd.DataFrame): Extended training data
        train_features_df (pd.DataFrame): Training features
        test_features_df (pd.DataFrame): Test features
        subtables (dict): Dictionary of target-specific data subsets
        filters (dict): Dictionary of feature filters for each target

    Returns:
        tuple: (oof_scores, final_test_predictions, wmae_score)
    """
    oof_scores = {}
    final_test_predictions = pd.DataFrame()

    for label in TARGETS:
        print(f"\n--- Processing Target: {label} ---")

        # Select data for the current target
        target_df = subtables[label]
        X = train_features_df.loc[target_df.index]
        y = target_df[label]

        # Apply the specific feature filter for this target
        X = X[filters[label]]
        X_test = test_features_df[filters[label]]

        print(f"  Training samples: {len(X)}")
        print(f"  Features: {len(X.columns)}")

        # K-Fold Cross-Validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        oof_preds_target = np.zeros(len(X))
        test_preds_target = []

        fold_count = 0
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            fold_count += 1
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            print(f"  Fold {fold + 1}/5...", end=" ", flush=True)

            # Initialize the model with pre-defined parameters
            model = XGBRegressor(**MODEL_PARAMS[label])

            # Fit model - using compatible parameters for different XGBoost versions
            try:
                # Try newer XGBoost API (2.0+)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    eval_metric='mae',
                    verbose=False,
                    early_stopping_rounds=50
                )
            except (TypeError, AttributeError):
                # Fallback to older XGBoost API (< 2.0)
                # For older versions, we use eval_set but without eval_metric
                try:
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        verbose=False,
                        early_stopping_rounds=50
                    )
                except TypeError:
                    # For very old versions, fit without eval_set
                    model.fit(
                        X_train, y_train,
                        verbose=False
                    )

            # Store predictions
            oof_preds_target[val_idx] = model.predict(X_val)
            test_preds_target.append(model.predict(X_test))

            print("✓")

        # Evaluate and store results for this target
        mae = mean_absolute_error(y, oof_preds_target)
        oof_scores[label] = mae
        print(f"  OOF MAE for {label}: {mae:.5f}")

        # Average predictions across all 5 folds for the final test prediction
        final_test_predictions[label] = np.mean(test_preds_target, axis=0)

    # Calculate final validation score
    wmae_weights = calculate_wmae_weights(train_extended, TARGETS)
    wmae_score = sum(wmae_weights[label] * oof_scores[label] for label in TARGETS)

    print(f"\n--- Validation Complete ---")
    print(f"Final Calculated OOF wMAE Score: {wmae_score:.6f}")

    return oof_scores, final_test_predictions, wmae_score
