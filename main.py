"""
Master Execution Pipeline for Polymer Property Prediction

This script runs the complete pipeline from data loading to model training,
evaluation, and prediction generation.
"""

import os
import sys
import time
import argparse
from datetime import datetime


def setup_environment():
    """Check and setup the execution environment."""
    print("="*80)
    print("ENVIRONMENT SETUP")
    print("="*80)

    # Check if required packages are installed
    required_packages = [
        'numpy', 'pandas', 'rdkit', 'networkx', 
        'sklearn', 'xgboost', 'joblib'
    ]

    missing_packages = []
    for package in required_packages:
        try:
            if package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} is NOT installed")

    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Please run: pip install -r requirements.txt")
        return False

    print("\n✅ All required packages are installed")
    return True


def run_full_pipeline(skip_training=False, skip_model_save=False, 
                      skip_prediction=False):
    """
    Executes the complete pipeline.

    Args:
        skip_training (bool): Skip training if models already exist
        skip_model_save (bool): Skip saving production models
        skip_prediction (bool): Skip generating predictions
    """
    start_time = time.time()

    print("\n" + "="*80)
    print("POLYMER PROPERTY PREDICTION PIPELINE")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # Step 1: Training
    if not skip_training:
        print("\n" + "="*80)
        print("PHASE 1: MODEL TRAINING")
        print("="*80)

        from train_pipeline import run_training_pipeline, create_submission

        try:
            oof_scores, predictions, wmae_score, test_df = run_training_pipeline()

            print(f"\n📊 Training Results:")
            print(f"   Overall wMAE Score: {wmae_score:.6f}")
            for target, score in oof_scores.items():
                print(f"   {target} MAE: {score:.5f}")

            # Create submission
            create_submission(predictions, test_df, 'submission.csv')

            print("\n✅ Phase 1 Complete: Training finished successfully")

        except Exception as e:
            print(f"\n❌ Error during training: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print("\n⏭️  Skipping training phase")

    # Step 2: Save Production Models
    if not skip_model_save:
        print("\n" + "="*80)
        print("PHASE 2: SAVING PRODUCTION MODELS")
        print("="*80)

        try:
            from save_production_models import (
                load_and_integrate_data, generate_features, 
                separate_subtables, save_production_models
            )

            print("Loading data for production model training...")
            train_extended, _ = load_and_integrate_data()

            print("Generating features...")
            train_features_df = generate_features(train_extended)

            print("Preparing subtables...")
            subtables = separate_subtables(train_extended)

            print("Training and saving production models...")
            save_production_models(train_extended, train_features_df, subtables)

            print("\n✅ Phase 2 Complete: Production models saved successfully")

        except Exception as e:
            print(f"\n❌ Error saving production models: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print("\n⏭️  Skipping production model save phase")

    # Step 3: Make Predictions (if test file is provided)
    if not skip_prediction:
        print("\n" + "="*80)
        print("PHASE 3: GENERATING PREDICTIONS")
        print("="*80)

        # Check if test file exists
        test_file = 'test.csv'
        if os.path.exists(test_file):
            try:
                from predict import predict_from_file

                print(f"Making predictions on {test_file}...")
                predict_from_file(test_file, 'final_predictions.csv')

                print("\n✅ Phase 3 Complete: Predictions generated successfully")

            except Exception as e:
                print(f"\n❌ Error during prediction: {e}")
                import traceback
                traceback.print_exc()
                return False
        else:
            print(f"⚠️  Test file '{test_file}' not found. Skipping predictions.")
    else:
        print("\n⏭️  Skipping prediction phase")

    # Pipeline Summary
    end_time = time.time()
    elapsed_time = end_time - start_time

    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"Total Execution Time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # List generated files
    print("\n📁 Generated Files:")
    output_files = [
        'submission.csv',
        'final_predictions.csv',
        'production_models/'
    ]

    for file in output_files:
        if os.path.exists(file):
            if os.path.isdir(file):
                num_files = len(os.listdir(file))
                print(f"   ✓ {file} (contains {num_files} files)")
            else:
                file_size = os.path.getsize(file) / 1024  # KB
                print(f"   ✓ {file} ({file_size:.2f} KB)")

    print("\n✅ All phases completed successfully!")
    print("="*80)

    return True


def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description='Polymer Property Prediction Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python main.py

  # Skip training (use existing models)
  python main.py --skip-training

  # Only train, don't save production models
  python main.py --skip-model-save

  # Run everything except predictions
  python main.py --skip-prediction
        """
    )

    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip the training phase'
    )

    parser.add_argument(
        '--skip-model-save',
        action='store_true',
        help='Skip saving production models'
    )

    parser.add_argument(
        '--skip-prediction',
        action='store_true',
        help='Skip the prediction phase'
    )

    parser.add_argument(
        '--check-env-only',
        action='store_true',
        help='Only check the environment and exit'
    )

    args = parser.parse_args()

    # Setup environment
    if not setup_environment():
        print("\n❌ Environment setup failed. Please install required packages.")
        sys.exit(1)

    if args.check_env_only:
        print("\n✅ Environment check complete. Exiting.")
        sys.exit(0)

    # Run pipeline
    success = run_full_pipeline(
        skip_training=args.skip_training,
        skip_model_save=args.skip_model_save,
        skip_prediction=args.skip_prediction
    )

    if success:
        sys.exit(0)
    else:
        print("\n❌ Pipeline failed. Check error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
