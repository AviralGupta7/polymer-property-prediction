"""
Setup Script for Polymer Property Prediction Pipeline
Helps with initial configuration and data file setup
"""

import os
import sys
import shutil
from pathlib import Path


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*80)
    print(text)
    print("="*80)


def print_success(text):
    """Print success message"""
    print(f"✓ {text}")


def print_error(text):
    """Print error message"""
    print(f"✗ {text}")


def print_info(text):
    """Print info message"""
    print(f"ℹ {text}")


def create_data_directory():
    """Create the data directory structure"""
    print_header("CREATING DATA DIRECTORY STRUCTURE")

    data_dir = r'polymer prediction'

    if os.path.exists(data_dir):
        print_info(f"Directory '{data_dir}' already exists")
    else:
        try:
            os.makedirs(data_dir, exist_ok=True)
            print_success(f"Created directory: {data_dir}")
        except Exception as e:
            print_error(f"Failed to create directory: {e}")
            return False

    return True


def check_data_files():
    """Check what data files are present"""
    print_header("CHECKING DATA FILES")

    data_dir = r'polymer prediction'
    required_files = [
        'train.csv',
        'test.csv'
    ]

    optional_files = [
        'Tc_SMILES.csv',
        'TgSS_enriched_cleaned.csv',
        'JCIM_sup_bigsmiles.csv',
        'data_tg3.xlsx',
        'data_dnst1.xlsx',
        'dataset4.csv'
    ]

    print("\nRequired files (essential for pipeline to run):")
    all_required_present = True
    for file in required_files:
        path = os.path.join(data_dir, file)
        if os.path.exists(path):
            file_size = os.path.getsize(path) / (1024*1024)  # MB
            print_success(f"{file} ({file_size:.1f} MB)")
        else:
            print_error(f"{file} - NOT FOUND")
            all_required_present = False

    print("\nOptional files (enhance model performance):")
    for file in optional_files:
        path = os.path.join(data_dir, file)
        if os.path.exists(path):
            file_size = os.path.getsize(path) / (1024*1024)  # MB
            print_success(f"{file} ({file_size:.1f} MB)")
        else:
            print_info(f"{file} - not found (optional)")

    return all_required_present


def show_setup_instructions():
    """Show instructions for setting up data files"""
    print_header("DATA FILE SETUP INSTRUCTIONS")

    instructions = """
1. REQUIRED FILES
   Place these files in the 'polymer prediction' directory:

   • train.csv          - Training dataset with SMILES and target values
   • test.csv           - Test dataset with SMILES strings

2. OPTIONAL FILES (recommended for better performance)

   • Tc_SMILES.csv                   - External Tc data
   • TgSS_enriched_cleaned.csv       - External Tg data (enriched)
   • JCIM_sup_bigsmiles.csv          - JCIM supplementary Tg data
   • data_tg3.xlsx                   - Additional Tg data
   • data_dnst1.xlsx                 - Density data
   • dataset4.csv                    - FFV data

3. DIRECTORY STRUCTURE
   Your project should look like:

   your_project/
   ├── polymer prediction/
   │   ├── train.csv
   │   ├── test.csv
   │   ├── Tc_SMILES.csv (optional)
   │   └── ... other optional files
   ├── config.py
   ├── data_utils.py
   ├── feature_engineering.py
   ├── model_utils.py
   ├── train_pipeline.py
   ├── save_production_models.py
   ├── predict.py
   └── main.py

4. RUNNING THE PIPELINE

   After placing files, run:

   python main.py

   Or check environment first:

   python main.py --check-env-only

5. CUSTOM DATA PATH (Optional)

   If your files are in a different location, you have options:

   a) Set environment variable:
      # Windows
      set POLYMER_DATA_DIR=C:\path\to\data

      # Linux/Mac
      export POLYMER_DATA_DIR=/path/to/data

      Then run: python main.py

   b) Modify in Python code:
      from config import set_data_paths
      set_data_paths('/your/data/path')

      Then run training pipeline

   c) Modify config.py directly:
      Edit LOCAL_DATA_PATH variable

6. TROUBLESHOOTING

   Issue: "No such file or directory"
   Fix: Make sure files are in 'polymer prediction' directory

   Issue: File encoding error when reading CSV
   Fix: Ensure CSV files are UTF-8 encoded

   Issue: Missing column in dataset
   Fix: Check that CSV has required columns (SMILES, target values)
    """

    print(instructions)


def show_file_format_examples():
    """Show examples of expected file formats"""
    print_header("EXPECTED FILE FORMATS")

    examples = """
1. TRAIN.CSV FORMAT

   Expected columns: SMILES, Tg, FFV, Tc, Density, Rg

   Example:
   SMILES,Tg,FFV,Tc,Density,Rg
   CC(C)C,100.5,,150.2,1.05,2.3
   CCCC,95.2,0.15,148.0,1.02,2.5
   C1CCCCC1,110.1,,152.5,1.08,

   Notes:
   - SMILES is required
   - Target values (Tg, FFV, Tc, Density, Rg) can be missing (NaN)
   - Can have additional columns (will be ignored)

2. TEST.CSV FORMAT

   Expected columns: id, SMILES (and optionally target columns)

   Example:
   id,SMILES
   0,CC(C)C
   1,CCCC
   2,C1CCCCC1

   Notes:
   - id column uniquely identifies each sample
   - Target columns will be predicted

3. EXTERNAL DATA FORMAT

   External datasets should have:
   - SMILES column (required)
   - Target column named after property (Tg, Tc, FFV, Density, Rg)

   Example (Tc_SMILES.csv):
   SMILES,TC_mean
   CC(C)C,150.2
   CCCC,148.0
    """

    print(examples)


def verify_installation():
    """Verify that all required packages are installed"""
    print_header("VERIFYING PACKAGE INSTALLATION")

    packages = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'rdkit': 'rdkit',
        'networkx': 'networkx',
        'sklearn': 'scikit-learn',
        'xgboost': 'xgboost',
        'joblib': 'joblib'
    }

    missing = []
    for import_name, display_name in packages.items():
        try:
            if import_name == 'sklearn':
                __import__('sklearn')
            else:
                __import__(import_name)
            print_success(f"{display_name}")
        except ImportError:
            print_error(f"{display_name}")
            missing.append(display_name)

    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("\nInstall using: pip install -r requirements.txt")
        return False

    return True


def main():
    """Main setup function"""
    print_header("POLYMER PROPERTY PREDICTION - SETUP WIZARD")

    print("""
This setup wizard will help you:
1. Create the data directory structure
2. Check for required and optional data files
3. Verify package installation
4. Show file format examples
5. Provide setup instructions
    """)

    # Step 1: Create directory
    if not create_data_directory():
        print_error("Failed to setup directories")
        sys.exit(1)

    # Step 2: Verify packages
    packages_ok = verify_installation()

    # Step 3: Check data files
    files_ok = check_data_files()

    # Step 4: Show instructions
    show_file_format_examples()
    show_setup_instructions()

    # Summary
    print_header("SETUP SUMMARY")

    if files_ok and packages_ok:
        print_success("All requirements met! Ready to run pipeline.")
        print_info("Run: python main.py")
    elif packages_ok and not files_ok:
        print_error("Some data files are missing.")
        print_info("Please download/place required files in 'polymer prediction' directory")
        print_info("Then run: python main.py")
    else:
        print_error("Some packages are missing.")
        print_info("Run: pip install -r requirements.txt")

    print()


if __name__ == "__main__":
    main()
