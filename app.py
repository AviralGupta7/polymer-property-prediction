"""
Streamlit Web App for Polymer Property Prediction
Fixed to work with local project structure
"""

import streamlit as st
import pandas as pd
import joblib
import os
import sys
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
import networkx as nx
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Polymer Property Predictor",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 0rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# FEATURE GENERATION & ARTIFACT LOADING
# ============================================================================

@st.cache_resource
def load_artifacts():
    """Load pre-trained models and feature configurations."""
    artifacts = {}
    TARGETS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

    model_dir = 'production_models'

    # Check if models exist
    if not os.path.exists(model_dir):
        st.error(f"❌ Models directory not found: {model_dir}")
        st.info("Please run the training pipeline first: python main.py")
        return None

    try:
        for target in TARGETS:
            model_path = f'{model_dir}/{target}_model.joblib'
            columns_path = f'{model_dir}/{target}_kept_columns.joblib'

            if not os.path.exists(model_path):
                st.error(f"Model not found: {model_path}")
                return None

            if not os.path.exists(columns_path):
                st.error(f"Feature columns not found: {columns_path}")
                return None

            artifacts[f'{target}_model'] = joblib.load(model_path)
            artifacts[f'{target}_kept_columns'] = joblib.load(columns_path)

        return artifacts
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None


def generate_single_features(smiles):
    """
    Generate features for a single SMILES string.

    Args:
        smiles (str): SMILES string

    Returns:
        pd.DataFrame: Feature dataframe or None if invalid SMILES
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Generate descriptors
        descriptors = {
            name: func(mol) 
            for name, func in Descriptors.descList
        }

        # Generate MACCS keys
        maccs_fp = {
            f'maccs_{i}': int(bit) 
            for i, bit in enumerate(MACCSkeys.GenMACCSKeys(mol))
        }

        # Generate Morgan fingerprints
        morgan_gen = GetMorganGenerator(radius=2, fpSize=128)
        morgan_fp = {
            f'morgan_{i}': int(bit) 
            for i, bit in enumerate(morgan_gen.GetFingerprint(mol))
        }

        # Generate graph features
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
            graph_features['graph_diameter'] = 0
            graph_features['avg_shortest_path'] = 0
            graph_features['num_cycles'] = 0

        # Combine all features
        all_features = {**descriptors, **maccs_fp, **morgan_fp, **graph_features}
        return pd.DataFrame([all_features])

    except Exception as e:
        st.error(f"Error generating features: {e}")
        return None


# ============================================================================
# MAIN APPLICATION UI
# ============================================================================

def main():
    """Main application function."""

    # Header
    st.title("🧪 Polymer Property Predictor")
    st.write("""
        **A Materials Informatics application** to predict key physical properties 
        of polymers from their chemical structure.

        This tool is powered by a suite of **XGBoost models** trained on data from 
        the **NeurIPS 2025 Open Polymer Prediction challenge**.
    """)

    # Load models
    st.sidebar.header("⚙️ Configuration")
    artifacts = load_artifacts()

    if artifacts is None:
        st.error("""
        ❌ **Models not found!**

        Please follow these steps:
        1. Run the training pipeline: `python main.py`
        2. Wait for models to be saved to `production_models/` directory
        3. Then run this app: `streamlit run app.py`
        """)
        return

    st.sidebar.success("✅ Models loaded successfully")

    # Sidebar for inputs
    st.sidebar.header("📋 Input Controls")

    # Polymer examples
    polymer_examples = {
        "Select an example...": "",
        "Polystyrene": "*CC(*)c1ccccc1",
        "Polycarbonate": "*OC(C)(C)c1ccccc1OC(=O)*",
        "PET": "*OCC(=O)c1ccccc1C(=O)O*",
        "PMMA": "*CC(C)(C(=O)OC)*",
        "Polypropylene": "*C(C)C*",
        "Nylon 6": "*CCCCCC(=O)N*"
    }

    # Select example
    selected_example = st.sidebar.selectbox(
        "📦 Choose an example polymer:",
        list(polymer_examples.keys())
    )

    # SMILES input
    user_input = st.sidebar.text_area(
        "✏️ Or enter your own SMILES string:",
        value=polymer_examples[selected_example],
        height=100,
        help="Enter a valid SMILES string representing the polymer unit or backbone"
    )

    # Predict button
    predict_button = st.sidebar.button(
        "🔮 Predict Properties",
        type="primary",
        use_container_width=True
    )

    st.sidebar.markdown("---")
    st.sidebar.header("ℹ️ Information")
    st.sidebar.markdown("""
    ### About This App
    - **Models**: XGBoost Regressors
    - **Training Data**: NeurIPS 2025 Polymer Challenge
    - **Features**: 515 molecular descriptors

    ### Predicted Properties
    - **Tg**: Glass Transition Temperature (°C)
    - **FFV**: Fractional Free Volume
    - **Tc**: thermal conductivity (w.m/k)
    - **Density**: Polymer Density (g/cm³)
    - **Rg**: Radius of Gyration (Å)
    """)

    # Main panel - Results
    if predict_button:
        if not user_input or user_input == "":
            st.warning("⚠️ Please select an example or enter a SMILES string")
        else:
            # Generate features
            with st.spinner("🔄 Calculating molecular features..."):
                features_df = generate_single_features(user_input)

            if features_df is None:
                st.error("❌ Invalid SMILES string. Please check your input and try again.")
            else:
                # Display predictions
                st.success("✅ Prediction successful!")

                st.header("1️⃣ Predicted Properties")

                # Create columns for metrics
                cols = st.columns(5)
                predictions = {}

                for i, target in enumerate(['Tg', 'FFV', 'Tc', 'Density', 'Rg']):
                    model = artifacts[f'{target}_model']
                    kept_columns = artifacts[f'{target}_kept_columns']

                    # Ensure all columns exist
                    for col in kept_columns:
                        if col not in features_df.columns:
                            features_df[col] = 0

                    # Make prediction
                    X_selected = features_df[kept_columns]
                    prediction = model.predict(X_selected)[0]
                    predictions[target] = prediction

                    # Display metric
                    with cols[i]:
                        units = {
                            "Tg": "°C",
                            "Tc": "w/m.k",
                            "Density": "g/cm³",
                            "Rg": "Å"
                        }
                        unit = units.get(target, "")
                        st.metric(
                            label=target,
                            value=f"{prediction:.3f}",
                            help=f"{target} prediction"
                        )
                        st.caption(unit)

                # Display SMILES
                st.header("2️⃣ Input Details")
                col1, col2 = st.columns([1, 2])

                with col1:
                    st.text_input("SMILES String:", value=user_input, disabled=True)

                with col2:
                    # Show molecular weight
                    try:
                        mol = Chem.MolFromSmiles(user_input)
                        if mol:
                            mw = Descriptors.MolWt(mol)
                            st.metric("Molecular Weight", f"{mw:.2f} g/mol")
                    except:
                        pass

                # Display predictions table
                st.header("3️⃣ Detailed Results")

                results_df = pd.DataFrame({
                    'Property': list(predictions.keys()),
                    'Predicted Value': list(predictions.values()),
                    'Unit': ['°C', 'none', 'W/m.K', 'g/cm³', 'Å']
                })

                st.table(results_df)

                # Model information
                st.header("4️⃣ Model Information")

                col1, col2 = st.columns(2)

                with col1:
                    st.info("""
                    ### ✨ Model Details
                    - **Algorithm**: XGBoost Regressor
                    - **Validation Method**: 5-Fold Cross-Validation
                    - **Total Features**: 515
                    - **Training Data**: 7,973 polymer samples
                    """)

                with col2:
                    st.warning("""
                    ### ⚠️ Limitations
                    - Model trained on research-grade polymers
                    - May have lower accuracy for commodities
                    - Best for complex polymer structures
                    - Dataset-specific predictions
                    """)

    else:
        # Show welcome message
        st.header("👋 Welcome!")

        col1, col2 = st.columns(2)

        with col1:
            st.info("""
            ### 🚀 How to Use
            1. Select an example polymer from the sidebar
            2. Or enter your own SMILES string
            3. Click "Predict Properties"
            4. View the predicted values and analysis
            """)

        with col2:
            st.success("""
            ### ✅ Quick Start
            - Use pre-loaded examples (recommended)
            - Enter valid SMILES strings
            - Predictions take a few seconds
            - Results shown with units
            """)

        st.header("📊 Example Polymers")

        examples_info = {
            "Polystyrene": "Common plastic, used in packaging and insulation",
            "Polycarbonate": "Strong, transparent, used in safety equipment",
            "PET": "Polyester, used in bottles and fabrics",
            "PMMA": "Acrylic, transparent, used in windows",
            "Polypropylene": "Flexible plastic, used in containers",
            "Nylon 6": "Engineering plastic, used in textiles"
        }

        col1, col2, col3 = st.columns(3)

        for i, (polymer, description) in enumerate(examples_info.items()):
            with [col1, col2, col3][i % 3]:
                st.write(f"**{polymer}**")
                st.caption(description)


# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()
