from typing import List, Dict
import streamlit as st
import deepchem as dc
from rdkit import Chem
import pandas as pd
import numpy as np
import plotly.express as px
import os
os.environ["TF_USE_LEGACY_KERAS"] = "True"  # Force TensorFlow to use legacy Keras
import tensorflow as tf  # Import TensorFlow

# Ensure DeepChem uses tf.keras from tf_keras
dc.models.tf_keras = tf.keras

def predict_toxicity(smiles_list: List[str], model_name: str = 'tox21') -> pd.DataFrame:
    """Predict toxicity using DeepChem models."""
    # Load the DeepChem model
    if model_name == 'tox21':
        tasks, datasets, transformers = dc.molnet.load_tox21()
        model = dc.models.GraphConvModel(len(tasks), mode='classification')
        checkpoint_path = "tox21_graphconvmodel"  # Path to pre-trained weights
    elif model_name == 'clintox':
        tasks, datasets, transformers = dc.molnet.load_clintox()
        model = dc.models.GraphConvModel(len(tasks), mode='classification')
        checkpoint_path = "clintox_graphconvmodel"  # Path to pre-trained weights
    elif model_name == 'herg':
        tasks, datasets, transformers = dc.molnet.load_herg()
        model = dc.models.GraphConvModel(len(tasks), mode='classification')
        checkpoint_path = "herg_graphconvmodel"  # Path to pre-trained weights
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    # Load the pre-trained model weights
    model.restore(checkpoint_path)  # Specify the checkpoint path
    
    # Convert SMILES to DeepChem molecules
    featurizer = dc.feat.ConvMolFeaturizer()
    molecules = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    features = featurizer.featurize(molecules)
    
    # Predict toxicity
    predictions = model.predict(features)
    
    # Create a DataFrame with results
    results = pd.DataFrame(predictions, columns=tasks, index=smiles_list)
    return results

# Function to predict solubility using DeepChem
def predict_solubility(smiles_list: List[str]) -> pd.DataFrame:
    """Predict aqueous solubility using DeepChem."""
    # Load the Delaney dataset and model
    tasks, datasets, transformers = dc.molnet.load_delaney()
    model = dc.models.GraphConvModel(len(tasks), mode='regression')
    model.restore()
    
    # Convert SMILES to DeepChem molecules
    featurizer = dc.feat.ConvMolFeaturizer()
    molecules = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    features = featurizer.featurize(molecules)
    
    # Predict solubility
    predictions = model.predict(features)
    
    # Create a DataFrame with results
    results = pd.DataFrame(predictions, columns=['Solubility'], index=smiles_list)
    return results

# Function to predict drug-likeness using DeepChem
def predict_drug_likeness(smiles_list: List[str]) -> pd.DataFrame:
    """Predict drug-likeness using DeepChem."""
    # Load the BBBP dataset and model
    tasks, datasets, transformers = dc.molnet.load_bbbp()
    model = dc.models.GraphConvModel(len(tasks), mode='classification')
    model.restore()
    
    # Convert SMILES to DeepChem molecules
    featurizer = dc.feat.ConvMolFeaturizer()
    molecules = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    features = featurizer.featurize(molecules)
    
    # Predict drug-likeness
    predictions = model.predict(features)
    
    # Create a DataFrame with results
    results = pd.DataFrame(predictions, columns=['Drug-Likeness'], index=smiles_list)
    return results

# Function to predict ADMET properties using DeepChem
def predict_admet(smiles_list: List[str]) -> pd.DataFrame:
    """Predict ADMET properties using DeepChem."""
    # Load the ADMET dataset and model
    tasks, datasets, transformers = dc.molnet.load_admet()
    model = dc.models.GraphConvModel(len(tasks), mode='classification')
    model.restore()
    
    # Convert SMILES to DeepChem molecules
    featurizer = dc.feat.ConvMolFeaturizer()
    molecules = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    features = featurizer.featurize(molecules)
    
    # Predict ADMET properties
    predictions = model.predict(features)
    
    # Create a DataFrame with results
    results = pd.DataFrame(predictions, columns=tasks, index=smiles_list)
    return results

# Function to predict binding affinity using DeepChem
def predict_binding_affinity(smiles_list: List[str]) -> pd.DataFrame:
    """Predict binding affinity using DeepChem."""
    # Load the PDBbind dataset and model
    tasks, datasets, transformers = dc.molnet.load_pdbbind()
    model = dc.models.GraphConvModel(len(tasks), mode='regression')
    model.restore()
    
    # Convert SMILES to DeepChem molecules
    featurizer = dc.feat.ConvMolFeaturizer()
    molecules = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    features = featurizer.featurize(molecules)
    
    # Predict binding affinity
    predictions = model.predict(features)
    
    # Create a DataFrame with results
    results = pd.DataFrame(predictions, columns=['Binding Affinity'], index=smiles_list)
    return results

# Streamlit App
def main():
    st.set_page_config(layout="wide")
    st.title("DeepChem Integration for Drug Discovery")
    
    # Input SMILES strings
    st.sidebar.header("Input Molecules")
    smiles_list = st.sidebar.text_area(
        "Enter SMILES strings (one per line):",
        "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5\nCCO\nC1=CC=CC=C1"
    ).split("\n")
    
    # Remove empty lines
    smiles_list = [smiles.strip() for smiles in smiles_list if smiles.strip()]
    
    # Display input molecules
    st.subheader("Input Molecules")
    st.write(smiles_list)
    
    # DeepChem Predictions
    st.header("DeepChem Predictions")
    
    # Toxicity Prediction (Tox21)
    st.subheader("Toxicity Prediction (Tox21)")
    toxicity_results = predict_toxicity(smiles_list, model_name='tox21')
    st.dataframe(toxicity_results)
    
    # Solubility Prediction (Delaney)
    st.subheader("Solubility Prediction (Delaney)")
    solubility_results = predict_solubility(smiles_list)
    st.dataframe(solubility_results)
    
    # Drug-Likeness Prediction (BBBP)
    st.subheader("Drug-Likeness Prediction (BBBP)")
    drug_likeness_results = predict_drug_likeness(smiles_list)
    st.dataframe(drug_likeness_results)
    
    # ADMET Property Prediction
    st.subheader("ADMET Property Prediction")
    admet_results = predict_admet(smiles_list)
    st.dataframe(admet_results)
    
    # Binding Affinity Prediction (PDBbind)
    st.subheader("Binding Affinity Prediction (PDBbind)")
    binding_affinity_results = predict_binding_affinity(smiles_list)
    st.dataframe(binding_affinity_results)
    
    # Visualization of Tox21 Predictions
    st.subheader("Tox21 Predictions Visualization")
    for smiles in smiles_list:
        st.write(f"**Molecule:** {smiles}")
        fig = px.bar(toxicity_results.loc[smiles], title="Tox21 Predictions")
        st.plotly_chart(fig)

if __name__ == "__main__":
    main()