import streamlit as st
import requests
import numpy as np
import pandas as pd
import cma
import deepchem as dc
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, Descriptors
from rdkit.Chem.QED import qed
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Descriptors, Crippen, AllChem
import sascorer  # Synthetic Accessibility Score module
from rdkit.Chem.FilterCatalog import FilterCatalogParams, FilterCatalog
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from rdkit.DataStructs import TanimotoSimilarity
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Tuple, Dict, Optional
import json
from pathlib import Path
import pickle
from datetime import datetime
from dataclasses import dataclass
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import pipeline
import torch
import os
os.environ['TF_USE_LEGACY_KERAS'] = 'True' # DeepChem uses legacy tf_keras 2x

@dataclass
class OptimizationResult:
    smiles: str
    qed_score: float
    tanimoto_score: float
    timestamp: datetime
    parameters: dict

class MolMIMClient:
    def __init__(self, host: str = "localhost", port: str = "8000"):
        self.base_url = f"http://{host}:{port}"
        
    def check_health(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/v1/health/ready")
            return response.status_code == 200
        except:
            return False
            
    def get_hidden_state(self, smiles: str) -> np.ndarray:
        response = requests.post(
            f"{self.base_url}/hidden",
            headers={'Content-Type': 'application/json'},
            json={"sequences": [smiles]}
        )
        return np.squeeze(np.array(response.json()["hiddens"]))
        
    def decode_hidden(self, hidden_states: np.ndarray) -> List[str]:
        hiddens_array = np.expand_dims(hidden_states, axis=1)
        response = requests.post(
            f"{self.base_url}/decode",
            headers={'Content-Type': 'application/json'},
            json={
                "hiddens": hiddens_array.tolist(),
                "mask": [[True] for _ in range(hiddens_array.shape[0])]
            }
        )
        return list(dict.fromkeys(response.json()['generated']))

class MoleculeOptimizer:
    def __init__(self, reference_smiles: str):
        self.reference_smiles = reference_smiles
        self.reference_mol = Chem.MolFromSmiles(reference_smiles)
        self.reference_qed = qed(self.reference_mol)
        self.history: List[OptimizationResult] = []
        
    def calculate_tanimoto(self, smiles: str) -> float:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
        fp1 = AllChem.GetMorganFingerprintAsBitVect(self.reference_mol, 2, 2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
        return TanimotoSimilarity(fp1, fp2)
        
    def scoring_function(self, qeds: np.ndarray, similarities: np.ndarray) -> np.ndarray:
        return -1.0 * (
            np.clip(np.array(qeds) / 0.9, a_min=0.0, a_max=1.0) + 
            np.clip(np.array(similarities) / 0.4, a_min=0.0, a_max=1.0)
        )
        
    def load_history(self, filepath: str):
        if Path(filepath).exists():
            with open(filepath, 'rb') as f:
                self.history = pickle.load(f)
                
    def save_history(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump(self.history, f)
            
    def save_result(self, result: OptimizationResult):
        self.history.append(result)

def create_comparison_table(smiles_list: List[str]) -> pd.DataFrame:
    properties = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        properties.append({
            'Molecular Weight': Descriptors.ExactMolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'H-Bond Donors': Descriptors.NumHDonors(mol),
            'H-Bond Acceptors': Descriptors.NumHAcceptors(mol),
            'Rotatable Bonds': Descriptors.NumRotatableBonds(mol),
            'TPSA': Descriptors.TPSA(mol),
            'QED': qed(mol)
        })
    return pd.DataFrame(properties, index=smiles_list)

class ConversationalMoleculeAnalyzer:
    def __init__(self):
        self.validator = MoleculeValidator()
        self.model_name = "meta-llama/Llama-3.2-1B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            torch_dtype=torch.float16
        ).to("cuda")
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=500,
            temperature=0.3,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id
        )
        self.conversation_history = []

    def add_to_history(self, role: str, content: str):
        self.conversation_history.append({"role": role, "content": content})
        
    def get_conversation_context(self) -> str:
        return "\n".join([f"[{msg['role']}]: {msg['content']}" 
                         for msg in self.conversation_history[-5:]])

    def analyze_with_context(self, prompt: str) -> str:
        context = self.get_conversation_context()
        full_prompt = f"""
        [INST] Previous conversation:
        {context}
        
        Current question:
        {prompt} [/INST]
        """
        response = self.pipe(full_prompt, max_new_tokens=300, temperature=0.3)[0]['generated_text']
        # Extract only the assistant's response (remove the prompt)
        assistant_response = response.split("[/INST]")[-1].strip()
        self.add_to_history("assistant", assistant_response)
        return assistant_response

    def analyze_input_molecule(self, smiles: str) -> str:
        validation_results = self.validator.validate_smiles(smiles)
        if not validation_results['valid']:
            return f"Invalid molecule: {validation_results['error']}"
            
        props = validation_results['properties']
        prompt = f"""
        [INST] As a pharmaceutical researcher, analyze this molecule:
        
        Chemical Properties:
        - MW: {props['molecular_weight']:.1f}
        - LogP: {props['logP']:.1f}
        - TPSA: {props['tpsa']:.1f}
        - HB Donors: {props['hbd']}
        - HB Acceptors: {props['hba']}
        - Rotatable Bonds: {props['rotatable_bonds']}
        
        Drug-likeness:
        - Lipinski Violations: {props['lipinski_violations']}
        - Synthetic Accessibility: {props['synthetic_accessibility']:.1f}
        
        ADMET Predictions:
        - Absorption: {props['admet']['absorption']}
        - Distribution: {props['admet']['distribution']}
        - Metabolism: {props['admet']['metabolism']}
        - Excretion: {props['admet']['excretion']}
        - Toxicity Risk: {props['admet']['toxicity']}
        
        Provide detailed analysis focusing on:
        1. Drug-likeness assessment
        2. ADMET properties implications
        3. Potential therapeutic applications
        4. Development challenges
        5. Optimization suggestions [/INST]
        """
        response = self.pipe(prompt, max_new_tokens=800, temperature=0.3)[0]['generated_text']
        # Extract only the assistant's response (remove the prompt)
        assistant_response = response.split("[/INST]")[-1].strip()
        return assistant_response

    def compare_molecules_comprehensive(self, original_smiles: str, new_smiles: str) -> str:
        original_props = self.validator.validate_smiles(original_smiles)['properties']
        new_props = self.validator.validate_smiles(new_smiles)['properties']
        
        prompt = f"""
        [INST] Compare these molecules as a medicinal chemist:
        
        Property Changes:
        - MW: {original_props['molecular_weight']:.1f} → {new_props['molecular_weight']:.1f}
        - LogP: {original_props['logP']:.1f} → {new_props['logP']:.1f}
        - TPSA: {original_props['tpsa']:.1f} → {new_props['tpsa']:.1f}
        - Drug-likeness: {original_props['lipinski_violations']} → {new_props['lipinski_violations']} violations
        
        ADMET Changes:
        - Absorption changes: {new_props['admet']['absorption']}
        - Distribution changes: {new_props['admet']['distribution']}
        - Metabolism impact: {new_props['admet']['metabolism']}
        - Toxicity risk change: {new_props['admet']['toxicity']}
        
        Analyze:
        1. Impact on drug-like properties
        2. ADMET profile changes
        3. Potential therapeutic implications
        4. Development advantages/challenges [/INST]
        """
        response = self.pipe(prompt, max_new_tokens=500, temperature=0.3)[0]['generated_text']
        # Extract only the assistant's response (remove the prompt)
        assistant_response = response.split("[/INST]")[-1].strip()
        return assistant_response
    
    def suggest_questions(self) -> List[str]:
        return [
            "How might changes in polarity affect the drug's absorption?",
            "What structural modifications could improve blood-brain barrier penetration?",
            "How do these changes impact the drug's metabolic stability?",
            "Could these modifications reduce known side effects?",
            "How might solubility differences affect bioavailability?",
            "What's the impact on target protein binding?",
            "Do these changes suggest different administration routes?",
            "How might these modifications affect drug resistance?",
            "Could these changes improve the therapeutic window?",
            "What potential new indications do these modifications suggest?"
        ]

class MoleculeValidator:
    def __init__(self):
        self.lipinski_rules = {
            'MW': 500,
            'LogP': 5,
            'HBD': 5,
            'HBA': 10,
            'TPSA': 140,
            'RotBonds': 10
        }

    def validate_smiles(self, smiles: str) -> dict:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {'valid': False, 'error': 'Invalid SMILES'}
                
            # Basic properties
            mw = Descriptors.ExactMolWt(mol)
            logp = Crippen.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            tpsa = Descriptors.TPSA(mol)
            rotatable_bonds = Descriptors.NumRotatableBonds(mol)
            
            # QED score
            qed_score = qed(mol)
            
            # Lipinski compliance
            lipinski_violations = sum([
                mw > self.lipinski_rules['MW'],
                logp > self.lipinski_rules['LogP'],
                hbd > self.lipinski_rules['HBD'],
                hba > self.lipinski_rules['HBA']
            ])
            
            # Synthetic Accessibility
            sa_score = 0.0  # Placeholder value (or use sascorer.calculateScore(mol) if available)
            
            # Check for reactive groups
            params = FilterCatalogParams()
            params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
            catalog = FilterCatalog(params)
            contains_reactive_groups = catalog.HasMatch(mol)
            
            # ADMET predictions
            admet_predictions = self.predict_admet(mol)
            
            return {
                'valid': True,
                'properties': {
                    'molecular_weight': round(mw, 2),
                    'logP': round(logp, 2),
                    'hbd': hbd,
                    'hba': hba,
                    'tpsa': round(tpsa, 2),
                    'rotatable_bonds': rotatable_bonds,
                    'qed': qed_score,  # Add QED score here
                    'lipinski_violations': lipinski_violations,
                    'synthetic_accessibility': round(sa_score, 2),
                    'reactive_groups': contains_reactive_groups,
                    'admet': admet_predictions
                }
            }
        except Exception as e:
            return {'valid': False, 'error': str(e)}

    def predict_admet(self, mol) -> dict:
        """Predict ADMET properties using RDKit descriptors."""
        # Absorption
        caco2_permeability = Descriptors.TPSA(mol) < 140 and 0 < Crippen.MolLogP(mol) < 5
        p_gp_substrate = Descriptors.NumHDonors(mol) > 5  # Example rule for P-gp substrate

        # Distribution
        bbb_penetration = (Descriptors.TPSA(mol) < 90 and 
                          Descriptors.NumRotatableBonds(mol) < 8 and
                          2 < Crippen.MolLogP(mol) < 4)
        plasma_protein_binding = Crippen.MolLogP(mol) > 3  # Example rule

        # Metabolism
        cyp_substrate = Descriptors.NumAromaticRings(mol) > 1
        cyp3a4_inhibition = Descriptors.NumHDonors(mol) > 2  # Example rule

        # Excretion
        clearance_likely = Descriptors.MolMR(mol) < 130
        half_life = Descriptors.MolWt(mol) < 500  # Example rule

        # Toxicity
        mutagenicity = Descriptors.NumAromaticRings(mol) > 3  # Example rule
        herg_inhibition = Crippen.MolLogP(mol) > 4  # Example rule
        hepatotoxicity = Descriptors.NumHDonors(mol) > 3  # Example rule

        return {
            'absorption': {
                'caco2_permeability': caco2_permeability,
                'p_gp_substrate': p_gp_substrate
            },
            'distribution': {
                'bbb_penetration': bbb_penetration,
                'plasma_protein_binding': plasma_protein_binding
            },
            'metabolism': {
                'cyp_substrate': cyp_substrate,
                'cyp3a4_inhibition': cyp3a4_inhibition
            },
            'excretion': {
                'clearance_likely': clearance_likely,
                'half_life': half_life
            },
            'toxicity': {
                'mutagenicity': mutagenicity,
                'herg_inhibition': herg_inhibition,
                'hepatotoxicity': hepatotoxicity
            }
        }

def create_similarity_heatmap(smiles_list: List[str]) -> go.Figure:
    """Create a heatmap of Tanimoto similarity scores."""
    fingerprints = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 2, 2048) for smiles in smiles_list]
    similarity_matrix = np.zeros((len(smiles_list), len(smiles_list)))
    
    for i in range(len(smiles_list)):
        for j in range(len(smiles_list)):
            similarity_matrix[i, j] = TanimotoSimilarity(fingerprints[i], fingerprints[j])
    
    fig = go.Figure(data=go.Heatmap(
        z=similarity_matrix,
        x=smiles_list,
        y=smiles_list,
        colorscale='Viridis',
        colorbar=dict(title='Tanimoto Similarity')
    ))
    fig.update_layout(
        title="Molecular Similarity Heatmap",
        xaxis_title="SMILES",
        yaxis_title="SMILES",
        width=800,
        height=800
    )
    return fig

def create_property_histograms(properties_df: pd.DataFrame) -> go.Figure:
    """Create histograms for molecular properties."""
    fig = go.Figure()
    for col in properties_df.columns:
        fig.add_trace(go.Histogram(
            x=properties_df[col],
            name=col,
            opacity=0.75
        ))
    fig.update_layout(
        title="Property Distributions",
        xaxis_title="Value",
        yaxis_title="Count",
        barmode='overlay'
    )
    return fig

def create_pca_scatter_plot(smiles_list: List[str]) -> go.Figure:
    """Create a PCA scatter plot for molecular diversity."""
    fingerprints = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 2, 2048) for smiles in smiles_list]
    fps_array = np.array([np.array(fp) for fp in fingerprints])
    
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(fps_array)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=reduced_data[:, 0],
        y=reduced_data[:, 1],
        mode='markers',
        text=smiles_list,
        marker=dict(size=10, color='blue', opacity=0.7)
    ))
    fig.update_layout(
        title="Molecular Diversity (PCA)",
        xaxis_title="PCA Component 1",
        yaxis_title="PCA Component 2",
        width=800,
        height=600
    )
    return fig

def create_parallel_coordinates_plot(properties_df: pd.DataFrame) -> go.Figure:
    """Create a parallel coordinates plot for molecular properties."""
    # Assign a numeric value to each SMILES string for coloring
    properties_df['color'] = range(len(properties_df))
    
    fig = go.Figure(data=go.Parcoords(
        line=dict(
            color=properties_df['color'],
            colorscale='Viridis',  # Use a valid colorscale
            showscale=True,
            colorbar=dict(title='Molecule Index')
        ),
        dimensions=[dict(label=col, values=properties_df[col]) for col in properties_df.columns if col != 'color']
    ))
    fig.update_layout(
        title="Parallel Coordinates Plot",
        width=1000,
        height=600
    )
    return fig

def create_toxicity_radar_chart(properties: dict) -> go.Figure:
    """Create a radar chart for toxicity properties."""
    categories = list(properties.keys())
    values = list(properties.values())
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Toxicity Profile'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=True,
        title="Toxicity Radar Chart",
        width=600,
        height=600
    )
    return fig

def create_violin_plot(properties_df: pd.DataFrame, property_name: str) -> go.Figure:
    """Create a violin plot for a specific property."""
    fig = go.Figure()
    fig.add_trace(go.Violin(
        y=properties_df[property_name],
        name=property_name,
        box_visible=True,
        meanline_visible=True
    ))
    fig.update_layout(
        title=f"{property_name} Distribution",
        yaxis_title=property_name,
        width=800,
        height=600
    )
    return fig

def create_dendrogram(smiles_list: List[str]):
    """Create a dendrogram for hierarchical clustering."""
    fingerprints = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 2, 2048) for smiles in smiles_list]
    fps_array = np.array([np.array(fp) for fp in fingerprints])
    
    # Perform hierarchical clustering
    Z = linkage(fps_array, method='ward')
    
    # Plot the dendrogram
    fig, ax = plt.subplots(figsize=(10, 6))
    dendrogram(Z, labels=smiles_list, ax=ax)
    plt.title("Molecular Clustering Dendrogram")
    plt.xlabel("SMILES")
    plt.ylabel("Distance")
    st.pyplot(fig)

# Function to predict toxicity using DeepChem
def predict_toxicity(smiles_list: List[str], model_name: str = 'tox21') -> pd.DataFrame:
    """Predict toxicity using DeepChem models."""
    # Load the DeepChem model
    if model_name == 'tox21':
        tasks, datasets, transformers = dc.molnet.load_tox21()
        model = dc.models.GraphConvModel(len(tasks), mode='classification')
    elif model_name == 'clintox':
        tasks, datasets, transformers = dc.molnet.load_clintox()
        model = dc.models.GraphConvModel(len(tasks), mode='classification')
    elif model_name == 'herg':
        tasks, datasets, transformers = dc.molnet.load_herg()
        model = dc.models.GraphConvModel(len(tasks), mode='classification')
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    # Load the pre-trained model weights
    model.restore()
    
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

def main():
    st.set_page_config(layout="wide")
    st.title("Advanced Drug Discovery Optimization Platform")
    
    # Sidebar configurations
    st.sidebar.header("Settings")
    host = st.sidebar.text_input("MolMIM Host", "localhost")
    port = st.sidebar.text_input("MolMIM Port", "8000")
    
    # Initialize MolMIM client
    client = MolMIMClient(host, port)
    
    # Check MolMIM server health
    if not client.check_health():
        st.error("⚠️ MolMIM server is not accessible. Please check your connection settings.")
        return
    
    st.success("✅ Connected to MolMIM server")
    
    # Tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["Optimization", "Analysis", "History", "AI Analysis"])
    
    with tab1:
        st.header("Molecule Input")
        seed_smiles = st.text_area(
            "Enter seed SMILES string",
            "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5"  # Imatinib
        )
        
        if not Chem.MolFromSmiles(seed_smiles):
            st.error("Invalid SMILES string")
            return
            
        # Display seed molecule
        mol = Chem.MolFromSmiles(seed_smiles)
        img = Draw.MolToImage(mol)
        st.image(img, caption="Seed Molecule", use_container_width=True)
        
        # Optimization parameters
        st.header("Optimization Parameters")
        col1, col2, col3 = st.columns(3)
        with col1:
            population_size = st.number_input("Population Size", 10, 100, 50)
        with col2:
            n_iterations = st.number_input("Number of Iterations", 10, 200, 50)
        with col3:
            sigma = st.number_input("CMA-ES Sigma", 0.1, 5.0, 1.0)
            
        # Additional features
        st.sidebar.header("Advanced Settings")
        custom_constraints = st.sidebar.expander("Molecular Constraints")
        with custom_constraints:
            max_molecular_weight = st.number_input("Max Molecular Weight", 0, 1000, 500)
            min_rotatable_bonds = st.number_input("Min Rotatable Bonds", 0, 20, 5)
            max_rotatable_bonds = st.number_input("Max Rotatable Bonds", 0, 50, 20)
            
        if st.button("Start Optimization"):
            optimizer = MoleculeOptimizer(seed_smiles)
            
            # Initialize CMA-ES with seed molecule hidden state
            hidden_state = client.get_hidden_state(seed_smiles)
            es = cma.CMAEvolutionStrategy(hidden_state, sigma, {'popsize': population_size})
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Results tracking
            results = []
            
            # Create plots
            col1, col2 = st.columns(2)
            with col1:
                scatter_plot = st.empty()
            with col2:
                trend_plot = st.empty()
                
            # Optimization loop
            for iteration in range(n_iterations):
                # Generate and evaluate candidates
                trial_encodings = es.ask(population_size)
                molecules = client.decode_hidden(np.array(trial_encodings))
                valid_molecules = [m for m in molecules if Chem.MolFromSmiles(m)]
                
                # Calculate scores
                qed_scores = [qed(Chem.MolFromSmiles(m)) for m in valid_molecules]
                tanimoto_scores = [optimizer.calculate_tanimoto(m) for m in valid_molecules]
                scores = optimizer.scoring_function(qed_scores, tanimoto_scores)
                
                # Update CMA-ES
                if len(valid_molecules) > population_size // 2:
                    es.tell(trial_encodings[:len(valid_molecules)], scores)
                    
                    # Store results
                    results.append({
                        'iteration': iteration,
                        'qed_median': np.median(qed_scores),
                        'tanimoto_median': np.median(tanimoto_scores),
                        'best_score': -min(scores)
                    })
                    
                    # Update plots
                    df = pd.DataFrame(results)
                    
                    # Scatter plot
                    fig1 = px.scatter(
                        x=qed_scores, 
                        y=tanimoto_scores,
                        title="Population Distribution",
                        labels={'x': 'QED Score', 'y': 'Tanimoto Similarity'}
                    )
                    scatter_plot.plotly_chart(fig1)
                    
                    # Trend plot
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=df['iteration'], y=df['qed_median'], name='Median QED'))
                    fig2.add_trace(go.Scatter(x=df['iteration'], y=df['tanimoto_median'], name='Median Tanimoto'))
                    fig2.update_layout(title="Optimization Progress", xaxis_title="Iteration", yaxis_title="Score")
                    trend_plot.plotly_chart(fig2)
                
                # Update progress
                progress = (iteration + 1) / n_iterations
                progress_bar.progress(progress)
                status_text.text(f"Iteration {iteration + 1}/{n_iterations}")
                
            # Display top 5 results
            st.header("Top 5 Optimized Molecules")
            
            # Get indices of top 5 molecules
            top_indices = np.argsort(scores)[:5]
            
            # Create columns for each molecule
            cols = st.columns(5)
            
            for idx, col in zip(top_indices, cols):
                molecule = valid_molecules[idx]
                mol = Chem.MolFromSmiles(molecule)
                
                with col:
                    st.image(Draw.MolToImage(mol), caption=f"Rank {idx+1}")
                    st.text_area("SMILES", molecule, height=70)
                    st.metric("QED Score", f"{qed_scores[idx]:.3f}")
                    st.metric("Tanimoto", f"{tanimoto_scores[idx]:.3f}")
                    
            # Save top molecules to session state
            st.session_state['top_molecules'] = [{
                'smiles': valid_molecules[idx],
                'qed': qed_scores[idx],
                'tanimoto': tanimoto_scores[idx]
            } for idx in top_indices]
            
    with tab2:
        st.header("Analysis Dashboard")
        
        if 'top_molecules' in st.session_state:
            smiles_list = [m['smiles'] for m in st.session_state['top_molecules']]
            
            # Group visualizations into tabs
            tab_analysis, tab_deepchem = st.tabs(["Basic Analysis", "DeepChem Predictions"])
            
            # Basic Analysis Tab
            with tab_analysis:
                st.subheader("Property Distributions")
                properties_df = create_comparison_table(smiles_list)
                
                # Histograms for property distributions
                st.plotly_chart(create_property_histograms(properties_df))
                
                # Box plot for property comparison
                st.subheader("Property Comparison")
                fig = px.box(properties_df)
                st.plotly_chart(fig)
                
                # Molecular Similarity Heatmap
                st.subheader("Molecular Similarity Heatmap")
                st.plotly_chart(create_similarity_heatmap(smiles_list))
                
                # PCA Scatter Plot for Molecular Diversity
                st.subheader("Molecular Diversity (PCA)")
                st.plotly_chart(create_pca_scatter_plot(smiles_list))
                
                # Toxicity Radar Chart
                st.subheader("Toxicity Radar Chart")
                toxicity_properties = {
                    'Mutagenicity': 0.8,
                    'Carcinogenicity': 0.6,
                    'Hepatotoxicity': 0.7,
                    'hERG Inhibition': 0.5
                }
                st.plotly_chart(create_toxicity_radar_chart(toxicity_properties))
                
                # Drug-Likeness Violin Plot
                st.subheader("Drug-Likeness (QED) Distribution")
                st.plotly_chart(create_violin_plot(properties_df, 'QED'))
                
                # Clustering Dendrogram
                st.subheader("Molecular Clustering Dendrogram")
                create_dendrogram(smiles_list)
            
            # DeepChem Predictions Tab
            with tab_deepchem:
                st.subheader("DeepChem Predictions")
                
                # Toxicity Prediction (Tox21)
                with st.expander("Toxicity Prediction (Tox21)", expanded=False):
                    toxicity_results = predict_toxicity(smiles_list, model_name='tox21')
                    st.dataframe(toxicity_results)
                    
                    # Visualization of Tox21 Predictions
                    st.subheader("Tox21 Predictions Visualization")
                    for smiles in smiles_list:
                        st.write(f"**Molecule:** {smiles}")
                        fig = px.bar(toxicity_results.loc[smiles], title="Tox21 Predictions")
                        st.plotly_chart(fig)
                
                # Solubility Prediction (Delaney)
                with st.expander("Solubility Prediction (Delaney)", expanded=False):
                    solubility_results = predict_solubility(smiles_list)
                    st.dataframe(solubility_results)
                
                # Drug-Likeness Prediction (BBBP)
                with st.expander("Drug-Likeness Prediction (BBBP)", expanded=False):
                    drug_likeness_results = predict_drug_likeness(smiles_list)
                    st.dataframe(drug_likeness_results)
                
                # ADMET Property Prediction
                with st.expander("ADMET Property Prediction", expanded=False):
                    admet_results = predict_admet(smiles_list)
                    st.dataframe(admet_results)
                
                # Binding Affinity Prediction (PDBbind)
                with st.expander("Binding Affinity Prediction (PDBbind)", expanded=False):
                    binding_affinity_results = predict_binding_affinity(smiles_list)
                    st.dataframe(binding_affinity_results)
        else:
            st.info("Run optimization first to see analysis")
            
    with tab3:
        st.header("Optimization History")
        
        if 'top_molecules' in st.session_state:
            # Filter and sort options
            col1, col2 = st.columns(2)
            with col1:
                sort_by = st.selectbox("Sort By", ["QED Score", "Tanimoto Similarity"])
            with col2:
                ascending = st.checkbox("Ascending")
                
            # Create history DataFrame
            history_df = pd.DataFrame(st.session_state['top_molecules'])
            
            # Sort based on user selection
            if sort_by == "QED Score":
                history_df = history_df.sort_values('qed', ascending=ascending)
            else:
                history_df = history_df.sort_values('tanimoto', ascending=ascending)
                
            st.dataframe(history_df)
        else:
            st.info("Run optimization first to see history")

    with tab4:
        st.header("Interactive Drug Analysis")
        
        if "analyzer" not in st.session_state:
            st.session_state.analyzer = ConversationalMoleculeAnalyzer()
        
        # Display generated molecules and their properties
        if 'top_molecules' in st.session_state:
            st.subheader("Generated Molecules")
            cols = st.columns(5)  # Display up to 5 molecules in a row
            
            for idx, mol_data in enumerate(st.session_state['top_molecules']):
                with cols[idx % 5]:
                    mol = Chem.MolFromSmiles(mol_data['smiles'])
                    st.image(Draw.MolToImage(mol), caption=f"Molecule {idx + 1}")
                    
                    # Add a button to trigger analysis
                    if st.button(f"Analyze Molecule {idx + 1}"):
                        st.session_state['selected_molecule'] = mol_data['smiles']
            
            # Display properties and analysis for the selected molecule
            if 'selected_molecule' in st.session_state:
                selected_smiles = st.session_state['selected_molecule']
                validator = MoleculeValidator()
                validation_results = validator.validate_smiles(selected_smiles)
                
                if validation_results['valid']:
                    props = validation_results['properties']
                    
                    # Display properties in an expandable section
                    with st.expander("View Properties", expanded=False):
                        st.write(f"**SMILES:** {selected_smiles}")
                        st.write(f"**QED:** {props['qed']:.3f}")
                        st.write(f"**Tanimoto Similarity:** {mol_data['tanimoto']:.3f}")
                        st.write(f"**Molecular Weight:** {props['molecular_weight']:.2f}")
                        st.write(f"**LogP:** {props['logP']:.2f}")
                        st.write(f"**H-Bond Donors:** {props['hbd']}")
                        st.write(f"**H-Bond Acceptors:** {props['hba']}")
                        st.write(f"**Rotatable Bonds:** {props['rotatable_bonds']}")
                        st.write(f"**TPSA:** {props['tpsa']:.2f}")
                        st.write(f"**Lipinski Violations:** {props['lipinski_violations']}")
                        st.write(f"**Synthetic Accessibility:** {props['synthetic_accessibility']:.2f}")
                        
                        # Display ADMET predictions
                        st.write("**ADMET Predictions:**")
                        st.write(f"- Absorption (Caco-2 Permeability): {props['admet']['absorption']['caco2_permeability']}")
                        st.write(f"- Distribution (BBB Penetration): {props['admet']['distribution']['bbb_penetration']}")
                        st.write(f"- Metabolism (CYP Substrate): {props['admet']['metabolism']['cyp_substrate']}")
                        st.write(f"- Excretion (Clearance Likely): {props['admet']['excretion']['clearance_likely']}")
                        st.write(f"- Toxicity (hERG Inhibition): {props['admet']['toxicity']['herg_inhibition']}")
                    
                    # Generate and display LLM analysis on the full page
                    st.subheader("Molecule Analysis")
                    with st.spinner("Generating analysis..."):
                        analysis = st.session_state.analyzer.analyze_input_molecule(selected_smiles)
                        st.write(analysis)
                else:
                    st.error(f"Invalid molecule: {validation_results['error']}")
        else:
            st.info("Run optimization first to see generated molecules")

if __name__ == "__main__":
    main()