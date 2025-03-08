
# """Advanced Drug Discovery Optimization Platform with integrated scoring and analysis."""

import streamlit as st
import numpy as np
import pandas as pd
import cma
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, Descriptors
from rdkit.Chem.QED import qed
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Descriptors import MolLogP as rdkit_logp  # noqa
from rdkit.DataStructs import TanimotoSimilarity
import logging
from multiprocessing import Pool
from typing import List, Callable, Dict
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from dataclasses import dataclass
from datetime import datetime
import pickle
from pathlib import Path
from openai import OpenAI
import os
import sys
import requests

# DeepSeek API setup
DEEPSEEK_API_KEY = "sk-cb3e2fbf705d4c1899a7cf53c49fbaa6" 
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

# RDKit Contrib for Synthetic Accessibility
try:
    from rdkit.Chem import RDConfig
    import sys
    sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
    from sascorer import calculateScore
    SA_SCORE_AVAILABLE = True
except ImportError:
    SA_SCORE_AVAILABLE = False
    logging.warning("SA_Score unavailable. Install RDKit Contrib for synthetic accessibility.")

logging.basicConfig(level=logging.WARNING)

@dataclass
class OptimizationResult:
    smiles: str
    qed_score: float
    tanimoto_score: float
    sa_score: float
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
            json={"hiddens": hiddens_array.tolist(), "mask": [[True] for _ in range(hiddens_array.shape[0])]}
        )
        return list(dict.fromkeys(response.json()['generated']))

def _score_mol(args):
    smi, scorer, default_val = args
    mol = Chem.MolFromSmiles(smi)
    return scorer(mol) if mol else default_val

def _iterate_and_score_smiles(
    smis: List[str], scorer: Callable[[Chem.Mol], float], default_val: float = 0.0, parallel: bool = False
) -> np.ndarray:
    """Batch-process SMILES with a scoring function, optionally in parallel."""
    if parallel and len(smis) > 100:
        with Pool() as pool:
            results = pool.map(_score_mol, [(smi, scorer, default_val) for smi in smis])
        return np.array(results)
    results = np.zeros((len(smis),)) + default_val
    for i, smi in enumerate(smis):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            logging.warning(f"Failed to parse SMILES at index {i}: {smi}")
            continue
        results[i] = scorer(mol)
    return results

class MoleculeOptimizer:
    def __init__(self, reference_smiles: str, radius: int = 2, fp_size: int = 2048):
        self.reference_smiles = reference_smiles
        self.reference_mol = Chem.MolFromSmiles(reference_smiles)
        self.reference_qed = qed(self.reference_mol)
        self.history: List[OptimizationResult] = []
        self.radius = radius
        self.fp_size = fp_size
        self.mfpgen = AllChem.GetMorganGenerator(radius=radius, fpSize=fp_size)
        self.reference_fp = self.mfpgen.GetFingerprint(self.reference_mol)

    def tanimoto_similarity(self, smis: List[str], parallel: bool = False) -> np.ndarray:
        """Compute Tanimoto similarity to the reference molecule."""
        def scorer(mol): return TanimotoSimilarity(self.mfpgen.GetFingerprint(mol), self.reference_fp)
        return _iterate_and_score_smiles(smis, scorer, default_val=0.0, parallel=parallel)

    def qed(self, smis: List[str], parallel: bool = False) -> np.ndarray:
        """Compute QED scores."""
        return _iterate_and_score_smiles(smis, qed, default_val=0.0, parallel=parallel)

    def synthetic_accessibility(self, smis: List[str], parallel: bool = False) -> np.ndarray:
        """Compute synthetic accessibility scores."""
        if not SA_SCORE_AVAILABLE:
            logging.error("SA_Score not available. Returning zeros.")
            return np.zeros((len(smis),))
        return _iterate_and_score_smiles(smis, calculateScore, default_val=0.0, parallel=parallel)

    def scoring_function(self, smis: List[str], parallel: bool = False) -> np.ndarray:
        """Combined score based on QED, Tanimoto similarity, and SA."""
        qeds = self.qed(smis, parallel)
        similarities = self.tanimoto_similarity(smis, parallel)
        sa_scores = self.synthetic_accessibility(smis, parallel)
        return -1.0 * (np.clip(qeds / 0.9, 0, 1) + np.clip(similarities / 0.4, 0, 1) - 0.1 * sa_scores)

    def save_history(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump(self.history, f)

    def save_result(self, smiles: str, qed_score: float, tanimoto_score: float, sa_score: float, params: dict):
        self.history.append(OptimizationResult(smiles, qed_score, tanimoto_score, sa_score, datetime.now(), params))

class MoleculeValidator:
    def __init__(self):
        self.lipinski_rules = {'MW': 500, 'LogP': 5, 'HBD': 5, 'HBA': 10}

    def validate_smiles(self, smiles: List[str], parallel: bool = False) -> Dict[str, dict]:
        """Validate and compute properties for a list of SMILES."""
        def get_properties(smi):
            mol = Chem.MolFromSmiles(smi)
            if not mol:
                return {'valid': False, 'error': 'Invalid SMILES'}
            props = {
                'molecular_weight': Descriptors.ExactMolWt(mol),
                'logP': Descriptors.MolLogP(mol),
                'hbd': Descriptors.NumHDonors(mol),
                'hba': Descriptors.NumHAcceptors(mol),
                'tpsa': Descriptors.TPSA(mol),
                'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                'qed': qed(mol),
                'lipinski_violations': sum([
                    Descriptors.ExactMolWt(mol) > self.lipinski_rules['MW'],
                    Descriptors.MolLogP(mol) > self.lipinski_rules['LogP'],
                    Descriptors.NumHDonors(mol) > self.lipinski_rules['HBD'],
                    Descriptors.NumHAcceptors(mol) > self.lipinski_rules['HBA']
                ]),
                'synthetic_accessibility': calculateScore(mol) if SA_SCORE_AVAILABLE else 0.0
            }
            return {'valid': True, 'properties': props}
        return {smi: get_properties(smi) for smi in smiles}

@st.cache_data
def analyze_with_deepseek(messages: List[Dict[str, str]]) -> str:
    """Call DeepSeek API using OpenAI client."""
    try:
        response = client.chat.completions.create(
            model="deepseek-reasoner",  # or "deepseek-reasoner" if preferred
            messages=[{"role": "system", "content": "You are a pharmaceutical research assistant."}] + messages,
            max_tokens=1000,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error contacting DeepSeek API: {str(e)}"

class ConversationalMoleculeAnalyzer:
    def __init__(self):
        self.validator = MoleculeValidator()
        self.conversation_history = []

    def add_to_history(self, role: str, content: str):
        self.conversation_history.append({"role": role, "content": content})

    def analyze_input_molecule(self, smiles: str) -> str:
        validation = self.validator.validate_smiles([smiles])[smiles]
        if not validation['valid']:
            return f"Invalid molecule: {validation['error']}"
        props = validation['properties']
        
        prompt = f"""
        As an expert pharmaceutical researcher, provide a detailed analysis of this molecule (SMILES: {smiles}):

        **Chemical Properties:**
        - Molecular Weight: {props['molecular_weight']:.1f}
        - LogP: {props['logP']:.1f}
        - TPSA: {props['tpsa']:.1f}
        - H-Bond Donors: {props['hbd']}
        - H-Bond Acceptors: {props['hba']}
        - Rotatable Bonds: {props['rotatable_bonds']}
        - QED: {props['qed']:.3f}
        - Synthetic Accessibility (SA): {props['synthetic_accessibility']:.1f}
        - Lipinski Violations: {props['lipinski_violations']}

        **Instructions:**
        1. Assess drug-likeness (e.g., compliance with Lipinski’s Rule of Five, QED interpretation).
        2. Discuss synthetic feasibility based on SA score (1-10 scale, lower is easier).
        3. Evaluate potential therapeutic applications based on properties (e.g., oral bioavailability, CNS penetration).
        4. Highlight development challenges (e.g., solubility, stability, toxicity risks).
        5. Suggest specific structural modifications to improve the molecule (e.g., functional group changes).
        6. Provide 3-5 follow-up questions I could ask to deepen my understanding or refine the molecule.

        Use clear, concise language suitable for a researcher, and ensure recommendations are actionable.
        """
        messages = [{"role": "user", "content": prompt}]
        response = analyze_with_deepseek(messages)
        self.add_to_history("user", prompt)
        self.add_to_history("assistant", response)
        return response

    def answer_follow_up(self, question: str) -> str:
        messages = self.conversation_history + [{"role": "user", "content": question}]
        response = analyze_with_deepseek(messages)
        self.add_to_history("user", question)
        self.add_to_history("assistant", response)
        return response

# Visualization Functions
@st.cache_data
def create_comparison_table(smiles_list: List[str]) -> pd.DataFrame:
    validator = MoleculeValidator()
    results = validator.validate_smiles(smiles_list)
    return pd.DataFrame({smi: res['properties'] for smi, res in results.items() if res['valid']}).T

def create_similarity_heatmap(smiles_list: List[str]) -> go.Figure:
    fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, 2048) for s in smiles_list]
    sim_matrix = np.array([[TanimotoSimilarity(f1, f2) for f2 in fps] for f1 in fps])
    fig = go.Figure(data=go.Heatmap(z=sim_matrix, x=smiles_list, y=smiles_list, colorscale='Viridis',
                                    text=[[f"{val:.2f}" for val in row] for row in sim_matrix], hoverinfo="text"))
    fig.update_layout(title="Tanimoto Similarity Heatmap", width=800, height=800)
    return fig

def create_property_box_plot(properties_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for col in ['qed', 'logP', 'synthetic_accessibility']:
        fig.add_trace(go.Box(y=properties_df[col], name=col, boxpoints='outliers'))
    fig.update_layout(title="Property Distribution with Outliers", yaxis_title="Value", width=800, height=600)
    return fig

def create_qed_vs_sa_scatter(properties_df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(properties_df, x='qed', y='synthetic_accessibility', text=properties_df.index,
                     labels={'qed': 'QED Score', 'synthetic_accessibility': 'SA Score'},
                     title="QED vs. Synthetic Accessibility")
    fig.update_traces(textposition='top center')
    fig.update_layout(width=800, height=600)
    return fig

def create_3d_pca_scatter(smiles_list: List[str]) -> go.Figure:
    fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, 2048) for s in smiles_list]
    fps_array = np.array([np.array(fp) for fp in fps])
    pca = PCA(n_components=3)
    reduced = pca.fit_transform(fps_array)
    fig = go.Figure(data=[go.Scatter3d(x=reduced[:, 0], y=reduced[:, 1], z=reduced[:, 2], mode='markers',
                                       marker=dict(size=8, color='blue', opacity=0.7), text=smiles_list)])
    fig.update_layout(title="3D PCA of Molecular Diversity", scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
                      width=800, height=600)
    return fig

def main():
    st.set_page_config(layout="wide")
    st.title("Advanced Drug Discovery Optimization Platform")

    # Sidebar
    st.sidebar.header("Settings")
    host = st.sidebar.text_input("MolMIM Host", "localhost")
    port = st.sidebar.text_input("MolMIM Port", "8000")
    radius = st.sidebar.slider("Fingerprint Radius", 1, 4, 2)
    fp_size = st.sidebar.selectbox("Fingerprint Size", [1024, 2048, 4096], index=1)
    parallel = st.sidebar.checkbox("Parallel Processing", False)

    # client = MolMIMClient(host, port)
    # if not client.check_health():
    #     st.error("⚠️ MolMIM server not accessible. Check connection or start with: `sudo docker run ...`")
    #     return
    # st.success("✅ Connected to MolMIM server")
    
     # Initialize MolMIM client
    client = MolMIMClient(host, port)
    
    # Check MolMIM server health
    if not client.check_health():
        st.error("⚠️ MolMIM server is not accessible. Please check your connection settings.")
        return
    
    st.success("✅ Connected to MolMIM server") 
    tab1, tab2, tab3 = st.tabs(["Optimization", "Analysis", "AI Analysis"])

    with tab1:
        st.header("Molecule Optimization")
        seed_smiles = st.text_area("Seed SMILES", "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5")
        if not Chem.MolFromSmiles(seed_smiles):
            st.error("Invalid SMILES")
            return
        st.image(Draw.MolToImage(Chem.MolFromSmiles(seed_smiles)), caption="Seed Molecule")

        col1, col2, col3 = st.columns(3)
        with col1:
            pop_size = st.number_input("Population Size", 10, 100, 50)
        with col2:
            n_iter = st.number_input("Iterations", 10, 200, 50)
        with col3:
            sigma = st.number_input("CMA-ES Sigma", 0.1, 5.0, 1.0)

        if st.button("Start Optimization"):
            optimizer = MoleculeOptimizer(seed_smiles, radius, fp_size)
            hidden_state = client.get_hidden_state(seed_smiles)
            es = cma.CMAEvolutionStrategy(hidden_state, sigma, {'popsize': pop_size})
            progress_bar = st.progress(0)
            results = []

            for i in range(n_iter):
                trials = es.ask(pop_size)
                molecules = client.decode_hidden(np.array(trials))
                valid_mols = [m for m in molecules if Chem.MolFromSmiles(m)]
                if not valid_mols:
                    continue
                qeds = optimizer.qed(valid_mols, parallel)
                tanimotos = optimizer.tanimoto_similarity(valid_mols, parallel)
                sa_scores = optimizer.synthetic_accessibility(valid_mols, parallel)
                scores = optimizer.scoring_function(valid_mols, parallel)
                es.tell(trials[:len(valid_mols)], scores)
                results.append({'iteration': i, 'qed_median': np.median(qeds), 'tanimoto_median': np.median(tanimotos)})
                for j, smi in enumerate(valid_mols):
                    optimizer.save_result(smi, qeds[j], tanimotos[j], sa_scores[j], {'pop_size': pop_size, 'iter': i})
                progress_bar.progress((i + 1) / n_iter)

            top_indices = np.argsort(scores)[:5]
            top_molecules = [{
                'smiles': valid_mols[idx], 'qed': qeds[idx], 'tanimoto': tanimotos[idx], 'sa_score': sa_scores[idx]
            } for idx in top_indices]
            st.session_state['top_molecules'] = top_molecules
            st.subheader("Top 5 Molecules")
            cols = st.columns(5)
            for idx, col in enumerate(cols):
                mol_data = top_molecules[idx]
                with col:
                    st.image(Draw.MolToImage(Chem.MolFromSmiles(mol_data['smiles'])), caption=f"Rank {idx+1}")
                    st.text_area(f"SMILES {idx+1}", mol_data['smiles'], height=70)
                    st.metric("QED", f"{mol_data['qed']:.3f}")
                    st.metric("Tanimoto", f"{mol_data['tanimoto']:.3f}")
                    st.metric("SA Score", f"{mol_data['sa_score']:.1f}")

    with tab2:
        st.header("Analysis Dashboard")
        if 'top_molecules' not in st.session_state:
            st.info("Run optimization first.")
        else:
            smiles_list = [m['smiles'] for m in st.session_state['top_molecules']]
            properties_df = create_comparison_table(smiles_list)
            st.subheader("Property Table")
            st.dataframe(properties_df)
            st.plotly_chart(create_similarity_heatmap(smiles_list))
            st.plotly_chart(create_property_box_plot(properties_df))
            st.plotly_chart(create_qed_vs_sa_scatter(properties_df))
            st.plotly_chart(create_3d_pca_scatter(smiles_list))

    with tab3:
        st.header("AI-Driven Molecule Analysis")
        if 'top_molecules' not in st.session_state:
            st.info("Run optimization first.")
        else:
            if "analyzer" not in st.session_state:
                st.session_state.analyzer = ConversationalMoleculeAnalyzer()
            smiles_list = [m['smiles'] for m in st.session_state['top_molecules']]
            cols = st.columns(5)
            for idx, col in enumerate(cols):
                with col:
                    st.image(Draw.MolToImage(Chem.MolFromSmiles(smiles_list[idx])), caption=f"Molecule {idx+1}")
                    if st.button(f"Analyze {idx+1}"):
                        st.session_state['selected_molecule'] = smiles_list[idx]
            if 'selected_molecule' in st.session_state:
                smiles = st.session_state['selected_molecule']
                with st.spinner("Generating analysis..."):
                    analysis = st.session_state.analyzer.analyze_input_molecule(smiles)
                    st.write(analysis)
                follow_up = st.text_input("Ask a follow-up question:")
                if follow_up:
                    with st.spinner("Processing..."):
                        response = st.session_state.analyzer.answer_follow_up(follow_up)
                        st.write(response)

if __name__ == "__main__":
    main()



# openai.api_key = "Sk-4c0c80a4056c412491e08a64523b4945"
# openai.api_base = "https://api.deepseek.com" #or your correct base url.

# Please install OpenAI SDK first: `pip3 install openai`

# Please install OpenAI SDK first: `pip3 install openai`

# from openai import OpenAI

# client = OpenAI(api_key="sk-cb3e2fbf705d4c1899a7cf53c49fbaa6", base_url="https://api.deepseek.com")

# response = client.chat.completions.create(
#     model="deepseek-reasoner",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant"},
#         {"role": "user", "content": "What is MolMIM"},
#     ],
#     stream=False
# )

# print(response.choices[0].message.content)
