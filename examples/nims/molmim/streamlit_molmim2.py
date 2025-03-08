import streamlit as st
import requests
import numpy as np
import pandas as pd
import cma
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, Descriptors
from rdkit.Chem.QED import qed
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Descriptors, Crippen, AllChem
import sascorer  # Synthetic Accessibility Score module
from rdkit.Chem.FilterCatalog import FilterCatalogParams, FilterCatalog
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
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
        self.add_to_history("assistant", response)
        return response

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
        return self.pipe(prompt, max_new_tokens=800, temperature=0.3)[0]['generated_text']

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
        return self.pipe(prompt, max_new_tokens=500, temperature=0.3)[0]['generated_text']
    
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
            
            # Lipinski compliance
            lipinski_violations = sum([
                mw > self.lipinski_rules['MW'],
                logp > self.lipinski_rules['LogP'],
                hbd > self.lipinski_rules['HBD'],
                hba > self.lipinski_rules['HBA']
            ])
            
            # Synthetic Accessibility
            sa_score = sascorer.calculateScore(mol)
            
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
        
        # Distribution
        bbb_penetration = (Descriptors.TPSA(mol) < 90 and 
                          Descriptors.NumRotatableBonds(mol) < 8 and
                          2 < Crippen.MolLogP(mol) < 4)
        
        # Metabolism
        cyp_substrate = Descriptors.NumAromaticRings(mol) > 1
        
        # Excretion
        clearance_likely = Descriptors.MolMR(mol) < 130
        
        # Toxicity
        potential_toxicity = (
            Descriptors.NumAromaticRings(mol) > 3 or
            Descriptors.ExactMolWt(mol) > 800 or
            Crippen.MolLogP(mol) > 7
        )
        
        return {
            'absorption': {'caco2_permeability': caco2_permeability},
            'distribution': {'bbb_penetration': bbb_penetration},
            'metabolism': {'cyp_substrate': cyp_substrate},
            'excretion': {'clearance_likely': clearance_likely},
            'toxicity': {'high_risk': potential_toxicity}
        }
    def cluster_and_visualize(self, smiles_list: list):
        """Cluster molecules using fingerprints and visualize diversity."""
        fingerprints = []
        molecules = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
        
        for mol in molecules:
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                fingerprints.append(fp)
        
        if not fingerprints:
            print("No valid molecules found for clustering.")
            return

        # Convert fingerprints to numpy array
        fps_array = np.array([np.array(list(map(int, fp.ToBitString()))) for fp in fingerprints])

        # Perform clustering
        n_clusters = min(len(fps_array), 5)  # Limit to 5 clusters for simplicity
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(fps_array)
        labels = kmeans.labels_

        # Perform PCA for visualization
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(fps_array)

        # Plot the clusters
        plt.figure(figsize=(10, 6))
        for cluster in range(n_clusters):
            cluster_points = reduced_data[labels == cluster]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster+1}")

        plt.title("Molecular Diversity Clustering")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.legend()
        plt.show()


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
            # Property distribution plots
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Property Distributions")
                properties_df = create_comparison_table([m['smiles'] for m in st.session_state['top_molecules']])
                fig = px.box(properties_df)
                st.plotly_chart(fig)
                
            with col2:
                st.subheader("Property Comparison")
                st.dataframe(properties_df)
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
    
    # Input molecule analysis with enhanced prompts
    st.subheader("Original Drug Analysis")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        input_mol = Chem.MolFromSmiles(seed_smiles)
        st.image(Draw.MolToImage(input_mol))
        input_qed = qed(input_mol)
        st.metric("Drug-likeness Score", f"{input_qed:.3f}")
    
    with col2:
        #analysis = st.session_state.analyzer.analyze_input_molecule(seed_smiles, input_qed)
        analysis = st.session_state.analyzer.analyze_input_molecule(seed_smiles)
        st.write(analysis)
    
    # Modified molecules analysis
    if 'top_molecules' in st.session_state:
        st.subheader("Modified Drug Candidates")
        
        for idx, mol_data in enumerate(st.session_state['top_molecules']):
            with st.expander(f"Candidate {idx+1}", expanded=True):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    mol = Chem.MolFromSmiles(mol_data['smiles'])
                    st.image(Draw.MolToImage(mol))
                    st.metric("Drug-likeness", f"{mol_data['qed']:.3f}")
                    st.metric("Structural Similarity", f"{mol_data['tanimoto']:.3f}")
                
                # with col2:
                #     analysis = st.session_state.analyzer.analyze_generated_molecule(
                #         seed_smiles,
                #         mol_data['smiles'],
                #         mol_data['qed'],
                #         mol_data['tanimoto'],
                #         idx + 1
                #     )
                #     st.write(analysis)
    
    # Suggested questions
    st.subheader("Research Questions")
    suggested_questions = st.session_state.analyzer.suggest_questions()
    selected_question = st.selectbox(
        "Select a research question:",
        suggested_questions
    )
    
    if selected_question:
        st.session_state.analyzer.add_to_history("user", selected_question)
        response = st.session_state.analyzer.analyze_with_context(selected_question)
        st.write(response)
    
    # Custom questions
    custom_question = st.text_input("Ask your own question:", key="custom_question")
    if custom_question:
        st.session_state.analyzer.add_to_history("user", custom_question)
        response = st.session_state.analyzer.analyze_with_context(custom_question)
        st.write(response)      

if __name__ == "__main__":
    main()