# # app.py
# import streamlit as st
# import requests
# import numpy as np
# import pandas as pd
# import cma
# from rdkit import Chem
# from rdkit.Chem import Draw, AllChem
# from rdkit.Chem.QED import qed
# from rdkit.DataStructs import TanimotoSimilarity
# import plotly.express as px
# import plotly.graph_objects as go
# from typing import List, Tuple, Dict
# import json

# class MolMIMClient:
#     def __init__(self, host: str = "localhost", port: str = "8000"):
#         self.base_url = f"http://{host}:{port}"
        
#     def check_health(self) -> bool:
#         try:
#             response = requests.get(f"{self.base_url}/v1/health/ready")
#             return response.status_code == 200
#         except:
#             return False
            
#     def get_hidden_state(self, smiles: str) -> np.ndarray:
#         response = requests.post(
#             f"{self.base_url}/hidden",
#             headers={'Content-Type': 'application/json'},
#             json={"sequences": [smiles]}
#         )
#         return np.squeeze(np.array(response.json()["hiddens"]))
        
#     def decode_hidden(self, hidden_states: np.ndarray) -> List[str]:
#         hiddens_array = np.expand_dims(hidden_states, axis=1)
#         response = requests.post(
#             f"{self.base_url}/decode",
#             headers={'Content-Type': 'application/json'},
#             json={
#                 "hiddens": hiddens_array.tolist(),
#                 "mask": [[True] for _ in range(hiddens_array.shape[0])]
#             }
#         )
#         return list(dict.fromkeys(response.json()['generated']))

# class MoleculeOptimizer:
#     def __init__(self, reference_smiles: str):
#         self.reference_smiles = reference_smiles
#         self.reference_mol = Chem.MolFromSmiles(reference_smiles)
#         self.reference_qed = qed(self.reference_mol)
        
#     def calculate_tanimoto(self, smiles: str) -> float:
#         mol = Chem.MolFromSmiles(smiles)
#         if mol is None:
#             return 0
#         fp1 = AllChem.GetMorganFingerprintAsBitVect(self.reference_mol, 2, 2048)
#         fp2 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
#         return TanimotoSimilarity(fp1, fp2)
        
#     def scoring_function(self, qeds: np.ndarray, similarities: np.ndarray) -> np.ndarray:
#         return -1.0 * (
#             np.clip(np.array(qeds) / 0.9, a_min=0.0, a_max=1.0) + 
#             np.clip(np.array(similarities) / 0.4, a_min=0.0, a_max=1.0)
#         )

# def main():
#     st.title("Drug Discovery Optimization Platform")
    
#     # Sidebar configurations
#     st.sidebar.header("Settings")
#     host = st.sidebar.text_input("MolMIM Host", "localhost")
#     port = st.sidebar.text_input("MolMIM Port", "8000")
    
#     # Initialize MolMIM client
#     client = MolMIMClient(host, port)
    
#     # Check MolMIM server health
#     if not client.check_health():
#         st.error("⚠️ MolMIM server is not accessible. Please check your connection settings.")
#         return
    
#     st.success("✅ Connected to MolMIM server")
    
#     # Input section
#     st.header("Molecule Input")
#     seed_smiles = st.text_area(
#         "Enter seed SMILES string",
#         "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5"  # Imatinib
#     )
    
#     if not Chem.MolFromSmiles(seed_smiles):
#         st.error("Invalid SMILES string")
#         return
        
#     # Display seed molecule
#     mol = Chem.MolFromSmiles(seed_smiles)
#     img = Draw.MolToImage(mol)
#     st.image(img, caption="Seed Molecule", use_container_width=True)
    
#     # Optimization parameters
#     st.header("Optimization Parameters")
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         population_size = st.number_input("Population Size", 10, 100, 50)
#     with col2:
#         n_iterations = st.number_input("Number of Iterations", 10, 200, 50)
#     with col3:
#         sigma = st.number_input("CMA-ES Sigma", 0.1, 5.0, 1.0)
        
#     if st.button("Start Optimization"):
#         optimizer = MoleculeOptimizer(seed_smiles)
        
#         # Initialize CMA-ES with seed molecule hidden state
#         hidden_state = client.get_hidden_state(seed_smiles)
#         es = cma.CMAEvolutionStrategy(hidden_state, sigma, {'popsize': population_size})
        
#         # Progress tracking
#         progress_bar = st.progress(0)
#         status_text = st.empty()
        
#         # Results tracking
#         results = []
        
#         # Create plots
#         col1, col2 = st.columns(2)
#         with col1:
#             scatter_plot = st.empty()
#         with col2:
#             trend_plot = st.empty()
            
#         # Optimization loop
#         for iteration in range(n_iterations):
#             # Generate and evaluate candidates
#             trial_encodings = es.ask(population_size)
#             molecules = client.decode_hidden(np.array(trial_encodings))
#             valid_molecules = [m for m in molecules if Chem.MolFromSmiles(m)]
            
#             # Calculate scores
#             qed_scores = [qed(Chem.MolFromSmiles(m)) for m in valid_molecules]
#             tanimoto_scores = [optimizer.calculate_tanimoto(m) for m in valid_molecules]
#             scores = optimizer.scoring_function(qed_scores, tanimoto_scores)
            
#             # Update CMA-ES
#             if len(valid_molecules) > population_size // 2:
#                 es.tell(trial_encodings[:len(valid_molecules)], scores)
                
#                 # Store results
#                 results.append({
#                     'iteration': iteration,
#                     'qed_median': np.median(qed_scores),
#                     'tanimoto_median': np.median(tanimoto_scores),
#                     'best_score': -min(scores)
#                 })
                
#                 # Update plots
#                 df = pd.DataFrame(results)
                
#                 # Scatter plot
#                 fig1 = px.scatter(
#                     x=qed_scores, 
#                     y=tanimoto_scores,
#                     title="Population Distribution",
#                     labels={'x': 'QED Score', 'y': 'Tanimoto Similarity'}
#                 )
#                 scatter_plot.plotly_chart(fig1)
                
#                 # Trend plot
#                 fig2 = go.Figure()
#                 fig2.add_trace(go.Scatter(x=df['iteration'], y=df['qed_median'], name='Median QED'))
#                 fig2.add_trace(go.Scatter(x=df['iteration'], y=df['tanimoto_median'], name='Median Tanimoto'))
#                 fig2.update_layout(title="Optimization Progress", xaxis_title="Iteration", yaxis_title="Score")
#                 trend_plot.plotly_chart(fig2)
            
#             # Update progress
#             progress = (iteration + 1) / n_iterations
#             progress_bar.progress(progress)
#             status_text.text(f"Iteration {iteration + 1}/{n_iterations}")
            
#         # Final results
#         st.header("Optimization Results")
#         best_idx = np.argmin(scores)
#         best_molecule = valid_molecules[best_idx]
#         best_mol = Chem.MolFromSmiles(best_molecule)
        
#         col1, col2 = st.columns(2)
#         with col1:
#             st.image(Draw.MolToImage(best_mol), caption="Best Molecule")
#         with col2:
#             st.text(f"SMILES: {best_molecule}")
#             st.text(f"QED Score: {qed_scores[best_idx]:.3f}")
#             st.text(f"Tanimoto Similarity: {tanimoto_scores[best_idx]:.3f}")

# if __name__ == "__main__":
#     main()

import deepchem
print(deepchem.__file__)