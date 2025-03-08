import streamlit as st
import py3Dmol
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, PandasTools
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import json
import glob
import os
import requests
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import time
from stmol import showmol
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class DockingConfig:
    """Configuration for molecular docking."""
    base_url: str = "http://0.0.0.0:8000/"
    num_poses: int = 10
    time_divisions: int = 20
    steps: int = 18
    save_trajectory: bool = False
    is_staged: bool = False

class MolecularDockingPipeline:
    """Main class for handling the molecular docking pipeline."""
    
    def __init__(self, config: DockingConfig):
        self.config = config
        self.query_url = config.base_url + "/molecular-docking/diffdock/generate"
        self.health_check_url = config.base_url + "/v1/health/ready"
    
    def check_server_health(self) -> bool:
        try:
            response = requests.get(self.health_check_url)
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def run_docking(self, protein_file, ligand_file) -> Dict:
        """Run molecular docking for a single ligand."""
        protein_data = protein_file.getvalue().decode('utf-8')
        ligand_data = ligand_file.getvalue().decode('utf-8')
        
        payload = {
            "ligand": ligand_data,
            "ligand_file_type": "sdf",
            "protein": protein_data,
            "num_poses": self.config.num_poses,
            "time_divisions": self.config.time_divisions,
            "steps": self.config.steps,
            "save_trajectory": self.config.save_trajectory,
            "is_staged": self.config.is_staged
        }
        
        response = requests.post(
            self.query_url,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=300
        )
        response.raise_for_status()
        return response.json()

class ResultsAnalyzer:
    """Class for analyzing and visualizing docking results."""
    
    @staticmethod
    def create_score_distribution(scores: List[float], title: str = "Score Distribution"):
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=scores, nbinsx=20))
        fig.update_layout(
            title=title,
            xaxis_title="Docking Score",
            yaxis_title="Count",
            showlegend=False
        )
        return fig
    
    @staticmethod
    def create_score_heatmap(scores: List[float], poses: List[int], title: str = "Score Heatmap"):
        score_matrix = np.array(scores).reshape(-1, 1)
        fig = px.imshow(score_matrix,
                       labels=dict(x="Pose", y="Ligand", color="Score"),
                       title=title)
        return fig
    
    @staticmethod
    def create_interaction_network(pose: Chem.Mol, protein_structure: str):
        """Create protein-ligand interaction network visualization"""
        # This is a placeholder for implementing detailed interaction analysis
        # You would typically analyze hydrogen bonds, hydrophobic interactions, etc.
        return None

def show_structure_viewer(protein_pdb: str, ligand_mol: Chem.Mol, size=(800, 600)):
    """Create and display 3D structure viewer."""
    view = py3Dmol.view(width=size[0], height=size[1])
    
    # Add protein
    view.addModel(protein_pdb, "pdb")
    view.setStyle({'model': 0}, {
        'cartoon': {'color': 'spectrum'},
        'surface': {'opacity': 0.3}
    })
    
    # Add ligand
    if ligand_mol:
        ligand_block = Chem.MolToMolBlock(ligand_mol)
        view.addModel(ligand_block, "mol")
        view.setStyle({'model': 1}, {
            'stick': {'colorscheme': 'spectral'},
            'sphere': {'radius': 0.3}
        })
    
    view.zoomTo()
    return view

def main():
    st.set_page_config(page_title="Molecular Docking Analysis", layout="wide")
    
    # Sidebar configuration
    st.sidebar.title("Configuration")
    
    # File uploaders
    protein_file = st.sidebar.file_uploader("Upload Protein Structure (PDB)", type=['pdb'])
    ligand_file = st.sidebar.file_uploader("Upload Ligand Structure (SDF)", type=['sdf'])
    
    # Docking parameters
    st.sidebar.subheader("Docking Parameters")
    num_poses = st.sidebar.slider("Number of Poses", 1, 20, 10)
    time_divisions = st.sidebar.slider("Time Divisions", 10, 50, 20)
    steps = st.sidebar.slider("Steps", 10, 30, 18)
    save_trajectory = st.sidebar.checkbox("Save Trajectory", False)
    
    # Main content
    st.title("Molecular Docking Analysis Platform")
    
    if protein_file and ligand_file:
        # Initialize docking pipeline
        config = DockingConfig(
            num_poses=num_poses,
            time_divisions=time_divisions,
            steps=steps,
            save_trajectory=save_trajectory
        )
        pipeline = MolecularDockingPipeline(config)
        
        # Check server health
        if not pipeline.check_server_health():
            st.error("Docking server is not available")
            return
        
        # Run docking
        if st.button("Run Docking"):
            with st.spinner("Running molecular docking..."):
                try:
                    results = pipeline.run_docking(protein_file, ligand_file)
                    st.session_state.results = results
                    st.success("Docking completed successfully!")
                except Exception as e:
                    st.error(f"Error during docking: {str(e)}")
                    return
        
        # Results visualization
        if 'results' in st.session_state:
            st.header("Docking Results")
            
            # Create tabs for different visualizations
            tab1, tab2, tab3, tab4 = st.tabs([
                "3D Visualization", 
                "Score Analysis", 
                "Interaction Analysis",
                "Export Results"
            ])
            
            with tab1:
                # 3D Structure viewer
                st.subheader("3D Structure Visualization")
                pose_idx = st.slider("Select Pose", 0, len(st.session_state.results['ligand_positions'])-1, 0)
                
                # Create molecule from SDF
                mol = Chem.MolFromMolBlock(st.session_state.results['ligand_positions'][pose_idx])
                view = show_structure_viewer(protein_file.getvalue().decode('utf-8'), mol)
                showmol(view, height=600)
            
            with tab2:
                # Score analysis
                st.subheader("Score Analysis")
                scores = st.session_state.results['position_confidence']
                
                col1, col2 = st.columns(2)
                with col1:
                    fig_dist = ResultsAnalyzer.create_score_distribution(scores)
                    st.plotly_chart(fig_dist)
                
                with col2:
                    fig_heat = ResultsAnalyzer.create_score_heatmap(scores, range(len(scores)))
                    st.plotly_chart(fig_heat)
                
                # Statistics
                st.subheader("Statistical Analysis")
                stats_df = pd.DataFrame({
                    'Metric': ['Best Score', 'Average Score', 'Score Std Dev'],
                    'Value': [
                        min(scores),
                        np.mean(scores),
                        np.std(scores)
                    ]
                })
                st.table(stats_df)
            
            with tab3:
                # Interaction analysis
                st.subheader("Protein-Ligand Interactions")
                st.info("Detailed interaction analysis will be implemented in future updates")
                
                # Placeholder for interaction network
                st.image("https://via.placeholder.com/600x400.png?text=Interaction+Network+Visualization")
            
            with tab4:
                # Export options
                st.subheader("Export Results")
                
                # Export poses
                if st.button("Download All Poses (SDF)"):
                    # Implementation for downloading poses
                    pass
                
                # Export scores
                if st.button("Download Score Analysis (CSV)"):
                    scores_df = pd.DataFrame({
                        'Pose': range(len(scores)),
                        'Score': scores
                    })
                    st.download_button(
                        "Download Scores",
                        scores_df.to_csv(index=False),
                        "docking_scores.csv",
                        "text/csv"
                    )
    else:
        st.info("Please upload both protein and ligand files to begin analysis")

if __name__ == "__main__":
    main()