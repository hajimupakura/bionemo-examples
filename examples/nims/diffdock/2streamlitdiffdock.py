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
from Bio.PDB import *
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import io
from openai import OpenAI
@dataclass
class AIConfig:
    """Configuration for AI assistant."""
    api_key: str = "sk-4c0c80a4056c412491e08a64523b4945"
    base_url: str = "https://api.deepseek.com/v1"
    system_prompt: str = """You are an expert molecular docking assistant. Help users understand:
    - The 3D visualization tab shows protein-ligand binding poses
    - Score analysis includes histograms and heatmaps of docking scores
    - Interaction analysis identifies hydrogen bonds, hydrophobic contacts
    - Explain molecular docking concepts in simple terms
    - Keep responses concise and focused on structural biology"""

class DockingAssistant:
    """Handles AI-powered conversation about docking results."""
    
    def __init__(self, config: AIConfig):
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )
        self.system_prompt = config.system_prompt
        
    def get_response(self, messages: List[Dict]) -> str:
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": self.system_prompt}] + messages,
            temperature=0.7,
            stream=True
        )
        
        full_response = []
        for chunk in response:
            if chunk.choices[0].delta.content:
                full_response.append(chunk.choices[0].delta.content)
        return "".join(full_response)

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
    """Class for analyzing and visualizing docking results with protein-ligand interactions."""
    
    @staticmethod
    def analyze_interactions(protein_pdb: str, ligand_mol: Chem.Mol, cutoff: float = 4.5) -> Dict:
        """Analyze protein-ligand interactions."""
        # Parse protein structure
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', io.StringIO(protein_pdb))
        
        # Get ligand atoms coordinates
        conf = ligand_mol.GetConformer()
        ligand_coords = []
        ligand_atoms = []
        for i in range(ligand_mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            ligand_coords.append([pos.x, pos.y, pos.z])
            ligand_atoms.append(ligand_mol.GetAtomWithIdx(i))
        ligand_coords = np.array(ligand_coords)
        
        # Analyze interactions
        interactions = {
            'hydrogen_bonds': [],
            'hydrophobic': [],
            'ionic': [],
            'pi_stacking': []
        }
        
        # Find close contacts
        for residue in structure.get_residues():
            for atom in residue.get_atoms():
                atom_coord = atom.get_coord()
                distances = np.linalg.norm(ligand_coords - atom_coord, axis=1)
                close_contacts = np.where(distances < cutoff)[0]
                
                for contact_idx in close_contacts:
                    ligand_atom = ligand_atoms[contact_idx]
                    interaction_type = ResultsAnalyzer._classify_interaction(
                        atom, ligand_atom, distances[contact_idx]
                    )
                    if interaction_type:
                        interactions[interaction_type].append({
                            'residue': f"{residue.get_resname()}{residue.get_id()[1]}",
                            'atom': atom.get_name(),
                            'ligand_atom': ligand_atom.GetSymbol(),
                            'distance': float(distances[contact_idx])
                        })
        
        return interactions
    
    @staticmethod
    def _classify_interaction(protein_atom, ligand_atom, distance):
        """Classify the type of protein-ligand interaction."""
        # This is a simplified classification - could be expanded
        if distance < 3.5:
            # Potential hydrogen bond
            if protein_atom.get_name().startswith(('O', 'N')) and \
               ligand_atom.GetSymbol() in ['O', 'N']:
                return 'hydrogen_bonds'
            
            # Hydrophobic interaction
            if protein_atom.get_name().startswith('C') and \
               ligand_atom.GetSymbol() == 'C':
                return 'hydrophobic'
                
            # Ionic interaction
            if (protein_atom.get_name().startswith(('O', 'N')) and \
                ligand_atom.GetSymbol() in ['O', 'N']):
                return 'ionic'
        
        return None
    
    @staticmethod
    def create_interaction_network(interactions: Dict[str, List[Dict]]) -> go.Figure:
        """Create an interactive network visualization of protein-ligand interactions.
        
        Args:
            interactions (Dict[str, List[Dict]]): Dictionary of interaction types and their details
                The structure should be:
                {
                    'interaction_type': [
                        {
                            'residue': str,
                            'atom': str,
                            'ligand_atom': str,
                            'distance': float
                        },
                        ...
                    ],
                    ...
                }
        
        Returns:
            go.Figure: Plotly figure object containing the interaction network
        """
        
        G = nx.Graph()
        
        # Add nodes and edges for each interaction type
        colors = {
            'hydrogen_bonds': 'blue',
            'hydrophobic': 'green',
            'ionic': 'red',
            'pi_stacking': 'purple'
        }
        
        edge_traces = []
        node_traces = []
        
        for interaction_type, interactions_list in interactions.items():
            for interaction in interactions_list:
                residue_node = interaction['residue']
                ligand_node = f"LIG_{interaction['ligand_atom']}"
                
                G.add_node(residue_node, node_type='residue')
                G.add_node(ligand_node, node_type='ligand')
                G.add_edge(residue_node, ligand_node, 
                          type=interaction_type,
                          distance=interaction['distance'])
        
        # Create layout
        pos = nx.spring_layout(G)
        
        # Create edge traces
        for interaction_type, color in colors.items():
            edge_x = []
            edge_y = []
            
            for edge in G.edges():
                if G.edges[edge]['type'] == interaction_type:
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                    
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color=color),
                hoverinfo='none',
                mode='lines',
                name=interaction_type
            )
            edge_traces.append(edge_trace)
        
        # Create node traces
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            node_color.append('red' if 'LIG' in node else 'lightblue')
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="top center",
            marker=dict(
                size=10,
                color=node_color,
                line_width=2
            )
        )
        
        # Create figure
        fig = go.Figure(data=[*edge_traces, node_trace])
        fig.update_layout(
            title='Protein-Ligand Interaction Network',
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        return fig
    
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

def show_structure_viewer(protein_pdb: str, ligand_mol: Chem.Mol, score: float = None, size=(1000, 700)):
    """Create and display enhanced 3D structure viewer with proper protein-ligand visualization."""
    view = py3Dmol.view(width=size[0], height=size[1])
    
    # Add protein with enhanced visualization
    view.addModel(protein_pdb, 'pdb')
    view.setStyle({'model': 0}, {'cartoon': {'color': 'white', 'opacity': 0.7}})
    view.setViewStyle({'style':'outline','color':'black','width':0.03})
    Prot = view.getModel()
    Prot.setStyle({'cartoon':{'arrows':True, 'tubes':True, 'style':'oval', 'color':'white'}})
    view.addSurface(py3Dmol.VDW,{'opacity':0.4,'color':'white'})
    
    # Add ligand with score-based coloring
    if ligand_mol:
        ligand_block = Chem.MolToMolBlock(ligand_mol)
        view.addModel(ligand_block, 'mol')
        
        # Determine color scheme based on score
        if score is not None:
            color_scheme = 'greenCarbon' if score > -0.5 else 'cyanCarbon' if score >= -1.5 else 'magentaCarbon'
            surface_color = '#90EE90' if score > -0.5 else '#87CEEB' if score >= -1.5 else '#FFB6C1'
        else:
            color_scheme = 'magentaCarbon'
            surface_color = '#FFB6C1'
            
        view.setStyle({'model': 1}, {'stick': {'radius': 0.3, 'colorscheme': color_scheme}})
        view.addSurface(py3Dmol.VDW, {'opacity': 0.7, 'color': surface_color}, {'model': 1})
    
    # Force render and zoom after all elements added
    view.render()
    view.zoomTo()
    return view

def main():
    st.set_page_config(page_title="Molecular Docking Analysis", layout="wide")

    # Initialize AI assistant
    ai_config = AIConfig()
    assistant = DockingAssistant(ai_config)
    
    # Add to existing tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "3D Visualization", 
        "Score Analysis", 
        "Interaction Analysis",
        "Export Results",
        "AI Assistant"
    ])
    
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
                showmol(view, width=1000, height=700)  # Adjusted size parameters
            
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
                
                # Select pose for interaction analysis
                pose_idx = st.slider("Select Pose for Interaction Analysis", 
                                   0, 
                                   len(st.session_state.results['ligand_positions'])-1, 
                                   0,
                                   key='interaction_pose_slider')
                
                # Analyze interactions for selected pose
                mol = Chem.MolFromMolBlock(st.session_state.results['ligand_positions'][pose_idx])
                interactions = ResultsAnalyzer.analyze_interactions(
                    protein_file.getvalue().decode('utf-8'),
                    mol
                )
                
                # Display interaction statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Hydrogen Bonds", len(interactions['hydrogen_bonds']))
                with col2:
                    st.metric("Hydrophobic", len(interactions['hydrophobic']))
                with col3:
                    st.metric("Ionic", len(interactions['ionic']))
                with col4:
                    st.metric("Pi-Stacking", len(interactions['pi_stacking']))
                
                # Display interaction network
                st.subheader("Interaction Network")
                try:
                    network_fig = ResultsAnalyzer.create_interaction_network(interactions)
                    st.plotly_chart(network_fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating interaction network: {str(e)}")
                    st.write("Detailed interactions are still available in the table below.")
                
                # Display detailed interaction table
                st.subheader("Detailed Interactions")
                all_interactions = []
                for int_type, ints in interactions.items():
                    for interaction in ints:
                        interaction['type'] = int_type
                        all_interactions.append(interaction)
                
                if all_interactions:
                    df = pd.DataFrame(all_interactions)
                    df['distance'] = df['distance'].round(2)
                    st.dataframe(df)
            
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

            with tab5:
                st.header("Docking AI Assistant")
                
                # Initialize chat history
                if "messages" not in st.session_state:
                    st.session_state.messages = []
                
                # Display chat messages
                for msg in st.session_state.messages:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])
                
                # Chat input
                if prompt := st.chat_input("Ask about the docking results..."):
                    # Add user message
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    
                    # Get AI response
                    with st.chat_message("assistant"):
                        response = assistant.get_response(st.session_state.messages)
                        st.markdown(response)
                    
                    # Add AI response to history
                    st.session_state.messages.append({"role": "assistant", "content": response})    
    else:
        st.info("Please upload both protein and ligand files to begin analysis")

if __name__ == "__main__":
    main()