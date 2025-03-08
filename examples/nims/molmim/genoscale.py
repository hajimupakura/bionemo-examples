import streamlit as st
import pandas as pd
import numpy as np
import os
import uuid
import json
import datetime
import time
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix
import networkx as nx
from Bio import SeqIO, Entrez
from io import StringIO
import subprocess
import sys
import zipfile
from typing import List, Dict, Tuple, Optional, Union, Any

# Set page configuration
st.set_page_config(
    page_title="GenomeAI: Drug Discovery Pipeline",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define application constants
APP_VERSION = "1.0.0"
DATA_DIR = "data"
MODELS_DIR = "models"
PROJECTS_DIR = os.path.join(os.path.dirname(__file__), "projects")
RESULTS_DIR = "results"

# Ensure required directories exist
for directory in [DATA_DIR, MODELS_DIR, PROJECTS_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #4f8bf9;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #444;
        margin-bottom: 1rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 0.7rem;
        background-color: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .metric-card {
        text-align: center;
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #e6f3ff;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .success-text {
        color: #28a745;
        font-weight: 600;
    }
    .warning-text {
        color: #ffc107;
        font-weight: 600;
    }
    .error-text {
        color: #dc3545;
        font-weight: 600;
    }
    .info-text {
        color: #17a2b8;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Define utility functions
def load_project(project_id: str) -> Dict:
    """Load a project from disk by its ID."""
    project_path = os.path.join(PROJECTS_DIR, f"{project_id}.json")
    if os.path.exists(project_path):
        with open(project_path, "r") as f:
            return json.load(f)
    return None

def save_project(project: Dict) -> None:
    """Save a project to disk."""
    project_path = os.path.join(PROJECTS_DIR, f"{project['id']}.json")
    with open(project_path, "w") as f:
        json.dump(project, f, indent=2)

import logging  # Ensure this is at the top if not already present

def get_all_projects() -> List[Dict]:
    """Get all projects from disk."""
    projects = []
    for filename in os.listdir(PROJECTS_DIR):
        if filename.endswith(".json"):
            project_path = os.path.join(PROJECTS_DIR, filename)
            try:
                with open(project_path, "r") as f:
                    project = json.load(f)
                # Set defaults for all expected keys
                if "last_updated" not in project:
                    project["last_updated"] = project.get("created_at", datetime.datetime.now().isoformat())
                if "status" not in project:
                    project["status"] = "created"
                if "genome_type" not in project:
                    project["genome_type"] = "Unknown"
                if "datasets" not in project:
                    project["datasets"] = []
                if "models" not in project:
                    project["models"] = []
                if "experiments" not in project:
                    project["experiments"] = []  # Add this line
                if "results" not in project:
                    project["results"] = []
                projects.append(project)
            except Exception as e:
                logging.error(f"Error loading project {filename}: {str(e)}")
                continue
    
    return sorted(projects, key=lambda x: x.get("last_updated", ""), reverse=True)

def create_new_project(name: str, description: str, genome_type: str) -> Dict:
    """Create a new project with the given parameters."""
    project_id = str(uuid.uuid4())
    created_at = datetime.datetime.now().isoformat()
    
    project = {
        "id": project_id,
        "name": name,
        "description": description,
        "genome_type": genome_type,
        "created_at": created_at,
        "last_updated": created_at,
        "status": "created",
        "datasets": [],
        "models": [],
        "experiments": [],
        "results": []
    }
    
    save_project(project)
    return project

def process_genome_file(file, file_type: str) -> Dict:
    """Process an uploaded genome file."""
    # Create unique filename
    filename = f"{uuid.uuid4()}_{file.name}"
    filepath = os.path.join(DATA_DIR, filename)
    
    # Save file
    with open(filepath, "wb") as f:
        f.write(file.getbuffer())
    
    # Get basic stats based on file type
    stats = {}
    
    if file_type == "fasta":
        sequences = list(SeqIO.parse(filepath, "fasta"))
        stats = {
            "sequence_count": len(sequences),
            "avg_length": sum(len(seq) for seq in sequences) / len(sequences) if sequences else 0,
            "min_length": min(len(seq) for seq in sequences) if sequences else 0,
            "max_length": max(len(seq) for seq in sequences) if sequences else 0,
        }
    elif file_type == "vcf":
        # Count variants
        variant_count = 0
        with open(filepath, "r") as f:
            for line in f:
                if not line.startswith("#"):
                    variant_count += 1
        stats = {"variant_count": variant_count}
    
    return {
        "id": str(uuid.uuid4()),
        "original_name": file.name,
        "filename": filename,
        "filepath": filepath,
        "file_type": file_type,
        "upload_date": datetime.datetime.now().isoformat(),
        "size_bytes": file.size,
        "stats": stats
    }

# Define ML Models
class CNNModel(nn.Module):
    """Convolutional Neural Network for sequence analysis."""
    def __init__(self, seq_length: int, n_features: int, n_filters: List[int], kernel_sizes: List[int], 
                 fc_layers: List[int], n_classes: int, dropout: float = 0.3):
        super(CNNModel, self).__init__()
        
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=n_features, out_channels=n_filters[i], 
                      kernel_size=kernel_sizes[i], padding=kernel_sizes[i]//2)
            for i in range(len(n_filters))
        ])
        
        # Calculate CNN output size
        cnn_output_size = seq_length * n_filters[-1]
        
        # Fully connected layers
        self.fc_layers = nn.ModuleList()
        fc_input_size = cnn_output_size
        
        for fc_size in fc_layers:
            self.fc_layers.append(nn.Linear(fc_input_size, fc_size))
            fc_input_size = fc_size
        
        self.output_layer = nn.Linear(fc_input_size, n_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x shape: [batch_size, n_features, seq_length]
        for conv in self.convs:
            x = F.relu(conv(x))
        
        # Flatten for FC layers
        x = x.view(x.size(0), -1)
        
        for fc in self.fc_layers:
            x = F.relu(fc(x))
            x = self.dropout(x)
        
        return self.output_layer(x)

class GNNModel(nn.Module):
    """Graph Neural Network for molecular structure analysis."""
    def __init__(self, num_node_features: int, hidden_channels: int, num_classes: int):
        super(GNNModel, self).__init__()
        
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        
        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, num_classes)
    
    def forward(self, x, edge_index, batch):
        # Node embedding
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        
        # Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        
        # Classification
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        
        return x

# Data processing functions
def encode_dna_sequence(sequence: str, one_hot: bool = True) -> np.ndarray:
    """Encode DNA sequence as one-hot or integer encoding."""
    if one_hot:
        mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 
                   'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1],
                   'N': [0.25, 0.25, 0.25, 0.25]}
        
        encoded = np.array([mapping.get(nucl, mapping['N']) for nucl in sequence.upper()])
    else:
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        encoded = np.array([mapping.get(nucl, 4) for nucl in sequence.upper()])
    
    return encoded

def sliding_window(sequence: str, window_size: int, stride: int) -> List[str]:
    """Create sliding windows from a sequence."""
    return [sequence[i:i+window_size] for i in range(0, len(sequence)-window_size+1, stride)]

def prepare_batch_data(sequences: List[str], labels: List[int], window_size: int = 1000, 
                     stride: int = 500, batch_size: int = 32):
    """Prepare batched data for CNN training."""
    all_windows = []
    all_labels = []
    
    for seq, label in zip(sequences, labels):
        windows = sliding_window(seq, window_size, stride)
        all_windows.extend(windows)
        all_labels.extend([label] * len(windows))
    
    # Encode sequences
    encoded_seqs = [encode_dna_sequence(seq) for seq in all_windows]
    
    # Create batches
    n_batches = len(encoded_seqs) // batch_size + (1 if len(encoded_seqs) % batch_size != 0 else 0)
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(encoded_seqs))
        
        batch_seqs = encoded_seqs[start_idx:end_idx]
        batch_labels = all_labels[start_idx:end_idx]
        
        # Convert to tensor
        batch_seqs = torch.tensor(np.array(batch_seqs), dtype=torch.float32)
        batch_seqs = batch_seqs.permute(0, 2, 1)  # [batch_size, n_features, seq_length]
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)
        
        yield batch_seqs, batch_labels


def delete_project(project_id: str) -> bool:
    """Delete a project from disk by its ID."""
    project_path = os.path.join(PROJECTS_DIR, f"{project_id}.json")
    if os.path.exists(project_path):
        os.remove(project_path)
        if 'current_project' in st.session_state and st.session_state.current_project['id'] == project_id:
            del st.session_state.current_project
        return True
    return False

def render_dashboard():
    """Render the dashboard page."""
    st.markdown("<h1>Genomic Analysis Dashboard</h1>", unsafe_allow_html=True)
    
    # Display projects count
    projects = get_all_projects()
    st.markdown(f"### Total Projects: {len(projects)}")
    
    # Recent projects section
    st.markdown("## Recent Projects")
    
    if not projects:
        st.info("No projects found. Create a new project to get started.")
        return
    
    # Display up to 5 most recent projects
    for project in projects[:5]:
        with st.expander(f"Project: {project.get('name', 'Untitled Project')}"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"""
                **Description:** {project.get('description', 'No description available')}
                
                **Created:** {project.get('created_at', 'Date not available')}
                
                **Genome Type:** {project.get('genome_type', 'Not specified')}
                """)
            
            with col2:
                st.markdown(f"""
                **Datasets:** {len(project.get('datasets', []))}
                
                **Models:** {len(project.get('models', []))}
                
                **Experiments:** {len(project.get('experiments', []))}
                """)
            
            # Add action buttons
            cols = st.columns(3)
            with cols[0]:
                if st.button('View Details', key=f"view_{project.get('id', '')}"):
                    st.session_state.current_project = project
                    st.session_state.page = 'project_details'
                    st.rerun()
            
            with cols[1]:
                if st.button('Edit', key=f"edit_{project.get('id', '')}"):
                    st.session_state.current_project = project
                    st.session_state.page = 'edit_project'
                    st.rerun()
            
            with cols[2]:
                if st.button('Delete', key=f"delete_{project.get('id', '')}"):
                    if delete_project(project.get('id', '')):
                        st.success("Project deleted successfully")
                        st.rerun()

def render_projects_page():
    st.markdown("<h1>Projects</h1>", unsafe_allow_html=True)
    
    # Show current project selection
    if 'current_project' in st.session_state:
        st.markdown(f"<h2>Selected Project: {st.session_state.current_project.get('name', 'Untitled')}</h2>", unsafe_allow_html=True)
    else:
        st.write("No project currently selected.")
    
    # Project creation form
    with st.form("new_project_form"):
        project_name = st.text_input("Project Name")
        project_description = st.text_area("Description")
        genome_type = st.selectbox("Genome Type", ["Human", "Bacterial", "Viral", "Other"])
        
        submit_button = st.form_submit_button("Create Project")
        
        if submit_button and project_name:
            new_project = create_new_project(project_name, project_description, genome_type)
            st.session_state.current_project = new_project
            st.success(f"Project '{project_name}' created successfully!")
            st.rerun()
    
    # Display existing projects
    projects = get_all_projects()
    if projects:
        st.markdown("## Existing Projects")
        for project in projects:
            with st.expander(f"Project: {project.get('name', 'Untitled')}"):
                st.markdown(f"""
                **Description:** {project.get('description', 'No description')}
                **Genome Type:** {project.get('genome_type', 'Unknown')}
                **Datasets:** {len(project.get('datasets', []))}
                """)
                if st.button("Select Project", key=f"select_{project.get('id', '')}"):
                    st.session_state.current_project = project
                    st.rerun()
    else:
        st.info("No projects found. Create your first project using the form above.")

def render_data_management():
    """Render the data management page."""
    st.markdown("<h1 class='main-header'>Data Management</h1>", unsafe_allow_html=True)
    
    # Check if we have a current project
    if 'current_project_id' not in st.session_state or not st.session_state.current_project_id:
        st.warning("Please select a project first.")
        if st.button("Go to Projects"):
            st.rerun()
        return
    
    project = st.session_state.current_project
    
    st.markdown(f"<h2 class='sub-header'>Project: {project['name']}</h2>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Upload Data", "Manage Datasets", "Data Preprocessing"])
    
    with tab1:
        st.markdown("<h3>Upload Genomic Data</h3>", unsafe_allow_html=True)
        
        # File upload section
        uploaded_file = st.file_uploader("Choose a file", type=['fasta', 'vcf', 'csv', 'txt'])
        
        if uploaded_file:
            file_type = st.selectbox(
                "Select file type",
                ["FASTA", "VCF", "CSV", "TXT"],
                index=["fasta", "vcf", "csv", "txt"].index(uploaded_file.name.split('.')[-1].lower())
            )
            
            if st.button("Process File"):
                with st.spinner("Processing file..."):
                    try:
                        dataset = process_genome_file(uploaded_file, file_type.lower())
                        
                        # Update project with new dataset
                        project['datasets'].append(dataset)
                        save_project(project)
                        
                        st.success(f"Successfully processed {uploaded_file.name}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error processing file: {str(e)}")
    
    with tab2:
        if not project.get('datasets'):
            st.info("No datasets available. Upload data using the Upload Data tab.")
        else:
            st.markdown("<h3>Available Datasets</h3>", unsafe_allow_html=True)
            
            for idx, dataset in enumerate(project['datasets']):
                with st.expander(f"Dataset: {dataset['original_name']}"):
                    st.json(dataset['stats'])
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Remove Dataset", key=f"remove_dataset_{idx}"):
                            project['datasets'].pop(idx)
                            save_project(project)
                            st.rerun()
                    
                    with col2:
                        if st.button("View Details", key=f"view_dataset_{idx}"):
                            st.session_state.selected_dataset = dataset
                            st.rerun()
    
    with tab3:
        if not project.get('datasets'):
            st.info("No datasets available for preprocessing. Upload data first.")
        else:
            st.markdown("<h3>Preprocessing Options</h3>", unsafe_allow_html=True)
            
            dataset = st.selectbox(
                "Select Dataset",
                options=project['datasets'],
                format_func=lambda x: x['original_name']
            )
            
            if dataset:
                preprocessing_options = st.multiselect(
                    "Select Preprocessing Steps",
                    ["Quality Control", "Sequence Filtering", "Feature Extraction"]
                )
                
                if preprocessing_options and st.button("Run Preprocessing"):
                    with st.spinner("Running preprocessing..."):
                        # Add preprocessing logic here
                        st.success("Preprocessing completed successfully")
                        st.rerun()

def render_model_training():
    """Render the model training page."""
    st.markdown("<h1 class='main-header'>Model Training</h1>", unsafe_allow_html=True)
    
    # Check if we have a current project
    if 'current_project_id' not in st.session_state or not st.session_state.current_project_id:
        st.warning("Please select a project first.")
        if st.button("Go to Projects"):
            st.rerun()
        return
    
    project = st.session_state.current_project
    
    st.markdown(f"<h2 class='sub-header'>Project: {project['name']}</h2>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Configure Model", "Training", "Model Evaluation"])
    
    with tab1:
        st.markdown("<h3>Model Configuration</h3>", unsafe_allow_html=True)
        
        if not project["datasets"]:
            st.warning("You need to add datasets to your project before configuring models.")
            return
        
        model_type = st.selectbox("Model Type", ["CNN (Sequence Analysis)", 
                                               "GNN (Molecular Structure)", 
                                               "Transformer (Genome-wide Features)"])
        
        st.markdown("<h4>Model Architecture</h4>", unsafe_allow_html=True)
        
        if model_type == "CNN (Sequence Analysis)":
            col1, col2 = st.columns(2)
            
            with col1:
                seq_length = st.number_input("Sequence Length", min_value=100, max_value=10000, value=1000)
                n_filters = st.text_input("Number of Filters (comma-separated)", value="64,128,256")
                kernel_sizes = st.text_input("Kernel Sizes (comma-separated)", value="3,5,7")
            
            with col2:
                fc_layers = st.text_input("FC Layer Sizes (comma-separated)", value="512,256")
                n_classes = st.number_input("Number of Output Classes", min_value=1, value=2)
                dropout = st.slider("Dropout Rate", min_value=0.0, max_value=0.9, value=0.3, step=0.1)
            
            # Parse input lists
            n_filters = [int(x.strip()) for x in n_filters.split(",")]
            kernel_sizes = [int(x.strip()) for x in kernel_sizes.split(",")]
            fc_layers = [int(x.strip()) for x in fc_layers.split(",")]
            
            # Display model summary
            st.markdown("<h4>Model Summary</h4>", unsafe_allow_html=True)
            st.code(f"""
CNN Model Architecture:
- Sequence Length: {seq_length}
- Conv Layers: {len(n_filters)} layers with {n_filters} filters
- Kernel Sizes: {kernel_sizes}
- Fully Connected Layers: {fc_layers}
- Output Classes: {n_classes}
- Dropout Rate: {dropout}
            """)
            
        elif model_type == "GNN (Molecular Structure)":
            col1, col2 = st.columns(2)
            
            with col1:
                num_node_features = st.number_input("Node Feature Dimension", min_value=1, value=32)
                hidden_channels = st.number_input("Hidden Channels", min_value=8, value=64)
            
            with col2:
                num_classes = st.number_input("Number of Output Classes", min_value=1, value=2)
                num_layers = st.number_input("Number of GNN Layers", min_value=1, max_value=5, value=3)
            
            # Display model summary
            st.markdown("<h4>Model Summary</h4>", unsafe_allow_html=True)
            st.code(f"""
GNN Model Architecture:
- Node Feature Dimension: {num_node_features}
- Hidden Channels: {hidden_channels}
- GNN Layers: {num_layers}
- Output Classes: {num_classes}
            """)
            
        elif model_type == "Transformer (Genome-wide Features)":
            col1, col2 = st.columns(2)
            
            with col1:
                embed_dim = st.number_input("Embedding Dimension", min_value=16, value=128)
                num_heads = st.number_input("Number of Attention Heads", min_value=1, value=8)
                ff_dim = st.number_input("Feed Forward Dimension", min_value=32, value=256)
            
            with col2:
                num_transformer_blocks = st.number_input("Number of Transformer Blocks", min_value=1, value=6)
                mlp_units = st.text_input("MLP Units (comma-separated)", value="128,64")
                mlp_dropout = st.slider("MLP Dropout", min_value=0.0, max_value=0.9, value=0.2, step=0.1)
                
            # Parse MLP units
            mlp_units = [int(x.strip()) for x in mlp_units.split(",")]
            
            # Display model summary
            st.markdown("<h4>Model Summary</h4>", unsafe_allow_html=True)
            st.code(f"""
Transformer Model Architecture:
- Embedding Dimension: {embed_dim}
- Attention Heads: {num_heads}
- Feed Forward Dimension: {ff_dim}
- Transformer Blocks: {num_transformer_blocks}
- MLP Units: {mlp_units}
- MLP Dropout: {mlp_dropout}
            """)
        
        # Training configuration
        st.markdown("<h4>Training Configuration</h4>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            batch_size = st.number_input("Batch Size", min_value=1, value=32)
            epochs = st.number_input("Epochs", min_value=1, value=50)
            learning_rate = st.number_input("Learning Rate", min_value=0.0001, max_value=0.1, value=0.001, format="%.4f")
        
        with col2:
            optimizer = st.selectbox("Optimizer", ["Adam", "SGD", "RMSprop", "AdamW"])
            loss_function = st.selectbox("Loss Function", ["CrossEntropy", "BCE", "MSE", "Focal"])
            early_stopping = st.checkbox("Early Stopping", value=True)
            
        # Dataset selection
        st.markdown("<h4>Dataset Selection</h4>", unsafe_allow_html=True)
        
        dataset_options = [dataset.get('name', dataset['original_name']) for dataset in project["datasets"]]
        training_datasets = st.multiselect("Select Training Datasets", dataset_options)
        
        validation_split = st.slider("Validation Split", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
        
        if st.button("Save Model Configuration"):
            if not training_datasets:
                st.error("Please select at least one dataset for training.")
            else:
                # Create model config
                model_config = {
                    "id": str(uuid.uuid4()),
                    "name": f"{model_type.split(' ')[0]}_Model_{len(project['models']) + 1}",
                    "type": model_type,
                    "created_at": datetime.datetime.now().isoformat(),
                    "architecture": {
                        "model_type": model_type.split(" ")[0],  # CNN, GNN, or Transformer
                    },
                    "training": {
                        "batch_size": batch_size,
                        "epochs": epochs,
                        "learning_rate": learning_rate,
                        "optimizer": optimizer,
                        "loss_function": loss_function,
                        "early_stopping": early_stopping,
                        "validation_split": validation_split
                    },
                    "datasets": training_datasets,
                    "status": "configured",
                    "metrics": {}
                }
                
                # Add architecture specific parameters
                if model_type == "CNN (Sequence Analysis)":
                    model_config["architecture"].update({
                        "seq_length": seq_length,
                        "n_filters": n_filters,
                        "kernel_sizes": kernel_sizes,
                        "fc_layers": fc_layers,
                        "n_classes": n_classes,
                        "dropout": dropout
                    })
                elif model_type == "GNN (Molecular Structure)":
                    model_config["architecture"].update({
                        "num_node_features": num_node_features,
                        "hidden_channels": hidden_channels,
                        "num_classes": num_classes,
                        "num_layers": num_layers
                    })
                elif model_type == "Transformer (Genome-wide Features)":
                    model_config["architecture"].update({
                        "embed_dim": embed_dim,
                        "num_heads": num_heads,
                        "ff_dim": ff_dim,
                        "num_transformer_blocks": num_transformer_blocks,
                        "mlp_units": mlp_units,
                        "mlp_dropout": mlp_dropout
                    })
                
                # Add to project
                project["models"].append(model_config)
                project["last_updated"] = datetime.datetime.now().isoformat()
                save_project(project)
                
                st.session_state.current_project = project
                st.success("Model configuration saved successfully!")
    
    with tab2:
        st.markdown("<h3>Model Training</h3>", unsafe_allow_html=True)
        
        if not project.get("models"):
            st.warning("No model configurations found. Please configure a model first.")
        else:
            model_options = {model["name"]: i for i, model in enumerate(project["models"])}
            
            selected_model_name = st.selectbox("Select Model", list(model_options.keys()))
            selected_model_idx = model_options[selected_model_name]
            selected_model = project["models"][selected_model_idx]
            
            st.markdown(f"<h4>Training {selected_model_name}</h4>", unsafe_allow_html=True)
            
            # Display model configuration
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"<h4>Model Type: {selected_model['type']}</h4>", unsafe_allow_html=True)
            st.markdown("<h4>Architecture:</h4>", unsafe_allow_html=True)
            
            for key, value in selected_model["architecture"].items():
                st.markdown(f"- **{key}**: {value}", unsafe_allow_html=True)
            
            st.markdown("<h4>Training Configuration:</h4>", unsafe_allow_html=True)
            
            for key, value in selected_model["training"].items():
                st.markdown(f"- **{key}**: {value}", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Training controls
            col1, col2 = st.columns([1, 3])
            
            with col1:
                use_gpu = st.checkbox("Use GPU", value=torch.cuda.is_available())
                use_amp = st.checkbox("Use Mixed Precision", value=True)
                
                if st.button("Start Training"):
                    # In a real application, this would trigger an actual training job
                    # For this demo, we'll simulate the training process
                    selected_model["status"] = "training"
                    project["last_updated"] = datetime.datetime.now().isoformat()
                    save_project(project)
                    st.session_state.current_project = project
                    
                    with st.spinner("Training in progress..."):
                        # Simulate training for a few seconds
                        progress_bar = st.progress(0)
                        for i in range(100):
                            time.sleep(0.05)
                            progress_bar.progress(i + 1)
                        
                        # After "training" update model status
                        selected_model["status"] = "trained"
                        selected_model["training_completed_at"] = datetime.datetime.now().isoformat()
                        
                        # Add mock metrics
                        selected_model["metrics"] = {
                            "accuracy": round(random.uniform(0.85, 0.98), 4),
                            "precision": round(random.uniform(0.8, 0.95), 4),
                            "recall": round(random.uniform(0.8, 0.95), 4),
                            "f1_score": round(random.uniform(0.8, 0.95), 4),
                            "auc_roc": round(random.uniform(0.85, 0.99), 4),
                            "training_time": round(random.uniform(120, 1800), 2),
                            "epochs_completed": selected_model["training"]["epochs"]
                        }
                        
                        # Save model path
                        selected_model["model_path"] = f"{MODELS_DIR}/{selected_model['id']}.pt"
                        
                        # Update project
                        project["models"][selected_model_idx] = selected_model
                        project["last_updated"] = datetime.datetime.now().isoformat()
                        save_project(project)
                        st.session_state.current_project = project
                        
                        st.success("Training completed successfully!")
                        st.rerun()
            
            with col2:
                if selected_model["status"] == "training":
                    st.info("Training in progress...")
                    st.markdown("<!-- Training log placeholder -->")
                
                elif selected_model["status"] == "trained":
                    st.success("Training completed")
                    
                    # Display metrics
                    metrics = selected_model.get("metrics", {})
                    if metrics:
                        st.markdown("<h4>Training Results:</h4>", unsafe_allow_html=True)
                        
                        metric_cols = st.columns(3)
                        metric_data = [
                            {"name": "Accuracy", "value": metrics.get("accuracy", "N/A"), "format": "{:.2%}"},
                            {"name": "Precision", "value": metrics.get("precision", "N/A"), "format": "{:.2%}"},
                            {"name": "Recall", "value": metrics.get("recall", "N/A"), "format": "{:.2%}"},
                            {"name": "F1 Score", "value": metrics.get("f1_score", "N/A"), "format": "{:.2%}"},
                            {"name": "AUC-ROC", "value": metrics.get("auc_roc", "N/A"), "format": "{:.2%}"},
                            {"name": "Training Time", "value": metrics.get("training_time", "N/A"), "format": "{:.2f}s"}
                        ]
                        
                        for i, metric in enumerate(metric_data):
                            col_idx = i % 3
                            with metric_cols[col_idx]:
                                try:
                                    formatted_value = metric["format"].format(metric["value"])
                                except (ValueError, TypeError):
                                    formatted_value = str(metric["value"])
                                
                                st.markdown(f"""
                                <div class='metric-card'>
                                    <h5>{metric["name"]}</h5>
                                    <h3 class='info-text'>{formatted_value}</h3>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Display mock loss curve
                        st.markdown("<h4>Training Progress:</h4>", unsafe_allow_html=True)
                        
                        # Generate mock training data
                        epochs = metrics.get("epochs_completed", 50)
                        train_loss = []
                        val_loss = []
                        
                        for i in range(epochs):
                            train_loss.append(1.0 * (1 - 0.98 * i / epochs) + random.uniform(-0.05, 0.05))
                            val_loss.append(1.2 * (1 - 0.95 * i / epochs) + random.uniform(-0.07, 0.07))
                        
                        # Plot
                        fig, ax = plt.subplots(figsize=(8, 4))
                        ax.plot(range(1, epochs + 1), train_loss, label='Training Loss')
                        ax.plot(range(1, epochs + 1), val_loss, label='Validation Loss')
                        ax.set_xlabel('Epoch')
                        ax.set_ylabel('Loss')
                        ax.legend()
                        ax.grid(True, linestyle='--', alpha=0.7)
                        st.pyplot(fig)
                
                else:
                    st.warning("Model has not been trained yet. Click 'Start Training' to begin.")
    
    with tab3:
        st.markdown("<h3>Model Evaluation</h3>", unsafe_allow_html=True)
        
        if not project.get("models"):
            st.warning("No model configurations found. Please configure a model first.")
        else:
            # Filter only trained models
            trained_models = [model for model in project["models"] if model.get("status") == "trained"]
            
            if not trained_models:
                st.warning("No trained models found. Please train a model first.")
            else:
                model_options = {model["name"]: i for i, model in enumerate(trained_models)}
                
                selected_model_name = st.selectbox("Select Trained Model", list(model_options.keys()))
                selected_model_idx = model_options[selected_model_name]
                selected_model = trained_models[selected_model_idx]
                
                st.markdown(f"<h4>Evaluating {selected_model_name}</h4>", unsafe_allow_html=True)
                
                # Select evaluation dataset
                dataset_options = [dataset.get('name', dataset['original_name']) for dataset in project["datasets"]]
                eval_dataset = st.selectbox("Select Evaluation Dataset", dataset_options)
                
                # Generate confusion matrix
                st.markdown("<h4>Confusion Matrix</h4>", unsafe_allow_html=True)
                
                # Create a dummy confusion matrix for visualization
                if "metrics" in selected_model:
                    metrics = selected_model["metrics"]
                    precision = metrics.get("precision", 0.9)
                    recall = metrics.get("recall", 0.9)
                    
                    # Calculate confusion matrix values based on metrics
                    tp = int(100 * recall)
                    fn = int(100 - tp)
                    fp = int(tp * (1 - precision) / precision)
                    tn = int(100 - fp)
                    
                    cm = np.array([[tn, fp], [fn, tp]])
                    
                    # Plot confusion matrix
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    ax.set_title('Confusion Matrix')
                    ax.set_xticklabels(['Negative', 'Positive'])
                    ax.set_yticklabels(['Negative', 'Positive'])
                    st.pyplot(fig)
                    
                    # ROC curve
                    st.markdown("<h4>ROC Curve</h4>", unsafe_allow_html=True)
                    
                    # Generate mock ROC curve data
                    fpr = np.linspace(0, 1, 100)
                    auc_val = metrics.get("auc_roc", 0.9)
                    
                    # Approximate ROC curve from AUC value
                    tpr = np.power(fpr, (1 / auc_val - 1))
                    
                    # Plot ROC curve
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.plot(fpr, tpr, 'b-', label=f'AUC = {auc_val:.3f}')
                    ax.plot([0, 1], [0, 1], 'r--', label='Random')
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.set_title('Receiver Operating Characteristic')
                    ax.legend(loc='lower right')
                    ax.grid(True, linestyle='--', alpha=0.7)
                    st.pyplot(fig)
                    
                    # Feature importance (for classification models)
                    st.markdown("<h4>Feature Importance</h4>", unsafe_allow_html=True)
                    
                    # Generate mock feature importance
                    if selected_model["type"].startswith("CNN"):
                        # For sequence models, show position-wise importance
                        positions = [f"Pos {i+1}" for i in range(10)]
                        importance = np.random.uniform(0.2, 1.0, size=10)
                        importance = importance / importance.sum()
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.bar(positions, importance)
                        ax.set_xlabel('Sequence Position')
                        ax.set_ylabel('Importance')
                        ax.set_title('Position Importance (Top 10)')
                        st.pyplot(fig)
                    elif selected_model["type"].startswith("GNN"):
                        # For graph models, show node importance
                        nodes = [f"Node {i+1}" for i in range(10)]
                        importance = np.random.uniform(0.2, 1.0, size=10)
                        importance = importance / importance.sum()
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.bar(nodes, importance)
                        ax.set_xlabel('Nodes')
                        ax.set_ylabel('Importance')
                        ax.set_title('Node Importance (Top 10)')
                        st.pyplot(fig)
                    
                    # Generate Report button
                    if st.button("Generate Evaluation Report"):
                        with st.spinner("Generating report..."):
                            time.sleep(2)  # Simulate report generation
                            
                            # Create a mock report result
                            report_id = str(uuid.uuid4())
                            report = {
                                "id": report_id,
                                "name": f"Evaluation_{selected_model_name}_{datetime.datetime.now().strftime('%Y%m%d')}",
                                "model_id": selected_model["id"],
                                "dataset": eval_dataset,
                                "created_at": datetime.datetime.now().isoformat(),
                                "metrics": selected_model["metrics"],
                                "report_path": f"{RESULTS_DIR}/{report_id}.pdf"
                            }
                            
                            # Add to project
                            if "evaluations" not in project:
                                project["evaluations"] = []
                            
                            project["evaluations"].append(report)
                            project["last_updated"] = datetime.datetime.now().isoformat()
                            save_project(project)
                            st.session_state.current_project = project
                            
                            st.success("Evaluation report generated successfully!")
                
                else:
                    st.warning("Model metrics not found. Re-train the model to generate metrics.")

def render_analysis():
    """Render the analysis page."""
    st.markdown("<h1 class='main-header'>Analysis</h1>", unsafe_allow_html=True)
    
    # Check if we have a current project
    if 'current_project_id' not in st.session_state or not st.session_state.current_project_id:
        st.warning("Please select a project first.")
        if st.button("Go to Projects"):
            st.rerun()
        return
    
    project = st.session_state.current_project
    
    st.markdown(f"<h2 class='sub-header'>Project: {project['name']}</h2>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Sequence Analysis", "Genome Visualization", "Experiment Results"])
    
    with tab1:
        st.markdown("<h3>Sequence Analysis</h3>", unsafe_allow_html=True)
        
        if not project["datasets"]:
            st.warning("No datasets in this project. Add data in the Data Management section.")
        else:
            # Filter only sequence datasets
            seq_datasets = [d for d in project["datasets"] if d["file_type"] in ["fasta", "fa"]]
            
            if not seq_datasets:
                st.warning("No sequence datasets found in this project.")
            else:
                dataset_options = {dataset.get('name', dataset['original_name']): i 
                                 for i, dataset in enumerate(seq_datasets)}
                
                selected_dataset_name = st.selectbox("Select Sequence Dataset", list(dataset_options.keys()))
                selected_dataset_idx = dataset_options[selected_dataset_name]
                selected_dataset = seq_datasets[selected_dataset_idx]
                
                st.markdown(f"<h4>Analyzing {selected_dataset_name}</h4>", unsafe_allow_html=True)
                
                analysis_type = st.selectbox("Analysis Type", [
                    "Sequence Composition", 
                    "GC Content", 
                    "Repeats", 
                    "Motif Search"
                ])
                
                st.markdown(f"<h4>{analysis_type}</h4>", unsafe_allow_html=True)
                
                if analysis_type == "Sequence Composition":
                    # Demo sequence composition analysis
                    # In a real application, this would parse the actual FASTA file
                    
                    nucleotides = ['A', 'C', 'G', 'T', 'N']
                    counts = [random.randint(100, 1000) for _ in range(5)]
                    total = sum(counts)
                    percentages = [count / total * 100 for count in counts]
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Chart
                        fig, ax = plt.subplots(figsize=(8, 5))
                        ax.bar(nucleotides, percentages, color=['green', 'blue', 'orange', 'red', 'gray'])
                        ax.set_xlabel('Nucleotide')
                        ax.set_ylabel('Percentage (%)')
                        ax.set_title('Nucleotide Composition')
                        ax.set_ylim(0, max(percentages) * 1.2)
                        
                        # Add percentage labels
                        for i, p in enumerate(percentages):
                            ax.annotate(f'{p:.1f}%', (i, p + 1), ha='center')
                        
                        st.pyplot(fig)
                    
                    with col2:
                        # Data table
                        composition_data = {
                            "Nucleotide": nucleotides,
                            "Count": counts,
                            "Percentage": [f"{p:.2f}%" for p in percentages]
                        }
                        st.dataframe(pd.DataFrame(composition_data))
                
                elif analysis_type == "GC Content":
                    # Demo GC content analysis
                    
                    # Generate mock GC content data
                    window_size = st.slider("Window Size", min_value=100, max_value=10000, value=1000, step=100)
                    
                    # Generate mock data for sliding windows
                    num_windows = 100
                    positions = list(range(0, num_windows * window_size, window_size))
                    gc_content = [40 + 15 * np.sin(i / 10) + random.uniform(-5, 5) for i in range(num_windows)]
                    
                    # Plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(positions, gc_content, '-o', markersize=3, alpha=0.7)
                    ax.set_xlabel(f'Position (window size: {window_size} bp)')
                    ax.set_ylabel('GC Content (%)')
                    ax.set_title('GC Content Distribution')
                    ax.axhline(y=np.mean(gc_content), color='r', linestyle='--', label=f'Mean: {np.mean(gc_content):.2f}%')
                    ax.legend()
                    ax.grid(True, linestyle='--', alpha=0.7)
                    st.pyplot(fig)
                    
                    # Statistics
                    st.markdown("<h4>GC Content Statistics</h4>", unsafe_allow_html=True)
                    
                    gc_stats = {
                        "Statistic": ["Mean", "Median", "Std Dev", "Min", "Max"],
                        "Value": [
                            f"{np.mean(gc_content):.2f}%",
                            f"{np.median(gc_content):.2f}%",
                            f"{np.std(gc_content):.2f}%",
                            f"{min(gc_content):.2f}%",
                            f"{max(gc_content):.2f}%"
                        ]
                    }
                    st.dataframe(pd.DataFrame(gc_stats))
                
                elif analysis_type == "Motif Search":
                    # Demo motif search interface
                    
                    motif = st.text_input("Enter Motif Sequence", value="GGCGTG")
                    search_reverse = st.checkbox("Search Reverse Complement", value=True)
                    
                    if st.button("Search Motif"):
                        with st.spinner("Searching for motif..."):
                            # Simulate search
                            time.sleep(2)
                            
                            # Generate mock results
                            num_results = random.randint(5, 15)
                            positions = sorted(random.sample(range(1, 10000), num_results))
                            
                            # Display results
                            st.success(f"Found {num_results} occurrences of motif '{motif}'")
                            
                            results_data = {
                                "Position": positions,
                                "Strand": [random.choice(["+", "-"]) for _ in range(num_results)],
                                "Context": [
                                    f"...{random_dna(10)}<b>{motif}</b>{random_dna(10)}..." 
                                    for _ in range(num_results)
                                ]
                            }
                            
                            # Convert to DataFrame
                            results_df = pd.DataFrame(results_data)
                            
                            # Display with HTML for highlighting
                            st.markdown(results_df.to_html(escape=False, index=False), unsafe_allow_html=True)
                            
                            # Distribution plot
                            fig, ax = plt.subplots(figsize=(10, 4))
                            ax.hist(positions, bins=20, alpha=0.7)
                            ax.set_xlabel('Position')
                            ax.set_ylabel('Frequency')
                            ax.set_title(f'Distribution of Motif "{motif}" Occurrences')
                            st.pyplot(fig)
    
    with tab2:
        st.markdown("<h3>Genome Visualization</h3>", unsafe_allow_html=True)
        
        visualization_type = st.selectbox("Visualization Type", [
            "Genome Browser", 
            "Chromosome Map", 
            "Gene Network"
        ])
        
        if visualization_type == "Genome Browser":
            st.markdown("<h4>Genome Browser</h4>", unsafe_allow_html=True)
            
            st.info("This is a mock genome browser visualization. In a full implementation, this would integrate with tools like IGV.js.")
            
            chromosome = st.selectbox("Chromosome", ["chr1", "chr2", "chr3", "chr4", "chr5", "chrX", "chrY"])
            start_pos = st.number_input("Start Position", min_value=1, value=1000000)
            window_size = st.slider("Window Size", min_value=1000, max_value=1000000, value=50000, step=1000)
            
            end_pos = start_pos + window_size
            
            # Mock genome browser visualization
            st.markdown(f"<h5>Viewing {chromosome}:{start_pos:,}-{end_pos:,}</h5>", unsafe_allow_html=True)
            
            # Create a mock genome browser visualization
            track_height = 100
            fig_height = 400
            
            fig, axs = plt.subplots(4, 1, figsize=(10, fig_height / 100), 
                                   gridspec_kw={'height_ratios': [1, 2, 2, 3]})
            
            # Scale bar
            positions = np.linspace(start_pos, end_pos, 6)
            axs[0].set_xlim(start_pos, end_pos)
            axs[0].set_yticks([])
            axs[0].set_xticks(positions)
            axs[0].set_xticklabels([f"{int(p):,}" for p in positions], rotation=45)
            axs[0].set_title(f"{chromosome}:{start_pos:,}-{end_pos:,}", fontsize=10)
            
            # Variants track
            axs[1].set_xlim(start_pos, end_pos)
            axs[1].set_yticks([])
            axs[1].set_ylabel("Variants")
            
            # Generate random variant positions
            variant_positions = sorted(np.random.randint(start_pos, end_pos, 50))
            variant_types = np.random.choice(["SNP", "INDEL", "SV"], 50, p=[0.7, 0.2, 0.1])
            
            # Plot variants
            for i, (pos, v_type) in enumerate(zip(variant_positions, variant_types)):
                if v_type == "SNP":
                    color = "green"
                    marker = "o"
                    size = 20
                elif v_type == "INDEL":
                    color = "blue"
                    marker = "s"
                    size = 30
                else:  # SV
                    color = "red"
                    marker = "D"
                    size = 50
                
                axs[1].scatter(pos, 0.5, color=color, marker=marker, s=size)
            
            axs[1].set_title("Variants", fontsize=8)
            
            # Genes track
            axs[2].set_xlim(start_pos, end_pos)
            axs[2].set_yticks([])
            axs[2].set_ylabel("Genes")
            
            # Generate random genes
            num_genes = 8
            gene_starts = sorted(np.random.randint(start_pos, end_pos - 5000, num_genes))
            gene_lengths = np.random.randint(2000, 8000, num_genes)
            gene_strands = np.random.choice(["+", "-"], num_genes)
            
            # Plot genes
            for i, (g_start, g_len, strand) in enumerate(zip(gene_starts, gene_lengths, gene_strands)):
                g_end = g_start + g_len
                if g_end > end_pos:
                    g_end = end_pos
                
                height = 0.3
                y_pos = i % 3 * 0.25 + 0.1
                
                if strand == "+":
                    color = "blue"
                else:
                    color = "red"
                
                # Gene body
                axs[2].add_patch(plt.Rectangle((g_start, y_pos), g_end - g_start, height, 
                                             color=color, alpha=0.5))
                
                # Add label
                if (g_end - g_start) > window_size / 20:  # Only label genes wide enough
                    axs[2].text((g_start + g_end) / 2, y_pos + height / 2, f"Gene{i+1}", 
                              ha='center', va='center', fontsize=8, color='black')
            
            axs[2].set_title("Gene Annotations", fontsize=8)
            
            # Coverage track
            axs[3].set_xlim(start_pos, end_pos)
            axs[3].set_ylabel("Coverage")
            
            # Generate mock coverage data
            positions = np.linspace(start_pos, end_pos, 1000)
            
            # Create some peaks and valleys in the coverage
            base_coverage = 30
            coverage = base_coverage + 15 * np.sin(np.linspace(0, 4 * np.pi, 1000))
            
            # Add some random noise
            coverage += np.random.normal(0, 5, 1000)
            coverage = np.maximum(coverage, 0)  # Ensure no negative values
            
            # Add some peaks at gene positions
            for g_start, g_len in zip(gene_starts, gene_lengths):
                g_end = g_start + g_len
                indices = ((positions >= g_start) & (positions <= g_end))
                coverage[indices] += np.random.uniform(10, 30)
            
            # Plot coverage
            axs[3].fill_between(positions, coverage, alpha=0.5, color='purple')
            axs[3].plot(positions, coverage, color='purple', linewidth=1)
            axs[3].set_title("Read Coverage", fontsize=8)
            axs[3].set_ylim(0, max(coverage) * 1.1)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        elif visualization_type == "Chromosome Map":
            st.markdown("<h4>Chromosome Map</h4>", unsafe_allow_html=True)
            
            species = st.selectbox("Species", ["Human", "Mouse", "Yeast", "E. coli", "Custom"])
            
            if species == "Human":
                chromosomes = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
            elif species == "Mouse":
                chromosomes = [f"chr{i}" for i in range(1, 20)] + ["chrX", "chrY"]
            elif species == "Yeast":
                chromosomes = [f"chr{i}" for i in range(1, 17)]
            elif species == "E. coli":
                chromosomes = ["chromosome"]
            else:
                chromosome_num = st.number_input("Number of Chromosomes", min_value=1, value=5)
                chromosomes = [f"chr{i}" for i in range(1, chromosome_num + 1)]
            
            # Create ideogram-style visualization
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Calculate layout
            max_height = 8
            chr_width = 0.6
            spacing = 0.4
            max_chr_height = 5
            
            # Scale chromosome heights based on relative sizes
            if species == "Human":
                # Human chromosome sizes (approximate, in Mb)
                chr_sizes = {
                    'chr1': 248, 'chr2': 242, 'chr3': 198, 'chr4': 190, 'chr5': 181,
                    'chr6': 170, 'chr7': 159, 'chr8': 146, 'chr9': 138, 'chr10': 133,
                    'chr11': 135, 'chr12': 133, 'chr13': 114, 'chr14': 107, 'chr15': 101,
                    'chr16': 90, 'chr17': 83, 'chr18': 80, 'chr19': 58, 'chr20': 64,
                    'chr21': 46, 'chr22': 50, 'chrX': 156, 'chrY': 57
                }
            else:
                # Generate random sizes for other species
                max_size = 200 if species == "Mouse" else 100
                chr_sizes = {chr_name: random.randint(30, max_size) for chr_name in chromosomes}
            
            # Normalize chromosome heights
            max_size = max(chr_sizes.values())
            chr_heights = {chr_name: (size / max_size) * max_chr_height 
                          for chr_name, size in chr_sizes.items()}
            
            # Calculate how many chromosomes per row
            chr_per_row = 8
            num_rows = (len(chromosomes) + chr_per_row - 1) // chr_per_row
            
            # Layout chromosomes
            for i, chr_name in enumerate(chromosomes):
                row = i // chr_per_row
                col = i % chr_per_row
                
                x_pos = col * (chr_width + spacing)
                y_pos = row * (max_chr_height + spacing)
                
                height = chr_heights[chr_name]
                
                # Draw chromosome
                rect = plt.Rectangle((x_pos, y_pos), chr_width, height, 
                                   facecolor='lightgray', edgecolor='black')
                ax.add_patch(rect)
                
                # Add centromere (randomly positioned)
                centro_pos = random.uniform(0.3, 0.7) * height
                centro = plt.Rectangle((x_pos, y_pos + centro_pos), chr_width, 0.2,
                                     facecolor='red', edgecolor='black')
                ax.add_patch(centro)
                
                # Add label
                ax.text(x_pos + chr_width / 2, y_pos - 0.4, chr_name, 
                      ha='center', va='center', fontsize=8)
                
                # Add size label (Mb)
                ax.text(x_pos + chr_width / 2, y_pos + height + 0.2, 
                      f"{chr_sizes[chr_name]} Mb", ha='center', va='center', fontsize=6)
                
                # Generate some mock gene density data
                num_bands = 20
                band_height = height / num_bands
                
                for j in range(num_bands):
                    band_y = y_pos + j * band_height
                    density = random.uniform(0, 1)  # Random gene density
                    
                    # Color based on gene density (darker = higher density)
                    color = plt.cm.Blues(density)
                    
                    band = plt.Rectangle((x_pos, band_y), chr_width, band_height,
                                       facecolor=color, edgecolor=None, alpha=0.7)
                    ax.add_patch(band)
            
            # Set plot limits
            ax.set_xlim(-0.5, chr_per_row * (chr_width + spacing))
            ax.set_ylim(-1, num_rows * (max_chr_height + spacing) + 1)
            
            # Remove axis
            ax.set_axis_off()
            
            # Add title
            ax.set_title(f"{species} Chromosome Map with Gene Density")
            
            # Add color bar
            cax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
            cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.Blues), cax=cax)
            cbar.set_label('Gene Density')
            
            st.pyplot(fig)
            
        elif visualization_type == "Gene Network":
            st.markdown("<h4>Gene Interaction Network</h4>", unsafe_allow_html=True)
            
            # Options
            network_size = st.slider("Network Size", min_value=10, max_value=100, value=30)
            min_connections = st.slider("Min Connections per Gene", min_value=1, max_value=10, value=2)
            layout_type = st.selectbox("Layout", ["Spring", "Circular", "Random", "Shell"])
            
            # Generate mock gene network
            G = nx.random_degree_sequence_graph([min_connections + random.randint(0, 5) for _ in range(network_size)])
            
            # Rename nodes to gene names
            mapping = {i: f"Gene{i+1}" for i in range(network_size)}
            G = nx.relabel_nodes(G, mapping)
            
            # Add random weights to edges
            for u, v in G.edges():
                G[u][v]['weight'] = random.uniform(0.1, 1.0)
            
            # Node properties
            node_sizes = []
            node_colors = []
            
            for node in G.nodes():
                # Size based on degree (number of connections)
                size = 100 + 20 * G.degree(node)
                node_sizes.append(size)
                
                # Color based on community
                node_colors.append(random.randint(0, 5))
            
            # Network visualization
            plt.figure(figsize=(10, 8))
            
            # Choose layout
            if layout_type == "Spring":
                pos = nx.spring_layout(G, k=0.3)
            elif layout_type == "Circular":
                pos = nx.circular_layout(G)
            elif layout_type == "Random":
                pos = nx.random_layout(G)
            else:  # Shell
                pos = nx.shell_layout(G)
            
            # Edge weights for thickness
            edge_widths = [G[u][v]['weight'] * 3 for u, v in G.edges()]
            
            # Draw network
            nx.draw_networkx_edges(G, pos, alpha=0.3, width=edge_widths)
            nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                                        node_color=node_colors, cmap=plt.cm.tab10,
                                        alpha=0.8)
            nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')
            
            # Add colorbar
            plt.colorbar(nodes)
            plt.axis('off')
            plt.title(f"Gene Interaction Network (n={network_size})")
            
            st.pyplot(plt)
            
            # Network statistics
            st.markdown("<h4>Network Statistics</h4>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Nodes", network_size)
                st.metric("Edges", G.number_of_edges())
            
            with col2:
                st.metric("Avg. Connections", f"{sum(dict(G.degree()).values()) / network_size:.2f}")
                st.metric("Density", f"{nx.density(G):.4f}")
            
            with col3:
                # Find most connected gene
                degrees = dict(G.degree())
                max_degree_gene = max(degrees, key=degrees.get)
                
                st.metric("Most Connected Gene", max_degree_gene)
                st.metric("Connections", degrees[max_degree_gene])
            
            # Community detection
            st.markdown("<h4>Gene Clusters</h4>", unsafe_allow_html=True)
            
            # Detect communities
            communities = nx.community.greedy_modularity_communities(G)
            
            # Display communities
            for i, community in enumerate(communities[:5]):  # Show top 5 communities
                genes = list(community)
                st.markdown(f"**Cluster {i+1}** ({len(genes)} genes): {', '.join(genes[:10])}" + 
                          ("..." if len(genes) > 10 else ""))
    
    with tab3:
        st.markdown("<h3>Experiment Results</h3>", unsafe_allow_html=True)
        
        # Check if we have trained models
        trained_models = [model for model in project.get("models", []) if model.get("status") == "trained"]
        
        if not trained_models:
            st.warning("No trained models found. Train a model in the Model Training section.")
        else:
            # Show experiment results
            st.markdown("<h4>Model Performance Comparison</h4>", unsafe_allow_html=True)
            
            # Create comparison table
            comparison_data = []
            
            for model in trained_models:
                metrics = model.get("metrics", {})
                
                if metrics:
                    comparison_data.append({
                        "Model": model["name"],
                        "Type": model["type"].split(" ")[0],
                        "Accuracy": metrics.get("accuracy", "N/A"),
                        "Precision": metrics.get("precision", "N/A"),
                        "Recall": metrics.get("recall", "N/A"),
                        "F1": metrics.get("f1_score", "N/A"),
                        "AUC": metrics.get("auc_roc", "N/A")
                    })
            
            if comparison_data:
                # Convert to DataFrame
                comparison_df = pd.DataFrame(comparison_data)
                
                # Display comparison table
                st.dataframe(comparison_df.style.format({
                    "Accuracy": "{:.2%}",
                    "Precision": "{:.2%}",
                    "Recall": "{:.2%}",
                    "F1": "{:.2%}",
                    "AUC": "{:.2%}"
                }))
                
                # Performance comparison chart
                st.markdown("<h4>Performance Metrics Comparison</h4>", unsafe_allow_html=True)
                
                metric_to_plot = st.selectbox("Select Metric", 
                                            ["Accuracy", "Precision", "Recall", "F1", "AUC"])
                
                # Convert metric to lowercase for data access
                metric_key = metric_to_plot.lower()
                if metric_key == "auc":
                    metric_key = "auc_roc"
                elif metric_key == "f1":
                    metric_key = "f1_score"
                
                # Prepare data for chart
                chart_data = []
                
                for model in trained_models:
                    metrics = model.get("metrics", {})
                    
                    if metrics and metric_key in metrics:
                        chart_data.append({
                            "Model": model["name"],
                            "Type": model["type"].split(" ")[0],
                            "Metric": metrics[metric_key]
                        })
                
                if chart_data:
                    chart_df = pd.DataFrame(chart_data)
                    
                    # Create bar chart
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = ax.bar(chart_df["Model"], chart_df["Metric"], color='skyblue')
                    
                    # Color by model type
                    type_colors = {"CNN": "skyblue", "GNN": "salmon", "Transformer": "lightgreen"}
                    for i, (_, row) in enumerate(chart_df.iterrows()):
                        bars[i].set_color(type_colors.get(row["Type"], "gray"))
                    
                    ax.set_xlabel('Model')
                    ax.set_ylabel(metric_to_plot)
                    ax.set_title(f'{metric_to_plot} Comparison Across Models')
                    
                    # Add value labels
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                              f'{height:.2%}', ha='center', va='bottom')
                    
                    # Adjust y-axis to show percentage
                    ax.set_ylim(0, max(chart_df["Metric"]) * 1.1)
                    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
                    
                    # Add legend
                    from matplotlib.patches import Patch
                    legend_elements = [Patch(facecolor=color, label=model_type)
                                      for model_type, color in type_colors.items()
                                      if model_type in chart_df["Type"].values]
                    
                    ax.legend(handles=legend_elements, title="Model Type")
                    
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Add prediction time comparison
                    st.markdown("<h4>Prediction Time Comparison</h4>", unsafe_allow_html=True)
                    
                    # Generate mock prediction times
                    pred_times = []
                    
                    for model in trained_models:
                        model_type = model["type"].split(" ")[0]
                        
                        # Different base times for different model types
                        if model_type == "CNN":
                            base_time = random.uniform(0.01, 0.05)
                        elif model_type == "GNN":
                            base_time = random.uniform(0.05, 0.15)
                        else:  # Transformer
                            base_time = random.uniform(0.1, 0.3)
                        
                        pred_times.append({
                            "Model": model["name"],
                            "Type": model_type,
                            "Time (ms)": base_time * 1000
                        })
                    
                    pred_df = pd.DataFrame(pred_times)
                    
                    # Create bar chart
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = ax.bar(pred_df["Model"], pred_df["Time (ms)"], color='lightgreen')
                    
                    # Color by model type (same colors as before)
                    for i, (_, row) in enumerate(pred_df.iterrows()):
                        bars[i].set_color(type_colors.get(row["Type"], "gray"))
                    
                    ax.set_xlabel('Model')
                    ax.set_ylabel('Prediction Time (ms)')
                    ax.set_title('Prediction Time Comparison')
                    
                    # Add value labels
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                              f'{height:.1f}', ha='center', va='bottom')
                    
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    st.pyplot(fig)
            else:
                st.info("No metrics available for trained models.")
            
            # Experiment logs
            st.markdown("<h4>Experiment Logs</h4>", unsafe_allow_html=True)
            
            logs = []
            
            for model in trained_models:
                # Generate mock logs
                training_time = model.get("metrics", {}).get("training_time", random.uniform(120, 1800))
                completed_at = model.get("training_completed_at", datetime.datetime.now().isoformat())
                
                logs.append({
                    "Timestamp": datetime.datetime.fromisoformat(completed_at).strftime('%Y-%m-%d %H:%M:%S'),
                    "Model": model["name"],
                    "Type": model["type"],
                    "Duration": f"{training_time:.1f}s",
                    "Status": "Completed",
                    "Details": f"Training completed with {model.get('metrics', {}).get('accuracy', 0):.2%} accuracy"
                })
            
            # Add some mock evaluation logs
            for i in range(min(3, len(trained_models))):
                model = trained_models[i]
                eval_time = datetime.datetime.fromisoformat(model.get("training_completed_at", datetime.datetime.now().isoformat()))

def render_project_details():
    """Render the project details page."""
    if 'current_project' not in st.session_state:
        st.warning("No project selected")
        if st.button("Return to Projects"):
            st.session_state.page = "Projects"
            st.rerun()
        return
    
    project = st.session_state.current_project
    
    st.markdown(f"<h1>Project Details: {project.get('name', 'Untitled')}</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        ### Description
        {project.get('description', 'No description available')}
        
        ### Genome Type
        {project.get('genome_type', 'Not specified')}
        
        ### Created
        {project.get('created_at', 'Date not available')}
        """)
    
    with col2:
        st.markdown(f"""
        ### Statistics
        - Datasets: {len(project.get('datasets', []))}
        - Models: {len(project.get('models', []))}
        - Experiments: {len(project.get('experiments', []))}
        """)
    
    if st.button("Return to Projects"):
        st.session_state.page = "Projects"
        st.rerun()

def render_project_edit():
    """Render the project edit page."""
    if 'current_project' not in st.session_state:
        st.warning("No project selected")
        if st.button("Return to Projects"):
            st.session_state.page = "Projects"
            st.rerun()
        return
    
    project = st.session_state.current_project
    
    st.markdown(f"<h1>Edit Project: {project.get('name', 'Untitled')}</h1>", unsafe_allow_html=True)
    
    with st.form("edit_project_form"):
        project_name = st.text_input("Project Name", value=project.get('name', ''))
        project_description = st.text_area("Description", value=project.get('description', ''))
        genome_type = st.selectbox("Genome Type", 
                                 ["Human", "Bacterial", "Viral", "Other"],
                                 index=["Human", "Bacterial", "Viral", "Other"].index(project.get('genome_type', 'Other')))
        
        if st.form_submit_button("Save Changes"):
            project['name'] = project_name
            project['description'] = project_description
            project['genome_type'] = genome_type
            save_project(project)
            st.success("Changes saved successfully!")
            st.session_state.page = "Projects"
            st.rerun()
    
    if st.button("Cancel"):
        st.session_state.page = "Projects"
        st.rerun()
        eval_time += datetime.timedelta(minutes=random.randint(5, 60))
        
        logs.append({
            "Timestamp": eval_time.strftime('%Y-%m-%d %H:%M:%S'),
            "Model": model["name"],
            "Type": "Evaluation",
            "Duration": f"{random.uniform(5, 30):.1f}s",
            "Status": "Completed",
            "Details": f"Evaluation on test data with AUC {model.get('metrics', {}).get('auc_roc', 0):.2%}"
        })
    
    # Sort logs by timestamp (newest first)
    logs.sort(key=lambda x: x["Timestamp"], reverse=True)
    
    # Display logs
    st.table(pd.DataFrame(logs))

def render_settings():
    """Render the settings page."""
    st.markdown("<h1 class='main-header'>Settings</h1>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Application", "Hardware", "About"])
    
    with tab1:
        st.markdown("<h3>Application Settings</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            theme = st.selectbox("Application Theme", ["Light", "Dark", "Auto"])
            language = st.selectbox("Language", ["English", "Spanish", "Chinese", "Japanese", "German"])
            notifications = st.checkbox("Enable Notifications", value=True)
        
        with col2:
            default_model = st.selectbox("Default Model Type", ["CNN", "GNN", "Transformer"])
            auto_save = st.checkbox("Auto-save Projects", value=True)
            debug_mode = st.checkbox("Debug Mode", value=False)
        
        # Data storage settings
        st.markdown("<h4>Data Storage</h4>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            storage_path = st.text_input("Data Storage Path", value=os.getcwd())
            backup_enabled = st.checkbox("Enable Automatic Backups", value=True)
        
        with col2:
            backup_frequency = st.selectbox("Backup Frequency", ["Daily", "Weekly", "Monthly"])
            max_storage = st.slider("Maximum Storage (GB)", min_value=1, max_value=100, value=20)
        
        if st.button("Save Settings"):
            st.success("Settings saved successfully!")
    
    with tab2:
        st.markdown("<h3>Hardware Settings</h3>", unsafe_allow_html=True)
        
        # Display GPU information if available
        if torch.cuda.is_available():
            gpu_info = f"GPU: {torch.cuda.get_device_name(0)}"
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
            memory_cached = torch.cuda.memory_reserved(0) / 1024**3  # GB
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"CUDA Version: {torch.version.cuda}")
                st.info(f"GPU Model: {torch.cuda.get_device_name(0)}")
                st.info(f"GPU Count: {torch.cuda.device_count()}")
            
            with col2:
                st.info(f"Memory Allocated: {memory_allocated:.2f} GB")
                st.info(f"Memory Reserved: {memory_cached:.2f} GB")
                st.info(f"PyTorch Version: {torch.__version__}")
        else:
            st.warning("No GPU detected. The application will run on CPU only.")
            st.info(f"PyTorch Version: {torch.__version__}")
        
        # Hardware utilization settings
        st.markdown("<h4>Hardware Utilization</h4>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            gpu_memory_limit = st.slider("GPU Memory Limit (%)", min_value=10, max_value=100, value=80)
            use_mixed_precision = st.checkbox("Use Mixed Precision (FP16)", value=True)
        
        with col2:
            num_workers = st.slider("Data Loading Workers", min_value=1, max_value=16, value=4)
            cpu_threads = st.slider("CPU Threads", min_value=1, max_value=32, value=8)
        
        if st.button("Apply Hardware Settings"):
            st.success("Hardware settings applied successfully!")
    
    with tab3:
        st.markdown("<h3>About GenomeAI</h3>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='card'>
            <h2>GenomeAI: End-to-End Drug Discovery Pipeline</h2>
            <p>Version {}</p>
            <p>A comprehensive platform for genomic data analysis and AI-driven drug discovery.</p>
            
            <h4>Features:</h4>
            <ul>
                <li>Genomic data processing and analysis</li>
                <li>Advanced AI models (CNN, GNN, Transformers)</li>
                <li>Interactive visualizations</li>
                <li>Project management system</li>
                <li>Integrated model training and evaluation</li>
            </ul>
            
            <h4>Technologies:</h4>
            <ul>
                <li>Python</li>
                <li>PyTorch</li>
                <li>Streamlit</li>
                <li>BioPython</li>
                <li>NetworkX</li>
                <li>Matplotlib/Seaborn</li>
            </ul>
        </div>
        """.format(APP_VERSION), unsafe_allow_html=True)

def random_dna(length: int) -> str:
    """Generate a random DNA sequence of given length."""
    return ''.join(random.choice(['A', 'C', 'G', 'T']) for _ in range(length))

# Define application's main structure
def main():
    # Initialize page state if not set
    if 'page' not in st.session_state:
        st.session_state.page = "Dashboard"
    
    # Check for special pages first
    if st.session_state.page == "project_details":
        render_project_details()
        return
    elif st.session_state.page == "edit_project":
        render_project_edit()
        return
    
    with st.sidebar:
        st.markdown(f"<h1 class='main-header'>GenomeAI 🧬</h1>", unsafe_allow_html=True)
        st.markdown(f"<p>Version {APP_VERSION}</p>", unsafe_allow_html=True)
        
        # Define valid navigation options
        nav_options = ["Dashboard", "Projects", "Data Management", 
                       "Model Training", "Analysis", "Settings"]
        
        # Ensure the current page is valid for the radio button; default to "Dashboard" if not
        current_page = st.session_state.page if st.session_state.page in nav_options else "Dashboard"
        
        page = st.radio("Navigate", nav_options, index=nav_options.index(current_page))
        st.session_state.page = page  # Update state with user selection
    
    # Render the selected page
    if st.session_state.page == "Dashboard":
        render_dashboard()
    elif st.session_state.page == "Projects":
        render_projects_page()
    elif st.session_state.page == "Data Management":
        render_data_management()
    elif st.session_state.page == "Model Training":
        render_model_training()
    elif st.session_state.page == "Analysis":
        render_analysis()
    elif st.session_state.page == "Settings":
        render_settings()

# Run the app
if __name__ == "__main__":
    main()
