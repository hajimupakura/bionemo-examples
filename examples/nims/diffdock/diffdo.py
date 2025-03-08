# app.py
import streamlit as st
import py3Dmol
import pandas as pd
import numpy as np
import tempfile
import os
import requests
from rdkit import Chem
from rdkit.Chem import Draw
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Table
from reportlab.lib.styles import getSampleStyleSheet
from stmol import showmol
from loguru import logger
from prolif.plotting.network import LigNetwork
import subprocess

# Configuration
DIFFDOCK_API_URL = "http://localhost:8000/molecular-docking/diffdock/generate"
COLOR_SCALE = [(0.8, 0, 0), (0.9, 0.6, 0), (0, 0.8, 0)]  # Red-Yellow-Green

def convert_logit_to_prob(logit):
    """Convert DiffDock logit to probability"""
    return 1 / (1 + np.exp(-logit))

def detect_pockets(pdb_path):
    """Run fpocket and parse results"""
    try:
        subprocess.run(["fpocket", "-f", pdb_path], capture_output=True, check=True)
        pocket_dir = os.path.splitext(pdb_path)[0] + "_out"
        pockets = []
        if os.path.exists(pocket_dir):
            for f in os.listdir(pocket_dir):
                if f.endswith(".pdb"):
                    pockets.append(os.path.join(pocket_dir, f))
        return sorted(pockets, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    except Exception as e:
        logger.error(f"Pocket detection failed: {e}")
        return []

def analyze_interactions(protein_file, ligand_sdf):
    """Analyze protein-ligand interactions using ProLIF"""
    from prolif.fingerprint import Fingerprint
    from prolif.utils import get_residues_near_ligand
    
    protein = Chem.MolFromPDBFile(protein_file)
    if not protein:
        logger.warning("Could not parse protein for interaction analysis.")
        return pd.DataFrame()
    
    ligand = Chem.MolFromMolBlock(ligand_sdf)
    if not ligand:
        logger.warning("Could not parse ligand for interaction analysis.")
        return pd.DataFrame()
    
    fp = Fingerprint()
    residues = get_residues_near_ligand(ligand, protein)
    fp.run_from_iterable([ligand], protein, residues=residues)
    return fp.to_dataframe()

def generate_report(protein_file, ligand_name, poses, confidences, interactions_df):
    """Generate PDF report"""
    pdf_path = f"{ligand_name}_report.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    story.append(Paragraph(f"Docking Report: {ligand_name}", styles["Title"]))
    
    # Summary Table
    data = [
        ["Metric", "Value"],
        ["Top Confidence", f"{max(confidences):.2%}"],
        ["Average Confidence", f"{np.mean(confidences):.2%}"],
        ["Poses Generated", str(len(poses))]
    ]
    story.append(Table(data))
    
    # Interaction Network (only if we have some interactions)
    if not interactions_df.empty:
        net = LigNetwork.from_ifp(interactions_df, ligand=Chem.MolFromMolBlock(poses[0]))
        net.save("interactions.html")
        story.append(Paragraph("Interaction Network", styles["Heading2"]))
        story.append(Image("interactions.html", width=400, height=300))
    else:
        story.append(Paragraph("No interactions detected or unable to parse data.", styles["Heading2"]))
    
    doc.build(story)
    return pdf_path

def main():
    st.set_page_config(page_title="DiffDock Advanced", layout="wide")
    st.title("DiffDock Advanced Docking Suite")
    
    # Sidebar Controls
    with st.sidebar:
        st.header("Input Parameters")
        protein_file = st.file_uploader("Upload Protein (PDB)", type="pdb")
        ligand_files = st.file_uploader("Upload Ligands (SDF)", type="sdf", accept_multiple_files=True)
        num_poses = st.slider("Number of Poses", 1, 20, 10)
        confidence_threshold = st.slider("Confidence Threshold (%)", 0, 100, 50)
    
    # Main Interface
    if protein_file and ligand_files:
        # Write protein to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb") as tmp_protein:
            tmp_protein.write(protein_file.getvalue())
            tmp_protein.flush()
            protein_temp_name = tmp_protein.name
        
        # Read the protein file contents once
        with open(protein_temp_name, "r") as f:
            protein_data_str = f.read()
        
        # Detect pockets
        pockets = detect_pockets(protein_temp_name)
        
        # Pocket Selection
        if pockets:
            with st.expander("Detected Binding Pockets"):
                cols = st.columns(3)
                for i, pocket in enumerate(pockets[:3]):
                    with cols[i]:
                        st.write(f"Pocket {i+1}")
                        view = py3Dmol.view(width=200, height=200)
                        view.addModel(open(pocket).read(), "pdb")
                        view.setStyle({"sphere": {"radius": 0.5}})
                        showmol(view)
        
        # Docking Execution
        if st.button("Run Docking"):
            results = []
            progress_bar = st.progress(0)
            
            for idx, ligand_file in enumerate(ligand_files):
                try:
                    ligand_str = ligand_file.getvalue().decode()
                    
                    # Submit to DiffDock (replace DIFFDOCK_API_URL with your actual endpoint)
                    response = requests.post(
                        DIFFDOCK_API_URL,
                        json={
                            "protein": protein_data_str,
                            "ligand": ligand_str,
                            "num_poses": num_poses
                        },
                        timeout=300
                    )
                    response.raise_for_status()
                    
                    docking_data = response.json()
                    
                    # Check for expected fields
                    if ("ligand_positions" not in docking_data or
                        "position_confidence" not in docking_data):
                        st.warning(f"No valid docking data returned for {ligand_file.name}")
                        continue
                    
                    poses = docking_data["ligand_positions"]
                    raw_confidences = docking_data["position_confidence"]
                    
                    # If no poses were returned, skip
                    if not poses:
                        st.warning(f"No poses returned for {ligand_file.name}")
                        continue
                    
                    confidences = [convert_logit_to_prob(logit) for logit in raw_confidences]
                    
                    # Analyze interactions for the top pose only
                    interactions_df = analyze_interactions(protein_temp_name, poses[0])
                    
                    # Store results
                    results.append({
                        "ligand": ligand_file.name,
                        "best_confidence": max(confidences),
                        "average_confidence": np.mean(confidences),
                        "interactions": len(interactions_df),
                        "top_pose": poses[0],
                        "confidences": confidences,
                        "all_poses": poses
                    })
                    
                    # Update progress
                    progress_bar.progress((idx + 1) / len(ligand_files))
                
                except Exception as e:
                    logger.error(f"Failed processing {ligand_file.name}: {e}")
                    st.error(f"Error processing {ligand_file.name}: {str(e)}")
            
            # Results Display
            st.header("Docking Results")
            df = pd.DataFrame(results)
            
            # If no rows in df, show a warning
            if df.empty:
                st.warning("No valid docking results found.")
                return
            
            # Filter by confidence
            df = df[df["best_confidence"] >= (confidence_threshold / 100)]
            
            # If filtering removes all rows, also show a warning
            if df.empty:
                st.warning("No results meet the confidence threshold.")
                return
            
            # Show a simplified table
            st.dataframe(
                df[["ligand", "best_confidence", "average_confidence", "interactions"]],
                use_container_width=True
            )
            
            # 3D Visualization Section
            st.header("3D Visualization")
            col1, col2 = st.columns([3, 1])
            
            # For demonstration, we'll visualize the first row's poses
            selected_row = df.iloc[0]
            poses = selected_row["all_poses"]
            confidences = selected_row["confidences"]
            
            with col1:
                view = py3Dmol.view(width=800, height=600)
                # Reload protein from file
                with open(protein_temp_name, "r") as prot_f:
                    view.addModel(prot_f.read(), "pdb")
                view.setStyle({"cartoon": {"color": "lightgrey"}})
                
                # Add each pose in a color gradient
                for i, pose in enumerate(poses):
                    view.addModel(pose, "sdf")
                    color = "#{:02x}{:02x}{:02x}".format(
                        int(255 * (1 - confidences[i])),
                        int(255 * confidences[i]),
                        0
                    )
                    view.setStyle({"model": -1}, {"stick": {"color": color}})
                
                view.zoomTo()
                showmol(view, height=600)
            
            with col2:
                st.subheader("Legend")
                for confidence, pose in zip(confidences, poses):
                    # Create a color pick to show the gradient
                    color = "#{:02x}{:02x}{:02x}".format(
                        int(255 * (1 - confidence)),
                        int(255 * confidence),
                        0
                    )
                    st.color_picker(f"Confidence: {confidence:.1%}", color, disabled=True)
                
                # 2D Interaction Diagram of top pose
                st.subheader("Top Pose Interactions")
                mol = Chem.MolFromMolBlock(poses[0])
                if mol:
                    img = Draw.MolToImage(mol)
                    st.image(img)
                else:
                    st.warning("Could not parse top pose for 2D visualization.")
            
            # Report Generation
            pdf_path = generate_report(
                protein_temp_name,
                selected_row["ligand"],
                poses,
                confidences,
                analyze_interactions(protein_temp_name, poses[0])
            )
            
            with open(pdf_path, "rb") as f:
                st.download_button(
                    "Download Full Report",
                    f,
                    file_name=os.path.basename(pdf_path),
                    help="Detailed PDF report with interaction analysis"
                )

if __name__ == "__main__":
    main()
