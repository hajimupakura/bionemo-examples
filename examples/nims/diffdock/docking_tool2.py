# import os
# import subprocess
# import streamlit as st
# import numpy as np
# from rdkit import Chem
# from rdkit.Chem import AllChem, Draw
# from typing import List, Tuple, Optional, Dict
# import multiprocessing
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from pathlib import Path
# import pandas as pd
# import plotly.graph_objects as go
# import plotly.express as px
# import py3Dmol
# import logging
# import tempfile
# import hashlib
# import pubchempy as pcp
# import io
# import base64
# import matplotlib.pyplot as plt
# import prolif as plf  # For interaction analysis
# import requests
# from dataclasses import dataclass

# # Set page config as the FIRST Streamlit command
# st.set_page_config(layout="wide", page_title="Advanced Molecular Docking")

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[logging.FileHandler("docking_log.log"), logging.StreamHandler()]
# )
# logger = logging.getLogger(__name__)

# # Custom CSS for modern Streamlit interface
# st.markdown("""
#     <style>
#     .main {background-color: #f8f9fa;}
#     .stButton>button {width: 100%; border-radius: 5px;}
#     .stTabs [data-baseweb="tab-list"] {gap: 10px;}
#     .stTabs [data-baseweb="tab"] {background-color: #e9ecef; border-radius: 5px; padding: 8px 16px;}
#     .stTabs [data-baseweb="tab"][aria-selected="true"] {background-color: #007bff; color: white;}
#     .sidebar .sidebar-content {background-color: #f1f3f5;}
#     </style>
# """, unsafe_allow_html=True)

# # DiffDock Server Configuration
# @dataclass
# class DockingConfig:
#     """Configuration for NVIDIA DiffDock server-based docking."""
#     base_url: str = "http://0.0.0.0:8000/"
#     num_poses: int = 10
#     time_divisions: int = 20
#     steps: int = 18
#     save_trajectory: bool = False
#     is_staged: bool = False

# class MolecularDockingPipeline:
#     """Main class for handling NVIDIA DiffDock server-based docking."""
    
#     def __init__(self, config: DockingConfig):
#         self.config = config
#         self.query_url = config.base_url + "molecular-docking/diffdock/generate"
#         self.health_check_url = config.base_url + "v1/health/ready"
    
#     def check_server_health(self) -> bool:
#         """Check if the DiffDock server is available."""
#         try:
#             response = requests.get(self.health_check_url)
#             return response.status_code == 200
#         except requests.RequestException:
#             return False
    
#     def run_docking(self, protein_file: bytes, ligand_file: bytes) -> Dict:
#         """Run molecular docking for a single ligand via DiffDock server."""
#         protein_data = protein_file.decode('utf-8')
#         ligand_data = ligand_file.decode('utf-8')
        
#         payload = {
#             "ligand": ligand_data,
#             "ligand_file_type": "sdf",
#             "protein": protein_data,
#             "num_poses": self.config.num_poses,
#             "time_divisions": self.config.time_divisions,
#             "steps": self.config.steps,
#             "save_trajectory": self.config.save_trajectory,
#             "is_staged": self.config.is_staged
#         }
        
#         try:
#             response = requests.post(
#                 self.query_url,
#                 headers={"Content-Type": "application/json"},
#                 json=payload,
#                 timeout=300
#             )
#             response.raise_for_status()
#             return response.json()
#         except requests.RequestException as e:
#             raise Exception(f"DiffDock server error: {str(e)}")

# # Dependency Check
# def check_dependencies(docking_method: str) -> Dict[str, bool]:
#     """Check if required tools for the selected docking method are available."""
#     dependencies = {'obabel': False}
#     if docking_method == "Vina":
#         dependencies['vina'] = False
#     elif docking_method == "DiffDock":
#         dependencies['requests'] = True  # Assuming requests is always available
    
#     for cmd in dependencies.keys():
#         if cmd == 'requests':
#             continue
#         try:
#             subprocess.run([cmd, '--help'], capture_output=True, check=True)
#             dependencies[cmd] = True
#         except (FileNotFoundError, subprocess.CalledProcessError):
#             logger.error(f"{cmd} not found or failed to run")
#     return dependencies

# # Input Validation
# def validate_smiles(smiles: str) -> Tuple[bool, str]:
#     """Validate SMILES string."""
#     try:
#         mol = Chem.MolFromSmiles(smiles, sanitize=True)
#         if mol is None:
#             return False, "Invalid SMILES string"
#         return True, "Valid SMILES"
#     except Exception as e:
#         return False, f"SMILES parsing error: {str(e)}"

# def validate_pdb(pdb_file: str) -> Tuple[bool, str]:
#     """Validate PDB file contents."""
#     try:
#         mol = Chem.MolFromPDBFile(pdb_file, removeHs=False)
#         if mol is None or mol.GetNumAtoms() < 50:
#             return False, "Invalid PDB or likely not a protein"
#         return True, "Valid PDB"
#     except Exception as e:
#         return False, f"PDB parsing error: {str(e)}"

# # Core Docking Functions - Vina
# def prepare_ligand_vina(smiles: str, ligand_output: str) -> Tuple[bool, str, Optional[Chem.Mol]]:
#     """Convert SMILES to 3D PDB structure for Vina."""
#     try:
#         mol = Chem.MolFromSmiles(smiles)
#         if mol is None:
#             return False, f"Invalid SMILES: {smiles}", None
#         mol = Chem.AddHs(mol)
#         embed_success = False
#         for seed in [42, 123, 987, 555]:
#             params = AllChem.ETKDGv3()
#             params.randomSeed = seed
#             if AllChem.EmbedMolecule(mol, params) >= 0:
#                 embed_success = True
#                 break
#         if not embed_success:
#             return False, "Failed to generate 3D conformation", None
#         if AllChem.MMFFOptimizeMolecule(mol, maxIters=2000) != 0:
#             if AllChem.UFFOptimizeMolecule(mol, maxIters=2000) != 0:
#                 return False, "Structure optimization failed", None
#         Chem.MolToPDBFile(mol, ligand_output)
#         return True, "Success", mol
#     except Exception as e:
#         return False, f"Error in prepare_ligand: {str(e)}", None

# def convert_to_pdbqt(input_file: str, output_file: str, is_ligand: bool = True) -> Tuple[bool, str]:
#     """Convert PDB to PDBQT format for Vina."""
#     try:
#         if not os.path.exists(input_file):
#             return False, f"Input file not found: {input_file}"
#         cmd = ["obabel", input_file, "-O", output_file]
#         if is_ligand:
#             cmd.extend(["-h", "--gen3d"])
#         else:
#             cmd.extend(["-xr"])
#         result = subprocess.run(cmd, capture_output=True, text=True, check=True)
#         if not os.path.exists(output_file):
#             return False, "Output file was not created"
#         return True, "Conversion successful"
#     except subprocess.CalledProcessError as e:
#         return False, f"Conversion failed: {e.stderr}"
#     except Exception as e:
#         return False, f"Error in conversion: {str(e)}"

# def create_vina_config(output_dir: Path, receptor_pdb: str, center: Optional[Tuple[float, float, float]] = None,
#                       size: Optional[Tuple[float, float, float]] = None, exhaustiveness: int = 8, num_modes: int = 9,
#                       energy_range: int = 3) -> Tuple[bool, str]:
#     """Create Vina configuration file."""
#     try:
#         mol = Chem.MolFromPDBFile(receptor_pdb, removeHs=False)
#         if mol is None:
#             return False, "Failed to load receptor structure"
#         conf = mol.GetConformer()
#         coords = conf.GetPositions()
#         if center is None:
#             center = np.mean(coords, axis=0)
#             size = np.ptp(coords, axis=0) + 10.0
#             size = np.maximum(size, [20.0, 20.0, 20.0])
#         else:
#             size = size or (20.0, 20.0, 20.0)
#         config_file = output_dir / "config.txt"
#         config_content = f"""
# receptor = receptor.pdbqt
# ligand = ligand.pdbqt
# center_x = {center[0]:.3f}
# center_y = {center[1]:.3f}
# center_z = {center[2]:.3f}
# size_x = {size[0]:.3f}
# size_y = {size[1]:.3f}
# size_z = {size[2]:.3f}
# exhaustiveness = {exhaustiveness}
# num_modes = {num_modes}
# energy_range = {energy_range}
# cpu = {max(1, multiprocessing.cpu_count() - 1)}
# """
#         with open(config_file, "w") as f:
#             f.write(config_content)
#         logger.info(f"Created Vina config:\n{config_content}")
#         return True, str(config_file)
#     except Exception as e:
#         return False, f"Error creating config: {str(e)}"

# def run_vina_docking(receptor_pdbqt: str, ligand_pdbqt: str, output_pdbqt: str, config_file: str) -> Tuple[bool, str, List[Dict]]:
#     """Run AutoDock Vina docking and parse all poses."""
#     try:
#         for filepath in [receptor_pdbqt, ligand_pdbqt, config_file]:
#             if not os.path.exists(filepath):
#                 return False, f"File not found: {filepath}", []
#         cmd = ["vina", "--receptor", receptor_pdbqt, "--ligand", ligand_pdbqt, "--out", output_pdbqt, "--config", config_file]
#         result = subprocess.run(cmd, capture_output=True, text=True, check=True)
#         if not os.path.exists(output_pdbqt) or os.path.getsize(output_pdbqt) == 0:
#             return False, "Docking output file issue", []
#         poses = []
#         with open(output_pdbqt, 'r') as f:
#             for line in f:
#                 if "REMARK VINA RESULT" in line:
#                     parts = line.split()
#                     poses.append({
#                         'affinity': float(parts[3]),
#                         'rmsd_lb': float(parts[4]),
#                         'rmsd_ub': float(parts[5])
#                     })
#         return True, result.stdout, poses
#     except subprocess.CalledProcessError as e:
#         return False, f"Docking failed: {e.stderr}", []
#     except Exception as e:
#         return False, f"Error in docking: {str(e)}", []

# # Core Docking Functions - DiffDock
# def prepare_ligand_diffdock(smiles: str, ligand_output: str) -> Tuple[bool, str, Optional[Chem.Mol]]:
#     """Prepare ligand SDF for DiffDock server."""
#     try:
#         mol = Chem.MolFromSmiles(smiles)
#         if mol is None:
#             return False, f"Invalid SMILES: {smiles}", None
#         mol = Chem.AddHs(mol)
#         AllChem.EmbedMolecule(mol, randomSeed=42)
#         AllChem.MMFFOptimizeMolecule(mol)
#         Chem.MolToMolFile(mol, ligand_output)  # DiffDock uses SDF
#         return True, "Success", mol
#     except Exception as e:
#         return False, f"Error in prepare_ligand_diffdock: {str(e)}", None

# def run_diffdock_docking(pipeline: MolecularDockingPipeline, receptor_pdb: bytes, ligand_sdf: bytes, 
#                          output_dir: Path) -> Tuple[bool, str, List[Dict]]:
#     """Run DiffDock docking via server."""
#     try:
#         result = pipeline.run_docking(receptor_pdb, ligand_sdf)
#         poses = []
#         for i, pose in enumerate(result.get('poses', [])):
#             pose_pdb = output_dir / f"docked_{i}.pdb"
#             with open(pose_pdb, 'w') as f:
#                 f.write(pose.get('pdb_content', ''))
#             poses.append({
#                 'affinity': pose.get('confidence', 0.0),  # Using confidence as proxy for affinity
#                 'rmsd_lb': 0.0,  # DiffDock server might not provide RMSD
#                 'rmsd_ub': 0.0
#             })
#         return True, "DiffDock server docking successful", poses
#     except Exception as e:
#         return False, f"DiffDock server docking failed: {str(e)}", []

# # Processing Functions
# def process_ligand(args: Tuple[str, Path, int, Dict, str, Optional[MolecularDockingPipeline], bytes]) -> Dict:
#     """Process a single ligand with selected docking method."""
#     smiles, output_dir, idx, params, docking_method, diffdock_pipeline, receptor_bytes = args
#     result = {
#         'smiles': smiles,
#         'affinity': None,
#         'status': 'failed',
#         'error': None,
#         'poses': [],
#         'interactions': None,
#         'mol': None,
#         'method': docking_method
#     }
#     with tempfile.TemporaryDirectory() as tmpdirname:
#         try:
#             if docking_method == "Vina":
#                 ligand_pdb = f"{tmpdirname}/ligand_{idx}.pdb"
#                 ligand_pdbqt = output_dir / f"ligand_{idx}.pdbqt"
#                 output_pdbqt = output_dir / f"docked_{idx}.pdbqt"
#                 success, msg, mol = prepare_ligand_vina(smiles, ligand_pdb)
#                 if not success:
#                     result['error'] = msg
#                     return result
#                 result['mol'] = mol
#                 success, msg = convert_to_pdbqt(ligand_pdb, str(ligand_pdbqt))
#                 if not success:
#                     result['error'] = msg
#                     return result
#                 success, msg, poses = run_vina_docking(
#                     str(output_dir / "receptor.pdbqt"), str(ligand_pdbqt), str(output_pdbqt), 
#                     str(output_dir / "config.txt")
#                 )
#             elif docking_method == "DiffDock":
#                 ligand_sdf = f"{tmpdirname}/ligand_{idx}.sdf"
#                 success, msg, mol = prepare_ligand_diffdock(smiles, ligand_sdf)
#                 if not success:
#                     result['error'] = msg
#                     return result
#                 result['mol'] = mol
#                 with open(ligand_sdf, 'rb') as f:
#                     ligand_bytes = f.read()
#                 success, msg, poses = run_diffdock_docking(diffdock_pipeline, receptor_bytes, ligand_bytes, output_dir)
            
#             if not success:
#                 result['error'] = msg
#                 return result
#             result['poses'] = poses
#             result['affinity'] = poses[0]['affinity'] if poses else None
#             result['status'] = 'success'
#             # Interaction analysis
#             try:
#                 output_file = output_dir / f"docked_{idx}.pdbqt" if docking_method == "Vina" else output_dir / f"docked_0.pdb"
#                 if output_file.exists():
#                     fp = plf.Fingerprint()
#                     mol_rec = plf.Molecule.from_mol2(Chem.MolFromPDBFile(str(output_dir / "receptor.pdb")))
#                     mol_lig = plf.Molecule.from_mol2(Chem.MolFromPDBFile(str(output_file)))
#                     interactions = fp.run(mol_rec, mol_lig)
#                     result['interactions'] = interactions
#             except Exception as e:
#                 logger.warning(f"Interaction analysis failed: {str(e)}")
#         except Exception as e:
#             result['error'] = f"Unexpected error: {str(e)}"
#     return result

# def batch_docking(smiles_list: List[str], receptor_pdb: str, receptor_bytes: bytes, params: Dict, docking_method: str, 
#                   output_dir: str = "docking_results") -> List[Dict]:
#     """Run batch docking with selected method."""
#     output_dir = Path(output_dir)
#     output_dir.mkdir(exist_ok=True)
#     deps = check_dependencies(docking_method)
#     if not all(deps.values()):
#         st.error(f"Missing dependencies for {docking_method}: {', '.join([k for k, v in deps.items() if not v])}")
#         return []
    
#     diffdock_pipeline = None
#     if docking_method == "DiffDock":
#         config = DockingConfig(
#             base_url=params.get('diffdock_url', "http://0.0.0.0:8000/"),
#             num_poses=params.get('num_modes', 10),
#             time_divisions=params.get('time_divisions', 20),
#             steps=params.get('steps', 18)
#         )
#         diffdock_pipeline = MolecularDockingPipeline(config)
#         if not diffdock_pipeline.check_server_health():
#             st.error("DiffDock server is not available. Check URL and server status.")
#             return []
#     elif docking_method == "Vina":
#         receptor_pdbqt = output_dir / "receptor.pdbqt"
#         if not Path(receptor_pdb).exists():
#             st.error(f"Receptor file {receptor_pdb} not found")
#             return []
#         success, msg = convert_to_pdbqt(receptor_pdb, str(receptor_pdbqt), is_ligand=False)
#         if not success:
#             st.error(f"Receptor preparation failed: {msg}")
#             return []
#         success, config_path = create_vina_config(
#             output_dir, receptor_pdb, params.get('center'), params.get('size'), 
#             params.get('exhaustiveness', 8), params.get('num_modes', 9), params.get('energy_range', 3)
#         )
#         if not success:
#             st.error(f"Config creation failed: {msg}")
#             return []
    
#     results = []
#     max_workers = min(multiprocessing.cpu_count(), len(smiles_list))
#     progress_bar = st.progress(0)
#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         futures = [executor.submit(process_ligand, (smiles, output_dir, idx, params, docking_method, diffdock_pipeline, receptor_bytes)) 
#                    for idx, smiles in enumerate(smiles_list)]
#         for i, future in enumerate(as_completed(futures)):
#             result = future.result()
#             results.append(result)
#             progress_bar.progress((i + 1) / len(smiles_list))
#     return results

# # Visualization Functions
# def mol_to_img(mol: Chem.Mol, size: Tuple[int, int] = (300, 300)) -> str:
#     """Convert RDKit molecule to base64 image."""
#     if mol:
#         img = Draw.MolToImage(mol, size=size)
#         buffered = io.BytesIO()
#         img.save(buffered, format="PNG")
#         return base64.b64encode(buffered.getvalue()).decode()
#     return None

# def plot_interactive_results(results: List[Dict]) -> Dict[str, go.Figure]:
#     """Create enhanced interactive plots."""
#     successful = [r for r in results if r['status'] == 'success']
#     if not successful:
#         return {}
#     plots = {}
#     # Affinity Distribution
#     affinities = [r['affinity'] for r in successful]
#     label = "Affinity (kcal/mol)" if successful[0]['method'] == "Vina" else "Confidence Score"
#     plots['affinity'] = px.histogram(x=affinities, nbins=20, title="Affinity Distribution", 
#                                      labels={'x': label, 'y': 'Count'})
#     # Pose Comparison
#     pose_data = []
#     for i, r in enumerate(successful):
#         for p in r['poses']:
#             pose_data.append({'Ligand': f"L{i+1}", 'Affinity': p['affinity'], 'RMSD': p['rmsd_lb'], 'Method': r['method']})
#     if pose_data:
#         df = pd.DataFrame(pose_data)
#         plots['pose'] = px.scatter(df, x='RMSD', y='Affinity', color='Ligand', facet_col='Method',
#                                    title="Pose Comparison", labels={'RMSD': 'RMSD (Å)', 'Affinity': 'Score'})
#     return plots

# def show_3d_view(receptor_file: str, ligand_file: str) -> str:
#     """Enhanced 3D visualization with interactions."""
#     view = py3Dmol.view(width=800, height=600)
#     with open(receptor_file, 'r') as f:
#         view.addModel(f.read(), 'pdb' if receptor_file.endswith('.pdb') else 'pdbqt')
#     view.setStyle({'model': -1}, {'cartoon': {'color': 'gray'}})
#     view.addSurface('VDW', {'opacity': 0.7, 'color': 'white'})
#     with open(ligand_file, 'r') as f:
#         view.addModel(f.read(), 'pdb' if ligand_file.endswith('.pdb') else 'pdbqt')
#     view.setStyle({'model': -1}, {'stick': {'colorscheme': 'greenCarbon'}})
#     view.zoomTo()
#     return f"<div style='height: 600px'>{view.render()}</div>"

# # Main Application
# def main():
#     st.sidebar.title("Docking Workflow")
    
#     # Sidebar Navigation
#     steps = ["Setup", "Input", "Parameters", "Run", "Results"]
#     step = st.sidebar.radio("Navigate:", steps, index=0)

#     # Step 1: Setup
#     if step == "Setup":
#         st.header("Setup Environment")
#         docking_method = st.session_state.get('docking_method', "Vina")
#         docking_method = st.selectbox("Docking Method", ["Vina", "DiffDock"], index=0 if docking_method == "Vina" else 1)
#         st.session_state['docking_method'] = docking_method
#         deps = check_dependencies(docking_method)
#         for dep, status in deps.items():
#             st.write(f"{dep}: {'✅' if status else '❌'}")
#         if not all(deps.values()):
#             st.error(f"Please install missing dependencies for {docking_method}.")
#         st.info(f"For Vina: Install Open Babel and AutoDock Vina. For DiffDock: Ensure NVIDIA DiffDock server is running at the specified URL.")
#         st.info(f"For Vina: Install Open Babel and AutoDock Vina. For DiffDock: Ensure NVIDIA DiffDock server is running at the specified URL.")

#     # Step 2: Input
#     elif step == "Input":
#         st.header("Input Structures")
#         col1, col2 = st.columns(2)
#         with col1:
#             receptor_file = st.file_uploader("Upload Receptor (PDB)", type=["pdb"])
#             if receptor_file:
#                 receptor_path = Path("docking_results/receptor.pdb")
#                 receptor_path.parent.mkdir(exist_ok=True)
#                 receptor_path.write_bytes(receptor_file.read())
#                 st.session_state['receptor_path'] = str(receptor_path)
#                 st.session_state['receptor_bytes'] = receptor_file.getvalue()
#                 if validate_pdb(str(receptor_path))[0]:
#                     st.success("Receptor validated")
#                 else:
#                     st.error("Invalid PDB file")
#         with col2:
#             input_method = st.selectbox("Ligand Input Method", ["Manual SMILES", "CSV File", "PubChem Fetch"])
#             if input_method == "Manual SMILES":
#                 ligand_input = st.text_area("Enter SMILES (one per line)", height=150)
#                 smiles_list = [s.strip() for s in ligand_input.split("\n") if s.strip()]
#             elif input_method == "CSV File":
#                 ligand_file = st.file_uploader("Upload CSV (SMILES column)", type=["csv"])
#                 if ligand_file:
#                     df = pd.read_csv(ligand_file)
#                     smiles_list = df['SMILES'].tolist()
#             else:
#                 compound_name = st.text_input("Enter Compound Name")
#                 if compound_name and st.button("Fetch from PubChem"):
#                     smiles = pcp.get_compounds(compound_name, 'name')[0].canonical_smiles
#                     st.session_state['smiles_list'] = [smiles]
#                     st.success(f"Fetched: {smiles}")
#                 smiles_list = st.session_state.get('smiles_list', [])
#             st.session_state['smiles_list'] = smiles_list

#     # Step 3: Parameters
#     elif step == "Parameters":
#         st.header("Docking Parameters")
#         docking_method = st.session_state.get('docking_method', "Vina")
#         preset = st.selectbox("Preset", ["Quick Screen", "High Accuracy", "Custom"])
#         params = {}
#         if preset == "Quick Screen":
#             params = {'exhaustiveness': 4, 'num_modes': 5, 'energy_range': 2} if docking_method == "Vina" else {'num_modes': 5, 'time_divisions': 10, 'steps': 10}
#         elif preset == "High Accuracy":
#             params = {'exhaustiveness': 16, 'num_modes': 20, 'energy_range': 5} if docking_method == "Vina" else {'num_modes': 20, 'time_divisions': 30, 'steps': 25}
#         with st.expander("Advanced Options", expanded=preset == "Custom"):
#             if docking_method == "Vina":
#                 params['exhaustiveness'] = st.slider("Exhaustiveness", 1, 32, params.get('exhaustiveness', 8))
#                 params['num_modes'] = st.slider("Number of Modes", 1, 50, params.get('num_modes', 9))
#                 params['energy_range'] = st.slider("Energy Range (kcal/mol)", 1, 10, params.get('energy_range', 3))
#                 use_custom_center = st.checkbox("Specify Binding Site")
#                 if use_custom_center:
#                     col1, col2, col3 = st.columns(3)
#                     params['center'] = (col1.number_input("X", value=0.0), col2.number_input("Y", value=0.0), 
#                                         col3.number_input("Z", value=0.0))
#                     params['size'] = (st.number_input("Box Size X", value=20.0), st.number_input("Y", value=20.0), 
#                                       st.number_input("Z", value=20.0))
#             elif docking_method == "DiffDock":
#                 params['diffdock_url'] = st.text_input("DiffDock Server URL", value="http://0.0.0.0:8000/")
#                 params['num_modes'] = st.slider("Number of Poses", 1, 50, params.get('num_modes', 10))
#                 params['time_divisions'] = st.slider("Time Divisions", 10, 50, params.get('time_divisions', 20))
#                 params['steps'] = st.slider("Steps", 10, 50, params.get('steps', 18))
#         st.session_state['params'] = params

#     # Step 4: Run
#     elif step == "Run":
#         st.header("Run Docking")
#         docking_method = st.session_state.get('docking_method', "Vina")
#         if 'receptor_path' not in st.session_state or 'smiles_list' not in st.session_state or 'receptor_bytes' not in st.session_state:
#             st.error("Complete Setup, Input, and Parameters first.")
#         else:
#             if st.button("Start Docking", type="primary"):
#                 results = batch_docking(st.session_state['smiles_list'], st.session_state['receptor_path'], 
#                                         st.session_state['receptor_bytes'], st.session_state.get('params', {}), docking_method)
#                 st.session_state['results'] = results
#                 st.success("Docking completed!")

#     # Step 5: Results
#     elif step == "Results":
#         st.header("Docking Results")
#         if 'results' not in st.session_state or not st.session_state['results']:
#             st.warning("No results available. Run docking first.")
#         else:
#             results = st.session_state['results']
#             successful = [r for r in results if r['status'] == 'success']
#             tabs = st.tabs(["Overview", "2D Analysis", "3D View", "Interactions"])
            
#             with tabs[0]:  # Overview
#                 df = pd.DataFrame([{
#                     'SMILES': r['smiles'], 'Affinity': r['affinity'], 'Poses': len(r['poses']),
#                     'Method': r['method'], 'Status': r['status'], 'Error': r['error'] or 'None'
#                 } for r in results])
#                 st.dataframe(df)
#                 for i, r in enumerate(successful[:5]):
#                     st.image(f"data:image/png;base64,{mol_to_img(r['mol'])}", 
#                              caption=f"Ligand {i+1}: {r['affinity']:.2f} ({r['method']})")

#             with tabs[1]:  # 2D Analysis
#                 plots = plot_interactive_results(results)
#                 for name, fig in plots.items():
#                     st.plotly_chart(fig, use_container_width=True)

#             with tabs[2]:  # 3D View
#                 ligand_idx = st.selectbox("Select Ligand", range(len(successful)), 
#                                           format_func=lambda i: f"Ligand {i+1}: {successful[i]['affinity']:.2f} ({successful[i]['method']})")
#                 if successful:
#                     receptor_file = str(Path("docking_results/receptor.pdbqt" if successful[ligand_idx]['method'] == "Vina" else "docking_results/receptor.pdb"))
#                     ligand_file = str(Path(f"docking_results/docked_{ligand_idx}.pdbqt" if successful[ligand_idx]['method'] == "Vina" else f"docking_results/docked_{ligand_idx}.pdb"))
#                     html = show_3d_view(receptor_file, ligand_file)
#                     st.components.v1.html(html, height=600)

#             with tabs[3]:  # Interactions
#                 for i, r in enumerate(successful):
#                     if r.get('interactions'):
#                         with st.expander(f"Ligand {i+1} ({r['method']})"):
#                             st.pyplot(r['interactions'].plot())
#                             st.download_button("Download Results", 
#                                               data=pd.DataFrame(r['poses']).to_csv().encode(), 
#                                               file_name=f"ligand_{i+1}_{r['method']}_results.csv")

# if __name__ == "__main__":
#     main()

import os
import subprocess
import streamlit as st
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from typing import List, Tuple, Optional, Dict
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import py3Dmol
import logging
import tempfile
import hashlib
import pubchempy as pcp
import io
import base64
import matplotlib.pyplot as plt
import prolif as plf
import requests
from dataclasses import dataclass
import shutil
import zipfile
import nglview as nv

# Set page config as the FIRST Streamlit command
st.set_page_config(layout="wide", page_title="Advanced Molecular Docking")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("docking_log.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Custom CSS for modern Streamlit interface
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stButton>button {width: 100%; border-radius: 5px;}
    .stTabs [data-baseweb="tab-list"] {gap: 10px;}
    .stTabs [data-baseweb="tab"] {background-color: #e9ecef; border-radius: 5px; padding: 8px 16px;}
    .stTabs [data-baseweb="tab"][aria-selected="true"] {background-color: #007bff; color: white;}
    .sidebar .sidebar-content {background-color: #f1f3f5;}
    </style>
""", unsafe_allow_html=True)

# DiffDock Server Configuration
@dataclass
class DockingConfig:
    base_url: str = "http://0.0.0.0:8000/"
    num_poses: int = 10
    time_divisions: int = 20
    steps: int = 18
    save_trajectory: bool = False
    is_staged: bool = False

class MolecularDockingPipeline:
    def __init__(self, config: DockingConfig):
        self.config = config
        self.query_url = config.base_url + "molecular-docking/diffdock/generate"
        self.health_check_url = config.base_url + "v1/health/ready"
    
    def check_server_health(self) -> bool:
        try:
            response = requests.get(self.health_check_url)
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def run_docking(self, protein_file: bytes, ligand_file: bytes) -> Dict:
        protein_data = protein_file.decode('utf-8')
        ligand_data = ligand_file.decode('utf-8')
        
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
        
        try:
            response = requests.post(
                self.query_url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=300
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise Exception(f"DiffDock server error: {str(e)}")

# Dependency Check
def check_dependencies(docking_method: str) -> Dict[str, bool]:
    dependencies = {'obabel': False}
    if docking_method == "Vina":
        dependencies['vina'] = False
    elif docking_method == "DiffDock":
        dependencies['requests'] = True
    
    for cmd in dependencies.keys():
        if cmd == 'requests':
            continue
        try:
            subprocess.run([cmd, '--help'], capture_output=True, check=True)
            dependencies[cmd] = True
        except (FileNotFoundError, subprocess.CalledProcessError):
            logger.error(f"{cmd} not found or failed to run")
    return dependencies

# Input Validation
def validate_smiles(smiles: str) -> Tuple[bool, str]:
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        if mol is None:
            return False, "Invalid SMILES string"
        return True, "Valid SMILES"
    except Exception as e:
        return False, f"SMILES parsing error: {str(e)}"

def validate_pdb(pdb_file: str) -> Tuple[bool, str]:
    try:
        mol = Chem.MolFromPDBFile(pdb_file, removeHs=False)
        if mol is None or mol.GetNumAtoms() < 50:
            return False, "Invalid PDB or likely not a protein"
        return True, "Valid PDB"
    except Exception as e:
        return False, f"PDB parsing error: {str(e)}"

# Core Docking Functions - Vina
def prepare_ligand_vina(smiles: str, ligand_output: str) -> Tuple[bool, str, Optional[Chem.Mol]]:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, f"Invalid SMILES: {smiles}", None
        mol = Chem.AddHs(mol)
        embed_success = False
        for seed in [42, 123, 987, 555]:
            params = AllChem.ETKDGv3()
            params.randomSeed = seed
            if AllChem.EmbedMolecule(mol, params) >= 0:
                embed_success = True
                break
        if not embed_success:
            return False, "Failed to generate 3D conformation", None
        if AllChem.MMFFOptimizeMolecule(mol, maxIters=2000) != 0:
            if AllChem.UFFOptimizeMolecule(mol, maxIters=2000) != 0:
                return False, "Structure optimization failed", None
        Chem.MolToPDBFile(mol, ligand_output)
        return True, "Success", mol
    except Exception as e:
        return False, f"Error in prepare_ligand: {str(e)}", None

def convert_to_pdbqt(input_file: str, output_file: str, is_ligand: bool = True) -> Tuple[bool, str]:
    try:
        if not os.path.exists(input_file):
            return False, f"Input file not found: {input_file}"
        cmd = ["obabel", input_file, "-O", output_file]
        if is_ligand:
            cmd.extend(["-h", "--gen3d"])
        else:
            cmd.extend(["-xr"])
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if not os.path.exists(output_file):
            return False, "Output file was not created"
        return True, "Conversion successful"
    except subprocess.CalledProcessError as e:
        return False, f"Conversion failed: {e.stderr}"
    except Exception as e:
        return False, f"Error in conversion: {str(e)}"

def create_vina_config(output_dir: Path, receptor_pdb: str, center: Optional[Tuple[float, float, float]] = None,
                      size: Optional[Tuple[float, float, float]] = None, exhaustiveness: int = 8, num_modes: int = 9,
                      energy_range: int = 3) -> Tuple[bool, str]:
    try:
        mol = Chem.MolFromPDBFile(receptor_pdb, removeHs=False)
        if mol is None:
            return False, "Failed to load receptor structure"
        conf = mol.GetConformer()
        coords = conf.GetPositions()
        if center is None:
            center = np.mean(coords, axis=0)
            size = np.ptp(coords, axis=0) + 10.0
            size = np.maximum(size, [20.0, 20.0, 20.0])
        else:
            size = size or (20.0, 20.0, 20.0)
        config_file = output_dir / "config.txt"
        config_content = f"""
receptor = receptor.pdbqt
ligand = ligand.pdbqt
center_x = {center[0]:.3f}
center_y = {center[1]:.3f}
center_z = {center[2]:.3f}
size_x = {size[0]:.3f}
size_y = {size[1]:.3f}
size_z = {size[2]:.3f}
exhaustiveness = {exhaustiveness}
num_modes = {num_modes}
energy_range = {energy_range}
cpu = {max(1, multiprocessing.cpu_count() - 1)}
"""
        with open(config_file, "w") as f:
            f.write(config_content)
        logger.info(f"Created Vina config:\n{config_content}")
        return True, str(config_file)
    except Exception as e:
        return False, f"Error creating config: {str(e)}"

def split_vina_poses(pdbqt_file: str, output_dir: Path, idx: int) -> List[str]:
    pose_files = []
    current_pose = []
    pose_num = 0
    with open(pdbqt_file, 'r') as f:
        for line in f:
            if "MODEL" in line:
                current_pose = [line]
            elif "ENDMDL" in line:
                current_pose.append(line)
                pose_file = output_dir / f"docked_ligand_{idx}_pose_{pose_num}.pdbqt"
                with open(pose_file, 'w') as pf:
                    pf.write("".join(current_pose))
                pose_files.append(str(pose_file))
                pose_num += 1
            else:
                current_pose.append(line)
    return pose_files

def run_vina_docking(receptor_pdbqt: str, ligand_pdbqt: str, output_pdbqt: str, config_file: str) -> Tuple[bool, str, List[Dict]]:
    try:
        for filepath in [receptor_pdbqt, ligand_pdbqt, config_file]:
            if not os.path.exists(filepath):
                return False, f"File not found: {filepath}", []
        cmd = ["vina", "--receptor", receptor_pdbqt, "--ligand", ligand_pdbqt, "--out", output_pdbqt, "--config", config_file]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if not os.path.exists(output_pdbqt) or os.path.getsize(output_pdbqt) == 0:
            return False, "Docking output file issue", []
        poses = []
        with open(output_pdbqt, 'r') as f:
            for line in f:
                if "REMARK VINA RESULT" in line:
                    parts = line.split()
                    poses.append({
                        'affinity': float(parts[3]),
                        'rmsd_lb': float(parts[4]),
                        'rmsd_ub': float(parts[5])
                    })
        return True, result.stdout, poses
    except subprocess.CalledProcessError as e:
        return False, f"Docking failed: {e.stderr}", []
    except Exception as e:
        return False, f"Error in docking: {str(e)}", []

# Core Docking Functions - DiffDock
def prepare_ligand_diffdock(smiles: str, ligand_output: str) -> Tuple[bool, str, Optional[Chem.Mol]]:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, f"Invalid SMILES: {smiles}", None
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        Chem.MolToMolFile(mol, ligand_output)
        return True, "Success", mol
    except Exception as e:
        return False, f"Error in prepare_ligand_diffdock: {str(e)}", None

def run_diffdock_docking(pipeline: MolecularDockingPipeline, receptor_pdb: bytes, ligand_sdf: bytes, 
                         output_dir: Path, ligand_idx: int) -> Tuple[bool, str, List[Dict]]:
    try:
        result = pipeline.run_docking(receptor_pdb, ligand_sdf)
        poses = []
        for i, pose in enumerate(result.get('poses', [])):
            pose_pdb = output_dir / f"docked_ligand_{ligand_idx}_pose_{i}.pdb"
            with open(pose_pdb, 'w') as f:
                f.write(pose.get('pdb_content', ''))
            poses.append({
                'affinity': pose.get('confidence', 0.0),
                'rmsd_lb': 0.0,
                'rmsd_ub': 0.0,
                'pose_index': i
            })
        logger.info(f"DiffDock returned {len(poses)} poses for ligand {ligand_idx}")
        return True, "DiffDock server docking successful", poses
    except Exception as e:
        return False, f"DiffDock server docking failed: {str(e)}", []

# Processing Functions
def process_ligand(args: Tuple[str, Path, int, Dict, str, Optional[MolecularDockingPipeline], bytes]) -> Dict:
    smiles, output_dir, idx, params, docking_method, diffdock_pipeline, receptor_bytes = args
    result = {
        'smiles': smiles,
        'affinity': None,
        'status': 'failed',
        'error': None,
        'poses': [],
        'interactions': None,
        'mol': None,
        'method': docking_method,
        'pose_files': []
    }
    with tempfile.TemporaryDirectory() as tmpdirname:
        try:
            if docking_method == "Vina":
                ligand_pdb = f"{tmpdirname}/ligand_{idx}.pdb"
                ligand_pdbqt = output_dir / f"ligand_{idx}.pdbqt"
                output_pdbqt = output_dir / f"docked_ligand_{idx}.pdbqt"
                success, msg, mol = prepare_ligand_vina(smiles, ligand_pdb)
                if not success:
                    result['error'] = msg
                    return result
                result['mol'] = mol
                success, msg = convert_to_pdbqt(ligand_pdb, str(ligand_pdbqt))
                if not success:
                    result['error'] = msg
                    return result
                success, msg, poses = run_vina_docking(
                    str(output_dir / "receptor.pdbqt"), str(ligand_pdbqt), str(output_pdbqt), 
                    str(output_dir / "config.txt")
                )
                if success:
                    result['pose_files'] = split_vina_poses(output_pdbqt, output_dir, idx)
            elif docking_method == "DiffDock":
                ligand_sdf = f"{tmpdirname}/ligand_{idx}.sdf"
                success, msg, mol = prepare_ligand_diffdock(smiles, ligand_sdf)
                if not success:
                    result['error'] = msg
                    return result
                result['mol'] = mol
                with open(ligand_sdf, 'rb') as f:
                    ligand_bytes = f.read()
                success, msg, poses = run_diffdock_docking(diffdock_pipeline, receptor_bytes, ligand_bytes, output_dir, idx)
                if success:
                    result['pose_files'] = [str(output_dir / f"docked_ligand_{idx}_pose_{i}.pdb") for i in range(len(poses))]
            
            if not success:
                result['error'] = msg
                return result
            result['poses'] = poses
            result['affinity'] = poses[0]['affinity'] if poses else None
            result['status'] = 'success'
            # Interaction analysis for the top pose
            if result['pose_files']:
                try:
                    fp = plf.Fingerprint()
                    rdkit_rec = Chem.MolFromPDBFile(str(output_dir / "receptor.pdb"), sanitize=True, removeHs=False)
                    if rdkit_rec is None:
                        raise ValueError("Failed to load receptor.pdb with RDKit")
                    mol_rec = plf.Molecule.from_rdkit(rdkit_rec)

                    if docking_method == "Vina":
                        ligand_pdb_temp = f"{tmpdirname}/ligand_{idx}_pose_0.pdb"
                        subprocess.run(["obabel", result['pose_files'][0], "-O", ligand_pdb_temp], check=True)
                        rdkit_lig = Chem.MolFromPDBFile(ligand_pdb_temp, sanitize=True, removeHs=False)
                    else:  # DiffDock
                        rdkit_lig = Chem.MolFromPDBFile(result['pose_files'][0], sanitize=True, removeHs=False)
                    if rdkit_lig is None:
                        raise ValueError(f"Failed to load ligand file {result['pose_files'][0]} with RDKit")
                    mol_lig = plf.Molecule.from_rdkit(rdkit_lig)

                    fp.run_from_mol(mol_lig, mol_rec)  # Corrected to run_from_mol
                    result['interactions'] = fp
                except Exception as e:
                    logger.warning(f"Interaction analysis failed for ligand {idx}: {str(e)}")
        except Exception as e:
            result['error'] = f"Unexpected error: {str(e)}"
    return result

def batch_docking(smiles_list: List[str], receptor_pdb: str, receptor_bytes: bytes, params: Dict, docking_method: str, 
                  output_dir: str = "docking_results", clear_previous: bool = False) -> List[Dict]:
    output_dir = Path(output_dir)
    if clear_previous and output_dir.exists():
        shutil.rmtree(output_dir)
        logger.info("Cleared previous docking results")
    output_dir.mkdir(exist_ok=True)

    # Ensure receptor file is written from bytes
    receptor_path = output_dir / "receptor.pdb"
    with open(receptor_path, 'wb') as f:
        f.write(receptor_bytes)
    if not receptor_path.exists():
        st.error(f"Failed to write receptor file to {receptor_path}")
        return []

    deps = check_dependencies(docking_method)
    if not all(deps.values()):
        st.error(f"Missing dependencies for {docking_method}: {', '.join([k for k, v in deps.items() if not v])}")
        return []
    
    diffdock_pipeline = None
    if docking_method == "DiffDock":
        config = DockingConfig(
            base_url=params.get('diffdock_url', "http://0.0.0.0:8000/"),
            num_poses=params.get('num_modes', 10),
            time_divisions=params.get('time_divisions', 20),
            steps=params.get('steps', 18)
        )
        diffdock_pipeline = MolecularDockingPipeline(config)
        if not diffdock_pipeline.check_server_health():
            st.error("DiffDock server is not available. Check URL and server status.")
            return []
    elif docking_method == "Vina":
        receptor_pdbqt = output_dir / "receptor.pdbqt"
        success, msg = convert_to_pdbqt(str(receptor_path), str(receptor_pdbqt), is_ligand=False)
        if not success:
            st.error(f"Receptor preparation failed: {msg}")
            return []
        success, config_path = create_vina_config(
            output_dir, str(receptor_path), params.get('center'), params.get('size'), 
            params.get('exhaustiveness', 8), params.get('num_modes', 9), params.get('energy_range', 3)
        )
        if not success:
            st.error(f"Config creation failed: {msg}")
            return []
    
    results = []
    max_workers = min(multiprocessing.cpu_count(), len(smiles_list))
    progress_bar = st.progress(0)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_ligand, (smiles, output_dir, idx, params, docking_method, diffdock_pipeline, receptor_bytes)) 
                   for idx, smiles in enumerate(smiles_list)]
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            results.append(result)
            progress_bar.progress((i + 1) / len(smiles_list))
    return results

# Visualization Functions
def mol_to_img(mol: Chem.Mol, size: Tuple[int, int] = (300, 300)) -> str:
    if mol:
        img = Draw.MolToImage(mol, size=size)
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    return None

def plot_interactive_results(results: List[Dict]) -> Dict[str, go.Figure]:
    successful = [r for r in results if r['status'] == 'success']
    if not successful:
        return {}
    plots = {}
    affinities = [r['affinity'] for r in successful]
    label = "Affinity (kcal/mol)" if successful[0]['method'] == "Vina" else "Confidence Score"
    plots['affinity'] = px.histogram(x=affinities, nbins=20, title="Affinity Distribution", 
                                     labels={'x': label, 'y': 'Count'})
    pose_data = []
    for i, r in enumerate(successful):
        for p in r['poses']:
            pose_data.append({'Ligand': f"L{i+1}", 'Affinity': p['affinity'], 'RMSD': p['rmsd_lb'], 'Method': r['method'], 'Pose': p.get('pose_index', 0)})
    if pose_data:
        df = pd.DataFrame(pose_data)
        plots['pose'] = px.scatter(df, x='RMSD', y='Affinity', color='Ligand', facet_col='Method',
                                   title="Pose Comparison", labels={'RMSD': 'RMSD (Å)', 'Affinity': 'Score'}, hover_data=['Pose'])
    return plots

def show_3d_view(receptor_file: str, ligand_file: str) -> str:
    view = py3Dmol.view(width=800, height=600)
    try:
        with open(receptor_file, 'r') as f:
            receptor_content = f.read()
            if not receptor_content.strip():
                return "<div>Receptor file is empty</div>"
            view.addModel(receptor_content, 'pdb' if receptor_file.endswith('.pdb') else 'pdbqt')
        view.setStyle({'model': -1}, {'cartoon': {'color': 'gray'}})
        view.addSurface('VDW', {'opacity': 0.7, 'color': 'white'})

        with open(ligand_file, 'r') as f:
            ligand_content = f.read()
            if not ligand_content.strip():
                return "<div>Ligand file is empty</div>"
            view.addModel(ligand_content, 'pdb' if ligand_file.endswith('.pdb') else 'pdbqt')
        view.setStyle({'model': -1}, {'stick': {'colorscheme': 'greenCarbon'}})
        view.zoomTo()
        return f"<div style='height: 600px'>{view.show()}</div>"
    except Exception as e:
        return f"<div>Error rendering 3D view: {str(e)}</div>"

def show_nglview(receptor_file: str, ligand_file: str) -> str:
    try:
        view = nv.NGLWidget()
        with open(receptor_file, 'r') as f:
            receptor_content = f.read()
            if not receptor_content.strip():
                return "<div>Receptor file is empty</div>"
        view.add_component(receptor_file, default_representation=False)
        view.add_representation('cartoon', selection='protein', color='grey')

        with open(ligand_file, 'r') as f:
            ligand_content = f.read()
            if not ligand_content.strip():
                return "<div>Ligand file is empty</div>"
        view.add_component(ligand_file, default_representation=False)
        view.add_representation('stick', selection='all', color='green')

        view.center()
        view.control.zoom(0.8)

        temp_html = tempfile.NamedTemporaryFile(delete=False, suffix='.html')
        nv.write_html(temp_html.name, view)
        with open(temp_html.name, 'r') as f:
            html = f.read()
        os.unlink(temp_html.name)
        return f"<div style='height: 600px'>{html}</div>"
    except Exception as e:
        return f"<div>Error rendering NGLView: {str(e)}</div>"

def zip_directory(directory: Path) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                zip_file.write(file_path, os.path.relpath(file_path, directory))
    buffer.seek(0)
    return buffer.getvalue()

# Main Application
def main():
    st.sidebar.title("Docking Workflow")
    
    steps = ["Setup", "Input", "Parameters", "Run", "Results"]
    step = st.sidebar.radio("Navigate:", steps, index=0)

    if step == "Setup":
        st.header("Setup Environment")
        docking_method = st.session_state.get('docking_method', "Vina")
        docking_method = st.selectbox("Docking Method", ["Vina", "DiffDock"], index=0 if docking_method == "Vina" else 1)
        st.session_state['docking_method'] = docking_method
        deps = check_dependencies(docking_method)
        for dep, status in deps.items():
            st.write(f"{dep}: {'✅' if status else '❌'}")
        if not all(deps.values()):
            st.error(f"Please install missing dependencies for {docking_method}.")
        st.info(f"For Vina: Install Open Babel and AutoDock Vina. For DiffDock: Ensure NVIDIA DiffDock server is running at the specified URL.")

    elif step == "Input":
        st.header("Input Structures")
        col1, col2 = st.columns(2)
        with col1:
            receptor_file = st.file_uploader("Upload Receptor (PDB)", type=["pdb"])
            if receptor_file:
                receptor_path = Path("docking_results/receptor.pdb")
                receptor_path.parent.mkdir(exist_ok=True)
                receptor_path.write_bytes(receptor_file.read())
                st.session_state['receptor_path'] = str(receptor_path)
                st.session_state['receptor_bytes'] = receptor_file.getvalue()
                if receptor_path.exists():
                    st.success(f"Receptor file saved at {receptor_path}")
                else:
                    st.error(f"Failed to save receptor file at {receptor_path}")
                if validate_pdb(str(receptor_path))[0]:
                    st.success("Receptor validated")
                else:
                    st.error("Invalid PDB file")
        with col2:
            input_method = st.selectbox("Ligand Input Method", ["Manual SMILES", "CSV File", "PubChem Fetch"])
            if input_method == "Manual SMILES":
                ligand_input = st.text_area("Enter SMILES (one per line)", height=150)
                smiles_list = [s.strip() for s in ligand_input.split("\n") if s.strip()]
            elif input_method == "CSV File":
                ligand_file = st.file_uploader("Upload CSV (SMILES column)", type=["csv"])
                if ligand_file:
                    df = pd.read_csv(ligand_file)
                    smiles_list = df['SMILES'].tolist()
            else:
                compound_name = st.text_input("Enter Compound Name")
                if compound_name and st.button("Fetch from PubChem"):
                    smiles = pcp.get_compounds(compound_name, 'name')[0].canonical_smiles
                    st.session_state['smiles_list'] = [smiles]
                    st.success(f"Fetched: {smiles}")
                smiles_list = st.session_state.get('smiles_list', [])
            st.session_state['smiles_list'] = smiles_list

    elif step == "Parameters":
        st.header("Docking Parameters")
        docking_method = st.session_state.get('docking_method', "Vina")
        preset = st.selectbox("Preset", ["Quick Screen", "High Accuracy", "Custom"])
        params = {}
        if preset == "Quick Screen":
            params = {'exhaustiveness': 4, 'num_modes': 5, 'energy_range': 2} if docking_method == "Vina" else {'num_modes': 5, 'time_divisions': 10, 'steps': 10}
        elif preset == "High Accuracy":
            params = {'exhaustiveness': 16, 'num_modes': 20, 'energy_range': 5} if docking_method == "Vina" else {'num_modes': 20, 'time_divisions': 30, 'steps': 25}
        with st.expander("Advanced Options", expanded=preset == "Custom"):
            if docking_method == "Vina":
                params['exhaustiveness'] = st.slider("Exhaustiveness", 1, 32, params.get('exhaustiveness', 8))
                params['num_modes'] = st.slider("Number of Modes", 1, 50, params.get('num_modes', 9))
                params['energy_range'] = st.slider("Energy Range (kcal/mol)", 1, 10, params.get('energy_range', 3))
                use_custom_center = st.checkbox("Specify Binding Site")
                if use_custom_center:
                    col1, col2, col3 = st.columns(3)
                    params['center'] = (col1.number_input("X", value=0.0), col2.number_input("Y", value=0.0), 
                                        col3.number_input("Z", value=0.0))
                    params['size'] = (st.number_input("Box Size X", value=20.0), st.number_input("Y", value=20.0), 
                                      st.number_input("Z", value=20.0))
            elif docking_method == "DiffDock":
                params['diffdock_url'] = st.text_input("DiffDock Server URL", value="http://0.0.0.0:8000/")
                params['num_modes'] = st.slider("Number of Poses", 1, 50, params.get('num_modes', 10))
                params['time_divisions'] = st.slider("Time Divisions", 10, 50, params.get('time_divisions', 20))
                params['steps'] = st.slider("Steps", 10, 50, params.get('steps', 18))
        st.session_state['params'] = params

    elif step == "Run":
        st.header("Run Docking")
        docking_method = st.session_state.get('docking_method', "Vina")
        if 'receptor_path' not in st.session_state or 'smiles_list' not in st.session_state or 'receptor_bytes' not in st.session_state:
            st.error("Complete Setup, Input, and Parameters first.")
        else:
            clear_previous = st.checkbox("Clear Previous Results Before Docking", value=False)
            if st.button("Start Docking", type="primary"):
                results = batch_docking(
                    st.session_state['smiles_list'], 
                    st.session_state['receptor_path'], 
                    st.session_state['receptor_bytes'], 
                    st.session_state.get('params', {}), 
                    docking_method,
                    clear_previous=clear_previous
                )
                st.session_state['results'] = results
                st.success("Docking completed!")

    elif step == "Results":
        st.header("Docking Results")
        if 'results' not in st.session_state or not st.session_state['results']:
            st.warning("No results available. Run docking first.")
        else:
            results = st.session_state['results']
            successful = [r for r in results if r['status'] == 'success']
            tabs = st.tabs(["Overview", "2D Analysis", "3D View", "Interactions", "Interactive View"])
            
            with tabs[0]:  # Overview
                df = pd.DataFrame([{
                    'SMILES': r['smiles'], 'Affinity': r['affinity'], 'Poses': len(r['poses']),
                    'Method': r['method'], 'Status': r['status'], 'Error': r['error'] or 'None'
                } for r in results])
                st.dataframe(df)
                for i, r in enumerate(successful[:5]):
                    st.image(f"data:image/png;base64,{mol_to_img(r['mol'])}", 
                             caption=f"Ligand {i+1}: {r['affinity']:.2f} ({r['method']})")
                st.download_button("Download All Results (ZIP)", 
                                   data=zip_directory(Path("docking_results")), 
                                   file_name="docking_results.zip")

            with tabs[1]:  # 2D Analysis
                plots = plot_interactive_results(results)
                for name, fig in plots.items():
                    st.plotly_chart(fig, use_container_width=True)

            with tabs[2]:  # 3D View
                ligand_idx = st.selectbox("Select Ligand", range(len(successful)), 
                                          format_func=lambda i: f"Ligand {i+1}: {successful[i]['affinity']:.2f} ({successful[i]['method']})")
                if successful:
                    receptor_file = str(Path("docking_results/receptor.pdbqt" if successful[ligand_idx]['method'] == "Vina" else "docking_results/receptor.pdb"))
                    pose_files = successful[ligand_idx]['pose_files']
                    pose_idx = st.selectbox("Select Pose", range(len(pose_files)), 
                                            format_func=lambda i: f"Pose {i}: {successful[ligand_idx]['poses'][i]['affinity']:.2f}")
                    ligand_file = pose_files[pose_idx]
                    if os.path.exists(receptor_file) and os.path.exists(ligand_file):
                        html = show_3d_view(receptor_file, ligand_file)
                        st.components.v1.html(html, height=600)
                    else:
                        st.error(f"Files missing: {receptor_file}, {ligand_file}")

            with tabs[3]:  # Interactions
                for i, r in enumerate(successful):
                    if r.get('interactions'):
                        with st.expander(f"Ligand {i+1} ({r['method']})"):
                            fp = r['interactions']
                            df = fp.to_dataframe()
                            if not df.empty:
                                fig, ax = plt.subplots()
                                df.sum().plot(kind='bar', ax=ax)
                                ax.set_title(f"Interaction Counts for Ligand {i+1}")
                                st.pyplot(fig)
                            else:
                                st.write("No interactions detected")
                            st.download_button("Download Results", 
                                              data=pd.DataFrame(r['poses']).to_csv().encode(), 
                                              file_name=f"ligand_{i+1}_{r['method']}_results.csv")

            with tabs[4]:  # Interactive View
                st.subheader("Interactive 3D Visualization with NGLView")
                ligand_idx = st.selectbox("Select Ligand", range(len(successful)), 
                                          format_func=lambda i: f"Ligand {i+1}: {successful[i]['affinity']:.2f} ({successful[i]['method']})", key="ngl_ligand")
                if successful:
                    receptor_file = str(Path("docking_results/receptor.pdbqt" if successful[ligand_idx]['method'] == "Vina" else "docking_results/receptor.pdb"))
                    pose_files = successful[ligand_idx]['pose_files']
                    pose_idx = st.selectbox("Select Pose", range(len(pose_files)), 
                                            format_func=lambda i: f"Pose {i}: {successful[ligand_idx]['poses'][i]['affinity']:.2f}", key="ngl_pose")
                    ligand_file = pose_files[pose_idx]
                    if os.path.exists(receptor_file) and os.path.exists(ligand_file):
                        html = show_nglview(receptor_file, ligand_file)
                        st.components.v1.html(html, height=600)
                    else:
                        st.error(f"Files missing: {receptor_file}, {ligand_file}")

if __name__ == "__main__":
    main()