# import os
# import subprocess
# import streamlit as st
# import numpy as np
# from rdkit import Chem
# from rdkit.Chem import AllChem
# from typing import List, Tuple, Optional, Dict
# import multiprocessing
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from pathlib import Path
# import pandas as pd
# import plotly.graph_objects as go
# import py3Dmol
# import logging
# from datetime import datetime
# import matplotlib.pyplot as plt

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# def check_dependencies() -> Dict[str, bool]:
#     """Check if required external tools are available."""
#     dependencies = {
#         'obabel': False,
#         'vina': False,
#     }
    
#     for cmd in dependencies.keys():
#         try:
#             subprocess.run([cmd, '--help'], capture_output=True)
#             dependencies[cmd] = True
#         except FileNotFoundError:
#             logger.error(f"{cmd} not found in PATH")
    
#     return dependencies

# def prepare_ligand(smiles: str, ligand_output: str) -> Tuple[bool, str]:
#     """Convert SMILES to 3D PDB structure."""
#     try:
#         mol = Chem.MolFromSmiles(smiles)
#         if mol is None:
#             return False, f"Invalid SMILES: {smiles}"
            
#         mol = Chem.AddHs(mol)
        
#         # Try different 3D conformer generation methods
#         embed_success = False
#         for seed in [42, 123, 987, 555]:
#             params = AllChem.ETKDGv3()
#             params.randomSeed = seed
#             params.useSmallRingTorsions = True
#             params.useBasicKnowledge = True
#             params.enforceChirality = True
            
#             if AllChem.EmbedMolecule(mol, params) >= 0:
#                 embed_success = True
#                 break
        
#         if not embed_success:
#             return False, "Failed to generate 3D conformation"
        
#         # Try MMFF94s optimization
#         if AllChem.MMFFOptimizeMolecule(mol, maxIters=2000, mmffVariant='MMFF94s') != 0:
#             # Fall back to UFF
#             if AllChem.UFFOptimizeMolecule(mol, maxIters=2000) != 0:
#                 return False, "Structure optimization failed"
        
#         Chem.MolToPDBFile(mol, ligand_output)
#         return True, "Success"
#     except Exception as e:
#         return False, f"Error in prepare_ligand: {str(e)}"

# def check_pdbqt_file(filepath: str, is_receptor: bool = False) -> Tuple[bool, str]:
#     """Validate PDBQT file content."""
#     try:
#         with open(filepath, 'r') as f:
#             content = f.read()
            
#         if not content.strip():
#             return False, "File is empty"
            
#         if "ATOM" not in content and "HETATM" not in content:
#             return False, "No atom records found"
            
#         if is_receptor and "ROOT" in content:
#             return False, "Receptor file contains flexible residue markers"
            
#         if not is_receptor and "ROOT" not in content:
#             return False, "Ligand file missing ROOT marker"
            
#         return True, "File is valid"
#     except Exception as e:
#         return False, f"Error checking file: {str(e)}"

# def convert_to_pdbqt(input_file: str, output_file: str, is_ligand: bool = True) -> Tuple[bool, str]:
#     """Convert PDB to PDBQT format."""
#     try:
#         if not os.path.exists(input_file):
#             return False, f"Input file not found: {input_file}"
            
#         cmd = ["obabel", input_file, "-O", output_file]
#         if is_ligand:
#             cmd.extend(["-h", "--gen3d"])
#         else:
#             cmd.extend(["-xr"])
            
#         logger.info(f"Running conversion: {' '.join(cmd)}")
        
#         result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
#         if not os.path.exists(output_file):
#             return False, "Output file was not created"
            
#         valid, msg = check_pdbqt_file(output_file, not is_ligand)
#         if not valid:
#             return False, f"Invalid PDBQT file: {msg}"
            
#         return True, "Conversion successful"
#     except subprocess.CalledProcessError as e:
#         return False, f"Conversion failed: {e.stderr}"
#     except Exception as e:
#         return False, f"Error in conversion: {str(e)}"

# def create_vina_config(output_dir: Path, receptor_pdb: str) -> Tuple[bool, str]:
#     """Create Vina configuration file."""
#     try:
#         mol = Chem.MolFromPDBFile(receptor_pdb, removeHs=False)
#         if mol is None:
#             return False, "Failed to load receptor structure"
            
#         conf = mol.GetConformer()
#         coords = conf.GetPositions()
        
#         center = np.mean(coords, axis=0)
#         size = np.ptp(coords, axis=0) + 10.0
#         size = np.maximum(size, [20.0, 20.0, 20.0])
        
#         config_file = output_dir / "config.txt"
#         config_content = f"""
# center_x = {center[0]:.3f}
# center_y = {center[1]:.3f}
# center_z = {center[2]:.3f}
# size_x = {size[0]:.3f}
# size_y = {size[1]:.3f}
# size_z = {size[2]:.3f}
# exhaustiveness = 8
# num_modes = 9
# energy_range = 3
# cpu = {max(1, multiprocessing.cpu_count() - 1)}
# """.strip()

#         with open(config_file, "w") as f:
#             f.write(config_content)
            
#         logger.info(f"Created Vina config:\n{config_content}")
#         return True, str(config_file)
#     except Exception as e:
#         return False, f"Error creating config: {str(e)}"

# def run_vina_docking(receptor_pdbqt: str, ligand_pdbqt: str, output_pdbqt: str, 
#                      config_file: str) -> Tuple[bool, str]:
#     """Run AutoDock Vina docking."""
#     try:
#         # Verify input files
#         for filepath in [receptor_pdbqt, ligand_pdbqt, config_file]:
#             if not os.path.exists(filepath):
#                 return False, f"File not found: {filepath}"

#         # Run Vina
#         cmd = [
#             "vina",
#             "--receptor", receptor_pdbqt,
#             "--ligand", ligand_pdbqt,
#             "--out", output_pdbqt,
#             "--config", config_file
#         ]
        
#         logger.info(f"Running Vina: {' '.join(cmd)}")
#         result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
#         if not os.path.exists(output_pdbqt):
#             return False, "Docking output file was not created"
            
#         if os.path.getsize(output_pdbqt) == 0:
#             return False, "Docking output file is empty"
            
#         return True, result.stdout
#     except subprocess.CalledProcessError as e:
#         return False, f"Docking failed: {e.stderr}"
#     except Exception as e:
#         return False, f"Error in docking: {str(e)}"

# def process_ligand(args: Tuple[str, Path, int]) -> Dict:
#     """Process a single ligand."""
#     smiles, output_dir, idx = args
#     result = {
#         'smiles': smiles,
#         'affinity': None,
#         'status': 'failed',
#         'error': None,
#         'details': []
#     }
    
#     try:
#         ligand_pdb = output_dir / f"ligand_{idx}.pdb"
#         ligand_pdbqt = output_dir / f"ligand_{idx}.pdbqt"
#         output_pdbqt = output_dir / f"docked_{idx}.pdbqt"
        
#         # Prepare ligand
#         success, msg = prepare_ligand(smiles, str(ligand_pdb))
#         result['details'].append(f"Ligand preparation: {msg}")
#         if not success:
#             result['error'] = msg
#             return result
            
#         # Convert to PDBQT
#         success, msg = convert_to_pdbqt(str(ligand_pdb), str(ligand_pdbqt))
#         result['details'].append(f"PDBQT conversion: {msg}")
#         if not success:
#             result['error'] = msg
#             return result
            
#         # Run docking
#         success, msg = run_vina_docking(
#             str(output_dir / "receptor.pdbqt"),
#             str(ligand_pdbqt),
#             str(output_pdbqt),
#             str(output_dir / "config.txt")
#         )
#         result['details'].append(f"Docking: {msg}")
#         if not success:
#             result['error'] = msg
#             return result
            
#         # Parse results
#         with open(output_pdbqt, 'r') as f:
#             for line in f:
#                 if "REMARK VINA RESULT" in line:
#                     result['affinity'] = float(line.split()[3])
#                     result['status'] = 'success'
#                     break
        
#         if result['affinity'] is None:
#             result['error'] = "No binding affinity found in output"
            
#     except Exception as e:
#         result['error'] = f"Unexpected error: {str(e)}"
#         logger.exception("Error in process_ligand")
        
#     return result

# def batch_docking(smiles_list: List[str], receptor_pdb: str, 
#                  output_dir: str = "docking_results") -> List[Dict]:
#     """Run batch docking process."""
#     output_dir = Path(output_dir)
#     output_dir.mkdir(exist_ok=True)
    
#     # Check dependencies
#     deps = check_dependencies()
#     missing_deps = [dep for dep, installed in deps.items() if not installed]
#     if missing_deps:
#         st.error(f"Missing required dependencies: {', '.join(missing_deps)}")
#         return []
    
#     # Prepare receptor
#     receptor_pdbqt = output_dir / "receptor.pdbqt"
#     if not Path(receptor_pdb).exists():
#         st.error(f"Receptor file {receptor_pdb} not found")
#         return []
        
#     success, msg = convert_to_pdbqt(receptor_pdb, str(receptor_pdbqt), is_ligand=False)
#     if not success:
#         st.error(f"Receptor preparation failed: {msg}")
#         return []

#     # Create Vina configuration
#     success, msg = create_vina_config(output_dir, receptor_pdb)
#     if not success:
#         st.error(f"Failed to create Vina configuration: {msg}")
#         return []

#     # Process ligands
#     results = []
#     max_workers = min(multiprocessing.cpu_count(), len(smiles_list))
    
#     progress_text = "Processing ligands..."
#     my_bar = st.progress(0, text=progress_text)
    
#     try:
#         with ThreadPoolExecutor(max_workers=max_workers) as executor:
#             futures = [
#                 executor.submit(process_ligand, (smiles, output_dir, idx))
#                 for idx, smiles in enumerate(smiles_list)
#             ]
            
#             for i, future in enumerate(as_completed(futures)):
#                 result = future.result()
#                 results.append(result)
                
#                 progress = (i + 1) / len(smiles_list)
#                 my_bar.progress(progress, text=f"{progress_text} ({i + 1}/{len(smiles_list)})")
                
#                 if result['status'] == 'failed':
#                     st.warning(f"Failed processing ligand {i+1}: {result['error']}")
#                     st.write("Details:")
#                     for detail in result['details']:
#                         st.write(f"- {detail}")
                
#     finally:
#         my_bar.progress(1.0, text="Processing complete!")
                
#     return results

# def plot_affinities(results: List[Tuple[str, float]]):
#     """Plot binding affinities as a bar chart."""
#     if not results:
#         st.warning("No results to plot")
#         return
#     smiles, affinities = zip(*results)
#     plt.figure(figsize=(10, 6))
#     plt.bar(range(len(affinities)), affinities, tick_label=[s[:10] + "..." for s in smiles])
#     plt.xlabel("Ligands (Truncated SMILES)")
#     plt.ylabel("Binding Affinity (kcal/mol)")
#     plt.title("Docking Results")
#     plt.xticks(rotation=45, ha="right")
#     st.pyplot(plt)

# def create_interactive_results(results: List[Dict]) -> Dict[str, go.Figure]:
#     """Create interactive plots for docking results."""
#     successful_results = [r for r in results if r['status'] == 'success']
    
#     if not successful_results:
#         return None
    
#     plots = {}
    
#     # Affinity distribution plot
#     affinity_fig = go.Figure(data=[
#         go.Histogram(
#             x=[r['affinity'] for r in successful_results],
#             nbinsx=20,
#             name='Affinity Distribution'
#         )
#     ])
#     affinity_fig.update_layout(
#         title="Binding Affinity Distribution",
#         xaxis_title="Binding Affinity (kcal/mol)",
#         yaxis_title="Count",
#         template="plotly_white"
#     )
#     plots['affinity_dist'] = affinity_fig
    
#     # Scatter plot of affinities
#     scatter_fig = go.Figure(data=[
#         go.Scatter(
#             x=list(range(len(successful_results))),
#             y=[r['affinity'] for r in successful_results],
#             mode='markers',
#             text=[f"SMILES: {r['smiles'][:30]}...<br>Affinity: {r['affinity']:.2f}" 
#                   for r in successful_results],
#             marker=dict(
#                 size=10,
#                 color=[r['affinity'] for r in successful_results],
#                 colorscale='Viridis',
#                 showscale=True
#             )
#         )
#     ])
#     scatter_fig.update_layout(
#         title="Binding Affinities by Ligand",
#         xaxis_title="Ligand Index",
#         yaxis_title="Binding Affinity (kcal/mol)",
#         template="plotly_white"
#     )
#     plots['scatter'] = scatter_fig
    
#     return plots

# def show_3dmol(pdbfile: str, pdbqtfile: str) -> str:
#     """Create a 3Dmol.js visualization."""
#     view = py3Dmol.view(width=800, height=600)
    
#     # Load and display receptor
#     with open(pdbfile, 'r') as f:
#         receptor_data = f.read()
#     view.addModel(receptor_data, "pdb")
#     view.setStyle({'model': -1}, {'cartoon': {'color': 'gray'}})
    
#     # Load and display ligand
#     with open(pdbqtfile, 'r') as f:
#         ligand_data = f.read()
#     view.addModel(ligand_data, "pdbqt")
#     view.setStyle({'model': -1}, {'stick': {'colorscheme': 'greenCarbon'}})
    
#     view.zoomTo()
    
#     # Get the HTML content
#     html_content = view.render()
    
#     # Wrap in a div with proper styling
#     return f"""
#     <div style="height: 600px; width: 100%; position: relative;">
#         {html_content}
#     </div>
#     """

# def main():
#     st.set_page_config(layout="wide", page_title="Molecular Docking")
#     st.title("Molecular Docking Tool")
    
#     # Initialize session state if not exists
#     if 'results' not in st.session_state:
#         st.session_state.results = None
#     if 'receptor_path' not in st.session_state:
#         st.session_state.receptor_path = None
    
#     # Main interface
#     col1, col2 = st.columns([2, 1])
    
#     with col1:
#         receptor_file = st.file_uploader("Upload Receptor (PDB)", type=["pdb"])
#         if receptor_file:
#             receptor_path = Path("docking_results") / "receptor.pdb"
#             receptor_path.parent.mkdir(exist_ok=True)
#             receptor_path.write_bytes(receptor_file.read())
#             st.session_state.receptor_path = receptor_path
        
#         ligand_input = st.text_area(
#             "Enter SMILES strings (one per line)", 
#             "CC(=O)OC1=CC=CC=C1C(=O)O\nCN1C=NC2=C1C(=O)N(C(=O)N2C)C"
#         )
#         smiles_list = [s.strip() for s in ligand_input.split("\n") if s.strip()]

#     if st.button("Run Docking", type="primary"):
#         if not st.session_state.receptor_path:
#             st.error("Please upload a receptor file")
#         elif not smiles_list:
#             st.error("Please enter at least one SMILES string")
#         else:
#             status_area = st.empty()
#             status_area.info("Initializing docking process...")
            
#             try:
#                 results = batch_docking(
#                     smiles_list, 
#                     str(st.session_state.receptor_path)
#                 )
#                 st.session_state.results = results
                
#                 if results:
#                     successful = [r for r in results if r['status'] == 'success']
#                     status_area.success(f"Docking completed! ({len(successful)}/{len(results)} successful)")
#                 else:
#                     status_area.error("Docking process failed to produce results")
#             except Exception as e:
#                 status_area.error(f"An error occurred: {str(e)}")
#                 st.error("Full error details:")
#                 st.code(str(e))
#                 raise

#     # Display results if available
#     if st.session_state.results:
#         results = st.session_state.results
#         successful = [r for r in results if r['status'] == 'success']
        
#         # Show results in tabs
#         tab1, tab2, tab3, tab4 = st.tabs(["Results", "2D Visualization", "3D Visualization", "Details"])
        
#         with tab1:
#             # Create a more detailed DataFrame
#             df = pd.DataFrame([{
#                 'SMILES': r['smiles'],
#                 'Status': r['status'],
#                 'Affinity': r['affinity'] if r['affinity'] is not None else 'N/A',
#                 'Error': r['error'] if r['error'] is not None else 'None'
#             } for r in results])
#             st.dataframe(df)
        
#         with tab2:
#             if len(successful) > 0:
#                 # Add interactive plots
#                 plots = create_interactive_results(results)
#                 if plots:
#                     col1, col2 = st.columns(2)
#                     with col1:
#                         st.plotly_chart(plots['affinity_dist'], use_container_width=True)
#                     with col2:
#                         st.plotly_chart(plots['scatter'], use_container_width=True)
                
#                 st.subheader("Matplotlib Visualization")
#                 # Create list of (SMILES, affinity) tuples for successful results
#                 affinity_data = [(r['smiles'], r['affinity']) for r in successful]
#                 plot_affinities(affinity_data)
#             else:
#                 st.warning("No successful docking results to visualize")
        
#         with tab3:
#             if len(successful) > 0:
#                 st.subheader("3D Structure Visualization")
                
#                 for i, result in enumerate(successful):
#                     with st.expander(f"Docking Result {i+1}", expanded=(i==0)):
#                         st.write(f"Binding Affinity: {result['affinity']:.2f} kcal/mol")
                        
#                         output_pdbqt = Path("docking_results") / f"docked_{i}.pdbqt"
#                         if output_pdbqt.exists() and st.session_state.receptor_path:
#                             try:
#                                 html_content = show_3dmol(
#                                     str(st.session_state.receptor_path),
#                                     str(output_pdbqt)
#                                 )
#                                 st.components.v1.html(html_content, height=600)
#                             except Exception as e:
#                                 st.error(f"Error displaying 3D structure: {str(e)}")
#                         else:
#                             st.warning("3D visualization not available - missing output files")
#             else:
#                 st.warning("No successful docking results to visualize")
        
#         with tab4:
#             for i, result in enumerate(results):
#                 with st.expander(f"Ligand {i+1} Details"):
#                     st.write(f"Status: {result['status']}")
#                     st.write(f"Affinity: {result['affinity']}")
#                     if result['error']:
#                         st.error(f"Error: {result['error']}")
#                     st.write("Processing Steps:")
#                     for detail in result['details']:
#                         st.write(f"- {detail}")

# if __name__ == "__main__":
#     main()



import os
import sys
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
from plotly.subplots import make_subplots
import py3Dmol
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import io
import base64

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_dependencies() -> Dict[str, bool]:
    """Check if required external tools are available."""
    dependencies = {
        'obabel': False,
        'vina': False,
    }
    
    for cmd in dependencies.keys():
        try:
            subprocess.run([cmd, '--help'], capture_output=True)
            dependencies[cmd] = True
        except FileNotFoundError:
            logger.error(f"{cmd} not found in PATH")
    
    return dependencies

def prepare_ligand(smiles: str, ligand_output: str) -> Tuple[bool, str]:
    """Convert SMILES to 3D PDB structure."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, f"Invalid SMILES: {smiles}"
            
        mol = Chem.AddHs(mol)
        
        # Try different 3D conformer generation methods
        embed_success = False
        for seed in [42, 123, 987, 555]:
            params = AllChem.ETKDGv3()
            params.randomSeed = seed
            params.useSmallRingTorsions = True
            params.useBasicKnowledge = True
            params.enforceChirality = True
            
            if AllChem.EmbedMolecule(mol, params) >= 0:
                embed_success = True
                break
        
        if not embed_success:
            return False, "Failed to generate 3D conformation"
        
        # Try MMFF94s optimization
        if AllChem.MMFFOptimizeMolecule(mol, maxIters=2000, mmffVariant='MMFF94s') != 0:
            # Fall back to UFF
            if AllChem.UFFOptimizeMolecule(mol, maxIters=2000) != 0:
                return False, "Structure optimization failed"
        
        Chem.MolToPDBFile(mol, ligand_output)
        return True, "Success"
    except Exception as e:
        return False, f"Error in prepare_ligand: {str(e)}"

def check_pdbqt_file(filepath: str, is_receptor: bool = False) -> Tuple[bool, str]:
    """Validate PDBQT file content."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            
        if not content.strip():
            return False, "File is empty"
            
        if "ATOM" not in content and "HETATM" not in content:
            return False, "No atom records found"
            
        if is_receptor and "ROOT" in content:
            return False, "Receptor file contains flexible residue markers"
            
        if not is_receptor and "ROOT" not in content:
            return False, "Ligand file missing ROOT marker"
            
        return True, "File is valid"
    except Exception as e:
        return False, f"Error checking file: {str(e)}"

def convert_to_pdbqt(input_file: str, output_file: str, is_ligand: bool = True) -> Tuple[bool, str]:
    """Convert PDB to PDBQT format."""
    try:
        if not os.path.exists(input_file):
            return False, f"Input file not found: {input_file}"
            
        cmd = ["obabel", input_file, "-O", output_file]
        if is_ligand:
            cmd.extend(["-h", "--gen3d"])
        else:
            cmd.extend(["-xr"])
            
        logger.info(f"Running conversion: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        if not os.path.exists(output_file):
            return False, "Output file was not created"
            
        valid, msg = check_pdbqt_file(output_file, not is_ligand)
        if not valid:
            return False, f"Invalid PDBQT file: {msg}"
            
        return True, "Conversion successful"
    except subprocess.CalledProcessError as e:
        return False, f"Conversion failed: {e.stderr}"
    except Exception as e:
        return False, f"Error in conversion: {str(e)}"

def create_vina_config(output_dir: Path, receptor_pdb: str) -> Tuple[bool, str]:
    """Create Vina configuration file."""
    try:
        mol = Chem.MolFromPDBFile(receptor_pdb, removeHs=False)
        if mol is None:
            return False, "Failed to load receptor structure"
            
        conf = mol.GetConformer()
        coords = conf.GetPositions()
        
        center = np.mean(coords, axis=0)
        size = np.ptp(coords, axis=0) + 10.0
        size = np.maximum(size, [20.0, 20.0, 20.0])
        
        config_file = output_dir / "config.txt"
        config_content = f"""
center_x = {center[0]:.3f}
center_y = {center[1]:.3f}
center_z = {center[2]:.3f}
size_x = {size[0]:.3f}
size_y = {size[1]:.3f}
size_z = {size[2]:.3f}
exhaustiveness = 8
num_modes = 9
energy_range = 3
cpu = {max(1, multiprocessing.cpu_count() - 1)}
""".strip()

        with open(config_file, "w") as f:
            f.write(config_content)
            
        logger.info(f"Created Vina config:\n{config_content}")
        return True, str(config_file)
    except Exception as e:
        return False, f"Error creating config: {str(e)}"

def run_vina_docking(receptor_pdbqt: str, ligand_pdbqt: str, output_pdbqt: str, 
                     config_file: str) -> Tuple[bool, str, List[Dict]]:
    """Run AutoDock Vina docking and parse all poses."""
    try:
        # Verify input files
        for filepath in [receptor_pdbqt, ligand_pdbqt, config_file]:
            if not os.path.exists(filepath):
                return False, f"File not found: {filepath}", []

        # Run Vina
        cmd = [
            "vina",
            "--receptor", receptor_pdbqt,
            "--ligand", ligand_pdbqt,
            "--out", output_pdbqt,
            "--config", config_file
        ]
        
        logger.info(f"Running Vina: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        if not os.path.exists(output_pdbqt):
            return False, "Docking output file was not created", []
            
        if os.path.getsize(output_pdbqt) == 0:
            return False, "Docking output file is empty", []
        
        # Parse all poses
        poses = []
        current_pose = None
        pose_count = 0
        
        with open(output_pdbqt, 'r') as f:
            for line in f:
                if "MODEL" in line:
                    pose_count += 1
                    current_pose = {
                        'model': pose_count,
                        'affinity': None,
                        'rmsd_lb': None,
                        'rmsd_ub': None
                    }
                elif "REMARK VINA RESULT" in line:
                    parts = line.split()
                    if len(parts) >= 6:
                        current_pose['affinity'] = float(parts[3])
                        current_pose['rmsd_lb'] = float(parts[4])
                        current_pose['rmsd_ub'] = float(parts[5])
                elif "ENDMDL" in line and current_pose is not None:
                    poses.append(current_pose)
                    current_pose = None
        
        if not poses:
            return False, "No valid poses found in output", []
            
        return True, result.stdout, poses
    except subprocess.CalledProcessError as e:
        return False, f"Docking failed: {e.stderr}", []
    except Exception as e:
        return False, f"Error in docking: {str(e)}", []

def extract_single_pose(input_pdbqt: str, output_pdbqt: str, model_num: int) -> bool:
    """Extract a single pose from a multi-model PDBQT file."""
    try:
        with open(input_pdbqt, 'r') as f:
            content = f.readlines()
        
        if not content:
            return False
            
        output_content = []
        in_target_model = False
        current_model = 0
        
        for line in content:
            if line.startswith("MODEL"):
                current_model += 1
                in_target_model = (current_model == model_num)
            
            if in_target_model or line.startswith("REMARK") or line.startswith("HEADER"):
                output_content.append(line)
                
            if line.startswith("ENDMDL") and in_target_model:
                in_target_model = False
        
        with open(output_pdbqt, 'w') as f:
            f.writelines(output_content)
            
        return True
    except Exception as e:
        logger.error(f"Error extracting pose: {str(e)}")
        return False

def process_ligand(args: Tuple[str, Path, int]) -> Dict:
    """Process a single ligand."""
    smiles, output_dir, idx = args
    result = {
        'smiles': smiles,
        'affinity': None,
        'status': 'failed',
        'error': None,
        'details': [],
        'poses': []
    }
    
    try:
        # Generate RDKit mol object for 2D structure
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            AllChem.Compute2DCoords(mol)
            result['mol'] = mol
        
        ligand_pdb = output_dir / f"ligand_{idx}.pdb"
        ligand_pdbqt = output_dir / f"ligand_{idx}.pdbqt"
        output_pdbqt = output_dir / f"docked_{idx}.pdbqt"
        
        # Prepare ligand
        success, msg = prepare_ligand(smiles, str(ligand_pdb))
        result['details'].append(f"Ligand preparation: {msg}")
        if not success:
            result['error'] = msg
            return result
            
        # Convert to PDBQT
        success, msg = convert_to_pdbqt(str(ligand_pdb), str(ligand_pdbqt))
        result['details'].append(f"PDBQT conversion: {msg}")
        if not success:
            result['error'] = msg
            return result
            
        # Run docking
        success, msg, poses = run_vina_docking(
            str(output_dir / "receptor.pdbqt"),
            str(ligand_pdbqt),
            str(output_pdbqt),
            str(output_dir / "config.txt")
        )
        
        result['details'].append(f"Docking: {msg}")
        if not success:
            result['error'] = msg
            return result
        
        # Store all poses
        result['poses'] = poses
        
        # Extract each pose to a separate file
        for pose in poses:
            pose_file = output_dir / f"docked_{idx}_pose_{pose['model']}.pdbqt"
            if extract_single_pose(str(output_pdbqt), str(pose_file), pose['model']):
                pose['file'] = str(pose_file)
        
        # Use the best affinity as the main result
        if poses:
            result['affinity'] = poses[0]['affinity']
            result['status'] = 'success'
            
    except Exception as e:
        result['error'] = f"Unexpected error: {str(e)}"
        logger.exception("Error in process_ligand")
        
    return result

def batch_docking(smiles_list: List[str], receptor_pdb: str, 
                 output_dir: str = "docking_results") -> List[Dict]:
    """Run batch docking process."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Check dependencies
    deps = check_dependencies()
    missing_deps = [dep for dep, installed in deps.items() if not installed]
    if missing_deps:
        st.error(f"Missing required dependencies: {', '.join(missing_deps)}")
        return []
    
    # Prepare receptor
    receptor_pdbqt = output_dir / "receptor.pdbqt"
    if not Path(receptor_pdb).exists():
        st.error(f"Receptor file {receptor_pdb} not found")
        return []
        
    success, msg = convert_to_pdbqt(receptor_pdb, str(receptor_pdbqt), is_ligand=False)
    if not success:
        st.error(f"Receptor preparation failed: {msg}")
        return []

    # Create Vina configuration
    success, msg = create_vina_config(output_dir, receptor_pdb)
    if not success:
        st.error(f"Failed to create Vina configuration: {msg}")
        return []

    # Process ligands
    results = []
    max_workers = min(multiprocessing.cpu_count(), len(smiles_list))
    
    progress_text = "Processing ligands..."
    my_bar = st.progress(0, text=progress_text)
    
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_ligand, (smiles, output_dir, idx))
                for idx, smiles in enumerate(smiles_list)
            ]
            
            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                results.append(result)
                
                progress = (i + 1) / len(smiles_list)
                my_bar.progress(progress, text=f"{progress_text} ({i + 1}/{len(smiles_list)})")
                
                if result['status'] == 'failed':
                    st.warning(f"Failed processing ligand {i+1}: {result['error']}")
                    st.write("Details:")
                    for detail in result['details']:
                        st.write(f"- {detail}")
                
    finally:
        my_bar.progress(1.0, text="Processing complete!")
                
    return results

def mol_to_img(mol, size=(300, 300)):
    """Convert an RDKit molecule to an image for Streamlit."""
    if mol is None:
        return None
    
    img = Draw.MolToImage(mol, size=size)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def plot_affinities(results: List[Dict]):
    """Plot binding affinities as a bar chart with improved styling."""
    if not results:
        st.warning("No results to plot")
        return
    
    # Prepare data
    successful = [r for r in results if r['status'] == 'success']
    if not successful:
        st.warning("No successful docking results to visualize")
        return
    
    # Sort by affinity (lowest/strongest first)
    sorted_results = sorted(successful, key=lambda x: x['affinity'])
    
    smiles = [r['smiles'] for r in sorted_results]
    affinities = [r['affinity'] for r in sorted_results]
    
    # Create a colormap based on affinity values (lower is better)
    norm = plt.Normalize(min(affinities), max(affinities))
    colors = cm.viridis(norm(affinities))
    colors = [(*c[:3], 0.8) for c in colors]  # Add transparency
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(affinities)), affinities, color=colors)
    
    # Add data labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{affinities[i]:.1f}', ha='center', va='bottom', rotation=0,
                fontsize=9)
    
    # Add grid lines and styling
    ax.set_xlabel("Ligands", fontsize=12)
    ax.set_ylabel("Binding Affinity (kcal/mol)", fontsize=12)
    ax.set_title("Docking Results (Lower Values Indicate Stronger Binding)", fontsize=14)
    ax.set_xticks(range(len(affinities)))
    
    # Determine label length based on number of compounds
    if len(smiles) <= 5:
        ax.set_xticklabels(smiles, rotation=45, ha="right")
    else:
        ax.set_xticklabels([s[:10] + "..." if len(s) > 10 else s for s in smiles], 
                           rotation=45, ha="right")
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Binding Affinity (kcal/mol)')
    
    plt.tight_layout()
    st.pyplot(fig)

def create_interactive_results(results: List[Dict]) -> Dict[str, go.Figure]:
    """Create interactive plots for docking results."""
    successful_results = [r for r in results if r['status'] == 'success']
    
    if not successful_results:
        return None
    
    plots = {}
    
    # Affinity distribution plot
    affinity_fig = go.Figure(data=[
        go.Histogram(
            x=[r['affinity'] for r in successful_results],
            nbinsx=20,
            marker_color='rgba(50, 168, 82, 0.7)',
            name='Affinity Distribution'
        )
    ])
    affinity_fig.update_layout(
        title="Binding Affinity Distribution",
        xaxis_title="Binding Affinity (kcal/mol)",
        yaxis_title="Count",
        template="plotly_white",
        hovermode="closest"
    )
    plots['affinity_dist'] = affinity_fig
    
    # Improved scatter plot of affinities
    scatter_fig = go.Figure()
    
    # Add a scatter trace
    scatter_fig.add_trace(go.Scatter(
        x=list(range(len(successful_results))),
        y=[r['affinity'] for r in successful_results],
        mode='markers',
        text=[f"SMILES: {r['smiles'][:30]}...<br>Affinity: {r['affinity']:.2f}" 
              for r in successful_results],
        marker=dict(
            size=15,
            color=[r['affinity'] for r in successful_results],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Affinity (kcal/mol)"),
            line=dict(width=1, color='black')
        ),
        name="Binding Affinities"
    ))
    
    scatter_fig.update_layout(
        title="Binding Affinities by Ligand",
        xaxis_title="Ligand Index",
        yaxis_title="Binding Affinity (kcal/mol)",
        template="plotly_white",
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(successful_results))),
            ticktext=[f"{i+1}" for i in range(len(successful_results))]
        )
    )
    plots['scatter'] = scatter_fig
    
    # Add radar chart for comparing poses
    # Find ligands with multiple poses
    ligands_with_poses = [r for r in successful_results if len(r.get('poses', [])) > 1]
    
    if ligands_with_poses:
        # Create a radar chart for the first ligand with multiple poses
        ligand = ligands_with_poses[0]
        
        radar_fig = go.Figure()
        
        categories = ['Affinity', 'RMSD LB', 'RMSD UB']
        for pose in ligand.get('poses', [])[:5]:  # Limit to 5 poses for clarity
            values = [
                pose.get('affinity', 0),
                pose.get('rmsd_lb', 0),
                pose.get('rmsd_ub', 0)
            ]
            
            radar_fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=f"Pose {pose.get('model', '?')}"
            ))
            
        radar_fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[
                        min([p.get('affinity', 0) for p in ligand.get('poses', [])]) - 1,
                        max([p.get('rmsd_ub', 0) for p in ligand.get('poses', [])]) + 1
                    ]
                )),
            title=f"Pose Comparison for First Ligand",
            template="plotly_white"
        )
        plots['radar'] = radar_fig
        
        # Create heatmap for all poses of all ligands
        heatmap_data = []
        for idx, result in enumerate(successful_results):
            for pose in result.get('poses', [])[:3]:  # Limit to top 3 poses
                heatmap_data.append({
                    'Ligand': f"Ligand {idx+1}",
                    'Pose': f"Pose {pose.get('model', '?')}",
                    'Affinity': pose.get('affinity', 0)
                })
        
        if heatmap_data:
            df_heatmap = pd.DataFrame(heatmap_data)
            pivot_table = df_heatmap.pivot(index='Ligand', columns='Pose', values='Affinity')
            
            heatmap_fig = px.imshow(
                pivot_table,
                labels=dict(x="Pose", y="Ligand", color="Affinity (kcal/mol)"),
                x=pivot_table.columns,
                y=pivot_table.index,
                color_continuous_scale="Viridis_r",  # Reversed so darker is stronger binding
                aspect="auto"
            )
            
            heatmap_fig.update_layout(
                title="Binding Affinity Heatmap Across Poses",
                template="plotly_white",
                coloraxis_colorbar=dict(title="Affinity (kcal/mol)")
            )
            
            # Add text annotations
            for i, ligand in enumerate(pivot_table.index):
                for j, pose in enumerate(pivot_table.columns):
                    value = pivot_table.iloc[i, j]
                    if not pd.isna(value):
                        heatmap_fig.add_annotation(
                            x=pose, y=ligand, text=f"{value:.2f}",
                            showarrow=False, font=dict(color="white" if value < -8 else "black")
                        )
            
            plots['heatmap'] = heatmap_fig
    
    # Add bubble chart comparing all compounds
    bubble_data = []
    for idx, result in enumerate(successful_results):
        if 'poses' in result and result['poses']:
            num_poses = len(result['poses'])
            best_affinity = result['affinity']
            rmsd_range = 0
            
            if num_poses > 1:
                rmsd_range = max([p.get('rmsd_ub', 0) for p in result['poses']]) - \
                            min([p.get('rmsd_lb', 0) for p in result['poses']])
            
            bubble_data.append({
                'Ligand': f"Ligand {idx+1}",
                'Affinity': best_affinity,
                'Poses': num_poses,
                'RMSD Range': rmsd_range,
                'SMILES': result['smiles']
            })
    
    if bubble_data:
        df_bubble = pd.DataFrame(bubble_data)
        bubble_fig = px.scatter(
            df_bubble,
            x='Affinity',
            y='RMSD Range',
            size='Poses',
            color='Affinity',
            hover_name='Ligand',
            hover_data=['SMILES'],
            color_continuous_scale='Viridis_r',
            size_max=30
        )
        
        bubble_fig.update_layout(
            title="Ligand Comparison - Affinity vs Conformational Diversity",
            xaxis_title="Binding Affinity (kcal/mol)",
            yaxis_title="RMSD Range (Å)",
            template="plotly_white"
        )
        
        plots['bubble'] = bubble_fig
    
    return plots
def show_basic_3dmol(receptor_pdbfile: str, ligand_pdbqtfile: str, viewer_height: int = 500) -> str:
    """Create a basic visualization using py3Dmol with download options."""
    try:
        # Check if files exist
        if not os.path.exists(receptor_pdbfile):
            return f"""
            <div style="height: {viewer_height}px; width: 100%; 
                       display: flex; justify-content: center; align-items: center; 
                       background-color: #f8f9fa; border: 1px solid #ddd;">
                <p>Receptor file not found: {receptor_pdbfile}</p>
            </div>
            """
            
        if not os.path.exists(ligand_pdbqtfile):
            return f"""
            <div style="height: {viewer_height}px; width: 100%; 
                       display: flex; justify-content: center; align-items: center; 
                       background-color: #f8f9fa; border: 1px solid #ddd;">
                <p>Ligand file not found: {ligand_pdbqtfile}</p>
            </div>
            """

        # Read file contents
        with open(receptor_pdbfile, 'r') as f:
            receptor_data = f.read()
            receptor_length = len(receptor_data)
            
        with open(ligand_pdbqtfile, 'r') as f:
            ligand_data = f.read()
            ligand_length = len(ligand_data)

        # Create py3Dmol view
        viewer = py3Dmol.view(width=800, height=viewer_height)
        viewer.addModel(receptor_data, 'pdb')
        viewer.setStyle({'model': 0}, {'cartoon': {'color': 'lightgray'}})
        viewer.addModel(ligand_data, 'pdbqt')
        viewer.setStyle({'model': 1}, {'stick': {'colorscheme': 'greenCarbon'}})
        viewer.zoomTo()
        viewer.setBackgroundColor('white')

        # Generate HTML with download buttons (using Streamlit components instead of JS)
        html_content = f"""
        <div style="text-align: center; padding: 10px;">
            <div style="font-size: 14px; margin-bottom: 10px;">
                <b>Debug Info:</b> Receptor file size: {receptor_length} bytes, Ligand file size: {ligand_length} bytes
            </div>
            {viewer.render()}
        </div>
        """

        # Return the HTML and let Streamlit handle downloads separately
        return html_content

    except Exception as e:
        logger.error(f"Error in basic 3D visualization: {str(e)}")
        return f"""
        <div style="height: {viewer_height}px; width: 100%; 
                   display: flex; justify-content: center; align-items: center; 
                   background-color: #f8f9fa; border: 1px solid #ddd;">
            <p>Error loading 3D visualization: {str(e)}</p>
        </div>
        """

def create_ngl_viewer(receptor_pdbfile: str, ligand_pdbqtfile: str, viewer_height: int = 500) -> str:
    """Create a visualization using NGL Viewer instead of 3Dmol.js."""
    try:
        # Check if files exist
        if not os.path.exists(receptor_pdbfile) or not os.path.exists(ligand_pdbqtfile):
            return show_basic_3dmol(receptor_pdbfile, ligand_pdbqtfile, viewer_height)
        
        # Use NGL Viewer which might be more reliable
        html_content = f"""
        <div id="viewport" style="width:100%; height:{viewer_height}px;"></div>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/ngl/2.0.0-dev.36/ngl.js"></script>
        <script>
            document.addEventListener("DOMContentLoaded", function() {{
                // Create NGL Stage object
                var stage = new NGL.Stage("viewport", {{backgroundColor: "white"}});

                // Load files from URLs or file paths when within the same origin
                Promise.all([
                    stage.loadFile('{receptor_pdbfile}'),
                    stage.loadFile('{ligand_pdbqtfile}')
                ]).then(function (components) {{
                    // receptor 
                    components[0].addRepresentation("cartoon", {{
                        color: "lightgrey",
                        opacity: 0.7
                    }});
                    
                    // ligand
                    components[1].addRepresentation("licorice", {{
                        colorScheme: "element",
                        radius: 0.5
                    }});
                    
                    // zoom to fit the viewport
                    stage.autoView();
                }});
            }});
        </script>
        """
        
        return html_content
    except Exception as e:
        logger.error(f"Error in NGL visualization: {str(e)}")
        return show_basic_3dmol(receptor_pdbfile, ligand_pdbqtfile, viewer_height)

def create_molstar_viewer(receptor_pdbfile: str, ligand_pdbqtfile: str, viewer_height: int = 500) -> str:
    """Create a visualization using Mol* Viewer as another alternative."""
    try:
        # Check if files exist
        if not os.path.exists(receptor_pdbfile) or not os.path.exists(ligand_pdbqtfile):
            return show_basic_3dmol(receptor_pdbfile, ligand_pdbqtfile, viewer_height)
        
        # Use Mol* Viewer which might be more reliable
        html_content = f"""
        <div id="molstar-viewer" style="width:100%; height:{viewer_height}px; position: relative;"></div>
        <script src="https://cdn.jsdelivr.net/npm/molstar/build/molstar.js"></script>
        <script>
            document.addEventListener("DOMContentLoaded", function() {{
                molstar.Viewer.create('molstar-viewer', {{
                    layoutIsExpanded: false,
                    layoutShowControls: false,
                    layoutShowSequence: false,
                    layoutShowLog: false,
                    layoutShowLeftPanel: false,
                }}).then(viewer => {{
                    Promise.all([
                        viewer.loadStructureFromUrl('{receptor_pdbfile}', 'pdb', {{ asTrajectory: false }}),
                        viewer.loadStructureFromUrl('{ligand_pdbqtfile}', 'pdbqt', {{ asTrajectory: false }})
                    ]).then(() => {{
                        viewer.updateStyle({{
                            chain: {{ A: {{ type: 'cartoon', color: 'uniform', params: {{ color: {{ r: 200, g: 200, b: 200 }} }} }} }}
                        }});
                    }});
                }});
            }});
        </script>
        """
        
        return html_content
    except Exception as e:
        logger.error(f"Error in Mol* visualization: {str(e)}")
        return show_basic_3dmol(receptor_pdbfile, ligand_pdbqtfile, viewer_height)

def show_3dmol(receptor_pdbfile: str, ligand_pdbqtfile: str, viewer_height: int = 500) -> str:
    """Create a 3Dmol.js visualization for a single pose."""
    try:
        if not os.path.exists(receptor_pdbfile) or not os.path.exists(ligand_pdbqtfile):
            return f"<div>Error: File(s) not found</div>"

        with open(receptor_pdbfile, 'r') as f:
            receptor_data = f.read().replace("'", "\\'").replace("`", "\\`")
        with open(ligand_pdbqtfile, 'r') as f:
            ligand_data = f.read().replace("'", "\\'").replace("`", "\\`")

        html_content = f"""
        <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.3/3Dmol-min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.3/jquery.min.js"></script>
        <style>
            .viewer_3Dmoljs {{
                position: relative;
                width: 100%;
                height: {viewer_height}px;
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: #f8f9fa;
            }}
            .mol-container {{
                width: 100%;
                height: 100%;
            }}
            .viewer-instructions {{
                position: absolute;
                bottom: 10px;
                left: 10px;
                background-color: rgba(255,255,255,0.7);
                padding: 5px;
                border-radius: 5px;
                font-size: 12px;
                z-index: 1000;
            }}
        </style>
        
        <div class="viewer_3Dmoljs">
            <div id="container-01" class="mol-container"></div>
            <div class="viewer-instructions">
                Scroll to zoom, drag to rotate, Shift+drag to translate
            </div>
        </div>
        
        <script>
            $(function() {{
                let element = $("#container-01");
                let config = {{ backgroundColor: "white" }};
                let viewer = $3Dmol.createViewer(element, config);
                
                viewer.addModel(`{receptor_data}`, "pdb");
                viewer.setStyle({{model: -1}}, {{cartoon: {{color: "lightgray", opacity: 0.8}}}});
                viewer.addSurface($3Dmol.SurfaceType.VDW, {{
                    opacity: 0.7,
                    color: "white"
                }}, {{model: -1}});
                
                viewer.addModel(`{ligand_data}`, "pdbqt");
                viewer.setStyle({{model: -1}}, {{
                    stick: {{
                        colorscheme: "greenCarbon",
                        radius: 0.3
                    }},
                    sphere: {{
                        scale: 0.3
                    }}
                }});
                
                viewer.zoomTo();
                viewer.rotate(90, 'y');
                viewer.render();
                
                element.dblclick(function() {{
                    let spin = viewer.isAnimated();
                    viewer.spin(!spin);
                }});
            }});
        </script>
        """
        return html_content
    except Exception as e:
        logger.error(f"Error in 3D visualization: {str(e)}")
        return f"<div>Error: {str(e)}</div>"

def show_multiple_poses(receptor_pdbfile: str, pose_files: List[str], viewer_height: int = 500) -> str:
    """Create a 3Dmol.js visualization with multiple poses."""
    try:
        if not pose_files or not os.path.exists(receptor_pdbfile):
            return f"<div>Error: No valid files</div>"

        valid_pose_files = [f for f in pose_files if os.path.exists(f)]
        if not valid_pose_files:
            return f"<div>No valid pose files</div>"

        with open(receptor_pdbfile, 'r') as f:
            receptor_data = f.read().replace("'", "\\'").replace("`", "\\`")

        colors = ['green', 'blue', 'red', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'white']
        colors = colors[:len(valid_pose_files)]
        pose_data_list = [open(pf, 'r').read().replace("'", "\\'").replace("`", "\\`") for pf in valid_pose_files]

        legend_html = "".join(f'<span style="color: {color}; margin-right: 10px;">● Pose {i+1}</span>' 
                            for i, color in enumerate(colors))

        poses_js = ""
        for i, pose_data in enumerate(pose_data_list):
            poses_js += f"""
                viewer.addModel(`{pose_data}`, "pdbqt");
                viewer.setStyle({{model: {i+1}}}, {{
                    stick: {{
                        color: "{colors[i]}",
                        radius: 0.3
                    }}
                }});
            """

        html_content = f"""
        <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.3/3Dmol-min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.3/jquery.min.js"></script>
        <style>
            .viewer_3Dmoljs {{
                position: relative;
                width: 100%;
                height: {viewer_height}px;
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: #f8f9fa;
            }}
            .mol-container {{
                width: 100%;
                height: 100%;
            }}
            .viewer-legend {{
                position: absolute;
                bottom: 10px;
                left: 10px;
                background-color: rgba(255,255,255,0.7);
                padding: 5px;
                border-radius: 5px;
                font-size: 12px;
                z-index: 1000;
            }}
        </style>
        
        <div class="viewer_3Dmoljs">
            <div id="container-02" class="mol-container"></div>
            <div class="viewer-legend">
                <p style="margin: 0; font-size: 12px;">Pose colors:</p>
                {legend_html}
            </div>
        </div>
        
        <script>
            $(function() {{
                let element = $("#container-02");
                let config = {{ backgroundColor: "white" }};
                let viewer = $3Dmol.createViewer(element, config);
                
                viewer.addModel(`{receptor_data}`, "pdb");
                viewer.setStyle({{model: 0}}, {{cartoon: {{color: "lightgray", opacity: 0.7}}}});
                viewer.addSurface($3Dmol.SurfaceType.VDW, {{
                    opacity: 0.5,
                    color: "white"
                }}, {{model: 0}});
                
                {poses_js}
                
                viewer.zoomTo();
                viewer.render();
                
                element.dblclick(function() {{
                    let spin = viewer.isAnimated();
                    viewer.spin(!spin);
                }});
            }});
        </script>
        """
        return html_content
    except Exception as e:
        logger.error(f"Error in multiple pose visualization: {str(e)}")
        return f"<div>Error: {str(e)}</div>"

def create_ligand_interaction_diagram(receptor_pdbfile: str, ligand_pdbqtfile: str, width=800, height=600):
    """
    Create a simplified 2D diagram of ligand-receptor interactions.
    This is a placeholder that would be enhanced with actual interaction analysis in production.
    """
    try:
        # In a real implementation, you would:
        # 1. Extract ligand and receptor structures
        # 2. Calculate interactions (H-bonds, hydrophobic, etc.)
        # 3. Create a 2D diagram using a library like RDKit or similar
        
        # For now, we'll create a simple placeholder
        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
        
        # Draw a placeholder circle for receptor pocket
        circle = plt.Circle((0.5, 0.5), 0.4, fill=False, color='gray', linestyle='--')
        ax.add_patch(circle)
        
        # Add text for interactions (this would be calculated in production)
        ax.text(0.5, 0.8, "Receptor Binding Pocket", ha='center', fontsize=14, color='gray')
        ax.text(0.5, 0.3, "Ligand", ha='center', fontsize=14, color='green')
        ax.text(0.3, 0.6, "H-bond", ha='center', fontsize=10, color='blue')
        ax.text(0.7, 0.6, "Hydrophobic", ha='center', fontsize=10, color='orange')
        ax.text(0.5, 0.5, "π-stacking", ha='center', fontsize=10, color='purple')
        
        # Add arrows for interactions
        ax.arrow(0.35, 0.55, 0, -0.1, head_width=0.02, head_length=0.02, fc='blue', ec='blue')
        ax.arrow(0.65, 0.55, 0, -0.1, head_width=0.02, head_length=0.02, fc='orange', ec='orange')
        ax.arrow(0.5, 0.45, 0, -0.03, head_width=0.02, head_length=0.02, fc='purple', ec='purple')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        return fig
    except Exception as e:
        logger.error(f"Error creating interaction diagram: {str(e)}")
        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
        ax.text(0.5, 0.5, f"Error creating diagram: {str(e)}", ha='center', va='center')
        ax.axis('off')
        return fig

def main():
    st.set_page_config(layout="wide", page_title="Molecular Docking Tool")
    
    # Add custom CSS
    st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .main .block-container {
        padding-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("Advanced Molecular Docking & Visualization Tool")
    st.write("Upload receptor, input ligand SMILES, and analyze docking results with enhanced visualizations")
    
    # Initialize session state if not exists
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'receptor_path' not in st.session_state:
        st.session_state.receptor_path = None
    if 'selected_ligand_idx' not in st.session_state:
        st.session_state.selected_ligand_idx = 0
    if 'selected_pose_idx' not in st.session_state:
        st.session_state.selected_pose_idx = 0
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        receptor_file = st.file_uploader("Upload Receptor (PDB)", type=["pdb"])
        if receptor_file:
            receptor_path = Path("docking_results") / "receptor.pdb"
            receptor_path.parent.mkdir(exist_ok=True)
            receptor_path.write_bytes(receptor_file.read())
            st.session_state.receptor_path = receptor_path
            st.success(f"Receptor uploaded: {receptor_file.name}")
    
    with col2:
        ligand_input = st.text_area(
            "Enter SMILES strings (one per line)", 
            "CC(=O)OC1=CC=CC=C1C(=O)O\nCN1C=NC2=C1C(=O)N(C(=O)N2C)C\nCCCC(=O)O",
            height=100
        )
        smiles_list = [s.strip() for s in ligand_input.split("\n") if s.strip()]
        
        if st.session_state.receptor_path:
            st.info(f"Ready to dock {len(smiles_list)} ligand(s) with receptor: {receptor_file.name if receptor_file else Path(st.session_state.receptor_path).name}")

    # Parameters expander
    with st.expander("Advanced Parameters", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            exhaustiveness = st.slider("Vina Exhaustiveness", 1, 16, 8, 
                help="Higher values give more thorough search but take longer")
            energy_range = st.slider("Energy Range (kcal/mol)", 1, 10, 3,
                help="Maximum energy difference between best and worst binding mode")
        with col2:
            num_modes = st.slider("Number of Binding Modes", 1, 20, 9,
                help="Maximum number of binding modes to generate")
            cpu_count = st.slider("CPU Cores", 1, multiprocessing.cpu_count(), 
                max(1, multiprocessing.cpu_count() - 1), 
                help="Number of CPU cores to use")

    run_col1, run_col2 = st.columns([3, 1])
    with run_col1:
        run_button = st.button("Run Docking", type="primary", use_container_width=True)
    with run_col2:
        help_button = st.button("Help & Info", type="secondary", use_container_width=True)
    
    if help_button:
        with st.expander("Help & Information", expanded=True):
            st.markdown("""
            ## How to Use This Tool
            1. **Upload a Receptor**: Provide a PDB file of your protein target
            2. **Enter SMILES**: Add one or more SMILES strings for the ligands you want to dock
            3. **Run Docking**: Click the Run Docking button to start the process
            4. **View Results**: Use the tabs to see different visualizations of the results
            
            ## Visualizations Explained
            - **Results Table**: Overview of all docking results
            - **2D Visualizations**: Various plots and charts analyzing binding affinities
            - **3D Visualization**: Interactive view of docked poses in 3D
            - **Multiple Poses**: View and compare different binding poses for each ligand
            - **Details**: Process logs and technical information
            
            ## About Binding Affinity
            Lower (more negative) binding affinity values indicate stronger binding. The unit is kcal/mol.
            """)

    if run_button:
        if not st.session_state.receptor_path:
            st.error("Please upload a receptor file")
        elif not smiles_list:
            st.error("Please enter at least one SMILES string")
        else:
            status_area = st.empty()
            status_area.info("Initializing docking process...")
            
            try:
                # Update config with slider values before running
                config_file = Path("docking_results") / "config.txt"
                if config_file.exists():
                    with open(config_file, "r") as f:
                        config_content = f.read()
                    
                    # Update configuration with slider values
                    config_content = config_content.replace("exhaustiveness = 8", f"exhaustiveness = {exhaustiveness}")
                    config_content = config_content.replace("num_modes = 9", f"num_modes = {num_modes}")
                    config_content = config_content.replace("energy_range = 3", f"energy_range = {energy_range}")
                    config_content = config_content.replace(f"cpu = {max(1, multiprocessing.cpu_count() - 1)}", f"cpu = {cpu_count}")
                    
                    with open(config_file, "w") as f:
                        f.write(config_content)
                
                results = batch_docking(
                    smiles_list, 
                    str(st.session_state.receptor_path)
                )
                st.session_state.results = results
                
                if results:
                    successful = [r for r in results if r['status'] == 'success']
                    status_area.success(f"Docking completed! ({len(successful)}/{len(results)} successful)")
                else:
                    status_area.error("Docking process failed to produce results")
            except Exception as e:
                status_area.error(f"An error occurred: {str(e)}")
                st.error("Full error details:")
                st.code(str(e))
                raise

    # Display results if available
    if st.session_state.results:
        results = st.session_state.results
        successful = [r for r in results if r['status'] == 'success']
        
        # Show results in tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔍 Results", 
            "📊 2D Visualization", 
            "🔄 3D Visualization", 
            "🧩 Multiple Poses", 
            "📋 Details"
        ])
        
        with tab1:
            # Create a more detailed DataFrame
            df = pd.DataFrame([{
                'SMILES': r['smiles'],
                'Status': r['status'],
                'Affinity (kcal/mol)': r['affinity'] if r['affinity'] is not None else 'N/A',
                'Poses': len(r.get('poses', [])),
                'Error': r['error'] if r['error'] is not None else 'None'
            } for r in results])
            
            # Display ligand structures if available
            st.subheader("Docking Results Overview")
            st.dataframe(df, use_container_width=True)
            
            if successful:
                st.subheader("Top Ligand Structures")
                
                # Sort by binding affinity
                sorted_results = sorted(successful, key=lambda x: x.get('affinity', 0))
                
                # Display top 5 ligands as a grid
                cols = st.columns(min(5, len(sorted_results)))
                
                for i, (col, result) in enumerate(zip(cols, sorted_results[:5])):
                    with col:
                        if 'mol' in result:
                            img = mol_to_img(result['mol'])
                            if img:
                                st.image(f"data:image/png;base64,{img}", 
                                         caption=f"Ligand {i+1}\nAffinity: {result['affinity']:.2f}")
                            else:
                                st.write(f"Ligand {i+1}\nAffinity: {result['affinity']:.2f}")
                        else:
                            st.write(f"Ligand {i+1}\nAffinity: {result['affinity']:.2f}")
        
        with tab2:
            if len(successful) > 0:
                st.subheader("Interactive Visualizations")
                
                # Add interactive plots
                plots = create_interactive_results(results)
                if plots:
                    # Display first row of plots
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(plots['affinity_dist'], use_container_width=True)
                    with col2:
                        st.plotly_chart(plots['scatter'], use_container_width=True)
                    
                    # Display second row if we have the other plots
                    if 'bubble' in plots or 'heatmap' in plots:
                        col1, col2 = st.columns(2)
                        if 'bubble' in plots:
                            with col1:
                                st.plotly_chart(plots['bubble'], use_container_width=True)
                        if 'heatmap' in plots:
                            with col2:
                                st.plotly_chart(plots['heatmap'], use_container_width=True)
                    
                    if 'radar' in plots:
                        st.plotly_chart(plots['radar'], use_container_width=True)
                
                st.subheader("Binding Affinities Visualization")
                plot_affinities(successful)
            else:
                st.warning("No successful docking results to visualize")
        
        with tab3:
            if len(successful) > 0:
                st.subheader("3D Structure Visualization")
                ligand_idx = st.session_state.selected_ligand_idx
                selected_result = successful[ligand_idx]
                pose_idx = st.session_state.selected_pose_idx
                pose_model = selected_result['poses'][pose_idx].get('model', 1)
                output_pdbqt = Path("docking_results") / f"docked_{ligand_idx}_pose_{pose_model}.pdbqt"
                if output_pdbqt.exists() and st.session_state.receptor_path:
                    html_content = show_3dmol(str(st.session_state.receptor_path), str(output_pdbqt), 600)
                    st.components.v1.html(html_content, height=600)
        
        with tab4:
            if len(successful) > 0:
                st.subheader("Multiple Pose Visualization")
                ligands_with_poses = [(i, r) for i, r in enumerate(successful) if len(r.get('poses', [])) > 1]
                if ligands_with_poses:
                    selected_idx = st.selectbox("Select ligand", range(len(ligands_with_poses)), 
                                            format_func=lambda x: f"Ligand {ligands_with_poses[x][0]+1}")
                    idx, result = ligands_with_poses[selected_idx]
                    max_poses = min(3, len(result['poses']))
                    pose_files = [str(Path("docking_results") / f"docked_{idx}_pose_{p['model']}.pdbqt") 
                                for p in result['poses'][:max_poses]]
                    html_content = show_multiple_poses(str(st.session_state.receptor_path), pose_files, 700)
                    st.components.v1.html(html_content, height=700)
        
        with tab5:
            st.subheader("Detailed Process Information")
            
            # Add configuration display
            config_file = Path("docking_results") / "config.txt"
            if config_file.exists():
                with st.expander("Docking Configuration", expanded=True):
                    with open(config_file, "r") as f:
                        config_content = f.read()
                    st.code(config_content, language="bash")
            
            # Add receptor information
            with st.expander("Receptor Information", expanded=False):
                if st.session_state.receptor_path and Path(st.session_state.receptor_path).exists():
                    try:
                        mol = Chem.MolFromPDBFile(str(st.session_state.receptor_path), removeHs=False)
                        if mol is not None:
                            st.write(f"**Atoms:** {mol.GetNumAtoms()}")
                            st.write(f"**Bonds:** {mol.GetNumBonds()}")
                            st.write(f"**Residues:** {len(Chem.SplitMolByPDBResidues(mol))}")
                    except Exception as e:
                        st.error(f"Error analyzing receptor: {str(e)}")
            
            # Individual ligand details
            for i, result in enumerate(results):
                with st.expander(f"Ligand {i+1} Details", expanded=(i==0)):
                    st.write(f"**SMILES:** {result['smiles']}")
                    st.write(f"**Status:** {result['status']}")
                    st.write(f"**Affinity:** {result['affinity']} kcal/mol" if result['affinity'] is not None else "**Affinity:** N/A")
                    
                    if result['error']:
                        st.error(f"**Error:** {result['error']}")
                    
                    st.write("**Processing Steps:**")
                    for detail in result['details']:
                        st.write(f"- {detail}")
                    
                    if 'poses' in result and result['poses']:
                        st.write(f"**Poses Found:** {len(result['poses'])}")
                        for j, pose in enumerate(result['poses']):
                            st.write(f"  - Pose {pose.get('model', '?')}: Affinity = {pose.get('affinity', 0):.2f}, "
                                    f"RMSD LB = {pose.get('rmsd_lb', 0):.2f}, RMSD UB = {pose.get('rmsd_ub', 0):.2f}")
                            
                            pose_file = Path("docking_results") / f"docked_{i}_pose_{pose.get('model', 1)}.pdbqt"
                            if pose_file.exists():
                                st.download_button(
                                    label=f"Download Pose {pose.get('model', '?')}",
                                    data=open(str(pose_file), "rb").read(),
                                    file_name=f"ligand_{i}_pose_{pose.get('model', 1)}.pdbqt",
                                    mime="chemical/x-pdbqt",
                                    key=f"download_pose_{i}_{pose.get('model', 1)}"
                                )
            
            # Add system information
            with st.expander("System Information", expanded=False):
                st.write(f"**CPU Cores:** {multiprocessing.cpu_count()}")
                import sys
                st.write(f"**Python Version:** {sys.version}")
                st.write(f"**RDKit Version:** {Chem.rdBase.rdkitVersion}")
                st.write(f"**Streamlit Version:** {st.__version__}")
                
                # Show dependency status
                deps = check_dependencies()
                for dep, installed in deps.items():
                    status = "✅ Installed" if installed else "❌ Not Found"
                    st.write(f"**{dep}:** {status}")

if __name__ == "__main__":
    main()