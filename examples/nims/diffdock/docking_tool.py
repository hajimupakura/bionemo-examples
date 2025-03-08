# import os
# import sys
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
# from plotly.subplots import make_subplots
# import py3Dmol
# import logging
# from datetime import datetime
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# import io
# import base64

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
#                      config_file: str) -> Tuple[bool, str, List[Dict]]:
#     """Run AutoDock Vina docking and parse all poses."""
#     try:
#         # Verify input files
#         for filepath in [receptor_pdbqt, ligand_pdbqt, config_file]:
#             if not os.path.exists(filepath):
#                 return False, f"File not found: {filepath}", []

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
#             return False, "Docking output file was not created", []
            
#         if os.path.getsize(output_pdbqt) == 0:
#             return False, "Docking output file is empty", []
        
#         # Parse all poses
#         poses = []
#         current_pose = None
#         pose_count = 0
        
#         with open(output_pdbqt, 'r') as f:
#             for line in f:
#                 if "MODEL" in line:
#                     pose_count += 1
#                     current_pose = {
#                         'model': pose_count,
#                         'affinity': None,
#                         'rmsd_lb': None,
#                         'rmsd_ub': None
#                     }
#                 elif "REMARK VINA RESULT" in line:
#                     parts = line.split()
#                     if len(parts) >= 6:
#                         current_pose['affinity'] = float(parts[3])
#                         current_pose['rmsd_lb'] = float(parts[4])
#                         current_pose['rmsd_ub'] = float(parts[5])
#                 elif "ENDMDL" in line and current_pose is not None:
#                     poses.append(current_pose)
#                     current_pose = None
        
#         if not poses:
#             return False, "No valid poses found in output", []
            
#         return True, result.stdout, poses
#     except subprocess.CalledProcessError as e:
#         return False, f"Docking failed: {e.stderr}", []
#     except Exception as e:
#         return False, f"Error in docking: {str(e)}", []

# def extract_single_pose(input_pdbqt: str, output_pdbqt: str, model_num: int) -> bool:
#     """Extract a single pose from a multi-model PDBQT file."""
#     try:
#         with open(input_pdbqt, 'r') as f:
#             content = f.readlines()
        
#         if not content:
#             return False
            
#         output_content = []
#         in_target_model = False
#         current_model = 0
        
#         for line in content:
#             if line.startswith("MODEL"):
#                 current_model += 1
#                 in_target_model = (current_model == model_num)
            
#             if in_target_model or line.startswith("REMARK") or line.startswith("HEADER"):
#                 output_content.append(line)
                
#             if line.startswith("ENDMDL") and in_target_model:
#                 in_target_model = False
        
#         with open(output_pdbqt, 'w') as f:
#             f.writelines(output_content)
            
#         return True
#     except Exception as e:
#         logger.error(f"Error extracting pose: {str(e)}")
#         return False

# def process_ligand(args: Tuple[str, Path, int]) -> Dict:
#     """Process a single ligand."""
#     smiles, output_dir, idx = args
#     result = {
#         'smiles': smiles,
#         'affinity': None,
#         'status': 'failed',
#         'error': None,
#         'details': [],
#         'poses': []
#     }
    
#     try:
#         # Generate RDKit mol object for 2D structure
#         mol = Chem.MolFromSmiles(smiles)
#         if mol is not None:
#             AllChem.Compute2DCoords(mol)
#             result['mol'] = mol
        
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
#         success, msg, poses = run_vina_docking(
#             str(output_dir / "receptor.pdbqt"),
#             str(ligand_pdbqt),
#             str(output_pdbqt),
#             str(output_dir / "config.txt")
#         )
        
#         result['details'].append(f"Docking: {msg}")
#         if not success:
#             result['error'] = msg
#             return result
        
#         # Store all poses
#         result['poses'] = poses
        
#         # Extract each pose to a separate file
#         for pose in poses:
#             pose_file = output_dir / f"docked_{idx}_pose_{pose['model']}.pdbqt"
#             if extract_single_pose(str(output_pdbqt), str(pose_file), pose['model']):
#                 pose['file'] = str(pose_file)
        
#         # Use the best affinity as the main result
#         if poses:
#             result['affinity'] = poses[0]['affinity']
#             result['status'] = 'success'
            
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

# def mol_to_img(mol, size=(300, 300)):
#     """Convert an RDKit molecule to an image for Streamlit."""
#     if mol is None:
#         return None
    
#     img = Draw.MolToImage(mol, size=size)
#     buffered = io.BytesIO()
#     img.save(buffered, format="PNG")
#     return base64.b64encode(buffered.getvalue()).decode()

# def plot_affinities(results: List[Dict]):
#     """Plot binding affinities as a bar chart with improved styling."""
#     if not results:
#         st.warning("No results to plot")
#         return
    
#     # Prepare data
#     successful = [r for r in results if r['status'] == 'success']
#     if not successful:
#         st.warning("No successful docking results to visualize")
#         return
    
#     # Sort by affinity (lowest/strongest first)
#     sorted_results = sorted(successful, key=lambda x: x['affinity'])
    
#     smiles = [r['smiles'] for r in sorted_results]
#     affinities = [r['affinity'] for r in sorted_results]
    
#     # Create a colormap based on affinity values (lower is better)
#     norm = plt.Normalize(min(affinities), max(affinities))
#     colors = cm.viridis(norm(affinities))
#     colors = [(*c[:3], 0.8) for c in colors]  # Add transparency
    
#     # Create the plot
#     fig, ax = plt.subplots(figsize=(12, 6))
#     bars = ax.bar(range(len(affinities)), affinities, color=colors)
    
#     # Add data labels
#     for i, bar in enumerate(bars):
#         height = bar.get_height()
#         ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
#                 f'{affinities[i]:.1f}', ha='center', va='bottom', rotation=0,
#                 fontsize=9)
    
#     # Add grid lines and styling
#     ax.set_xlabel("Ligands", fontsize=12)
#     ax.set_ylabel("Binding Affinity (kcal/mol)", fontsize=12)
#     ax.set_title("Docking Results (Lower Values Indicate Stronger Binding)", fontsize=14)
#     ax.set_xticks(range(len(affinities)))
    
#     # Determine label length based on number of compounds
#     if len(smiles) <= 5:
#         ax.set_xticklabels(smiles, rotation=45, ha="right")
#     else:
#         ax.set_xticklabels([s[:10] + "..." if len(s) > 10 else s for s in smiles], 
#                            rotation=45, ha="right")
    
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.grid(axis='y', linestyle='--', alpha=0.7)
    
#     # Add a colorbar
#     sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
#     sm.set_array([])
#     cbar = plt.colorbar(sm, ax=ax)
#     cbar.set_label('Binding Affinity (kcal/mol)')
    
#     plt.tight_layout()
#     st.pyplot(fig)

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
#             marker_color='rgba(50, 168, 82, 0.7)',
#             name='Affinity Distribution'
#         )
#     ])
#     affinity_fig.update_layout(
#         title="Binding Affinity Distribution",
#         xaxis_title="Binding Affinity (kcal/mol)",
#         yaxis_title="Count",
#         template="plotly_white",
#         hovermode="closest"
#     )
#     plots['affinity_dist'] = affinity_fig
    
#     # Improved scatter plot of affinities
#     scatter_fig = go.Figure()
    
#     # Add a scatter trace
#     scatter_fig.add_trace(go.Scatter(
#         x=list(range(len(successful_results))),
#         y=[r['affinity'] for r in successful_results],
#         mode='markers',
#         text=[f"SMILES: {r['smiles'][:30]}...<br>Affinity: {r['affinity']:.2f}" 
#               for r in successful_results],
#         marker=dict(
#             size=15,
#             color=[r['affinity'] for r in successful_results],
#             colorscale='Viridis',
#             showscale=True,
#             colorbar=dict(title="Affinity (kcal/mol)"),
#             line=dict(width=1, color='black')
#         ),
#         name="Binding Affinities"
#     ))
    
#     scatter_fig.update_layout(
#         title="Binding Affinities by Ligand",
#         xaxis_title="Ligand Index",
#         yaxis_title="Binding Affinity (kcal/mol)",
#         template="plotly_white",
#         xaxis=dict(
#             tickmode='array',
#             tickvals=list(range(len(successful_results))),
#             ticktext=[f"{i+1}" for i in range(len(successful_results))]
#         )
#     )
#     plots['scatter'] = scatter_fig
    
#     # Add radar chart for comparing poses
#     # Find ligands with multiple poses
#     ligands_with_poses = [r for r in successful_results if len(r.get('poses', [])) > 1]
    
#     if ligands_with_poses:
#         # Create a radar chart for the first ligand with multiple poses
#         ligand = ligands_with_poses[0]
        
#         radar_fig = go.Figure()
        
#         categories = ['Affinity', 'RMSD LB', 'RMSD UB']
#         for pose in ligand.get('poses', [])[:5]:  # Limit to 5 poses for clarity
#             values = [
#                 pose.get('affinity', 0),
#                 pose.get('rmsd_lb', 0),
#                 pose.get('rmsd_ub', 0)
#             ]
            
#             radar_fig.add_trace(go.Scatterpolar(
#                 r=values,
#                 theta=categories,
#                 fill='toself',
#                 name=f"Pose {pose.get('model', '?')}"
#             ))
            
#         radar_fig.update_layout(
#             polar=dict(
#                 radialaxis=dict(
#                     visible=True,
#                     range=[
#                         min([p.get('affinity', 0) for p in ligand.get('poses', [])]) - 1,
#                         max([p.get('rmsd_ub', 0) for p in ligand.get('poses', [])]) + 1
#                     ]
#                 )),
#             title=f"Pose Comparison for First Ligand",
#             template="plotly_white"
#         )
#         plots['radar'] = radar_fig
        
#         # Create heatmap for all poses of all ligands
#         heatmap_data = []
#         for idx, result in enumerate(successful_results):
#             for pose in result.get('poses', [])[:3]:  # Limit to top 3 poses
#                 heatmap_data.append({
#                     'Ligand': f"Ligand {idx+1}",
#                     'Pose': f"Pose {pose.get('model', '?')}",
#                     'Affinity': pose.get('affinity', 0)
#                 })
        
#         if heatmap_data:
#             df_heatmap = pd.DataFrame(heatmap_data)
#             pivot_table = df_heatmap.pivot(index='Ligand', columns='Pose', values='Affinity')
            
#             heatmap_fig = px.imshow(
#                 pivot_table,
#                 labels=dict(x="Pose", y="Ligand", color="Affinity (kcal/mol)"),
#                 x=pivot_table.columns,
#                 y=pivot_table.index,
#                 color_continuous_scale="Viridis_r",  # Reversed so darker is stronger binding
#                 aspect="auto"
#             )
            
#             heatmap_fig.update_layout(
#                 title="Binding Affinity Heatmap Across Poses",
#                 template="plotly_white",
#                 coloraxis_colorbar=dict(title="Affinity (kcal/mol)")
#             )
            
#             # Add text annotations
#             for i, ligand in enumerate(pivot_table.index):
#                 for j, pose in enumerate(pivot_table.columns):
#                     value = pivot_table.iloc[i, j]
#                     if not pd.isna(value):
#                         heatmap_fig.add_annotation(
#                             x=pose, y=ligand, text=f"{value:.2f}",
#                             showarrow=False, font=dict(color="white" if value < -8 else "black")
#                         )
            
#             plots['heatmap'] = heatmap_fig
    
#     # Add bubble chart comparing all compounds
#     bubble_data = []
#     for idx, result in enumerate(successful_results):
#         if 'poses' in result and result['poses']:
#             num_poses = len(result['poses'])
#             best_affinity = result['affinity']
#             rmsd_range = 0
            
#             if num_poses > 1:
#                 rmsd_range = max([p.get('rmsd_ub', 0) for p in result['poses']]) - \
#                             min([p.get('rmsd_lb', 0) for p in result['poses']])
            
#             bubble_data.append({
#                 'Ligand': f"Ligand {idx+1}",
#                 'Affinity': best_affinity,
#                 'Poses': num_poses,
#                 'RMSD Range': rmsd_range,
#                 'SMILES': result['smiles']
#             })
    
#     if bubble_data:
#         df_bubble = pd.DataFrame(bubble_data)
#         bubble_fig = px.scatter(
#             df_bubble,
#             x='Affinity',
#             y='RMSD Range',
#             size='Poses',
#             color='Affinity',
#             hover_name='Ligand',
#             hover_data=['SMILES'],
#             color_continuous_scale='Viridis_r',
#             size_max=30
#         )
        
#         bubble_fig.update_layout(
#             title="Ligand Comparison - Affinity vs Conformational Diversity",
#             xaxis_title="Binding Affinity (kcal/mol)",
#             yaxis_title="RMSD Range (Å)",
#             template="plotly_white"
#         )
        
#         plots['bubble'] = bubble_fig
    
#     return plots
# def show_basic_3dmol(receptor_pdbfile: str, ligand_pdbqtfile: str, viewer_height: int = 500) -> str:
#     """Create a basic visualization using py3Dmol with download options."""
#     try:
#         # Check if files exist
#         if not os.path.exists(receptor_pdbfile):
#             return f"""
#             <div style="height: {viewer_height}px; width: 100%; 
#                        display: flex; justify-content: center; align-items: center; 
#                        background-color: #f8f9fa; border: 1px solid #ddd;">
#                 <p>Receptor file not found: {receptor_pdbfile}</p>
#             </div>
#             """
            
#         if not os.path.exists(ligand_pdbqtfile):
#             return f"""
#             <div style="height: {viewer_height}px; width: 100%; 
#                        display: flex; justify-content: center; align-items: center; 
#                        background-color: #f8f9fa; border: 1px solid #ddd;">
#                 <p>Ligand file not found: {ligand_pdbqtfile}</p>
#             </div>
#             """

#         # Read file contents
#         with open(receptor_pdbfile, 'r') as f:
#             receptor_data = f.read()
#             receptor_length = len(receptor_data)
            
#         with open(ligand_pdbqtfile, 'r') as f:
#             ligand_data = f.read()
#             ligand_length = len(ligand_data)

#         # Create py3Dmol view
#         viewer = py3Dmol.view(width=800, height=viewer_height)
#         viewer.addModel(receptor_data, 'pdb')
#         viewer.setStyle({'model': 0}, {'cartoon': {'color': 'lightgray'}})
#         viewer.addModel(ligand_data, 'pdbqt')
#         viewer.setStyle({'model': 1}, {'stick': {'colorscheme': 'greenCarbon'}})
#         viewer.zoomTo()
#         viewer.setBackgroundColor('white')

#         # Generate HTML with download buttons (using Streamlit components instead of JS)
#         html_content = f"""
#         <div style="text-align: center; padding: 10px;">
#             <div style="font-size: 14px; margin-bottom: 10px;">
#                 <b>Debug Info:</b> Receptor file size: {receptor_length} bytes, Ligand file size: {ligand_length} bytes
#             </div>
#             {viewer.render()}
#         </div>
#         """

#         # Return the HTML and let Streamlit handle downloads separately
#         return html_content

#     except Exception as e:
#         logger.error(f"Error in basic 3D visualization: {str(e)}")
#         return f"""
#         <div style="height: {viewer_height}px; width: 100%; 
#                    display: flex; justify-content: center; align-items: center; 
#                    background-color: #f8f9fa; border: 1px solid #ddd;">
#             <p>Error loading 3D visualization: {str(e)}</p>
#         </div>
#         """

# def create_ngl_viewer(receptor_pdbfile: str, ligand_pdbqtfile: str, viewer_height: int = 500) -> str:
#     """Create a visualization using NGL Viewer instead of 3Dmol.js."""
#     try:
#         # Check if files exist
#         if not os.path.exists(receptor_pdbfile) or not os.path.exists(ligand_pdbqtfile):
#             return show_basic_3dmol(receptor_pdbfile, ligand_pdbqtfile, viewer_height)
        
#         # Use NGL Viewer which might be more reliable
#         html_content = f"""
#         <div id="viewport" style="width:100%; height:{viewer_height}px;"></div>
#         <script src="https://cdnjs.cloudflare.com/ajax/libs/ngl/2.0.0-dev.36/ngl.js"></script>
#         <script>
#             document.addEventListener("DOMContentLoaded", function() {{
#                 // Create NGL Stage object
#                 var stage = new NGL.Stage("viewport", {{backgroundColor: "white"}});

#                 // Load files from URLs or file paths when within the same origin
#                 Promise.all([
#                     stage.loadFile('{receptor_pdbfile}'),
#                     stage.loadFile('{ligand_pdbqtfile}')
#                 ]).then(function (components) {{
#                     // receptor 
#                     components[0].addRepresentation("cartoon", {{
#                         color: "lightgrey",
#                         opacity: 0.7
#                     }});
                    
#                     // ligand
#                     components[1].addRepresentation("licorice", {{
#                         colorScheme: "element",
#                         radius: 0.5
#                     }});
                    
#                     // zoom to fit the viewport
#                     stage.autoView();
#                 }});
#             }});
#         </script>
#         """
        
#         return html_content
#     except Exception as e:
#         logger.error(f"Error in NGL visualization: {str(e)}")
#         return show_basic_3dmol(receptor_pdbfile, ligand_pdbqtfile, viewer_height)

# def create_molstar_viewer(receptor_pdbfile: str, ligand_pdbqtfile: str, viewer_height: int = 500) -> str:
#     """Create a visualization using Mol* Viewer as another alternative."""
#     try:
#         # Check if files exist
#         if not os.path.exists(receptor_pdbfile) or not os.path.exists(ligand_pdbqtfile):
#             return show_basic_3dmol(receptor_pdbfile, ligand_pdbqtfile, viewer_height)
        
#         # Use Mol* Viewer which might be more reliable
#         html_content = f"""
#         <div id="molstar-viewer" style="width:100%; height:{viewer_height}px; position: relative;"></div>
#         <script src="https://cdn.jsdelivr.net/npm/molstar/build/molstar.js"></script>
#         <script>
#             document.addEventListener("DOMContentLoaded", function() {{
#                 molstar.Viewer.create('molstar-viewer', {{
#                     layoutIsExpanded: false,
#                     layoutShowControls: false,
#                     layoutShowSequence: false,
#                     layoutShowLog: false,
#                     layoutShowLeftPanel: false,
#                 }}).then(viewer => {{
#                     Promise.all([
#                         viewer.loadStructureFromUrl('{receptor_pdbfile}', 'pdb', {{ asTrajectory: false }}),
#                         viewer.loadStructureFromUrl('{ligand_pdbqtfile}', 'pdbqt', {{ asTrajectory: false }})
#                     ]).then(() => {{
#                         viewer.updateStyle({{
#                             chain: {{ A: {{ type: 'cartoon', color: 'uniform', params: {{ color: {{ r: 200, g: 200, b: 200 }} }} }} }}
#                         }});
#                     }});
#                 }});
#             }});
#         </script>
#         """
        
#         return html_content
#     except Exception as e:
#         logger.error(f"Error in Mol* visualization: {str(e)}")
#         return show_basic_3dmol(receptor_pdbfile, ligand_pdbqtfile, viewer_height)

# def show_3dmol(receptor_pdbfile: str, ligand_pdbqtfile: str, viewer_height: int = 500) -> str:
#     """Create a 3D visualization using py3Dmol."""
#     try:
#         # Check if files exist
#         if not os.path.exists(receptor_pdbfile):
#             return f"""
#             <div style="height: {viewer_height}px; width: 100%; 
#                        display: flex; justify-content: center; align-items: center; 
#                        background-color: #f8f9fa; border: 1px solid #ddd;">
#                 <p>Receptor file not found: {receptor_pdbfile}</p>
#             </div>
#             """
            
#         if not os.path.exists(ligand_pdbqtfile):
#             return f"""
#             <div style="height: {viewer_height}px; width: 100%; 
#                        display: flex; justify-content: center; align-items: center; 
#                        background-color: #f8f9fa; border: 1px solid #ddd;">
#                 <p>Ligand file not found: {ligand_pdbqtfile}</p>
#             </div>
#             """

#         # Read file contents
#         with open(receptor_pdbfile, 'r') as f:
#             receptor_data = f.read()
            
#         with open(ligand_pdbqtfile, 'r') as f:
#             ligand_data = f.read()

#         # Create py3Dmol view
#         viewer = py3Dmol.view(width=800, height=viewer_height)
        
#         # Add receptor
#         viewer.addModel(receptor_data, 'pdb')
#         viewer.setStyle({'model': 0}, {
#             'cartoon': {'color': 'lightgray', 'opacity': 0.8}
#         })
#         viewer.addSurface('VDW', {
#             'opacity': 0.7,
#             'color': 'white'
#         }, {'model': 0})

#         # Add ligand
#         viewer.addModel(ligand_data, 'pdbqt')
#         viewer.setStyle({'model': 1}, {
#             'stick': {'colorscheme': 'greenCarbon', 'radius': 0.3},
#             'sphere': {'scale': 0.3}
#         })

#         # Set view parameters
#         viewer.zoomTo()
#         viewer.rotate(90, {'x': 0, 'y': 1, 'z': 0})
#         viewer.setBackgroundColor('white')

#         # Generate HTML
#         html_content = f"""
#         <div style="position: relative; width: 100%; height: {viewer_height}px; 
#                     border: 1px solid #ccc; border-radius: 4px; background-color: #f8f9fa;">
#             {viewer.render()}
#             <div style="position: absolute; bottom: 10px; left: 10px; 
#                        background-color: rgba(255,255,255,0.7); padding: 5px; 
#                        border-radius: 5px; font-size: 12px; z-index: 1000;">
#                 Scroll to zoom, drag to rotate, Shift+drag to translate
#             </div>
#         </div>
#         """

#         return html_content

#     except Exception as e:
#         logger.error(f"Error in 3D visualization: {str(e)}")
#         return f"""
#         <div style="height: {viewer_height}px; width: 100%; 
#                    display: flex; justify-content: center; align-items: center; 
#                    background-color: #f8f9fa; border: 1px solid #ddd;">
#             <p>Error loading 3D visualization: {str(e)}</p>
#         </div>
#         """
# import logging
# def show_multiple_poses(receptor_pdbfile: str, pose_files: List[str], viewer_height: int = 500) -> str:
#     logger = logging.getLogger(__name__) 
#     logger.info(f"Receptor path: {receptor_pdbfile}, Exists: {os.path.exists(receptor_pdbfile)}")
#     logger.info(f"Pose files: {pose_files}")
#     for pf in pose_files:
#         logger.info(f"Pose file {pf} exists: {os.path.exists(pf)}")
#     """Create a py3Dmol visualization with multiple poses."""
#     try:
#         # Check inputs
#         if not pose_files:
#             return f"""
#             <div style="height: {viewer_height}px; width: 100%; 
#                        display: flex; justify-content: center; align-items: center; 
#                        background-color: #f8f9fa; border: 1px solid #ddd;">
#                 <p>No pose files found to visualize</p>
#             </div>
#             """

#         if not os.path.exists(receptor_pdbfile):
#             return f"""
#             <div style="height: {viewer_height}px; width: 100%; 
#                        display: flex; justify-content: center; align-items: center; 
#                        background-color: #f8f9fa; border: 1px solid #ddd;">
#                 <p>Receptor file not found: {receptor_pdbfile}</p>
#             </div>
#             """

#         # Filter valid pose files
#         valid_pose_files = [f for f in pose_files if os.path.exists(f)]
#         if not valid_pose_files:
#             return f"""
#             <div style="height: {viewer_height}px; width: 100%; 
#                        display: flex; justify-content: center; align-items: center; 
#                        background-color: #f8f9fa; border: 1px solid #ddd;">
#                 <p>No valid pose files exist</p>
#             </div>
#             """

#         # Read receptor
#         with open(receptor_pdbfile, 'r') as f:
#             receptor_data = f.read()

#         # Create py3Dmol viewer
#         viewer = py3Dmol.view(width=800, height=viewer_height)

#         # Add receptor
#         viewer.addModel(receptor_data, 'pdb')
#         viewer.setStyle({'model': 0}, {
#             'cartoon': {'color': 'lightgray', 'opacity': 0.7}
#         })
#         viewer.addSurface('VDW', {
#             'opacity': 0.5,
#             'color': 'white'
#         }, {'model': 0})

#         # Colors for poses
#         colors = ['green', 'blue', 'red', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'white']
#         colors = colors[:len(valid_pose_files)]

#         # Add poses
#         for i, pose_file in enumerate(valid_pose_files):
#             with open(pose_file, 'r') as f:
#                 pose_data = f.read()
            
#             viewer.addModel(pose_data, 'pdbqt')
#             viewer.setStyle({'model': i + 1}, {
#                 'stick': {'colorscheme': f'{colors[i]}Carbon', 'radius': 0.3}
#             })

#         # Set view parameters
#         viewer.zoomTo()
#         viewer.setBackgroundColor('white')

#         # Create legend
#         legend_html = ""
#         for i, color in enumerate(colors):
#             pose_num = i + 1
#             legend_html += f'<span style="color: {color}; margin-right: 10px;">● Pose {pose_num}</span>'

#         # Generate HTML
#         html_content = f"""
#         <div style="position: relative; width: 100%; height: {viewer_height}px; 
#                     border: 1px solid #ccc; border-radius: 4px; background-color: #f8f9fa;">
#             {viewer.render()}
#             <div style="position: absolute; bottom: 10px; left: 10px; 
#                        background-color: rgba(255,255,255,0.7); padding: 5px; 
#                        border-radius: 5px; font-size: 12px; z-index: 1000;">
#                 <p style="margin: 0; font-size: 12px;">Pose colors:</p>
#                 {legend_html}
#             </div>
#         </div>
#         """

#         return html_content

#     except Exception as e:
#         logger.error(f"Error in multiple pose visualization: {str(e)}")
#         return f"""
#         <div style="height: {viewer_height}px; width: 100%; 
#                    display: flex; justify-content: center; align-items: center; 
#                    background-color: #f8f9fa; border: 1px solid #ddd;">
#             <p>Error loading multiple pose visualization: {str(e)}</p>
#         </div>
#         """

# def create_ligand_interaction_diagram(receptor_pdbfile: str, ligand_pdbqtfile: str, width=800, height=600):
#     """
#     Create a simplified 2D diagram of ligand-receptor interactions.
#     This is a placeholder that would be enhanced with actual interaction analysis in production.
#     """
#     try:
#         # In a real implementation, you would:
#         # 1. Extract ligand and receptor structures
#         # 2. Calculate interactions (H-bonds, hydrophobic, etc.)
#         # 3. Create a 2D diagram using a library like RDKit or similar
        
#         # For now, we'll create a simple placeholder
#         fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
        
#         # Draw a placeholder circle for receptor pocket
#         circle = plt.Circle((0.5, 0.5), 0.4, fill=False, color='gray', linestyle='--')
#         ax.add_patch(circle)
        
#         # Add text for interactions (this would be calculated in production)
#         ax.text(0.5, 0.8, "Receptor Binding Pocket", ha='center', fontsize=14, color='gray')
#         ax.text(0.5, 0.3, "Ligand", ha='center', fontsize=14, color='green')
#         ax.text(0.3, 0.6, "H-bond", ha='center', fontsize=10, color='blue')
#         ax.text(0.7, 0.6, "Hydrophobic", ha='center', fontsize=10, color='orange')
#         ax.text(0.5, 0.5, "π-stacking", ha='center', fontsize=10, color='purple')
        
#         # Add arrows for interactions
#         ax.arrow(0.35, 0.55, 0, -0.1, head_width=0.02, head_length=0.02, fc='blue', ec='blue')
#         ax.arrow(0.65, 0.55, 0, -0.1, head_width=0.02, head_length=0.02, fc='orange', ec='orange')
#         ax.arrow(0.5, 0.45, 0, -0.03, head_width=0.02, head_length=0.02, fc='purple', ec='purple')
        
#         ax.set_xlim(0, 1)
#         ax.set_ylim(0, 1)
#         ax.axis('off')
        
#         return fig
#     except Exception as e:
#         logger.error(f"Error creating interaction diagram: {str(e)}")
#         fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
#         ax.text(0.5, 0.5, f"Error creating diagram: {str(e)}", ha='center', va='center')
#         ax.axis('off')
#         return fig

# def main():
#     st.set_page_config(layout="wide", page_title="Molecular Docking Tool")
    
#     # Add custom CSS
#     st.markdown("""
#     <style>
#     .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
#         font-size: 1.2rem;
#     }
#     .stTabs [data-baseweb="tab-list"] {
#         gap: 8px;
#     }
#     .main .block-container {
#         padding-top: 2rem;
#     }
#     </style>
#     """, unsafe_allow_html=True)
    
#     st.title("Advanced Molecular Docking & Visualization Tool")
#     st.write("Upload receptor, input ligand SMILES, and analyze docking results with enhanced visualizations")
    
#     # Initialize session state if not exists
#     if 'results' not in st.session_state:
#         st.session_state.results = None
#     if 'receptor_path' not in st.session_state:
#         st.session_state.receptor_path = None
#     if 'selected_ligand_idx' not in st.session_state:
#         st.session_state.selected_ligand_idx = 0
#     if 'selected_pose_idx' not in st.session_state:
#         st.session_state.selected_pose_idx = 0
    
#     # Main interface
#     col1, col2 = st.columns([1, 1])
    
#     with col1:
#         receptor_file = st.file_uploader("Upload Receptor (PDB)", type=["pdb"])
#         if receptor_file:
#             receptor_path = Path("docking_results") / "receptor.pdb"
#             receptor_path.parent.mkdir(exist_ok=True)
#             receptor_path.write_bytes(receptor_file.read())
#             st.session_state.receptor_path = receptor_path
#             st.success(f"Receptor uploaded: {receptor_file.name}")
    
#     with col2:
#         ligand_input = st.text_area(
#             "Enter SMILES strings (one per line)", 
#             "CC(=O)OC1=CC=CC=C1C(=O)O\nCN1C=NC2=C1C(=O)N(C(=O)N2C)C\nCCCC(=O)O",
#             height=100
#         )
#         smiles_list = [s.strip() for s in ligand_input.split("\n") if s.strip()]
        
#         if st.session_state.receptor_path:
#             st.info(f"Ready to dock {len(smiles_list)} ligand(s) with receptor: {receptor_file.name if receptor_file else Path(st.session_state.receptor_path).name}")

#     # Parameters expander
#     with st.expander("Advanced Parameters", expanded=False):
#         col1, col2 = st.columns(2)
#         with col1:
#             exhaustiveness = st.slider("Vina Exhaustiveness", 1, 16, 8, 
#                 help="Higher values give more thorough search but take longer")
#             energy_range = st.slider("Energy Range (kcal/mol)", 1, 10, 3,
#                 help="Maximum energy difference between best and worst binding mode")
#         with col2:
#             num_modes = st.slider("Number of Binding Modes", 1, 20, 9,
#                 help="Maximum number of binding modes to generate")
#             cpu_count = st.slider("CPU Cores", 1, multiprocessing.cpu_count(), 
#                 max(1, multiprocessing.cpu_count() - 1), 
#                 help="Number of CPU cores to use")

#     run_col1, run_col2 = st.columns([3, 1])
#     with run_col1:
#         run_button = st.button("Run Docking", type="primary", use_container_width=True)
#     with run_col2:
#         help_button = st.button("Help & Info", type="secondary", use_container_width=True)
    
#     if help_button:
#         with st.expander("Help & Information", expanded=True):
#             st.markdown("""
#             ## How to Use This Tool
#             1. **Upload a Receptor**: Provide a PDB file of your protein target
#             2. **Enter SMILES**: Add one or more SMILES strings for the ligands you want to dock
#             3. **Run Docking**: Click the Run Docking button to start the process
#             4. **View Results**: Use the tabs to see different visualizations of the results
            
#             ## Visualizations Explained
#             - **Results Table**: Overview of all docking results
#             - **2D Visualizations**: Various plots and charts analyzing binding affinities
#             - **3D Visualization**: Interactive view of docked poses in 3D
#             - **Multiple Poses**: View and compare different binding poses for each ligand
#             - **Details**: Process logs and technical information
            
#             ## About Binding Affinity
#             Lower (more negative) binding affinity values indicate stronger binding. The unit is kcal/mol.
#             """)

#     if run_button:
#         if not st.session_state.receptor_path:
#             st.error("Please upload a receptor file")
#         elif not smiles_list:
#             st.error("Please enter at least one SMILES string")
#         else:
#             status_area = st.empty()
#             status_area.info("Initializing docking process...")
            
#             try:
#                 # Update config with slider values before running
#                 config_file = Path("docking_results") / "config.txt"
#                 if config_file.exists():
#                     with open(config_file, "r") as f:
#                         config_content = f.read()
                    
#                     # Update configuration with slider values
#                     config_content = config_content.replace("exhaustiveness = 8", f"exhaustiveness = {exhaustiveness}")
#                     config_content = config_content.replace("num_modes = 9", f"num_modes = {num_modes}")
#                     config_content = config_content.replace("energy_range = 3", f"energy_range = {energy_range}")
#                     config_content = config_content.replace(f"cpu = {max(1, multiprocessing.cpu_count() - 1)}", f"cpu = {cpu_count}")
                    
#                     with open(config_file, "w") as f:
#                         f.write(config_content)
                
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
#         tab1, tab2, tab3, tab4, tab5 = st.tabs([
#             "🔍 Results", 
#             "📊 2D Visualization", 
#             "🔄 3D Visualization", 
#             "🧩 Multiple Poses", 
#             "📋 Details"
#         ])
        
#         with tab1:
#             # Create a more detailed DataFrame
#             df = pd.DataFrame([{
#                 'SMILES': r['smiles'],
#                 'Status': r['status'],
#                 'Affinity (kcal/mol)': r['affinity'] if r['affinity'] is not None else 'N/A',
#                 'Poses': len(r.get('poses', [])),
#                 'Error': r['error'] if r['error'] is not None else 'None'
#             } for r in results])
            
#             # Display ligand structures if available
#             st.subheader("Docking Results Overview")
#             st.dataframe(df, use_container_width=True)
            
#             if successful:
#                 st.subheader("Top Ligand Structures")
                
#                 # Sort by binding affinity
#                 sorted_results = sorted(successful, key=lambda x: x.get('affinity', 0))
                
#                 # Display top 5 ligands as a grid
#                 cols = st.columns(min(5, len(sorted_results)))
                
#                 for i, (col, result) in enumerate(zip(cols, sorted_results[:5])):
#                     with col:
#                         if 'mol' in result:
#                             img = mol_to_img(result['mol'])
#                             if img:
#                                 st.image(f"data:image/png;base64,{img}", 
#                                          caption=f"Ligand {i+1}\nAffinity: {result['affinity']:.2f}")
#                             else:
#                                 st.write(f"Ligand {i+1}\nAffinity: {result['affinity']:.2f}")
#                         else:
#                             st.write(f"Ligand {i+1}\nAffinity: {result['affinity']:.2f}")
        
#         with tab2:
#             if len(successful) > 0:
#                 st.subheader("Interactive Visualizations")
                
#                 # Add interactive plots
#                 plots = create_interactive_results(results)
#                 if plots:
#                     # Display first row of plots
#                     col1, col2 = st.columns(2)
#                     with col1:
#                         st.plotly_chart(plots['affinity_dist'], use_container_width=True)
#                     with col2:
#                         st.plotly_chart(plots['scatter'], use_container_width=True)
                    
#                     # Display second row if we have the other plots
#                     if 'bubble' in plots or 'heatmap' in plots:
#                         col1, col2 = st.columns(2)
#                         if 'bubble' in plots:
#                             with col1:
#                                 st.plotly_chart(plots['bubble'], use_container_width=True)
#                         if 'heatmap' in plots:
#                             with col2:
#                                 st.plotly_chart(plots['heatmap'], use_container_width=True)
                    
#                     if 'radar' in plots:
#                         st.plotly_chart(plots['radar'], use_container_width=True)
                
#                 st.subheader("Binding Affinities Visualization")
#                 plot_affinities(successful)
#             else:
#                 st.warning("No successful docking results to visualize")
        
#         with tab3:
#             if len(successful) > 0:
#                 st.subheader("3D Structure Visualization")
                
#                 # Add ligand selector
#                 ligand_options = [f"Ligand {i+1} (Affinity: {r['affinity']:.2f})" 
#                                 for i, r in enumerate(successful)]
#                 selected_ligand = st.selectbox(
#                     "Select ligand to visualize:", 
#                     ligand_options,
#                     index=min(st.session_state.selected_ligand_idx, len(ligand_options)-1)
#                 )
                
#                 # Update selected ligand index
#                 st.session_state.selected_ligand_idx = ligand_options.index(selected_ligand)
#                 ligand_idx = st.session_state.selected_ligand_idx
                
#                 # Add pose selector
#                 selected_result = successful[ligand_idx]
#                 if 'poses' in selected_result and selected_result['poses']:
#                     pose_options = [f"Pose {p.get('model', '?')} (Affinity: {p.get('affinity', 0):.2f})" 
#                                   for p in selected_result['poses']]
                    
#                     selected_pose = st.selectbox(
#                         "Select pose to visualize:", 
#                         pose_options,
#                         index=min(st.session_state.selected_pose_idx, len(pose_options)-1)
#                     )
#                     st.session_state.selected_pose_idx = pose_options.index(selected_pose)
#                     pose_idx = selected_result['poses'][st.session_state.selected_pose_idx].get('model', 1)
                    
#                     # Get the pose file
#                     output_pdbqt = Path("docking_results") / f"docked_{ligand_idx}_pose_{pose_idx}.pdbqt"
                    
#                     if output_pdbqt.exists() and st.session_state.receptor_path:
#                         try:
#                             st.write("Drag to rotate, scroll to zoom")
#                             html_content = show_3dmol(
#                                 str(st.session_state.receptor_path),
#                                 str(output_pdbqt),
#                                 viewer_height=600
#                             )
#                             st.components.v1.html(html_content, height=600)
                            
#                             # Add a download button for the current pose
#                             if output_pdbqt.exists():
#                                 with open(str(output_pdbqt), "rb") as f:
#                                     pose_bytes = f.read()
#                                 st.download_button(
#                                     label="Download Current Pose (PDBQT)",
#                                     data=pose_bytes,
#                                     file_name=f"ligand_{ligand_idx}_pose_{pose_idx}.pdbqt",
#                                     mime="chemical/x-pdbqt"
#                                 )
#                         except Exception as e:
#                             st.error(f"Error displaying 3D structure: {str(e)}")
#                     else:
#                         st.warning("3D visualization not available - missing output files")
                
#                 # Draw a simplified interaction diagram (this would be enhanced in production)
#                 st.subheader("Ligand-Receptor Interactions (Schematic)")
#                 interaction_fig = create_ligand_interaction_diagram(
#                     str(st.session_state.receptor_path),
#                     str(Path("docking_results") / f"docked_{ligand_idx}.pdbqt")
#                 )
#                 st.pyplot(interaction_fig)
#             else:
#                 st.warning("No successful docking results to visualize")
        
#         with tab4:
#             if len(successful) > 0:
#                 st.subheader("Multiple Pose Visualization")
                
#                 # Filter ligands with multiple poses
#                 ligands_with_poses = [(i, r) for i, r in enumerate(successful) 
#                                       if 'poses' in r and len(r.get('poses', [])) > 1]
                
#                 if ligands_with_poses:
#                     # Create options for the dropdown
#                     ligand_options = [
#                         f"Ligand {i+1} (Affinity: {r['affinity']:.2f}, Poses: {len(r.get('poses', []))})" 
#                         for i, r in ligands_with_poses
#                     ]
                    
#                     # Show dropdown for selecting ligand
#                     selected_multi_ligand = st.selectbox(
#                         "Select ligand with multiple poses:", 
#                         ligand_options,
#                         index=0
#                     )
                    
#                     # Extract index and result for selected ligand
#                     selected_idx = ligand_options.index(selected_multi_ligand)
#                     multi_ligand_idx, selected_result = ligands_with_poses[selected_idx]
                    
#                     # Filter poses to display
#                     max_poses = st.slider("Maximum number of poses to display", 1, 
#                                          min(9, len(selected_result.get('poses', []))), 
#                                          min(3, len(selected_result.get('poses', []))))
                    
#                     # Collect pose files
#                     pose_files = []
#                     valid_poses = []
                    
#                     for pose in selected_result.get('poses', [])[:max_poses]:
#                         pose_model = pose.get('model', 1)
#                         pose_file = Path("docking_results") / f"docked_{multi_ligand_idx}_pose_{pose_model}.pdbqt"
                        
#                         if pose_file.exists():
#                             pose_files.append(str(pose_file))
#                             valid_poses.append(pose)
                    
#                     if pose_files and st.session_state.receptor_path:
#                         try:
#                             st.write("Multiple binding poses shown simultaneously (different colors)")
                            
#                             # Debug information
#                             st.info(f"Found {len(pose_files)} valid pose files to display")
                            
#                             # Use the more robust function
#                             html_content = show_multiple_poses(
#                                 str(st.session_state.receptor_path),
#                                 pose_files,
#                                 viewer_height=700
#                             )
#                             st.components.v1.html(html_content, height=700)
                            
#                             # Add a binding site analysis
#                             st.subheader("Binding Site Analysis")
                            
#                             # Create columns for pose properties
#                             col1, col2, col3 = st.columns(3)
                            
#                             # Display properties of each pose
#                             for i, pose in enumerate(valid_poses):
#                                 if i % 3 == 0:
#                                     with col1:
#                                         st.write(f"**Pose {pose.get('model', '?')}**")
#                                         st.write(f"Affinity: {pose.get('affinity', 0):.2f} kcal/mol")
#                                         st.write(f"RMSD LB: {pose.get('rmsd_lb', 0):.2f} Å")
#                                         st.write(f"RMSD UB: {pose.get('rmsd_ub', 0):.2f} Å")
#                                         st.write("---")
#                                 elif i % 3 == 1:
#                                     with col2:
#                                         st.write(f"**Pose {pose.get('model', '?')}**")
#                                         st.write(f"Affinity: {pose.get('affinity', 0):.2f} kcal/mol")
#                                         st.write(f"RMSD LB: {pose.get('rmsd_lb', 0):.2f} Å")
#                                         st.write(f"RMSD UB: {pose.get('rmsd_ub', 0):.2f} Å")
#                                         st.write("---")
#                                 else:
#                                     with col3:
#                                         st.write(f"**Pose {pose.get('model', '?')}**")
#                                         st.write(f"Affinity: {pose.get('affinity', 0):.2f} kcal/mol")
#                                         st.write(f"RMSD LB: {pose.get('rmsd_lb', 0):.2f} Å")
#                                         st.write(f"RMSD UB: {pose.get('rmsd_ub', 0):.2f} Å")
#                                         st.write("---")
                                        
#                             # Add pose comparison visualization
#                             if len(valid_poses) > 1:
#                                 st.subheader("Pose Comparison")
                                
#                                 # Create a side-by-side plot comparing RMSD vs Affinity
#                                 comparison_data = [
#                                     {
#                                         'Pose': f"Pose {p.get('model', '?')}",
#                                         'Affinity': p.get('affinity', 0),
#                                         'RMSD': p.get('rmsd_lb', 0)
#                                     }
#                                     for p in valid_poses
#                                 ]
                                
#                                 if comparison_data:
#                                     df_comparison = pd.DataFrame(comparison_data)
                                    
#                                     fig = px.scatter(
#                                         df_comparison,
#                                         x='RMSD',
#                                         y='Affinity',
#                                         text='Pose',
#                                         color='Affinity',
#                                         color_continuous_scale='Viridis_r',
#                                         size=[20] * len(df_comparison),
#                                         title="Pose Comparison: RMSD vs Binding Affinity"
#                                     )
                                    
#                                     fig.update_traces(
#                                         textposition='top center',
#                                         marker=dict(line=dict(width=1, color='DarkSlateGrey'))
#                                     )
                                    
#                                     fig.update_layout(
#                                         xaxis_title="RMSD from Best Mode (Å)",
#                                         yaxis_title="Binding Affinity (kcal/mol)",
#                                         template="plotly_white"
#                                     )
                                    
#                                     st.plotly_chart(fig, use_container_width=True)
#                         except Exception as e:
#                             st.error(f"Error displaying multiple poses: {str(e)}")
#                             st.code(traceback.format_exc())
#                     else:
#                         st.warning("Multiple pose visualization not available - missing output files")
#                         st.info(f"Looking for pose files in: {Path('docking_results')}")
#                         st.info(f"Receptor path: {st.session_state.receptor_path}")
#                 else:
#                     st.info("No ligands with multiple poses found. Run docking with more exhaustive parameters to generate multiple poses.")
#             else:
#                 st.warning("No successful docking results to visualize")
        
#         with tab5:
#             st.subheader("Detailed Process Information")
            
#             # Add configuration display
#             config_file = Path("docking_results") / "config.txt"
#             if config_file.exists():
#                 with st.expander("Docking Configuration", expanded=True):
#                     with open(config_file, "r") as f:
#                         config_content = f.read()
#                     st.code(config_content, language="bash")
            
#             # Add receptor information
#             with st.expander("Receptor Information", expanded=False):
#                 if st.session_state.receptor_path and Path(st.session_state.receptor_path).exists():
#                     try:
#                         mol = Chem.MolFromPDBFile(str(st.session_state.receptor_path), removeHs=False)
#                         if mol is not None:
#                             st.write(f"**Atoms:** {mol.GetNumAtoms()}")
#                             st.write(f"**Bonds:** {mol.GetNumBonds()}")
#                             st.write(f"**Residues:** {len(Chem.SplitMolByPDBResidues(mol))}")
#                     except Exception as e:
#                         st.error(f"Error analyzing receptor: {str(e)}")
            
#             # Individual ligand details
#             for i, result in enumerate(results):
#                 with st.expander(f"Ligand {i+1} Details", expanded=(i==0)):
#                     st.write(f"**SMILES:** {result['smiles']}")
#                     st.write(f"**Status:** {result['status']}")
#                     st.write(f"**Affinity:** {result['affinity']} kcal/mol" if result['affinity'] is not None else "**Affinity:** N/A")
                    
#                     if result['error']:
#                         st.error(f"**Error:** {result['error']}")
                    
#                     st.write("**Processing Steps:**")
#                     for detail in result['details']:
#                         st.write(f"- {detail}")
                    
#                     if 'poses' in result and result['poses']:
#                         st.write(f"**Poses Found:** {len(result['poses'])}")
#                         for j, pose in enumerate(result['poses']):
#                             st.write(f"  - Pose {pose.get('model', '?')}: Affinity = {pose.get('affinity', 0):.2f}, "
#                                     f"RMSD LB = {pose.get('rmsd_lb', 0):.2f}, RMSD UB = {pose.get('rmsd_ub', 0):.2f}")
                            
#                             pose_file = Path("docking_results") / f"docked_{i}_pose_{pose.get('model', 1)}.pdbqt"
#                             if pose_file.exists():
#                                 st.download_button(
#                                     label=f"Download Pose {pose.get('model', '?')}",
#                                     data=open(str(pose_file), "rb").read(),
#                                     file_name=f"ligand_{i}_pose_{pose.get('model', 1)}.pdbqt",
#                                     mime="chemical/x-pdbqt",
#                                     key=f"download_pose_{i}_{pose.get('model', 1)}"
#                                 )
            
#             # Add system information
#             with st.expander("System Information", expanded=False):
#                 st.write(f"**CPU Cores:** {multiprocessing.cpu_count()}")
#                 import sys
#                 st.write(f"**Python Version:** {sys.version}")
#                 st.write(f"**RDKit Version:** {Chem.rdBase.rdkitVersion}")
#                 st.write(f"**Streamlit Version:** {st.__version__}")
                
#                 # Show dependency status
#                 deps = check_dependencies()
#                 for dep, installed in deps.items():
#                     status = "✅ Installed" if installed else "❌ Not Found"
#                     st.write(f"**{dep}:** {status}")

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

# Set page config as the FIRST Streamlit command
st.set_page_config(layout="wide", page_title="Advanced Molecular Docking")

# Configure logging
logging.basicBasicConfig(
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

            # Replace the interaction analysis section in the process_ligand function with this code:
            # Look for the section starting with "# Interaction analysis for the top pose"

            if result['pose_files']:
                try:
                    # Check ProLiF version and adjust approach
                    import prolif
                    prolif_version = getattr(prolif, "__version__", "0.3.0")  # Default to older version if not found
                    
                    # Different approach based on version
                    if hasattr(prolif.Molecule, 'from_rdkit'):
                        # Newer version of ProLiF (>= 0.3.0)
                        fp = prolif.Fingerprint()
                        
                        # Load receptor with RDKit and convert to prolif.Molecule
                        rdkit_rec = Chem.MolFromPDBFile(str(output_dir / "receptor.pdb"), removeHs=False)
                        if rdkit_rec is None:
                            raise ValueError("Failed to load receptor PDB with RDKit")
                        mol_rec = prolif.Molecule.from_rdkit(rdkit_rec)
                        
                        # Load ligand with RDKit and convert to prolif.Molecule
                        if docking_method == "Vina":
                            rdkit_lig = Chem.MolFromPDBQTFile(result['pose_files'][0], removeHs=False)
                        else:  # DiffDock
                            rdkit_lig = Chem.MolFromPDBFile(result['pose_files'][0], removeHs=False)
                        
                        if rdkit_lig is None:
                            raise ValueError("Failed to load ligand file with RDKit")
                        mol_lig = prolif.Molecule.from_rdkit(rdkit_lig)
                        
                        # Generate interactions
                        interactions = fp.generate(mol_rec, mol_lig)
                        result['interactions'] = fp
                        
                    else:
                        # Older version of ProLiF (< 0.3.0)
                        fp = prolif.Fingerprint()
                        
                        if docking_method == "Vina":
                            complex_file = result['pose_files'][0]
                            # Create a complex file that includes both receptor and ligand
                            with open(str(output_dir / "receptor.pdb"), 'r') as f_rec:
                                receptor_content = f_rec.read()
                            with open(complex_file, 'r') as f_lig:
                                ligand_content = f_lig.read()
                            
                            temp_complex = str(output_dir / f"temp_complex_{idx}.pdb")
                            with open(temp_complex, 'w') as f_out:
                                f_out.write(receptor_content + "\n" + ligand_content)
                            
                            # Generate interactions directly from the complex file
                            interactions = fp.run(temp_complex, ligand_residue="UNL")
                            result['interactions'] = fp
                        else:
                            # For DiffDock, try direct loading
                            complex_file = result['pose_files'][0]
                            interactions = fp.run(complex_file, ligand_residue="LIG")
                            result['interactions'] = fp
                            
                except Exception as e:
                    logger.warning(f"Interaction analysis failed for ligand {idx}: {str(e)}")
                    logger.warning(f"ProLiF version: {prolif_version}")
            # Interaction analysis for the top pose
    #         if result['pose_files']:
    #             try:
    #                 fp = plf.Fingerprint()
    #                 # Load receptor with RDKit and convert to prolif.Molecule
    #                 rdkit_rec = Chem.MolFromPDBFile(str(output_dir / "receptor.pdb"), removeHs=False)
    #                 if rdkit_rec is None:
    #                     raise ValueError("Failed to load receptor PDB with RDKit")
    #                 mol_rec = plf.Molecule.from_rdkit(rdkit_rec)

    #                 # Load ligand with RDKit and convert to prolif.Molecule
    #                 if docking_method == "Vina":
    #                     rdkit_lig = Chem.MolFromPDBQTFile(result['pose_files'][0], removeHs=False)
    #                 else:  # DiffDock
    #                     rdkit_lig = Chem.MolFromPDBFile(result['pose_files'][0], removeHs=False)
    #                 if rdkit_lig is None:
    #                     raise ValueError("Failed to load ligand file with RDKit")
    #                 mol_lig = plf.Molecule.from_rdkit(rdkit_lig)

    #                 # Generate interactions
    #                 interactions = fp.generate(mol_rec, mol_lig)
    #                 result['interactions'] = fp
    #             except Exception as e:
    #                 logger.warning(f"Interaction analysis failed for ligand {idx}: {str(e)}")
    #     except Exception as e:
    #         result['error'] = f"Unexpected error: {str(e)}"
    # return result

def batch_docking(smiles_list: List[str], receptor_pdb: str, receptor_bytes: bytes, params: Dict, docking_method: str, 
                  output_dir: str = "docking_results") -> List[Dict]:
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)  # Clear previous results
    output_dir.mkdir(exist_ok=True)
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
        if not Path(receptor_pdb).exists():
            st.error(f"Receptor file {receptor_pdb} not found")
            return []
        success, msg = convert_to_pdbqt(receptor_pdb, str(receptor_pdbqt), is_ligand=False)
        if not success:
            st.error(f"Receptor preparation failed: {msg}")
            return []
        success, config_path = create_vina_config(
            output_dir, receptor_pdb, params.get('center'), params.get('size'), 
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

def create_pymol_script(receptor_file: str, ligand_file: str, output_dir: Path) -> str:
    script_content = f"""
load {receptor_file}, receptor
load {ligand_file}, ligand
hide everything
show cartoon, receptor
color gray, receptor
show sticks, ligand
color green, ligand
zoom
bg_color white
"""
    script_path = output_dir / "view_in_pymol.pml"
    with open(script_path, 'w') as f:
        f.write(script_content)
    return str(script_path)

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
            if st.button("Clear Previous Results"):
                if Path("docking_results").exists():
                    shutil.rmtree("docking_results")
                    st.success("Previous docking results cleared.")
            if st.button("Start Docking", type="primary"):
                results = batch_docking(st.session_state['smiles_list'], st.session_state['receptor_path'], 
                                        st.session_state['receptor_bytes'], st.session_state.get('params', {}), docking_method)
                st.session_state['results'] = results
                st.success("Docking completed!")

    elif step == "Results":
        st.header("Docking Results")
        if 'results' not in st.session_state or not st.session_state['results']:
            st.warning("No results available. Run docking first.")
        else:
            results = st.session_state['results']
            successful = [r for r in results if r['status'] == 'success']
            tabs = st.tabs(["Overview", "2D Analysis", "3D View", "Interactions", "PyMOL/Chimera View"])
            
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

            # with tabs[4]:  # PyMOL/Chimera View
            #     st.subheader("PyMOL Visualization")
            #     st.write("Generate a PyMOL script to view docking results locally.")
            #     ligand_idx = st.selectbox("Select Ligand for PyMOL", range(len(successful)), 
            #                               format_func=lambda i: f"Ligand {i+1}: {successful[i]['affinity']:.2f} ({successful[i]['method']})", key="pymol_ligand")
            #     if successful:
            #         receptor_file = str(Path("docking_results/receptor.pdbqt" if successful[ligand_idx]['method'] == "Vina" else "docking_results/receptor.pdb"))
            #         pose_files = successful[ligand_idx]['pose_files']
            #         pose_idx = st.selectbox("Select Pose for PyMOL", range(len(pose_files)), 
            #                                 format_func=lambda i: f"Pose {i}: {successful[ligand_idx]['poses'][i]['affinity']:.2f}", key="pymol_pose")
            #         ligand_file = pose_files[pose_idx]
            #         if os.path.exists(receptor_file) and os.path.exists(ligand_file):
            #             pymol_script = create_pymol_script(receptor_file, ligand_file, Path("docking_results"))
            #             st.download_button("Download PyMOL Script", 
            #                                data=open(pymol_script, 'rb').read(), 
            #                                file_name="view_in_pymol.pml")
            #             st.write("**Instructions**: Download the ZIP file from the 'Overview' tab, extract it, download this script, and run `pymol view_in_pymol.pml` in the extracted folder.")
            #         else:
            #             st.error(f"Files missing for PyMOL: {receptor_file}, {ligand_file}")

            # Replace the PyMOL tab section in the main function with this code:
# Look for the section within tabs[4] in the Results section

        with tabs[4]:  # PyMOL/Chimera View
            st.subheader("Export for External Visualization")
            st.write("Generate scripts to view docking results in PyMOL or UCSF Chimera")
            
            if not successful:
                st.warning("No successful docking results to export")
            else:
                # Create a more reliable PyMOL script generator
                ligand_idx = st.selectbox("Select Ligand", range(len(successful)), 
                                        format_func=lambda i: f"Ligand {i+1}: {successful[i]['affinity']:.2f}", 
                                        key="pymol_select")
                
                selected_result = successful[ligand_idx]
                if 'pose_files' in selected_result and selected_result['pose_files']:
                    pose_idx = st.selectbox("Select Pose", range(len(selected_result['pose_files'])), 
                                        format_func=lambda i: f"Pose {i+1}", key="pose_select")
                    
                    receptor_path = Path("docking_results/receptor.pdb")
                    ligand_path = Path(selected_result['pose_files'][pose_idx])
                    
                    if receptor_path.exists() and ligand_path.exists():
                        # Create PyMOL script content
                        pymol_script = f"""# PyMOL script for visualizing docking results
                                                load {receptor_path.absolute()}, receptor
                                                load {ligand_path.absolute()}, ligand
                                                hide everything
                                                show cartoon, receptor
                                                color gray80, receptor
                                                show sticks, ligand
                                                color green, ligand
                                                show surface, receptor
                                                set transparency, 0.5
                                                set surface_quality, 1
                                                center ligand
                                                zoom ligand, 5
                                                bg_color white
                                                """
                        # Save script to file
                        script_path = Path("docking_results/view_in_pymol.pml")
                        with open(script_path, 'w') as f:
                            f.write(pymol_script)
                        
                        # Provide download button
                        with open(script_path, 'r') as f:
                            script_content = f.read()
                        st.download_button(
                            label="Download PyMOL Script",
                            data=script_content,
                            file_name="view_in_pymol.pml",
                            mime="text/plain"
                        )
                        
                        # Create a simple structure archive for download
                        st.write("### Download Structure Files")
                        
                        # Create zip with just the necessary files
                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                            zip_file.write(receptor_path, receptor_path.name)
                            zip_file.write(ligand_path, ligand_path.name)
                            zip_file.write(script_path, script_path.name)
                        
                        zip_buffer.seek(0)
                        st.download_button(
                            label="Download Structure Files (ZIP)",
                            data=zip_buffer.getvalue(),
                            file_name=f"docking_complex_{ligand_idx}_pose_{pose_idx}.zip",
                            mime="application/zip"
                        )
                        
                        st.info("""
                        **To use in PyMOL:**
                        1. Download the ZIP file
                        2. Extract the files
                        3. Open PyMOL
                        4. Run: File → Run Script → Select the .pml file
                        """)
                    else:
                        st.error(f"Required files not found. Receptor: {receptor_path.exists()}, Ligand: {ligand_path.exists()}")
                else:
                    st.error("No pose files available for this ligand")

if __name__ == "__main__":
    main()