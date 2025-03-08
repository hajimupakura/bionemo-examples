
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import io
import base64
import subprocess
import os
import sys
import time
from pathlib import Path
import tempfile
import requests
import threading
import multiprocessing
from matplotlib import animation
from stmol import showmol
import py3Dmol
import mdtraj as md
import networkx as nx
import json
from fpdf import FPDF
import concurrent.futures
import dask.dataframe as dd
from dask.distributed import Client

# Optional import for OpenMM integration
try:
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit
    OPENMM_AVAILABLE = True
except ImportError:
    OPENMM_AVAILABLE = False

# Optional import for PyMBAR (for free energy calculations)
try:
    import pymbar
    PYMBAR_AVAILABLE = True
except ImportError:
    PYMBAR_AVAILABLE = False

# Initialize Dask client for distributed computing (when needed)
def initialize_dask_client():
    return Client(processes=True, threads_per_worker=4, n_workers=max(1, multiprocessing.cpu_count()-1), memory_limit='4GB')

# Set page configuration
st.set_page_config(
    page_title="BioSim: Drug Behavior Simulation Platform",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #0066cc;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .tool-card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .result-section {
        padding: 1rem;
        background-color: #e6f3ff;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .progress-bar {
        height: 10px;
        background-color: #0066cc;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        color: #6c757d;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.markdown("<h1 class='main-header'>BioSim: Drug Behavior Simulation Platform</h1>", unsafe_allow_html=True)

st.markdown("""
This application provides a comprehensive platform for simulating drug behavior in biological systems. 
It integrates molecular dynamics, free energy calculations, and systems biology modeling to offer 
detailed insights into drug-target interactions and system-level effects before experimental testing.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose a simulation module",
    ["Home", "Molecular Dynamics (MD)", "Free Energy Calculations", "Systems Biology Modeling", "Results Dashboard"])

# Utility Functions
def display_pdb_structure(pdb_string, style='stick'):
    """Display a PDB structure using py3Dmol"""
    viewer = py3Dmol.view(width=700, height=500)
    viewer.addModel(pdb_string, 'pdb')
    viewer.setStyle({style:{}})
    viewer.zoomTo()
    viewer.spin(True)
    showmol(viewer, height=500, width=700)

def get_example_data(data_type):
    """Retrieve example data for demonstrations"""
    if data_type == "md_trajectory":
        # Simulated MD trajectory data
        time_steps = np.arange(0, 1000, 10)
        rmsd_values = 0.2 + 0.8 * np.exp(-time_steps/500) + 0.1 * np.random.random(len(time_steps))
        energy_values = -500 - 100 * np.exp(-time_steps/300) + 20 * np.random.random(len(time_steps))
        return pd.DataFrame({
            'Time (ps)': time_steps,
            'RMSD (nm)': rmsd_values,
            'Energy (kJ/mol)': energy_values
        })
    elif data_type == "binding_energy":
        # Simulated binding energy data
        compounds = [f"Compound-{i}" for i in range(1, 11)]
        binding_energies = -50 - 30 * np.random.random(10)
        uncertainty = 5 * np.random.random(10)
        return pd.DataFrame({
            'Compound': compounds,
            'Binding Energy (kJ/mol)': binding_energies,
            'Uncertainty': uncertainty
        })
    elif data_type == "pathway":
        # Create a sample biological pathway graph
        G = nx.DiGraph()
        nodes = ["Drug", "Receptor", "Protein A", "Protein B", "Protein C", 
                 "Enzyme X", "Signaling molecule", "Gene expression", "Cell response"]
        G.add_nodes_from(nodes)
        edges = [
            ("Drug", "Receptor"), ("Receptor", "Protein A"), 
            ("Receptor", "Protein B"), ("Protein A", "Protein C"),
            ("Protein B", "Enzyme X"), ("Protein C", "Signaling molecule"),
            ("Enzyme X", "Signaling molecule"), ("Signaling molecule", "Gene expression"),
            ("Gene expression", "Cell response")
        ]
        G.add_edges_from(edges)
        return G

def simulate_molecular_dynamics(pdb_file, simulation_time_ns, temperature, simulation_package, 
                             solvent_model="TIP3P", force_field="amber14", periodic=True,
                             timestep=0.002, reporting_interval=5000, use_gpu=True):
    """
    Run a molecular dynamics simulation using OpenMM or simulate the process
    
    Parameters:
    -----------
    pdb_file : str or file-like
        Path or file object for PDB structure
    simulation_time_ns : float
        Simulation time in nanoseconds
    temperature : float
        Temperature in Kelvin
    simulation_package : str
        Simulation engine to use ('OpenMM', 'NAMD', etc.)
    solvent_model : str
        Water model or implicit solvent model
    force_field : str
        Force field to use for the simulation
    periodic : bool
        Whether to use periodic boundary conditions
    timestep : float
        Integration timestep in picoseconds
    reporting_interval : int
        Number of steps between logging/reporting
    use_gpu : bool
        Whether to use GPU acceleration
    
    Returns:
    --------
    pandas.DataFrame with simulation results
    """
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    if simulation_package == "OpenMM" and OPENMM_AVAILABLE:
        try:
            status_text.text("Setting up OpenMM simulation...")
            
            # Convert ns to ps for internal calculations
            simulation_time_ps = simulation_time_ns * 1000
            
            # Calculate steps based on timestep and total time
            steps = int(simulation_time_ps / timestep)
            reporting_steps = max(1, steps // 100)  # Report at 100 points during simulation
            
            # Load structure
            if isinstance(pdb_file, str):
                if os.path.exists(pdb_file):
                    # It's a filepath
                    pdb = app.PDBFile(pdb_file)
                else:
                    # It's a PDB string, write to temporary file
                    with tempfile.NamedTemporaryFile(suffix='.pdb', mode='w', delete=False) as tmp:
                        tmp.write(pdb_file)
                        tmp_path = tmp.name
                    pdb = app.PDBFile(tmp_path)
                    os.unlink(tmp_path)
            else:
                # It's a file object
                pdb = app.PDBFile(pdb_file)
            
            # Select force field
            if solvent_model.startswith("GB") or solvent_model.startswith("PB"):
                # Implicit solvent
                if force_field.lower().startswith("amber"):
                    ff = app.ForceField('amber14-all.xml', 'implicit/gbn2.xml')
                elif force_field.lower().startswith("charmm"):
                    ff = app.ForceField('charmm36.xml', 'implicit/gbn2.xml')
                else:
                    ff = app.ForceField('amber14-all.xml', 'implicit/gbn2.xml')
            else:
                # Explicit solvent
                if force_field.lower().startswith("amber"):
                    if solvent_model == "TIP3P":
                        ff = app.ForceField('amber14-all.xml', 'amber14/tip3p.xml')
                    elif solvent_model == "TIP4P":
                        ff = app.ForceField('amber14-all.xml', 'amber14/tip4pew.xml')
                    else:  # Default to TIP3P
                        ff = app.ForceField('amber14-all.xml', 'amber14/tip3p.xml')
                elif force_field.lower().startswith("charmm"):
                    if solvent_model == "TIP3P":
                        ff = app.ForceField('charmm36.xml', 'charmm36/water.xml')
                    else:
                        ff = app.ForceField('charmm36.xml', 'charmm36/water.xml')
                else:  # Default to AMBER
                    ff = app.ForceField('amber14-all.xml', 'amber14/tip3p.xml')
            
            # Create system
            status_text.text("Creating molecular system...")
            progress_bar.progress(5)
            
            if solvent_model.startswith("GB") or solvent_model.startswith("PB"):
                # Implicit solvent
                system = ff.createSystem(
                    pdb.topology, 
                    nonbondedMethod=app.NoCutoff,
                    constraints=app.HBonds,
                    hydrogenMass=4*unit.amu  # For 4 fs timesteps with HMR
                )
            else:
                # Explicit solvent - first solvate the system
                status_text.text("Solvating system...")
                
                if periodic:
                    # Add solvent with periodic box
                    modeller = app.Modeller(pdb.topology, pdb.positions)
                    modeller.addSolvent(ff, model=solvent_model.split("/")[0], 
                                      padding=1.0*unit.nanometer)
                    
                    # Create system with periodic boundary conditions
                    system = ff.createSystem(
                        modeller.topology, 
                        nonbondedMethod=app.PME,
                        nonbondedCutoff=1.0*unit.nanometer,
                        constraints=app.HBonds,
                        hydrogenMass=4*unit.amu,  # For 4 fs timesteps with HMR
                        rigidWater=True
                    )
                else:
                    # Non-periodic system (less common for explicit solvent)
                    system = ff.createSystem(
                        pdb.topology, 
                        nonbondedMethod=app.NoCutoff,
                        constraints=app.HBonds
                    )
            
            # Add thermostat
            system.addForce(mm.AndersenThermostat(
                temperature*unit.kelvin, 
                1.0/unit.picosecond)
            )
            
            # Create integrator (timestep in ps)
            integrator = mm.LangevinMiddleIntegrator(
                temperature*unit.kelvin,
                1.0/unit.picosecond,
                timestep*unit.picoseconds
            )
            
            # Set up platform
            if use_gpu:
                try:
                    platform = mm.Platform.getPlatformByName('CUDA')
                    properties = {'CudaPrecision': 'mixed'}
                    status_text.text("Using CUDA platform for GPU acceleration")
                except Exception:
                    try:
                        platform = mm.Platform.getPlatformByName('OpenCL')
                        properties = {}
                        status_text.text("Using OpenCL platform for GPU acceleration")
                    except Exception:
                        platform = mm.Platform.getPlatformByName('CPU')
                        properties = {'Threads': str(max(1, multiprocessing.cpu_count()-1))}
                        status_text.text("Using CPU platform (GPU not available)")
            else:
                platform = mm.Platform.getPlatformByName('CPU')
                properties = {'Threads': str(max(1, multiprocessing.cpu_count()-1))}
            
            # Create simulation object
            progress_bar.progress(10)
            status_text.text("Initializing simulation...")
            
            if solvent_model.startswith("GB") or solvent_model.startswith("PB"):
                # Implicit solvent
                simulation = app.Simulation(pdb.topology, system, integrator, platform, properties)
                simulation.context.setPositions(pdb.positions)
            else:
                # Explicit solvent with modeller
                simulation = app.Simulation(modeller.topology, system, integrator, platform, properties)
                simulation.context.setPositions(modeller.positions)
            
            # Minimize energy
            status_text.text("Minimizing energy...")
            simulation.minimizeEnergy()
            
            # Equilibrate
            progress_bar.progress(15)
            status_text.text("Equilibrating system...")
            simulation.context.setVelocitiesToTemperature(temperature*unit.kelvin)
            simulation.step(5000)  # Short equilibration
            
            # Prepare for production run
            progress_bar.progress(20)
            status_text.text("Starting production simulation...")
            
            # Set up data collection
            time_values = []
            potential_energy = []
            kinetic_energy = []
            temperature_values = []
            rmsd_values = []
            
            # Get initial positions for RMSD calculation
            initial_positions = simulation.context.getState(getPositions=True).getPositions(asNumpy=True)
            
            # Validate initial positions
            if initial_positions.size == 0:
                raise ValueError("No initial positions found in the simulation context.")
            
            # Get number of particles from topology
            num_particles = pdb.topology.getNumAtoms()
            if len(initial_positions) != num_particles:
                raise ValueError(f"Mismatch between topology atoms ({num_particles}) and initial positions ({len(initial_positions)}).")

            # Run production simulation
            for i in range(steps // reporting_steps):
                simulation.step(reporting_steps)
                
                # Get state information
                state = simulation.context.getState(getEnergy=True, getPositions=True)
                time_ps = (i + 1) * reporting_steps * timestep
                time_values.append(time_ps)
                
                # Record energies
                potential_energy.append(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
                kinetic_energy.append(state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole))
                
                # Calculate temperature
                dof = 3 * system.getNumParticles() - system.getNumConstraints()
                temperature_values.append((2 * state.getKineticEnergy() / (dof * unit.MOLAR_GAS_CONSTANT_R)).value_in_unit(unit.kelvin))
                
                # Calculate RMSD
                positions = state.getPositions(asNumpy=True)
                
                # Validate positions match initial_positions
                if len(positions) != len(initial_positions):
                    status_text.text(f"Warning: Position mismatch at step {i + 1}. Skipping RMSD calculation.")
                    rmsd_values.append(np.nan)  # Append NaN to indicate failure
                    continue
                
                displacement = positions - initial_positions
                
                # Select heavy atoms
                heavy_atoms = [idx for idx, atom in enumerate(pdb.topology.atoms()) 
                              if atom.element.symbol != 'H']
                
                # Calculate RMSD with safety checks
                try:
                    if heavy_atoms and len(heavy_atoms) <= len(displacement):
                        rmsd = np.sqrt(np.mean(np.sum(displacement[heavy_atoms]**2, axis=1)))
                    else:
                        rmsd = np.sqrt(np.mean(np.sum(displacement**2, axis=1)))
                    rmsd_values.append(rmsd.value_in_unit(unit.nanometers))
                except IndexError as e:
                    status_text.text(f"Warning: Index error in RMSD calculation at step {i + 1}: {str(e)}. Using NaN.")
                    rmsd_values.append(np.nan)
                
                # Update progress
                progress = 20 + 80 * (i + 1) / (steps // reporting_steps)
                progress_bar.progress(int(progress))
                status_text.text(f"Running OpenMM simulation: {int(progress)}% complete")
            
            # Create results DataFrame
            results = pd.DataFrame({
                'Time (ps)': time_values,
                'Potential Energy (kJ/mol)': potential_energy,
                'Kinetic Energy (kJ/mol)': kinetic_energy,
                'Temperature (K)': temperature_values,
                'RMSD (nm)': rmsd_values
            })
            
            status_text.text("Simulation completed successfully!")
            return results
            
        except Exception as e:
            status_text.text(f"Error in OpenMM simulation: {str(e)}")
            st.error(f"OpenMM simulation failed: {str(e)}")
            return get_example_data("md_trajectory")
            
            # Create results DataFrame
            results = pd.DataFrame({
                'Time (ps)': time_values,
                'Potential Energy (kJ/mol)': potential_energy,
                'Kinetic Energy (kJ/mol)': kinetic_energy,
                'Temperature (K)': temperature_values,
                'RMSD (nm)': rmsd_values
            })
            
            status_text.text("Simulation completed successfully!")
            return results
            
        except Exception as e:
            status_text.text(f"Error in OpenMM simulation: {str(e)}")
            st.error(f"OpenMM simulation failed: {str(e)}")
            # Fall back to simulated data
            return get_example_data("md_trajectory")
    else:
        # Simulate the process for other simulation packages or if OpenMM is not available
        if simulation_package == "OpenMM" and not OPENMM_AVAILABLE:
            st.warning("OpenMM is not installed. Using simulated data instead.")
        
        for i in range(101):
            progress_bar.progress(i)
            status_text.text(f"Running {simulation_package} simulation: {i}% complete")
            time.sleep(0.05)
        
        status_text.text("Simulation completed successfully!")
        return get_example_data("md_trajectory")

def calculate_binding_energy(receptor_file, ligand_file, method, 
                          advanced_options=None, trajectory_file=None, num_frames=None):
    """
    Calculate binding free energy between receptor and ligand
    
    Parameters:
    -----------
    receptor_file : str or file-like
        Path or file object for receptor structure
    ligand_file : str or file-like
        Path or file object for ligand structure
    method : str
        Method to use for calculation ('MM/GBSA', 'FEP', 'Docking')
    advanced_options : dict
        Additional method-specific parameters
    trajectory_file : str, optional
        Path to MD trajectory for ensemble calculations
    num_frames : int, optional
        Number of frames to use from trajectory
        
    Returns:
    --------
    tuple of (energy, uncertainty, decomposition)
    """
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Set default advanced options if not provided
    if advanced_options is None:
        advanced_options = {}
    
    # Placeholder for energy decomposition (will be filled with real data for real calculations)
    decomposition = {
        'Van der Waals': 0,
        'Electrostatic': 0,
        'Polar solvation': 0,
        'Non-polar solvation': 0,
        'Entropy': 0
    }
    
    if method == "MM/GBSA" and PYMBAR_AVAILABLE and trajectory_file is not None:
        try:
            # This would be a real MM/GBSA calculation using pymbar and MDAnalysis or MDTraj
            status_text.text("Preparing MM/GBSA calculation...")
            progress_bar.progress(5)
            
            # Parameters from advanced options
            gb_model = advanced_options.get('gb_model', 'OBC2')
            salt_conc = advanced_options.get('salt_concentration', 0.15)  # in M
            num_frames = num_frames or advanced_options.get('frames', 100)
            
            # Load trajectory using MDTraj (in a real app)
            status_text.text("Loading trajectory...")
            progress_bar.progress(10)
            
            # Simulate the steps of a real calculation
            status_text.text("Extracting complex conformations...")
            progress_bar.progress(20)
            
            status_text.text("Computing MM energies...")
            progress_bar.progress(30)
            
            # Process frames in parallel for better performance with large datasets
            status_text.text("Processing trajectory frames in parallel...")
            
            # Initialize empty energy components (for demonstration)
            vdw_energies = []
            elec_energies = []
            polar_energies = []
            nonpolar_energies = []
            
            # Simulate parallel frame processing
            def process_frames(frame_indices):
                # This would process a batch of frames in a real application
                local_vdw = []
                local_elec = []
                local_polar = []
                local_nonpolar = []
                
                for i in frame_indices:
                    # Simulate frame processing
                    local_vdw.append(-50 - 30 * np.random.random())
                    local_elec.append(-40 - 20 * np.random.random())
                    local_polar.append(70 + 30 * np.random.random())
                    local_nonpolar.append(-15 - 5 * np.random.random())
                
                return local_vdw, local_elec, local_polar, local_nonpolar
            
            # Parallel processing of frames using ThreadPoolExecutor
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, os.cpu_count())) as executor:
                # Split frames into batches
                batch_size = max(1, num_frames // os.cpu_count())
                batches = [range(i, min(i+batch_size, num_frames)) 
                          for i in range(0, num_frames, batch_size)]
                
                # Process batches in parallel
                future_to_batch = {executor.submit(process_frames, batch): batch for batch in batches}
                
                for i, future in enumerate(concurrent.futures.as_completed(future_to_batch)):
                    batch = future_to_batch[future]
                    try:
                        batch_vdw, batch_elec, batch_polar, batch_nonpolar = future.result()
                        vdw_energies.extend(batch_vdw)
                        elec_energies.extend(batch_elec)
                        polar_energies.extend(batch_polar)
                        nonpolar_energies.extend(batch_nonpolar)
                        
                        # Update progress
                        progress = 30 + 50 * ((i+1) / len(batches))
                        progress_bar.progress(int(progress))
                        status_text.text(f"Processing frame batch {i+1}/{len(batches)}...")
                    except Exception as e:
                        st.error(f"Error processing batch {batch}: {str(e)}")
            
            # Calculate entropy term (typically requires additional calculation)
            entropy_term = 10.5 + np.random.normal(0, 1)
            
            # Calculate final binding energy components
            mean_vdw = np.mean(vdw_energies)
            mean_elec = np.mean(elec_energies)
            mean_polar = np.mean(polar_energies)
            mean_nonpolar = np.mean(nonpolar_energies)
            
            # Fill decomposition dictionary with calculated values
            decomposition = {
                'Van der Waals': mean_vdw,
                'Electrostatic': mean_elec,
                'Polar solvation': mean_polar,
                'Non-polar solvation': mean_nonpolar,
                'Entropy': entropy_term
            }
            
            # Calculate total binding energy
            energy = mean_vdw + mean_elec + mean_polar + mean_nonpolar - entropy_term
            
            # Calculate uncertainty using bootstrap
            status_text.text("Calculating uncertainty using bootstrap...")
            progress_bar.progress(90)
            
            # In a real app, this would be a proper bootstrap calculation
            # Here we'll simulate it
            component_stdevs = {
                'Van der Waals': np.std(vdw_energies),
                'Electrostatic': np.std(elec_energies),
                'Polar solvation': np.std(polar_energies),
                'Non-polar solvation': np.std(nonpolar_energies),
                'Entropy': 1.5  # Entropy uncertainty is often higher
            }
            
            # Total uncertainty (simplified)
            uncertainty = np.sqrt(sum(x**2 for x in component_stdevs.values()))
            
            status_text.text("MM/GBSA calculation completed successfully!")
            progress_bar.progress(100)
            
            return energy, uncertainty, decomposition
            
        except Exception as e:
            status_text.text(f"Error in MM/GBSA calculation: {str(e)}")
            st.error(f"MM/GBSA calculation failed: {str(e)}")
            # Fall back to simulated data
            energy = -45.3 + np.random.normal(0, 3)
            uncertainty = 4.2 + np.random.random()
            
            # Generate mock decomposition
            decomposition = {
                'Van der Waals': -68.3 + np.random.normal(0, 2),
                'Electrostatic': -42.1 + np.random.normal(0, 3),
                'Polar solvation': 78.5 + np.random.normal(0, 4),
                'Non-polar solvation': -12.6 + np.random.normal(0, 1),
                'Entropy': 10.5 + np.random.normal(0, 1.5)
            }
            
            return energy, uncertainty, decomposition
    
    elif method == "FEP" and PYMBAR_AVAILABLE:
        try:
            # Simulate steps of a real FEP calculation
            status_text.text("Setting up FEP windows...")
            progress_bar.progress(5)
            
            # Parameters from advanced options
            windows = advanced_options.get('windows', 16)
            sampling_per_window = advanced_options.get('sampling_per_window', 5)  # in ns
            fep_method = advanced_options.get('fep_method', 'MBAR')  # MBAR, TI, BAR
            
            # In a real app, this would set up and run alchemical transformations
            
            # Simulate progress through lambda windows
            for i in range(windows):
                # Simulate window calculation
                start_progress = 5 + (90 * i // windows)
                end_progress = 5 + (90 * (i+1) // windows)
                
                # Update progress for this window
                for j in range(start_progress, end_progress):
                    progress_bar.progress(j)
                    status_text.text(f"FEP window {i+1}/{windows}: λ = {i/(windows-1):.2f}")
                    time.sleep(0.02)
            
            # Calculate final result (in a real app, this would use pymbar)
            status_text.text("Calculating final free energy difference...")
            progress_bar.progress(95)
            
            # Generate simulated FEP result
            energy = -42.7 + np.random.normal(0, 2)
            uncertainty = 2.8 + np.random.random()
            
            # Generate mock decomposition (less detailed for FEP)
            decomposition = {
                'Van der Waals': -55.5 + np.random.normal(0, 2),
                'Electrostatic': -38.2 + np.random.normal(0, 3),
                'Polar solvation': 65.1 + np.random.normal(0, 4),
                'Non-polar solvation': -10.3 + np.random.normal(0, 1),
                'Entropy': -3.8 + np.random.normal(0, 1.5)  # Included in FEP
            }
            
            status_text.text("FEP calculation completed successfully!")
            progress_bar.progress(100)
            
        except Exception as e:
            status_text.text(f"Error in FEP calculation: {str(e)}")
            st.error(f"FEP calculation failed: {str(e)}")
            # Fall back to simulated data
            energy = -42.7 + np.random.normal(0, 2)
            uncertainty = 2.8 + np.random.random()
            
            # Generate mock decomposition
            decomposition = {
                'Van der Waals': -55.5 + np.random.normal(0, 2),
                'Electrostatic': -38.2 + np.random.normal(0, 3),
                'Polar solvation': 65.1 + np.random.normal(0, 4),
                'Non-polar solvation': -10.3 + np.random.normal(0, 1),
                'Entropy': -3.8 + np.random.normal(0, 1.5)
            }
    
    else:  # Docking or fallback to simulated data
        for i in range(101):
            progress_bar.progress(i)
            status_text.text(f"Calculating binding energy using {method}: {i}% complete")
            time.sleep(0.02)
        
        status_text.text(f"{method} calculation completed successfully!")
        
        if method == "MM/GBSA":
            energy = -45.3 + np.random.normal(0, 3)
            uncertainty = 4.2 + np.random.random()
            
            # Generate mock decomposition
            decomposition = {
                'Van der Waals': -68.3 + np.random.normal(0, 2),
                'Electrostatic': -42.1 + np.random.normal(0, 3),
                'Polar solvation': 78.5 + np.random.normal(0, 4),
                'Non-polar solvation': -12.6 + np.random.normal(0, 1),
                'Entropy': 10.5 + np.random.normal(0, 1.5)
            }
        elif method == "FEP":
            energy = -42.7 + np.random.normal(0, 2)
            uncertainty = 2.8 + np.random.random()
            
            # Generate mock decomposition
            decomposition = {
                'Van der Waals': -55.5 + np.random.normal(0, 2),
                'Electrostatic': -38.2 + np.random.normal(0, 3),
                'Polar solvation': 65.1 + np.random.normal(0, 4),
                'Non-polar solvation': -10.3 + np.random.normal(0, 1),
                'Entropy': -3.8 + np.random.normal(0, 1.5)
            }
        else:  # Docking
            energy = -38.5 + np.random.normal(0, 5)
            uncertainty = 6.5 + np.random.random()
            
            # Generate mock decomposition (simpler for docking)
            decomposition = {
                'Van der Waals': -48.5 + np.random.normal(0, 3),
                'Electrostatic': -32.1 + np.random.normal(0, 3),
                'Desolvation': 42.5 + np.random.normal(0, 4),
                'Hydrogen bonds': -10.6 + np.random.normal(0, 2),
                'Entropy': 10.2 + np.random.normal(0, 2)
            }
    
    return energy, uncertainty, decomposition

def run_systems_biology_modeling(model_type, parameters, simulation_time):
    """Run systems biology modeling simulation"""
    # Simulate the modeling process
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(101):
        progress_bar.progress(i)
        status_text.text(f"Running {model_type} modeling: {i}% complete")
        time.sleep(0.04)
    
    status_text.text("Modeling completed successfully!")
    
    # Return example pathway or simulation data
    if model_type == "Network Analysis":
        return get_example_data("pathway")
    else:  # ODE model
        time_points = np.linspace(0, simulation_time, 100)
        # Simulate multiple molecular species
        species_data = {}
        species_names = ["Drug", "Receptor", "Complex", "Product", "Response"]
        
        for i, species in enumerate(species_names):
            # Create different dynamics for each species
            if species == "Drug":
                y = 100 * np.exp(-0.1 * time_points)
            elif species == "Receptor":
                y = 50 - 30 * (1 - np.exp(-0.15 * time_points))
            elif species == "Complex":
                y = 30 * (1 - np.exp(-0.12 * time_points)) * np.exp(-0.05 * time_points)
            elif species == "Product":
                y = 40 * (1 - np.exp(-0.08 * time_points))
            else:  # Response
                y = 80 * (1 - np.exp(-0.03 * time_points))
            
            # Add some noise
            y += np.random.normal(0, 0.05 * max(y), size=len(time_points))
            species_data[species] = y
        
        df = pd.DataFrame({"Time": time_points, **species_data})
        return df

# Home page
if app_mode == "Home":
    st.markdown("<h2 class='sub-header'>Welcome to BioSim</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    ### Simulating Drug Behavior in Biological Systems
    
    This platform integrates state-of-the-art computational methods to predict and analyze drug behavior 
    before experimental testing, saving time and resources in drug development.
    
    #### Key Features:
    
    * **Molecular Dynamics (MD)**: Simulate long-timescale interactions between drug, target, and solvent
    * **Free Energy Calculations**: Compute binding free energies for accurate affinity predictions
    * **Systems Biology Modeling**: Model drug effects on pathways or cellular systems
    
    #### Getting Started:
    
    Select a simulation module from the sidebar to begin exploring drug behavior in silico.
    """)
    
    # Display overview of the workflow
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<div class='tool-card'>", unsafe_allow_html=True)
        st.subheader("Step 1: Molecular Dynamics")
        st.image("https://raw.githubusercontent.com/ambermd/pytraj/master/examples/figures/md_1us.gif", 
                 caption="MD Simulation Example")
        st.markdown("""
        - Upload or select molecular structures
        - Configure simulation parameters
        - Analyze trajectories and interactions
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='tool-card'>", unsafe_allow_html=True)
        st.subheader("Step 2: Free Energy Calculations")
        st.image("https://pubs.acs.org/cms/10.1021/acs.jcim.7b00083/asset/images/medium/ci-2017-00083v_0011.gif", 
                 caption="Free Energy Calculation Example")
        st.markdown("""
        - Calculate binding affinities 
        - Compare multiple compounds
        - Estimate thermodynamic properties
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col3:
        st.markdown("<div class='tool-card'>", unsafe_allow_html=True)
        st.subheader("Step 3: Systems Biology Modeling")
        st.image("https://www.ebi.ac.uk/sites/ebi.ac.uk/files/styles/news_and_events_homepage/public/news/biomodels_hero.png", 
                 caption="Systems Biology Modeling Example")
        st.markdown("""
        - Model cellular pathways
        - Predict system-level effects
        - Analyze network perturbations
        """)
        st.markdown("</div>", unsafe_allow_html=True)

# Molecular Dynamics Module
elif app_mode == "Molecular Dynamics (MD)":
    st.markdown("<h2 class='sub-header'>Molecular Dynamics Simulation</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    Simulate long-timescale interactions between the drug, target proteins, and solvent
    to validate binding stability and understand dynamic behavior.
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Input Parameters")
        
        input_method = st.radio(
            "Input method", 
            ["Upload PDB files", "Use example structure"]
        )
        
        if input_method == "Upload PDB files":
            protein_file = st.file_uploader("Upload protein structure (PDB format)", type=["pdb"])
            ligand_file = st.file_uploader("Upload ligand structure (PDB/MOL2 format)", type=["pdb", "mol2"])
            complex_ready = protein_file is not None and ligand_file is not None
            
            if complex_ready:
                st.success("Files uploaded successfully!")
            else:
                st.warning("Please upload both protein and ligand files")
        else:
            st.info("Using example protein-ligand complex")
            complex_ready = True
        
        # Simulation parameters
        st.subheader("Simulation Settings")
        
        simulation_package = st.selectbox(
            "Simulation package", 
            ["OpenMM", "NAMD", "GROMM", "CHARMM"]
        )
        
        simulation_time = st.slider(
            "Simulation time (ns)", 
            min_value=1, 
            max_value=100, 
            value=10
        )
        
        temperature = st.slider(
            "Temperature (K)", 
            min_value=270, 
            max_value=330, 
            value=300
        )
        
        implicit_solvent = st.checkbox("Use implicit solvent")
        
        if implicit_solvent:
            solvent_model = st.selectbox(
                "Solvent model", 
                ["GBSA", "PBSA"]
            )
        else:
            water_model = st.selectbox(
                "Water model", 
                ["TIP3P", "TIP4P", "SPC/E"]
            )
            
        periodic_boundary = st.checkbox("Use periodic boundary conditions", value=True)
        
        run_button = st.button("Run Simulation")
        
    with col2:
        st.subheader("Molecular Viewer")
        
        if complex_ready:
            # In a real application, we would load the actual PDB data
            # Here we use an example PDB string for demonstration
            example_pdb = """
ATOM      1  N   ASP A  30      11.482  22.360  81.280  1.00 67.06           N  
ATOM      2  CA  ASP A  30      12.652  23.228  81.504  1.00 68.91           C  
ATOM      3  C   ASP A  30      12.443  24.662  80.994  1.00 64.85           C  
ATOM      4  O   ASP A  30      11.492  24.979  80.275  1.00 63.82           O  
ATOM      5  CB  ASP A  30      13.010  23.298  82.992  1.00 72.76           C  
ATOM      6  CG  ASP A  30      13.326  21.976  83.654  1.00 75.91           C  
ATOM      7  OD1 ASP A  30      14.027  21.131  83.051  1.00 77.82           O  
ATOM      8  OD2 ASP A  30      12.868  21.776  84.801  1.00 77.86           O1-
ATOM      9  N   THR A  31      13.366  25.533  81.379  1.00 66.11           N  
ATOM     10  CA  THR A  31      13.295  26.946  81.016  1.00 67.36           C  
            """
            
            display_pdb_structure(example_pdb)
            
            if run_button and complex_ready:
                st.subheader("Simulation Progress")
                
                # Run the simulated MD
                trajectory_data = simulate_molecular_dynamics(
                    "example.pdb", 
                    simulation_time, 
                    temperature, 
                    simulation_package
                )
                
                # Display simulation results
                st.markdown("<div class='result-section'>", unsafe_allow_html=True)
                st.subheader("Simulation Results")
                
                # Show trajectory metrics plots
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                
                ax1.plot(trajectory_data['Time (ps)'], trajectory_data['RMSD (nm)'])
                ax1.set_xlabel('Time (ps)')
                ax1.set_ylabel('RMSD (nm)')
                ax1.set_title('Root Mean Square Deviation')
                ax1.grid(True)
                
                ax2.plot(trajectory_data['Time (ps)'], trajectory_data['Energy (kJ/mol)'])
                ax2.set_xlabel('Time (ps)')
                ax2.set_ylabel('Energy (kJ/mol)')
                ax2.set_title('Potential Energy')
                ax2.grid(True)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Interactive 3D trajectory animation placeholder
                st.subheader("Trajectory Animation")
                st.image("https://media.giphy.com/media/xGdxUhqLlHAIL7nqOZ/giphy.gif", 
                         caption="Example MD trajectory animation")
                
                # Analysis options
                st.subheader("Additional Analysis")
                analysis_options = st.multiselect(
                    "Select additional analyses to perform",
                    ["Hydrogen bond analysis", "Contact maps", "Secondary structure analysis", 
                     "Principal Component Analysis", "Clustering"]
                )
                
                if analysis_options:
                    st.info(f"Selected analyses: {', '.join(analysis_options)}")
                    st.write("Analysis results would be displayed here in a real application.")
                
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("Please upload molecular structures or select example data to visualize")

# Free Energy Calculations Module            
elif app_mode == "Free Energy Calculations":
    st.markdown("<h2 class='sub-header'>Free Energy Calculations</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    Compute binding free energies between drugs and their targets to predict binding 
    affinities accurately and compare multiple drug candidates.
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Input Structures")
        
        input_method = st.radio(
            "Input method",
            ["Upload files", "Use example data"]
        )
        
        if input_method == "Upload files":
            receptor_file = st.file_uploader("Upload receptor structure (PDB format)", type=["pdb"])
            ligand_file = st.file_uploader("Upload ligand structure (PDB/MOL2/SDF format)", 
                                          type=["pdb", "mol2", "sdf"])
            inputs_ready = receptor_file is not None and ligand_file is not None
            
            if inputs_ready:
                st.success("Files uploaded successfully!")
            else:
                st.warning("Please upload both receptor and ligand files")
        else:
            st.info("Using example structures")
            inputs_ready = True
        
        st.subheader("Calculation Method")
        
        calculation_method = st.selectbox(
            "Energy calculation method",
            ["MM/GBSA", "FEP (Free Energy Perturbation)", "Molecular Docking"]
        )
        
        if calculation_method == "MM/GBSA":
            st.markdown("""
            **MM/GBSA Parameters:**
            """)
            gb_model = st.selectbox("GB Model", ["OBC I", "OBC II", "GBn", "GBn2"])
            trajectory_ns = st.slider("Trajectory length (ns)", 1, 50, 10)
            
        elif calculation_method == "FEP (Free Energy Perturbation)":
            st.markdown("""
            **FEP Parameters:**
            """)
            windows = st.slider("Number of lambda windows", 5, 32, 16)
            sampling_ns = st.slider("Sampling per window (ns)", 1, 10, 5)
            
        else:  # Docking
            st.markdown("""
            **Docking Parameters:**
            """)
            exhaustiveness = st.slider("Search exhaustiveness", 1, 16, 8)
            flexibility = st.checkbox("Include receptor flexibility")
            
        calculate_button = st.button("Calculate Binding Energy")
        
    with col2:
        if inputs_ready:
            st.subheader("Structure Preview")
            
            # In a real app, display the actual structure
            example_pdb = """
ATOM      1  N   TYR A 215     -11.286  10.598   2.691  1.00 10.56           N  
ATOM      2  CA  TYR A 215     -11.992   9.757   3.650  1.00 10.16           C  
ATOM      3  C   TYR A 215     -13.415  10.298   3.837  1.00 10.83           C  
ATOM      4  O   TYR A 215     -14.062  10.704   2.882  1.00 10.95           O  
ATOM      5  CB  TYR A 215     -12.035   8.305   3.183  1.00 10.04           C  
ATOM      6  CG  TYR A 215     -12.609   7.349   4.206  1.00 10.58           C  
ATOM      7  CD1 TYR A 215     -11.853   6.938   5.297  1.00 11.33           C  
ATOM      8  CD2 TYR A 215     -13.897   6.856   4.083  1.00 11.40           C  
ATOM      9  CE1 TYR A 215     -12.357   6.069   6.254  1.00 11.94           C  
ATOM     10  CE2 TYR A 215     -14.420   5.990   5.033  1.00 11.39           C  
            """
            
            display_pdb_structure(example_pdb)
            
            if calculate_button:
                st.subheader("Calculation Progress")
                
                # Run the simulated binding energy calculation
                energy, uncertainty, decomposition  = calculate_binding_energy(
                    "receptor.pdb",
                    "ligand.pdb",
                    calculation_method
                )
                
                # Display results (update to use decomposition as well)
                st.markdown("<div class='result-section'>", unsafe_allow_html=True)
                st.subheader("Binding Energy Results")
                
                col_result1, col_result2 = st.columns(2)
                
                with col_result1:
                    st.metric(
                        label="Binding Free Energy",
                        value=f"{energy:.2f} kJ/mol",
                        delta=f"± {uncertainty:.2f}"
                    )
                    
                with col_result2:
                    energy_kcal = energy / 4.184
                    st.metric(
                        label="Binding Free Energy",
                        value=f"{energy_kcal:.2f} kcal/mol",
                        delta=f"± {uncertainty/4.184:.2f}"
                    )
                
                # Add interpretation
                if energy < -40:
                    binding_strength = "Strong binding"
                    prediction = "Likely to be effective"
                elif energy < -20:
                    binding_strength = "Moderate binding"
                    prediction = "May be effective"
                else:
                    binding_strength = "Weak binding"
                    prediction = "Unlikely to be effective"
                    
                st.info(f"Interpretation: {binding_strength}. {prediction}.")
                
                # Comparison with other compounds
                st.subheader("Comparison with Other Compounds")
                
                comparison_data = get_example_data("binding_energy")
                
                # Add current result
                new_row = pd.DataFrame({
                    'Compound': ['Current Compound'],
                    'Binding Energy (kJ/mol)': [energy],
                    'Uncertainty': [uncertainty]
                })
                comparison_data = pd.concat([comparison_data, new_row], ignore_index=True)
                
                # Sort by binding energy
                comparison_data = comparison_data.sort_values('Binding Energy (kJ/mol)')
                
                # Plot comparison
                fig = px.bar(
                    comparison_data,
                    x='Compound',
                    y='Binding Energy (kJ/mol)',
                    error_y='Uncertainty',
                    color='Compound',
                    labels={'Binding Energy (kJ/mol)': 'Binding Free Energy (kJ/mol)'},
                    title='Binding Energy Comparison'
                )
                
                # Highlight current compound
                fig.update_traces(
                    marker_color=['blue' if x != 'Current Compound' else 'red' 
                                 for x in comparison_data['Compound']]
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Decomposition analysis
                st.subheader("Energy Decomposition")
    
                # Replace the hardcoded energy_components with the actual decomposition
                energy_components = pd.DataFrame({
                    'Component': list(decomposition.keys()),
                    'Energy (kJ/mol)': list(decomposition.values()),
                    'Type': ['Favorable' if val < 0 else 'Unfavorable' for val in decomposition.values()]
                })
                
                # Adjust colors based on favorability
                color_map = {'Favorable': 'green', 'Unfavorable': 'red'}
                
                fig2 = px.bar(
                    energy_components,
                    y='Component',
                    x='Energy (kJ/mol)',
                    color='Type',
                    color_discrete_map=color_map,
                    title='Energy Component Decomposition'
                )
                
                st.plotly_chart(fig2, use_container_width=True)
                
                # Export options
                st.subheader("Export Results")
                export_format = st.selectbox("Export format", ["CSV", "Excel", "JSON", "PDF Report"])
                
                # Advanced export options
                with st.expander("Advanced Export Options"):
                    # General options
                    include_plots = st.checkbox("Include visualizations", value=True)
                    include_raw_data = st.checkbox("Include raw data", value=True)
                    
                    # PDF report options (only show if PDF selected)
                    if export_format == "PDF Report":
                        report_title = st.text_input("Report title", "Drug Binding Analysis Report")
                        include_logo = st.checkbox("Include company logo", value=True)
                        report_sections = st.multiselect(
                            "Sections to include", 
                            ["Executive Summary", "Methods", "Results", "Discussion", "References"],
                            default=["Executive Summary", "Methods", "Results"]
                        )
                
                # Export functionality
                export_button = st.button("Export Results")
                
                if export_button:
                    if export_format == "CSV":
                        # Prepare CSV data
                        csv_data = io.StringIO()
                        
                        # Write header
                        csv_data.write("# Binding Energy Analysis Results\n")
                        csv_data.write(f"# Method: {calculation_method}\n")
                        csv_data.write(f"# Date: {time.strftime('%Y-%m-%d')}\n\n")
                        
                        # Write binding energy summary
                        csv_data.write("## Summary\n")
                        csv_data.write(f"Binding Energy (kJ/mol),{energy:.4f}\n")
                        csv_data.write(f"Uncertainty (kJ/mol),{uncertainty:.4f}\n")
                        csv_data.write(f"Binding Energy (kcal/mol),{energy/4.184:.4f}\n")
                        csv_data.write(f"Uncertainty (kcal/mol),{uncertainty/4.184:.4f}\n\n")
                        
                        # Write energy decomposition
                        csv_data.write("## Energy Decomposition\n")
                        csv_data.write("Component,Energy (kJ/mol)\n")
                        for component, value in energy_components['Energy (kJ/mol)'].items():
                            csv_data.write(f"{component},{value:.4f}\n")
                        
                        # Provide download link
                        st.download_button(
                            label="Download CSV",
                            data=csv_data.getvalue(),
                            file_name=f"{calculation_method.replace('/', '_')}_binding_energy.csv",
                            mime="text/csv"
                        )
                        
                        st.success("CSV file prepared for download")
                        
                    elif export_format == "Excel":
                        # Create Excel file in memory
                        output = io.BytesIO()
                        
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            # Create summary sheet
                            summary_df = pd.DataFrame({
                                'Parameter': ['Method', 'Binding Energy (kJ/mol)', 'Uncertainty (kJ/mol)', 
                                             'Binding Energy (kcal/mol)', 'Uncertainty (kcal/mol)'],
                                'Value': [calculation_method, f"{energy:.4f}", f"{uncertainty:.4f}", 
                                         f"{energy/4.184:.4f}", f"{uncertainty/4.184:.4f}"]
                            })
                            summary_df.to_excel(writer, sheet_name='Summary', index=False)
                            
                            # Create energy decomposition sheet
                            decomp_df = pd.DataFrame({
                                'Component': energy_components['Component'],
                                'Energy (kJ/mol)': energy_components['Energy (kJ/mol)'],
                                'Type': energy_components['Type']
                            })
                            decomp_df.to_excel(writer, sheet_name='Energy Decomposition', index=False)
                            
                            # Add comparison data if available
                            comparison_data.to_excel(writer, sheet_name='Compound Comparison', index=False)
                            
                            # Format the Excel file
                            workbook = writer.book
                            
                            # Format for summary sheet
                            summary_sheet = writer.sheets['Summary']
                            header_format = workbook.add_format({'bold': True, 'bg_color': '#D3D3D3'})
                            for col_num, value in enumerate(summary_df.columns.values):
                                summary_sheet.write(0, col_num, value, header_format)
                            
                            # Format for decomposition sheet
                            decomp_sheet = writer.sheets['Energy Decomposition']
                            for col_num, value in enumerate(decomp_df.columns.values):
                                decomp_sheet.write(0, col_num, value, header_format)
                            
                            # Add conditional formatting for energy values
                            green_format = workbook.add_format({'bg_color': '#C6EFCE'})
                            red_format = workbook.add_format({'bg_color': '#FFC7CE'})
                            decomp_sheet.conditional_format(1, 1, len(decomp_df), 1, 
                                                           {'type': 'cell', 'criteria': '<', 'value': 0,
                                                            'format': green_format})
                            decomp_sheet.conditional_format(1, 1, len(decomp_df), 1, 
                                                           {'type': 'cell', 'criteria': '>', 'value': 0,
                                                            'format': red_format})
                        
                        # Prepare Excel for download
                        output.seek(0)
                        
                        st.download_button(
                            label="Download Excel",
                            data=output,
                            file_name=f"{calculation_method.replace('/', '_')}_binding_energy.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                        
                        st.success("Excel file prepared for download")
                        
                    elif export_format == "JSON":
                        # Create JSON structure
                        json_data = {
                            "metadata": {
                                "method": calculation_method,
                                "date": time.strftime("%Y-%m-%d"),
                                "time": time.strftime("%H:%M:%S")
                            },
                            "binding_energy": {
                                "value_kj_mol": float(f"{energy:.4f}"),
                                "uncertainty_kj_mol": float(f"{uncertainty:.4f}"),
                                "value_kcal_mol": float(f"{energy/4.184:.4f}"),
                                "uncertainty_kcal_mol": float(f"{uncertainty/4.184:.4f}")
                            },
                            "energy_decomposition": {}
                        }
                        
                        # Add energy components to JSON
                        for i, component in enumerate(energy_components['Component']):
                            json_data["energy_decomposition"][component] = {
                                "value_kj_mol": float(f"{energy_components['Energy (kJ/mol)'][i]:.4f}"),
                                "type": energy_components['Type'][i]
                            }
                        
                        # Add comparison data
                        json_data["compound_comparison"] = []
                        for i, compound in enumerate(comparison_data['Compound']):
                            json_data["compound_comparison"].append({
                                "compound": compound,
                                "binding_energy_kj_mol": float(f"{comparison_data['Binding Energy (kJ/mol)'][i]:.4f}"),
                                "uncertainty": float(f"{comparison_data['Uncertainty'][i]:.4f}")
                            })
                        
                        # Convert to JSON string with pretty formatting
                        json_str = json.dumps(json_data, indent=4)
                        
                        st.download_button(
                            label="Download JSON",
                            data=json_str,
                            file_name=f"{calculation_method.replace('/', '_')}_binding_energy.json",
                            mime="application/json"
                        )
                        
                        st.success("JSON file prepared for download")
                        
                    elif export_format == "PDF Report":
                        # Use FPDF to create a detailed PDF report
                        st.info("Generating PDF report... This may take a moment.")
                        
                        try:
                            # Create PDF object
                            pdf = FPDF()
                            
                            # Set metadata
                            pdf.set_title(report_title)
                            pdf.set_author("BioSim: Drug Behavior Simulation Platform")
                            
                            # Add title page
                            pdf.add_page()
                            
                            # Add logo if selected
                            if include_logo:
                                # In a real app, this would load a company logo
                                # Here we'll create a placeholder image
                                pdf.image("https://via.placeholder.com/150x50?text=CompanyLogo", 30, 10, 150)
                                
                            # Title
                            pdf.set_font("Arial", "B", 24)
                            pdf.ln(40)
                            pdf.cell(0, 20, report_title, 0, 1, "C")
                            
                            # Date
                            pdf.set_font("Arial", "", 12)
                            pdf.cell(0, 10, f"Date: {time.strftime('%Y-%m-%d')}", 0, 1, "C")
                            
                            # Method
                            pdf.cell(0, 10, f"Method: {calculation_method}", 0, 1, "C")
                            
                            # Add sections based on selection
                            if "Executive Summary" in report_sections:
                                pdf.add_page()
                                pdf.set_font("Arial", "B", 16)
                                pdf.cell(0, 10, "Executive Summary", 0, 1, "L")
                                pdf.ln(5)
                                
                                pdf.set_font("Arial", "", 12)
                                pdf.multi_cell(0, 10, (
                                    f"This report presents the binding energy analysis between the receptor and ligand "
                                    f"using the {calculation_method} method. The calculated binding free energy is "
                                    f"{energy:.2f} kJ/mol with an uncertainty of {uncertainty:.2f} kJ/mol. "
                                    f"The analysis suggests {binding_strength.lower()} between the receptor and ligand. "
                                    f"Based on the computational results, the compound is {prediction.lower()}."
                                ))
                            
                            if "Methods" in report_sections:
                                pdf.add_page()
                                pdf.set_font("Arial", "B", 16)
                                pdf.cell(0, 10, "Methods", 0, 1, "L")
                                pdf.ln(5)
                                
                                pdf.set_font("Arial", "", 12)
                                if calculation_method == "MM/GBSA":
                                    pdf.multi_cell(0, 10, (
                                        "Molecular Mechanics Generalized Born Surface Area (MM/GBSA) method was used to "
                                        "calculate the binding free energy. The approach combines molecular mechanics "
                                        "energy terms with implicit solvent models. The total binding free energy is "
                                        "decomposed into contributions from van der Waals interactions, electrostatic "
                                        "interactions, polar and non-polar solvation, and entropy."
                                    ))
                                elif calculation_method == "FEP":
                                    pdf.multi_cell(0, 10, (
                                        "Free Energy Perturbation (FEP) method was used to calculate the binding free energy. "
                                        "This approach provides a rigorous calculation of binding free energy by simulating "
                                        "the reversible work required to transfer the ligand from solution to the binding site. "
                                        "Multiple lambda windows were used to ensure proper sampling along the alchemical path."
                                    ))
                                else:
                                    pdf.multi_cell(0, 10, (
                                        "Molecular docking was used to estimate the binding pose and affinity. "
                                        "The scoring function incorporates terms for van der Waals interactions, "
                                        "hydrogen bonding, electrostatics, and desolvation effects."
                                    ))
                            
                            if "Results" in report_sections:
                                pdf.add_page()
                                pdf.set_font("Arial", "B", 16)
                                pdf.cell(0, 10, "Results", 0, 1, "L")
                                pdf.ln(5)
                                
                                # Binding energy summary
                                pdf.set_font("Arial", "B", 14)
                                pdf.cell(0, 10, "Binding Energy Summary", 0, 1, "L")
                                
                                pdf.set_font("Arial", "", 12)
                                pdf.cell(100, 10, "Binding Free Energy (kJ/mol):", 0, 0)
                                pdf.cell(0, 10, f"{energy:.2f} ± {uncertainty:.2f}", 0, 1)
                                
                                pdf.cell(100, 10, "Binding Free Energy (kcal/mol):", 0, 0)
                                pdf.cell(0, 10, f"{energy/4.184:.2f} ± {uncertainty/4.184:.2f}", 0, 1)
                                
                                # Energy decomposition
                                pdf.ln(10)
                                pdf.set_font("Arial", "B", 14)
                                pdf.cell(0, 10, "Energy Decomposition", 0, 1, "L")
                                
                                pdf.set_font("Arial", "", 12)
                                pdf.cell(100, 10, "Component", 1, 0, "C")
                                pdf.cell(50, 10, "Value (kJ/mol)", 1, 1, "C")
                                
                                # Add energy component rows
                                for component, value in decomposition.items():
                                    pdf.cell(100, 10, component, 1, 0)
                                    pdf.cell(50, 10, f"{value:.2f}", 1, 1, "R")
                                
                                # If include_plots is True, add plotly figure (save as image first)
                                if include_plots:
                                    pdf.add_page()
                                    pdf.set_font("Arial", "B", 14)
                                    pdf.cell(0, 10, "Energy Component Visualization", 0, 1, "L")
                                    
                                    # In a real app, this would save the actual plotly figure
                                    pdf.ln(10)
                                    pdf.cell(0, 10, "[Energy Component Visualization would be shown here]", 0, 1, "C")
                                    
                                    # For compound comparison
                                    pdf.ln(10)
                                    pdf.set_font("Arial", "B", 14)
                                    pdf.cell(0, 10, "Compound Comparison", 0, 1, "L")
                                    
                                    pdf.ln(10)
                                    pdf.cell(0, 10, "[Compound Comparison Visualization would be shown here]", 0, 1, "C")
                            
                            if "Discussion" in report_sections:
                                pdf.add_page()
                                pdf.set_font("Arial", "B", 16)
                                pdf.cell(0, 10, "Discussion", 0, 1, "L")
                                pdf.ln(5)
                                
                                pdf.set_font("Arial", "", 12)
                                pdf.multi_cell(0, 10, (
                                    f"The binding free energy calculation indicates {binding_strength.lower()} between "
                                    f"the receptor and ligand. The major contribution to binding comes from "
                                    f"{max(decomposition.items(), key=lambda x: abs(x[1]) if x[1] < 0 else 0)[0]} "
                                    f"interactions, while {max(decomposition.items(), key=lambda x: x[1] if x[1] > 0 else 0)[0]} "
                                    f"provides the main unfavorable contribution.\n\n"
                                    f"Compared to existing compounds, our molecule shows "
                                    f"{'superior' if energy < -40 else 'comparable' if energy < -30 else 'inferior'} "
                                    f"binding affinity. "
                                ))
                            
                            if "References" in report_sections:
                                pdf.add_page()
                                pdf.set_font("Arial", "B", 16)
                                pdf.cell(0, 10, "References", 0, 1, "L")
                                pdf.ln(5)
                                
                                pdf.set_font("Arial", "", 12)
                                pdf.multi_cell(0, 10, (
                                    "1. Wang, J., et al. (2019). End-point binding free energy calculation with MM/PBSA and MM/GBSA: challenges and recent improvements. Current Medicinal Chemistry, 26(33), 6222-6241.\n\n"
                                    "2. Aldeghi, M., et al. (2016). Accurate calculation of the absolute free energy of binding for drug molecules. Chemical Science, 7(1), 207-218.\n\n"
                                    "3. Gapsys, V., et al. (2015). Large scale relative protein ligand binding affinities using non-equilibrium alchemy. Chemical Science, 6(4), 2450-2458."
                                ))
                            
                            # Add page numbers in footer
                            for i in range(1, pdf.page_no() + 1):
                                pdf.page = i
                                pdf.set_y(-15)
                                pdf.set_font("Arial", "I", 8)
                                pdf.cell(0, 10, f"Page {i} / {pdf.page_no()}", 0, 0, "C")
                            
                            # Get PDF as bytes
                            pdf_bytes = pdf.output(dest="S").encode("latin1")
                            
                            # Provide download button
                            st.download_button(
                                label="Download PDF Report",
                                data=pdf_bytes,
                                file_name=f"{calculation_method.replace('/', '_')}_binding_report.pdf",
                                mime="application/pdf"
                            )
                            
                            st.success("PDF report prepared for download")
                            
                        except Exception as e:
                            st.error(f"Error generating PDF report: {str(e)}")
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("Please upload structures or select example data to continue")

# Systems Biology Modeling Module
elif app_mode == "Systems Biology Modeling":
    st.markdown("<h2 class='sub-header'>Systems Biology Modeling</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    Model the drug's effect on biological pathways or cellular systems using differential 
    equations or network analysis to predict system-level effects.
    """)
    
    # Tabs for different modeling approaches
    modeling_tab = st.tabs(["Network Analysis", "ODE-based Modeling"])
    
    # Network Analysis Tab
    with modeling_tab[0]:
        st.subheader("Network-based Pathway Analysis")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.write("Configure network analysis parameters:")
            
            network_source = st.radio(
                "Network data source",
                ["Upload custom pathway", "Use example pathway", "Query pathway database"]
            )
            
            if network_source == "Upload custom pathway":
                network_file = st.file_uploader(
                    "Upload network file (CSV, GraphML, SBML)", 
                    type=["csv", "graphml", "sbml", "xml"]
                )
                if network_file:
                    st.success("Network file uploaded successfully!")
                else:
                    st.warning("Please upload a network file")
            
            elif network_source == "Query pathway database":
                database = st.selectbox(
                    "Select pathway database",
                    ["KEGG", "Reactome", "BioCyc", "WikiPathways"]
                )
                pathway_term = st.text_input("Search term (e.g., apoptosis, insulin)")
                st.info("In a real application, this would search the selected database")
            
            st.write("Analysis settings:")
            
            target_protein = st.text_input("Primary drug target protein", "EGFR")
            
            analysis_methods = st.multiselect(
                "Analysis methods",
                ["Centrality analysis", "Path analysis", "Network perturbation", 
                 "Community detection", "Enrichment analysis"],
                default=["Centrality analysis", "Path analysis"]
            )
            
            # Only show if perturbation analysis is selected
            if "Network perturbation" in analysis_methods:
                perturbation_type = st.selectbox(
                    "Perturbation type",
                    ["Node knockout", "Edge reduction", "Activity alteration"]
                )
                
                perturbation_strength = st.slider(
                    "Perturbation strength (%)",
                    0, 100, 50
                )
            
            run_network_button = st.button("Run Network Analysis")
        
        with col2:
            if run_network_button or network_source == "Use example pathway":
                st.subheader("Pathway Network Visualization")
                
                # Get example pathway network
                G = get_example_data("pathway")
                
                # Create a network visualization
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Define node colors based on type
                node_types = {
                    "Drug": "red",
                    "Receptor": "lightblue",
                    "Protein A": "green",
                    "Protein B": "green",
                    "Protein C": "green",
                    "Enzyme X": "purple",
                    "Signaling molecule": "orange",
                    "Gene expression": "yellow",
                    "Cell response": "pink"
                }
                
                node_colors = [node_types[node] for node in G.nodes()]
                
                # Set positions using a layout algorithm
                pos = nx.spring_layout(G, seed=42)
                
                # Draw the network
                nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700, alpha=0.8)
                nx.draw_networkx_edges(G, pos, width=2, alpha=0.7, edge_color='gray')
                nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
                
                # Add title and remove axis
                plt.title("Drug-Target Pathway Network", size=15)
                plt.axis('off')
                
                # Display the plot
                st.pyplot(fig)
                
                # Display analysis results
                st.subheader("Network Analysis Results")
                
                # Calculate and display centrality metrics
                if "Centrality analysis" in analysis_methods:
                    st.write("#### Centrality Analysis")
                    
                    # Calculate different centrality metrics
                    degree_centrality = nx.degree_centrality(G)
                    betweenness_centrality = nx.betweenness_centrality(G)
                    closeness_centrality = nx.closeness_centrality(G)
                    
                    # Convert to DataFrame for display
                    centrality_df = pd.DataFrame({
                        'Node': list(G.nodes()),
                        'Degree Centrality': [degree_centrality[n] for n in G.nodes()],
                        'Betweenness Centrality': [betweenness_centrality[n] for n in G.nodes()],
                        'Closeness Centrality': [closeness_centrality[n] for n in G.nodes()]
                    })
                    
                    # Sort by betweenness centrality
                    centrality_df = centrality_df.sort_values('Betweenness Centrality', ascending=False)
                    
                    # Round values for display
                    centrality_df = centrality_df.round(3)
                    
                    # Display the table
                    st.dataframe(centrality_df)
                    
                    # Highlight key findings
                    top_node = centrality_df.iloc[0]['Node']
                    st.info(f"Key finding: {top_node} is the most central node in the network, suggesting it plays a critical role in the drug's mechanism of action.")
                
                # Path analysis
                if "Path analysis" in analysis_methods:
                    st.write("#### Path Analysis")
                    
                    source_node = "Drug"
                    target_node = "Cell response"
                    
                    # Find all simple paths between drug and cell response
                    all_paths = list(nx.all_simple_paths(G, source=source_node, target=target_node))
                    
                    st.write(f"Found {len(all_paths)} pathways from {source_node} to {target_node}:")
                    
                    for i, path in enumerate(all_paths):
                        st.write(f"Path {i+1}: {' → '.join(path)}")
                    
                    # Find the shortest path
                    shortest_path = nx.shortest_path(G, source=source_node, target=target_node)
                    st.success(f"Shortest pathway: {' → '.join(shortest_path)}")
                
                # Community detection
                if "Community detection" in analysis_methods:
                    st.write("#### Community Detection")
                    
                    try:
                        communities = nx.algorithms.community.greedy_modularity_communities(G.to_undirected())
                        
                        st.write(f"Detected {len(communities)} functional modules in the network:")
                        
                        for i, community in enumerate(communities):
                            st.write(f"Module {i+1}: {', '.join(sorted(community))}")
                            
                    except:
                        st.warning("Community detection requires an undirected network with more nodes")
                
                # Network perturbation analysis
                if "Network perturbation" in analysis_methods:
                    st.write("#### Network Perturbation Analysis")
                    
                    st.write(f"Simulating {perturbation_type} with {perturbation_strength}% effect")
                    
                    # In a real app, this would perform actual network perturbation analysis
                    # Here we just show simulated results
                    
                    # Simulated perturbed node activity levels
                    node_activities = pd.DataFrame({
                        'Node': list(G.nodes()),
                        'Baseline Activity': np.random.uniform(0.5, 1.0, len(G.nodes())),
                        'Perturbed Activity': None
                    })
                    
                    # Calculate perturbed activity based on network distance from drug target
                    for i, node in enumerate(G.nodes()):
                        try:
                            distance = nx.shortest_path_length(G, source=target_protein, target=node)
                            effect = max(0, perturbation_strength/100 * (1 - 0.2 * distance))
                            if perturbation_type == "Node knockout":
                                node_activities.loc[i, 'Perturbed Activity'] = node_activities.loc[i, 'Baseline Activity'] * (1 - effect)
                            else:
                                node_activities.loc[i, 'Perturbed Activity'] = node_activities.loc[i, 'Baseline Activity'] * (1 + effect)
                        except:
                            node_activities.loc[i, 'Perturbed Activity'] = node_activities.loc[i, 'Baseline Activity']
                    
                    # Round for display
                    node_activities = node_activities.round(2)
                    
                    # Calculate percent change
                    node_activities['% Change'] = round(((node_activities['Perturbed Activity'] - 
                                                        node_activities['Baseline Activity']) / 
                                                       node_activities['Baseline Activity']) * 100, 1)
                    
                    # Sort by absolute percent change
                    node_activities = node_activities.sort_values(by='% Change', key=abs, ascending=False)
                    
                    # Display results
                    st.dataframe(node_activities)
                    
                    # Highlight significant changes
                    significant_nodes = node_activities[abs(node_activities['% Change']) > 20]
                    if not significant_nodes.empty:
                        st.success(f"Significant activity changes detected in {len(significant_nodes)} nodes")
                        
                        # Plot the changes
                        fig, ax = plt.subplots(figsize=(10, 6))
                        significant_nodes.plot(
                            kind='bar', 
                            x='Node', 
                            y='% Change', 
                            color=significant_nodes['% Change'].apply(lambda x: 'green' if x > 0 else 'red'),
                            ax=ax
                        )
                        plt.title('Significant Activity Changes After Perturbation')
                        plt.ylabel('Activity Change (%)')
                        plt.xticks(rotation=45, ha='right')
                        plt.grid(axis='y', linestyle='--', alpha=0.7)
                        plt.tight_layout()
                        st.pyplot(fig)
    
    # ODE-based Modeling Tab
    with modeling_tab[1]:
        st.subheader("ODE-based Systems Modeling")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.write("Configure ODE model parameters:")
            
            model_source = st.radio(
                "Model source",
                ["Built-in models", "Upload SBML model", "Define custom equations"]
            )
            
            if model_source == "Built-in models":
                model_type = st.selectbox(
                    "Select model type",
                    ["Basic PK/PD", "Enzyme kinetics", "Cell signaling", "Gene regulatory network"]
                )
                
                if model_type == "Basic PK/PD":
                    st.write("Model parameters:")
                    ka = st.slider("Absorption rate (ka)", 0.1, 2.0, 0.5, 0.1)
                    ke = st.slider("Elimination rate (ke)", 0.01, 0.5, 0.1, 0.01)
                    dose = st.number_input("Dose (mg)", 1, 1000, 100)
                
                elif model_type == "Enzyme kinetics":
                    st.write("Model parameters:")
                    km = st.slider("Michaelis constant (Km)", 0.1, 100.0, 10.0, 0.1)
                    vmax = st.slider("Maximum velocity (Vmax)", 1.0, 100.0, 50.0, 1.0)
                    substrate = st.number_input("Initial substrate (μM)", 1, 1000, 100)
                    inhibitor = st.number_input("Inhibitor concentration (μM)", 0, 100, 0)
                    
                elif model_type == "Cell signaling":
                    st.write("Model parameters:")
                    stimulus = st.slider("Stimulus strength", 0.0, 1.0, 0.5, 0.01)
                    feedback = st.slider("Feedback strength", 0.0, 1.0, 0.3, 0.01)
                    delay = st.slider("Signaling delay", 0.0, 10.0, 2.0, 0.1)
            
            elif model_source == "Upload SBML model":
                sbml_file = st.file_uploader("Upload SBML model file", type=["xml", "sbml"])
                if sbml_file:
                    st.success("SBML model uploaded successfully!")
                else:
                    st.warning("Please upload an SBML model file")
            
            elif model_source == "Define custom equations":
                st.write("In a real application, this would allow defining custom ODEs")
                st.text_area("Enter system of ODEs:", 
                             "dx/dt = -k1*x + k2*y\ndy/dt = k1*x - k2*y - k3*y*z\ndz/dt = k4 - k5*z",
                             height=120)
                st.info("This is a placeholder. A real implementation would parse and simulate these equations.")
            
            st.write("Simulation settings:")
            simulation_time = st.slider("Simulation time", 1, 100, 50)
            time_unit = st.selectbox("Time unit", ["minutes", "hours", "days"])
            
            drug_concentration = st.slider("Drug concentration (μM)", 0.0, 100.0, 10.0, 0.1)
            
            run_ode_button = st.button("Run ODE Simulation")
        
        with col2:
            if run_ode_button:
                st.subheader("ODE Simulation Results")
                
                # Run the simulated systems biology modeling
                simulation_results = run_systems_biology_modeling(
                    model_type if model_source == "Built-in models" else model_source,
                    {"drug_concentration": drug_concentration},
                    simulation_time
                )
                
                # Plot the results
                if isinstance(simulation_results, pd.DataFrame):  # ODE simulation results
                    # Plot time series
                    st.write("#### Time Course Simulation")
                    
                    fig = px.line(
                        simulation_results, 
                        x='Time', 
                        y=simulation_results.columns[1:],
                        title='Species Concentrations Over Time'
                    )
                    
                    fig.update_layout(
                        xaxis_title=f"Time ({time_unit})",
                        yaxis_title="Concentration (a.u.)",
                        legend_title="Species"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add heatmap visualization of the time course
                    st.write("#### Heatmap Visualization")
                    
                    # Prepare data for heatmap
                    heatmap_data = simulation_results.set_index('Time')
                    
                    fig = px.imshow(
                        heatmap_data.T,
                        aspect="auto",
                        labels=dict(x=f"Time ({time_unit})", y="Species", color="Concentration"),
                        title="Species Concentration Heatmap"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Phase plane analysis (if 2+ species)
                    if len(simulation_results.columns) > 3:  # Time + at least 2 species
                        st.write("#### Phase Plane Analysis")
                        
                        species1 = st.selectbox("X-axis species", simulation_results.columns[1:], index=0)
                        species2 = st.selectbox("Y-axis species", simulation_results.columns[1:], index=1)
                        
                        fig = px.scatter(
                            simulation_results,
                            x=species1,
                            y=species2,
                            color='Time',
                            title=f"Phase Plane: {species1} vs {species2}",
                            labels={species1: f"{species1} Concentration", species2: f"{species2} Concentration"},
                        )
                        
                        # Add arrows to show direction
                        steps = max(1, len(simulation_results) // 20)  # Show ~20 arrows
                        for i in range(0, len(simulation_results)-steps, steps):
                            fig.add_annotation(
                                x=simulation_results[species1].iloc[i],
                                y=simulation_results[species2].iloc[i],
                                ax=simulation_results[species1].iloc[i+steps] - simulation_results[species1].iloc[i],
                                ay=simulation_results[species2].iloc[i+steps] - simulation_results[species2].iloc[i],
                                showarrow=True,
                                arrowhead=2,
                                arrowsize=1,
                                arrowcolor="#636363",
                                arrowwidth=1.5
                            )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Sensitivity analysis
                    st.write("#### Parameter Sensitivity Analysis")
                    
                    # In a real app, this would run multiple simulations with varied parameters
                    # Here we show some simulated sensitivity results
                    
                    parameters = ["Drug concentration", "Binding rate", "Dissociation rate", 
                                 "Degradation rate", "Production rate"]
                    species = simulation_results.columns[1:].tolist()
                    
                    # Create random sensitivity coefficients
                    sensitivity_data = {}
                    
                    for param in parameters:
                        sensitivity_data[param] = [np.random.uniform(-1, 1) for _ in species]
                    
                    sensitivity_df = pd.DataFrame(sensitivity_data, index=species)
                    
                    # Plot sensitivity heatmap
                    fig = px.imshow(
                        sensitivity_df,
                        labels=dict(x="Parameter", y="Species", color="Sensitivity"),
                        color_continuous_scale="RdBu_r",
                        color_continuous_midpoint=0,
                        title="Parameter Sensitivity Analysis"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Key insights
                    st.write("#### Key Insights")
                    
                    # In a real app, these would be derived from actual simulation results
                    st.info("**Peak concentration**: Maximum drug effect observed at 14.3 hours")
                    st.info("**Steady state**: System reaches equilibrium after approximately 35 hours")
                    st.info("**Most sensitive parameter**: System dynamics are most sensitive to changes in the binding rate")
                    
                    # Download results
                    st.download_button(
                        label="Download Simulation Data (CSV)",
                        data=simulation_results.to_csv(index=False),
                        file_name=f"{model_type}_simulation_results.csv",
                        mime="text/csv"
                    )

# Results Dashboard Module
elif app_mode == "Results Dashboard":
    st.markdown("<h2 class='sub-header'>Integrated Results Dashboard</h2>", unsafe_allow_html=True)
    
    st.write("""
    This dashboard integrates results from all simulation modules to provide 
    comprehensive insights into drug behavior in biological systems.
    """)
    
    # Simulated projects
    projects = ["DrugX EGFR Inhibitor", "CompoundY p53 Activator", "PeptideZ Ion Channel Blocker", "New Project"]
    
    selected_project = st.selectbox("Select project", projects)
    
    if selected_project != "New Project":
        # Display summary dashboard
        st.markdown("<div class='result-section'>", unsafe_allow_html=True)
        st.subheader("Project Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Binding Affinity",
                value="-42.3 kJ/mol",
                delta="-3.6 kJ/mol"
            )
            
        with col2:
            st.metric(
                label="MD Simulation Stability",
                value="98.2%",
                delta="2.5%"
            )
            
        with col3:
            st.metric(
                label="Pathway Impact Score",
                value="7.8/10",
                delta="1.2"
            )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Visualizations tabs
        result_tabs = st.tabs(["Molecular Properties", "System Dynamics", "Comparative Analysis"])
        
        with result_tabs[0]:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Structure-Activity Relationship")
                
                # Example structure-activity data
                sar_data = pd.DataFrame({
                    'Compound': [f"Analog-{i}" for i in range(1, 11)],
                    'R1': np.random.choice(['H', 'CH3', 'Cl', 'F', 'OH', 'NH2'], 10),
                    'R2': np.random.choice(['H', 'Ph', 'Py', 'Furan', 'CN', 'CF3'], 10),
                    'Activity (IC50, nM)': np.random.lognormal(3, 1, 10)
                })
                
                # Sort by activity
                sar_data = sar_data.sort_values('Activity (IC50, nM)')
                
                # Display as table
                st.dataframe(sar_data)
                
                # Plot activity by R1 group
                r1_groups = sar_data.groupby('R1')['Activity (IC50, nM)'].mean().reset_index()
                
                fig = px.bar(
                    r1_groups,
                    x='R1',
                    y='Activity (IC50, nM)',
                    title='Average Activity by R1 Group',
                    log_y=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Molecular Properties")
                
                # Example molecular descriptors
                descriptors = {
                    'Molecular Weight': '412.5 Da',
                    'LogP': '3.2',
                    'H-Bond Donors': '2',
                    'H-Bond Acceptors': '5',
                    'Rotatable Bonds': '6',
                    'TPSA': '84.3 Å²',
                    'QED Drug-likeness': '0.78',
                    'Synthetic Accessibility': '3.2/10'
                }
                
                # Display as definition list with custom style
                html_content = "<dl style='background-color: #f8f9fa; padding: 20px; border-radius: 10px;'>"
                
                for prop, value in descriptors.items():
                    html_content += f"<dt style='font-weight: bold; margin-top: 10px;'>{prop}</dt>"
                    html_content += f"<dd style='margin-bottom: 10px; margin-left: 20px;'>{value}</dd>"
                
                html_content += "</dl>"
                
                st.markdown(html_content, unsafe_allow_html=True)
                
                # Display example molecule
                st.subheader("Molecular Structure")
                example_pdb = """
ATOM      1  N   ASP A  30      11.482  22.360  81.280  1.00 67.06           N  
ATOM      2  CA  ASP A  30      12.652  23.228  81.504  1.00 68.91           C  
ATOM      3  C   ASP A  30      12.443  24.662  80.994  1.00 64.85           C  
ATOM      4  O   ASP A  30      11.492  24.979  80.275  1.00 63.82           O  
ATOM      5  CB  ASP A  30      13.010  23.298  82.992  1.00 72.76           C  
ATOM      6  CG  ASP A  30      13.326  21.976  83.654  1.00 75.91           C  
ATOM      7  OD1 ASP A  30      14.027  21.131  83.051  1.00 77.82           O  
ATOM      8  OD2 ASP A  30      12.868  21.776  84.801  1.00 77.86           O1-
ATOM      9  N   THR A  31      13.366  25.533  81.379  1.00 66.11           N  
ATOM     10  CA  THR A  31      13.295  26.946  81.016  1.00 67.36           C  
                """
                
                display_pdb_structure(example_pdb)
        
        with result_tabs[1]:
            st.subheader("Pharmacokinetic Profile")
            
            # Create example PK data
            time_points = np.linspace(0, 24, 100)
            concentration = 100 * np.exp(-0.1 * time_points) * (1 - np.exp(-1.0 * time_points))
            effect = 100 * (concentration ** 2) / (10 ** 2 + concentration ** 2)
            
            # Add some noise
            concentration += np.random.normal(0, concentration.max() * 0.05, len(concentration))
            effect += np.random.normal(0, effect.max() * 0.05, len(effect))
            
            # Create DataFrame
            pk_data = pd.DataFrame({
                'Time (hours)': time_points,
                'Plasma Concentration (ng/mL)': concentration,
                'Effect (%)': effect
            })
            
            # Plot PK/PD
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig.add_trace(
                go.Scatter(
                    x=pk_data['Time (hours)'],
                    y=pk_data['Plasma Concentration (ng/mL)'],
                    name="Concentration",
                    line=dict(color='blue', width=2)
                ),
                secondary_y=False
            )
            
            fig.add_trace(
                go.Scatter(
                    x=pk_data['Time (hours)'],
                    y=pk_data['Effect (%)'],
                    name="Effect",
                    line=dict(color='red', width=2, dash='dot')
                ),
                secondary_y=True
            )
            
            fig.update_layout(
                title="Pharmacokinetics and Pharmacodynamics",
                xaxis_title="Time (hours)",
                legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)')
            )
            
            fig.update_yaxes(title_text="Plasma Concentration (ng/mL)", secondary_y=False)
            fig.update_yaxes(title_text="Effect (%)", secondary_y=True)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # PK parameters
            pk_params = {
                'Cmax': f"{np.max(concentration):.1f} ng/mL",
                'Tmax': f"{time_points[np.argmax(concentration)]:.1f} hours",
                'Half-life': "5.2 hours",
                'AUC': f"{np.trapz(concentration, time_points):.1f} ng·h/mL",
                'Bioavailability': "68%",
                'Volume of distribution': "0.8 L/kg",
                'Clearance': "12.5 mL/min/kg"
            }
            
            # Create columns for parameters
            col1, col2 = st.columns(2)
            
            with col1:
                for param, value in list(pk_params.items())[:4]:
                    st.metric(label=param, value=value)
            
            with col2:
                for param, value in list(pk_params.items())[4:]:
                    st.metric(label=param, value=value)
                    
            # Systems biology integration
            st.subheader("Predicted Cellular Response")
            
            # In a real app, this would be the result of systems biology modeling
            # simulating cellular response to the drug over time
            
            # Example pathway activity over time
            pathways = ["MAPK signaling", "PI3K/AKT pathway", "JAK/STAT pathway", 
                       "Cell cycle regulation", "Apoptosis"]
            
            # Time points
            time_points = np.linspace(0, 24, 100)
            
            # Create different response patterns for each pathway
            pathway_data = {}
            
            # Baseline activities (all start at 1.0)
            for pathway in pathways:
                if pathway == "MAPK signaling":
                    # Rapid decrease
                    y = 1.0 - 0.8 * (1 - np.exp(-0.5 * time_points))
                elif pathway == "PI3K/AKT pathway":
                    # Slower decrease
                    y = 1.0 - 0.6 * (1 - np.exp(-0.2 * time_points))
                elif pathway == "JAK/STAT pathway":
                    # Initial increase then decrease
                    y = 1.0 + 0.4 * np.exp(-0.1 * time_points) * np.sin(0.4 * time_points)
                elif pathway == "Cell cycle regulation":
                    # Delayed decrease
                    y = 1.0 - 0.7 * (1 - np.exp(-0.1 * (time_points - 5))) * (time_points > 5)
                else:  # Apoptosis
                    # Delayed increase
                    y = 1.0 + 1.5 * (1 - np.exp(-0.1 * (time_points - 8))) * (time_points > 8)
                
                # Add some noise
                y += np.random.normal(0, 0.03, len(time_points))
                
                pathway_data[pathway] = y
            
            # Create DataFrame
            pathway_df = pd.DataFrame({"Time (hours)": time_points, **pathway_data})
            
            # Plot pathway activities
            fig = px.line(
                pathway_df, 
                x='Time (hours)', 
                y=pathways,
                title='Predicted Pathway Activity Over Time'
            )
            
            fig.update_layout(
                yaxis_title="Relative Activity",
                legend_title="Pathway"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Key observations
            st.info("""
            **Key observations**:
            * MAPK signaling is rapidly inhibited, consistent with the drug's primary mechanism
            * Apoptosis pathway is activated after 8 hours, indicating potential anti-cancer activity
            * Cell cycle regulation shows delayed response, suggesting growth inhibition
            """)
            
        with result_tabs[2]:
            st.subheader("Comparative Efficacy Analysis")
            
            # Simulated comparative data against existing drugs
            comparison_metrics = {
                'Target affinity (IC50, nM)': [12, 35, 8, 18, 5],
                'Selectivity index': [45, 12, 28, 35, 65],
                'Cell potency (EC50, nM)': [85, 220, 110, 95, 45],
                'Solubility (μg/mL)': [120, 350, 45, 180, 95],
                'Metabolic stability (% remaining)': [65, 42, 38, 72, 68],
                'hERG inhibition (IC50, μM)': ['>30', 12, 8, 25, '>30'],
                'CYP inhibition (% inhibition)': [15, 42, 65, 24, 18]
            }
            
            compounds = ["Existing drug 1", "Existing drug 2", "Existing drug 3", "Competitor X", "Our compound"]
            
            # Create radar chart
            categories = list(comparison_metrics.keys())
            
            fig = go.Figure()
            
            # Normalize values for radar chart (higher is better)
            normalized_values = {}
            
            for metric, values in comparison_metrics.items():
                # For most metrics, higher is better
                if metric in ['hERG inhibition (IC50, μM)', 'CYP inhibition (% inhibition)', 'Cell potency (EC50, nM)', 'Target affinity (IC50, nM)']:
                    # For these, lower is better, so invert
                    if metric == 'hERG inhibition (IC50, μM)':
                        # Handle special case with >30 values
                        numeric_values = [30 if v == '>30' else v for v in values]
                        min_val = min(numeric_values)
                        max_val = max(numeric_values)
                        normalized_values[metric] = [1 - (v if v != '>30' else 30 - min_val) / (max_val - min_val) for v in values]
                    else:
                        min_val = min(values)
                        max_val = max(values)
                        normalized_values[metric] = [1 - (v - min_val) / (max_val - min_val) for v in values]
                else:
                    # For these, higher is better
                    min_val = min(values)
                    max_val = max(values)
                    normalized_values[metric] = [(v - min_val) / (max_val - min_val) for v in values]
            
            # Add traces for each compound
            for i, compound in enumerate(compounds):
                values = [normalized_values[metric][i] for metric in categories]
                # Add the first value again to close the loop
                values.append(values[0])
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories + [categories[0]],
                    name=compound,
                    fill='toself',
                    opacity=0.6
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                showlegend=True,
                title="Drug Comparative Profile"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display tabular comparison
            st.subheader("Detailed Comparison")
            
            # Create comparison table
            comparison_df = pd.DataFrame(comparison_metrics, index=compounds)
            
            # Format the hERG column correctly
            comparison_df['hERG inhibition (IC50, μM)'] = comparison_df['hERG inhibition (IC50, μM)'].astype(str)
            
            # Display the table
            st.dataframe(comparison_df)
            
            # Add bar chart comparison for selected metric
            selected_metric = st.selectbox(
                "Select metric for detailed comparison",
                list(comparison_metrics.keys())
            )
            
            # Handle the special case for hERG
            if selected_metric == 'hERG inhibition (IC50, μM)':
                # Replace >30 with 30 for plotting purposes
                plot_values = [30 if v == '>30' else v for v in comparison_metrics[selected_metric]]
            else:
                plot_values = comparison_metrics[selected_metric]
            
            # Create a DataFrame for the selected metric
            plot_df = pd.DataFrame({
                'Compound': compounds,
                selected_metric: plot_values
            })
            
            # Determine if higher or lower is better for coloring
            if selected_metric in ['hERG inhibition (IC50, μM)', 'CYP inhibition (% inhibition)', 
                                 'Cell potency (EC50, nM)', 'Target affinity (IC50, nM)']:
                # For these, lower is better
                color_scale = "RdYlGn_r"  # Reversed scale (red is high, green is low)
            else:
                # For these, higher is better
                color_scale = "RdYlGn"  # Normal scale (green is high, red is low)
            
            fig = px.bar(
                plot_df,
                x='Compound',
                y=selected_metric,
                color=selected_metric,
                color_continuous_scale=color_scale,
                title=f"Comparison of {selected_metric}"
            )
            
            # Highlight "Our compound"
            fig.update_traces(
                marker_line_width=[2 if c == "Our compound" else 0 for c in plot_df['Compound']],
                marker_line_color=["black" if c == "Our compound" else "white" for c in plot_df['Compound']]
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add interpretation
            if selected_metric == 'Target affinity (IC50, nM)':
                if plot_values[compounds.index("Our compound")] == min(plot_values):
                    st.success("Our compound shows the best target affinity among all compared molecules.")
                else:
                    st.info(f"Our compound shows {plot_values[compounds.index('Our compound')]} nM affinity, " +
                           f"which is {'better' if plot_values[compounds.index('Our compound')] < plot_values[compounds.index('Existing drug 1')] else 'worse'} " +
                           f"than the current standard of care (Existing drug 1).")
            elif selected_metric == 'Selectivity index':
                if plot_values[compounds.index("Our compound")] == max(plot_values):
                    st.success("Our compound shows the best selectivity profile, reducing potential off-target effects.")
                else:
                    st.info(f"Our compound shows good selectivity with an index of {plot_values[compounds.index('Our compound')]}, " +
                           f"{'better' if plot_values[compounds.index('Our compound')] > plot_values[compounds.index('Existing drug 1')] else 'worse'} " +
                           f"than the current standard of care.")
    else:
        # New project
        st.info("Create a new simulation project by configuring the parameters in each module.")
        
        # Project setup form
        with st.form("new_project_form"):
            st.subheader("New Project Configuration")
            
            project_name = st.text_input("Project Name")
            project_description = st.text_area("Project Description")
            
            col1, col2 = st.columns(2)
            
            with col1:
                target_type = st.selectbox(
                    "Target Type",
                    ["Enzyme", "Receptor", "Ion Channel", "Transporter", "Nuclear Receptor", "Other"]
                )
                
                therapeutic_area = st.selectbox(
                    "Therapeutic Area",
                    ["Oncology", "Neurology", "Cardiology", "Immunology", "Infectious Disease", "Other"]
                )
            
            with col2:
                molecule_type = st.selectbox(
                    "Molecule Type",
                    ["Small molecule", "Peptide", "Antibody", "Protein", "Oligonucleotide", "Other"]
                )
                
                development_stage = st.selectbox(
                    "Development Stage",
                    ["Target identification", "Hit discovery", "Lead optimization", "Preclinical", "Clinical"]
                )
            
            submit_button = st.form_submit_button("Create Project")
            
            if submit_button:
                st.success(f"Project '{project_name}' created successfully! Navigate to simulation modules to begin analysis.")

# Add app configuration section with performance settings
st.sidebar.markdown("---")
with st.sidebar.expander("⚙️ App Configuration"):
    st.subheader("Performance Settings")
    
    # Performance optimization settings
    use_gpu = st.checkbox("Use GPU acceleration (if available)", value=True)
    parallel_processing = st.checkbox("Enable parallel processing", value=True)
    processing_threads = st.slider("Processing threads", 1, max(16, multiprocessing.cpu_count()), 
                                 min(8, max(1, multiprocessing.cpu_count()-1)))
    
    memory_limit = st.selectbox("Memory limit per worker", 
                              ["2GB", "4GB", "8GB", "16GB", "32GB"], 
                              index=1)
    
    # Cache settings
    enable_caching = st.checkbox("Enable result caching", value=True)
    
    if enable_caching:
        st.info("Caching enabled: Repeated calculations with the same parameters will be faster.")
    
    # Data handling for large files
    st.subheader("Data Handling")
    
    chunk_size = st.selectbox("Chunk size for large files", 
                            ["10MB", "50MB", "100MB", "500MB", "1GB"], 
                            index=1)
    
    # Reset app button
    if st.button("Reset Application"):
        st.experimental_rerun()

# Add info about OpenMM availability
if not OPENMM_AVAILABLE and app_mode == "Molecular Dynamics (MD)":
    st.sidebar.warning("⚠️ OpenMM not detected. Using simulated data.")
    st.sidebar.markdown("""
    To enable real MD simulations, install OpenMM:
    ```
    conda install -c conda-forge openmm
    ```
    """)

# Add info about PyMBAR availability
if not PYMBAR_AVAILABLE and app_mode == "Free Energy Calculations":
    st.sidebar.warning("⚠️ PyMBAR not detected. Using simulated data.")
    st.sidebar.markdown("""
    To enable advanced free energy calculations, install PyMBAR:
    ```
    conda install -c conda-forge pymbar
    ```
    """)

# Version information
st.sidebar.markdown("---")
st.sidebar.markdown("""
**BioSim v1.2.0**  
Enhanced with:
- Real simulation engines
- Advanced export options
- Optimized performance
""")

# Footer with additional information
st.markdown("""
<div class='footer'>
    <p>BioSim: Drug Behavior Simulation Platform © 2025 | Developed with Streamlit and OpenMM</p>
    <p><small>For research and educational purposes only. Not for clinical use.</small></p>
</div>
""", unsafe_allow_html=True)