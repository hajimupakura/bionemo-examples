# import streamlit as st
# import os
# import pandas as pd
# import numpy as np
# from rdkit import Chem
# from rdkit.Chem import AllChem, Descriptors, Crippen, Draw  # Added Draw here
# from biopandas.pdb import PandasPdb
# import py3Dmol
# import matplotlib.pyplot as plt
# import seaborn as sns
# import logging
# from typing import Dict, List, Tuple
# import requests
# import json
# from io import StringIO

# # Setup logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # Streamlit App Configuration
# st.set_page_config(page_title="Drug Discovery Pipeline", layout="wide")

# # Helper Functions
# def load_pdb_or_pdbqt(uploaded_file) -> Tuple[PandasPdb, str]:
#     """Load uploaded PDB or PDBQT file with format verification."""
#     if uploaded_file is None:
#         st.error("Please upload a PDB or PDBQT file.")
#         return None, None
    
#     # Verify file extension
#     valid_extensions = ['.pdb', '.pdbqt']
#     if not any(uploaded_file.name.endswith(ext) for ext in valid_extensions):
#         st.error("Invalid file format. Please upload a .pdb or .pdbqt file.")
#         logging.error(f"Uploaded file {uploaded_file.name} is not a .pdb or .pdbqt file.")
#         return None, None
    
#     try:
#         # Read file content into memory
#         file_content = uploaded_file.getvalue().decode("utf-8")
#         file_lines = StringIO(file_content)
        
#         # Load with PandasPdb (works for both .pdb and .pdbqt, though .pdbqt may have extra columns)
#         ppdb = PandasPdb().read_pdb_from_list(file_lines.readlines())
#         st.success(f"Loaded file: {uploaded_file.name}")
#         return ppdb, file_content
#     except Exception as e:
#         st.error(f"Error loading file: {e}")
#         logging.error(f"File loading failed: {e}")
#         return None, None

# def visualize_3d_structure(file_content: str, file_type: str, style: Dict = None, width: int = 800, height: int = 400) -> None:
#     """Generic 3D visualization for PDB or PDBQT using py3Dmol."""
#     view = py3Dmol.view(width=width, height=height)
#     view.addModel(file_content, file_type)  # Use file_type to specify format
#     if style:
#         view.setStyle(style)
#     else:
#         view.setStyle({'cartoon': {'color': 'spectrum'}})
#     view.zoomTo()
#     st.components.v1.html(view._make_html(), width=width, height=height)

# def save_to_file(data, filename: str, format: str = "csv"):
#     """Save data to file and provide download button."""
#     if format == "csv":
#         data.to_csv(filename, index=False)
#         with open(filename, "rb") as f:
#             st.download_button(f"Download {filename}", f, file_name=filename)
#     elif format == "sdf":
#         writer = Chem.SDWriter(filename)
#         writer.write(data)
#         writer.close()
#         with open(filename, "rb") as f:
#             st.download_button(f"Download {filename}", f, file_name=filename)

# # Tabbed Interface
# tab1, tab2, tab3 = st.tabs(["Target Characterization", "Pharmacophore Modeling", "Virtual Screening"])

# # --- Target Characterization Tab ---
# with tab1:
#     st.header("Target Characterization")
#     st.write("Analyze protein structure and identify binding pockets.")

#     # File Upload
#     uploaded_file = st.file_uploader("Upload Protein Structure (PDB or PDBQT format)", type=["pdb", "pdbqt"])
#     ppdb, file_content = load_pdb_or_pdbqt(uploaded_file)
#     if ppdb is None or file_content is None:
#         st.stop()

#     # Determine file type for visualization
#     file_type = "pdb" if uploaded_file.name.endswith('.pdb') else "pdbqt"

#     # Visualize Structure
#     st.subheader("Protein Structure")
#     visualize_3d_structure(file_content, file_type)

#     # Pocket Analysis
#     st.subheader("Binding Pocket Analysis")
#     @st.cache_data
#     def analyze_pockets(_ppdb: PandasPdb) -> List[Dict]:
#         """Analyze binding pockets with key properties."""
#         hetatm_data = _ppdb.df['HETATM']
#         protein_atoms = _ppdb.df['ATOM']
#         pockets = []
#         for (residue, chain), group in hetatm_data.groupby(['residue_name', 'chain_id']):
#             if residue == 'HOH':  # Skip water molecules
#                 continue
#             center = (group['x_coord'].mean(), group['y_coord'].mean(), group['z_coord'].mean())
#             volume = np.prod(group[['x_coord', 'y_coord', 'z_coord']].max() - group[['x_coord', 'y_coord', 'z_coord']].min())
#             distances = np.sqrt(((protein_atoms[['x_coord', 'y_coord', 'z_coord']] - center)**2).sum(axis=1))
#             nearby = protein_atoms[distances <= 5.0]
#             pocket = {
#                 'residue': residue,
#                 'chain': chain,
#                 'center': center,
#                 'volume': volume,
#                 'nearby_residues': set(zip(nearby['residue_name'], nearby['chain_id'])),
#                 'hydrophobicity': sum(1 for res in nearby['residue_name'] if res in ['ALA', 'VAL', 'LEU', 'ILE', 'PHE'])
#             }
#             pockets.append(pocket)
#         return pockets

#     pockets = analyze_pockets(ppdb)
#     if pockets:
#         pocket_df = pd.DataFrame(pockets)
#         st.write(f"Detected {len(pockets)} binding pockets.")
#         st.dataframe(pocket_df[['residue', 'chain', 'volume', 'hydrophobicity']])
#         save_to_file(pocket_df, "pocket_analysis.csv")

#         # Visualize Pockets
#         st.subheader("Pocket Visualization")
#         view = py3Dmol.view(width=800, height=400)
#         view.addModel(file_content, file_type)
#         view.setStyle({'cartoon': {'color': 'lightgray', 'opacity': 0.5}})
#         for i, pocket in enumerate(pockets):
#             view.addSphere({'center': {'x': pocket['center'][0], 'y': pocket['center'][1], 'z': pocket['center'][2]},
#                            'radius': 2.0, 'color': 'purple', 'alpha': 0.7})
#             view.addLabel(f"Pocket {i+1}", {'position': {'x': pocket['center'][0], 'y': pocket['center'][1], 'z': pocket['center'][2]}})
#         view.zoomTo()
#         st.components.v1.html(view._make_html(), width=800, height=400)

#         # Bar Plot of Pocket Volumes
#         st.subheader("Pocket Volume Distribution")
#         fig, ax = plt.subplots()
#         sns.barplot(x=pocket_df.index, y='volume', data=pocket_df, ax=ax)
#         ax.set_xlabel("Pocket Index")
#         ax.set_ylabel("Volume (Å³)")
#         st.pyplot(fig)
#     else:
#         st.warning("No binding pockets detected in the uploaded file.")

# # --- Pharmacophore Modeling Tab ---
# with tab2:
#     st.header("Pharmacophore Modeling")
#     st.write("Generate a pharmacophore model based on pocket features.")

#     # Pocket Selection
#     if 'pockets' not in locals():
#         st.warning("Run Target Characterization first to identify pockets.")
#         st.stop()
#     pocket_idx = st.selectbox("Select Pocket", range(len(pockets)), format_func=lambda x: f"Pocket {x+1} ({pockets[x]['residue']})")

#     # Pharmacophore Generation
#     class PharmacophoreGenerator:
#         def __init__(self):
#             self.feature_thresholds = {'hydrophobicity': 3, 'polarity': 2}

#         def generate_features(self, pocket: Dict) -> List[str]:
#             features = []
#             if pocket['hydrophobicity'] >= self.feature_thresholds['hydrophobicity']:
#                 features.append('c1ccccc1')  # Aromatic hydrophobic
#             if len(pocket['nearby_residues']) >= self.feature_thresholds['polarity']:
#                 features.append('NC=O')  # Hydrogen bond acceptor/donor
#             return features

#         def create_3d_model(self, features: List[str]) -> Chem.Mol:
#             mols = [Chem.MolFromSmiles(f) for f in features if Chem.MolFromSmiles(f)]
#             if not mols:
#                 return None
#             for mol in mols:
#                 mol = Chem.AddHs(mol)
#                 AllChem.EmbedMolecule(mol, randomSeed=42)
#             return Chem.CombineMols(*mols) if len(mols) > 1 else mols[0]

#     generator = PharmacophoreGenerator()
#     features = generator.generate_features(pockets[pocket_idx])
#     st.write("Generated Features (SMILES):", features)

#     pharmacophore = generator.create_3d_model(features)
#     if pharmacophore:
#         st.subheader("Pharmacophore Structure")
#         img = Draw.MolToImage(pharmacophore)
#         st.image(img, caption="2D Pharmacophore")

#         # Save Pharmacophore
#         save_to_file(pharmacophore, "pharmacophore.sdf", format="sdf")
#     else:
#         st.warning("Could not generate a valid pharmacophore model from the selected pocket.")

# # --- Virtual Screening Tab ---
# with tab3:
#     st.header("Virtual Screening")
#     st.write("Screen compounds against the pharmacophore and drug-likeness criteria.")

#     # Compound File Upload
#     ligand_file = st.file_uploader("Upload Compound Library (SMILES format)", type=["smi", "txt"])
#     if ligand_file is None:
#         st.warning("Please upload a SMILES file to proceed.")
#         st.stop()
    
#     # Verify file format
#     if not (ligand_file.name.endswith('.smi') or ligand_file.name.endswith('.txt')):
#         st.error("Invalid file format. Please upload a .smi or .txt file with SMILES strings.")
#         st.stop()

#     ligand_data = ligand_file.getvalue().decode("utf-8").splitlines()

#     # Screening
#     class CompoundScreener:
#         def __init__(self):
#             self.criteria = {'MW_max': 500, 'LogP_range': (0, 5), 'TPSA_max': 140}

#         def screen(self, ligand_data: List[str], pharmacophore_features: List[str]) -> pd.DataFrame:
#             results = []
#             for smiles in ligand_data:
#                 smiles = smiles.strip()
#                 if not smiles:
#                     continue
#                 mol = Chem.MolFromSmiles(smiles)
#                 if mol:
#                     props = {
#                         'SMILES': smiles,
#                         'MW': Descriptors.MolWt(mol),
#                         'LogP': Crippen.MolLogP(mol),
#                         'TPSA': Descriptors.TPSA(mol)
#                     }
#                     if (props['MW'] <= self.criteria['MW_max'] and
#                         self.criteria['LogP_range'][0] <= props['LogP'] <= self.criteria['LogP_range'][1] and
#                         props['TPSA'] <= self.criteria['TPSA_max']):
#                         results.append(props)
#             return pd.DataFrame(results)

#     if 'features' not in locals():
#         st.warning("Run Pharmacophore Modeling first to generate features.")
#         st.stop()

#     screener = CompoundScreener()
#     screened_df = screener.screen(ligand_data, features)
#     if not screened_df.empty:
#         st.dataframe(screened_df)
#         save_to_file(screened_df, "screened_compounds.csv")

#         # Visualization
#         st.subheader("Property Distribution")
#         fig, ax = plt.subplots()
#         sns.scatterplot(x='LogP', y='TPSA', size='MW', data=screened_df, ax=ax)
#         ax.set_title("Screened Compounds: LogP vs TPSA")
#         st.pyplot(fig)
#     else:
#         st.warning("No compounds passed the screening criteria.")

# # Footer
# st.markdown("---")


import streamlit as st
import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, Draw, FilterCatalog, MolStandardize, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Geometry import Point3D  # Import Point3D for 3D coordinates
from biopandas.pdb import PandasPdb
import py3Dmol
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import requests
import json
from io import StringIO
from datetime import datetime
import pickle
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from PIL import Image
import base64
from stmol import showmol

# Setup more comprehensive logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"drug_discovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("drug_discovery_pipeline")

# Constants
PDB_BANK_API = "https://data.rcsb.org/rest/v1/core/entry"
LIPINSKI_RULES = {
    "MW": 500,
    "LogP": 5,
    "HBD": 5,
    "HBA": 10,
    "RotBonds": 10
}

# Streamlit App Configuration
st.set_page_config(
    page_title="Advanced Drug Discovery Pipeline",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main .block-container {padding-top: 2rem;}
    h1, h2, h3 {color: #2C3E50;}
    .stTabs [data-baseweb="tab-list"] {gap: 24px; margin-bottom: 1rem;}
    .stTabs [data-baseweb="tab"] {padding: 10px 20px;}
    .css-1oe6o3n {padding: 0.5rem 0.5rem;}
    div[data-testid="stSidebarNav"] {background-image: linear-gradient(#2e7bcf,#2e7bcf); 
                                      padding-top: 1.5rem; border-radius: 0.5rem;}
    div.stButton > button {background-color: #2e7bcf; color:white; border-radius:5px;}
    div.stButton > button:hover {background-color: #1a5889; color:white;}
    .reportview-container .main footer {visibility: hidden;}
    .css-ffhzg2 {background-color: #ecf0f1; padding: 1rem; border-radius: 0.5rem;}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'analyzed_proteins' not in st.session_state:
    st.session_state.analyzed_proteins = {}
if 'pharmacophores' not in st.session_state:
    st.session_state.pharmacophores = {}
if 'screened_compounds' not in st.session_state:
    st.session_state.screened_compounds = {}
if 'predictions' not in st.session_state:
    st.session_state.predictions = {}
if 'current_protein' not in st.session_state:
    st.session_state.current_protein = None

# Sidebar for configuration and global settings
with st.sidebar:
    st.title("Controls & Settings")
    
    # Theme selection
    selected_theme = st.selectbox(
        "Color Theme",
        ["Blues", "Greens", "Reds", "Purples", "Oranges"],
        index=0
    )
    
    # Global parameters for analysis
    st.subheader("Analysis Parameters")
    pocket_detection_method = st.selectbox(
        "Pocket Detection Method",
        ["Geometric", "Energy-based", "Conservation"],
        index=0
    )
    
    druglikeness_filter = st.multiselect(
        "Druglikeness Filters", 
        ["Lipinski's Rule of 5", "Veber Rules", "PAINS Filter", "Ghose Filter"],
        default=["Lipinski's Rule of 5"]
    )
    
    visualization_quality = st.select_slider(
        "Visualization Quality",
        options=["Low", "Medium", "High"],
        value="Medium"
    )
    
    st.subheader("Export Options")
    export_format = st.radio(
        "Export Format",
        ["CSV", "SDF", "JSON"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### Session Info")
    if st.session_state.current_protein:
        st.info(f"Current protein: {st.session_state.current_protein}")
    else:
        st.warning("No protein loaded")
    
    if st.button("Clear Session Data"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.experimental_rerun()
    
    st.markdown("---")
    st.markdown("<small>v2.0.0 - Updated March 2025</small>", unsafe_allow_html=True)

# Helper Functions
def log_action(action: str, details: str) -> None:
    """Log user actions for better tracking and debugging."""
    logger.info(f"ACTION: {action} - {details}")
    st.session_state.history.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "action": action,
        "details": details
    })

def add_fingerprint_to_mol(mol: Chem.Mol) -> Chem.Mol:
    """Add Morgan fingerprints to molecule as a property."""
    if mol is not None:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        mol.SetProp("MorganFP", fp.ToBitString())
    return mol

def load_pdb_or_pdbqt(uploaded_file) -> Tuple[Optional[PandasPdb], Optional[str]]:
    """Load uploaded PDB or PDBQT file with format verification and error handling."""
    if uploaded_file is None:
        return None, None
    
    try:
        # Verify file extension
        file_name = uploaded_file.name
        valid_extensions = ['.pdb', '.pdbqt']
        
        if not any(file_name.endswith(ext) for ext in valid_extensions):
            st.error("Invalid file format. Please upload a .pdb or .pdbqt file.")
            logger.error(f"Invalid file format: {file_name}")
            return None, None
        
        # Read file content into memory
        file_content = uploaded_file.getvalue().decode("utf-8")
        file_lines = StringIO(file_content)
        
        # Load with PandasPdb (works for both .pdb and .pdbqt)
        ppdb = PandasPdb().read_pdb_from_list(file_lines.readlines())
        
        # Validate file content has expected dataframes
        if ppdb.df['ATOM'].empty:
            st.warning("File contains no ATOM records. Please check the file.")
            logger.warning(f"No ATOM records in {file_name}")
            return None, None
            
        log_action("File_Load", f"Loaded {file_name}")
        st.session_state.current_protein = file_name.split('.')[0]
        return ppdb, file_content
    
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        logger.error(f"File loading failed: {str(e)}", exc_info=True)
        return None, None

def fetch_protein_from_pdb(pdb_id: str) -> Tuple[Optional[PandasPdb], Optional[str]]:
    """Fetch protein structure from PDB."""
    try:
        # First fetch metadata
        response = requests.get(f"{PDB_BANK_API}/{pdb_id}")
        if response.status_code != 200:
            st.error(f"Could not fetch PDB ID {pdb_id}. Please check the ID.")
            logger.error(f"PDB fetch failed: {response.status_code}")
            return None, None
        
        # Then fetch structure
        pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        response = requests.get(pdb_url)
        
        if response.status_code != 200:
            st.error(f"Could not download PDB ID {pdb_id}.")
            logger.error(f"PDB download failed: {response.status_code}")
            return None, None
        
        # Process the PDB file
        file_content = response.text
        file_lines = StringIO(file_content)
        ppdb = PandasPdb().read_pdb_from_list(file_lines.readlines())
        
        log_action("PDB_Fetch", f"Fetched {pdb_id} from RCSB")
        st.session_state.current_protein = pdb_id
        return ppdb, file_content
    
    except Exception as e:
        st.error(f"Error fetching PDB: {str(e)}")
        logger.error(f"PDB fetch failed: {str(e)}", exc_info=True)
        return None, None

def visualize_3d_structure(
    file_content: str, 
    file_type: str, 
    style: Dict = None, 
    width: int = 800, 
    height: int = 500, 
    surface: bool = False,
    spin: bool = False,
    highlighted_residues: List = None
) -> None:
    """Enhanced 3D visualization for PDB or PDBQT using py3Dmol."""
    try:
        # Quality settings based on user preference
        quality_settings = {
            "Low": {"width": 600, "height": 400},
            "Medium": {"width": 800, "height": 500},
            "High": {"width": 1000, "height": 600}
        }
        
        if visualization_quality in quality_settings:
            width = quality_settings[visualization_quality]["width"]
            height = quality_settings[visualization_quality]["height"]
        
        view = py3Dmol.view(width=width, height=height)
        view.addModel(file_content, file_type)
        
        # Set style based on parameters or default
        if style:
            view.setStyle(style)
        else:
            view.setStyle({'cartoon': {'color': 'spectrum', 'arrows': True}})
            view.addStyle({'hetflag': True}, {'stick': {'colorscheme': 'greenCarbon', 'radius': 0.3}})
        
        # Add surface if requested
        if surface:
            view.addSurface(py3Dmol.VDW, {'opacity': 0.7, 'color': 'white'})
        
        # Highlight specific residues if provided
        if highlighted_residues:
            for residue in highlighted_residues:
                view.addStyle(
                    {'chain': residue['chain'], 'resi': residue['resnum']},
                    {'stick': {'colorscheme': 'yellowCarbon', 'radius': 0.5}}
                )
        
        # Configure view options
        view.zoomTo()
        if spin:
            view.spin(True)
        
        # Create HTML representation directly with py3Dmol
        view_html = view._make_html()
        st.components.v1.html(view_html, width=width, height=height)
    
    except Exception as e:
        st.error(f"Error visualizing structure: {str(e)}")
        logger.error(f"Visualization failed: {str(e)}", exc_info=True)

def save_to_file(data: Any, filename: str, format: str = "csv") -> None:
    """Save data to file and provide download button with more formats."""
    try:
        if format.lower() == "csv":
            if isinstance(data, pd.DataFrame):
                csv = data.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
            else:
                st.error("Data must be a DataFrame for CSV export")
                return
                
        elif format.lower() == "sdf":
            # Handle RDKit molecule or list of molecules
            temp_file = "temp.sdf"
            if isinstance(data, Chem.Mol):
                writer = Chem.SDWriter(temp_file)
                writer.write(data)
                writer.close()
            elif isinstance(data, list) and all(isinstance(m, Chem.Mol) for m in data):
                writer = Chem.SDWriter(temp_file)
                for mol in data:
                    writer.write(mol)
                writer.close()
            else:
                st.error("Data must be an RDKit molecule or list of molecules for SDF export")
                return
                
            with open(temp_file, "rb") as f:
                bytes_data = f.read()
                b64 = base64.b64encode(bytes_data).decode()
            os.remove(temp_file)
            
        elif format.lower() == "json":
            if isinstance(data, pd.DataFrame):
                json_str = data.to_json(orient="records")
                b64 = base64.b64encode(json_str.encode()).decode()
            elif isinstance(data, dict) or isinstance(data, list):
                json_str = json.dumps(data, indent=2)
                b64 = base64.b64encode(json_str.encode()).decode()
            else:
                st.error("Data must be a DataFrame, dict, or list for JSON export")
                return
        else:
            st.error(f"Unsupported file format: {format}")
            return
            
        href = f'<a href="data:file/{format};base64,{b64}" download="{filename}" class="download-button">Download {filename}</a>'
        st.markdown(href, unsafe_allow_html=True)
        log_action("File_Export", f"Exported {filename}")
        
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        logger.error(f"File save failed: {str(e)}", exc_info=True)

def calculate_molecule_descriptors(mol: Chem.Mol) -> Dict[str, float]:
    """Calculate comprehensive set of molecular descriptors."""
    if mol is None:
        return {}
        
    descriptors = {
        'MW': Descriptors.MolWt(mol),
        'LogP': Crippen.MolLogP(mol),
        'TPSA': Descriptors.TPSA(mol),
        'HBD': Descriptors.NumHDonors(mol),
        'HBA': Descriptors.NumHAcceptors(mol),
        'RotBonds': Descriptors.NumRotatableBonds(mol),
        'AromaticRings': Descriptors.NumAromaticRings(mol),
        'HeavyAtoms': mol.GetNumHeavyAtoms(),
        'Rings': Descriptors.RingCount(mol),
        'Fsp3': Descriptors.FractionCSP3(mol),
        'QED': Descriptors.qed(mol),
        'MolFormula': Chem.rdMolDescriptors.CalcMolFormula(mol),
        'NumHeteroatoms': Descriptors.NumHeteroatoms(mol),
        'LabuteASA': Descriptors.LabuteASA(mol),
        'BalabanJ': Descriptors.BalabanJ(mol) if mol.GetNumHeavyAtoms() > 1 else 0,
        'BertzCT': Descriptors.BertzCT(mol)
    }
    
    # Add Lipinski violations count
    violations = 0
    if descriptors['MW'] > LIPINSKI_RULES['MW']: violations += 1
    if descriptors['LogP'] > LIPINSKI_RULES['LogP']: violations += 1
    if descriptors['HBD'] > LIPINSKI_RULES['HBD']: violations += 1
    if descriptors['HBA'] > LIPINSKI_RULES['HBA']: violations += 1
    descriptors['Lipinski_Violations'] = violations
    
    return descriptors

def generate_molecule_grid(mols: List[Chem.Mol], legends: List[str] = None, molsPerRow: int = 4) -> Image.Image:
    """Generate a grid of molecule images."""
    if not mols:
        return None
        
    if legends is None:
        legends = [f"Mol_{i}" for i in range(len(mols))]
    
    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=molsPerRow,
        subImgSize=(300, 300),
        legends=legends,
        useSVG=False
    )
    return img

def check_druglikeness(mol: Chem.Mol, filters: List[str]) -> Dict[str, Union[bool, str]]:
    """Check molecule against various druglikeness filters."""
    results = {}
    
    # Calculate descriptors
    desc = calculate_molecule_descriptors(mol)
    
    # Lipinski's Rule of 5
    if "Lipinski's Rule of 5" in filters:
        violations = desc['Lipinski_Violations']
        results["Lipinski"] = {
            "pass": violations <= 1,
            "violations": violations,
            "details": f"{violations} violation(s)"
        }
    
    # Veber Rules
    if "Veber Rules" in filters:
        veber_pass = desc['RotBonds'] <= 10 and desc['TPSA'] <= 140
        results["Veber"] = {
            "pass": veber_pass,
            "details": f"RotBonds: {desc['RotBonds']}, TPSA: {desc['TPSA']:.1f}"
        }
    
    # PAINS filter (checks for problematic substructures)
    if "PAINS Filter" in filters:
        pains_filter = FilterCatalog.FilterCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
        pains_pass = not pains_filter.HasMatch(mol)
        matches = []
        
        if not pains_pass:
            matches = [match.GetDescription() for match in pains_filter.GetMatches(mol)]
            
        results["PAINS"] = {
            "pass": pains_pass,
            "details": "No alerts" if pains_pass else f"{len(matches)} alert(s): {', '.join(matches[:3])}"
        }
    
    # Ghose Filter
    if "Ghose Filter" in filters:
        ghose_pass = (
            160 <= desc['MW'] <= 480 and
            -0.4 <= desc['LogP'] <= 5.6 and
            20 <= desc['HeavyAtoms'] <= 70 and
            0 <= desc['LabuteASA'] <= 150
        )
        results["Ghose"] = {
            "pass": ghose_pass,
            "details": f"MW: {desc['MW']:.1f}, LogP: {desc['LogP']:.1f}"
        }
    
    # Overall assessment
    all_tests = [r["pass"] for r in results.values()]
    results["overall_pass"] = all(all_tests) if all_tests else False
    
    return results

def extract_scaffolds(mols: List[Chem.Mol]) -> Dict[str, List[int]]:
    """Extract and count scaffolds from molecules."""
    scaffolds = {}
    for i, mol in enumerate(mols):
        if mol is None:
            continue
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold_smiles = Chem.MolToSmiles(scaffold)
        
        if scaffold_smiles not in scaffolds:
            scaffolds[scaffold_smiles] = []
        scaffolds[scaffold_smiles].append(i)
    
    # Sort by frequency
    return {k: v for k, v in sorted(scaffolds.items(), key=lambda item: len(item[1]), reverse=True)}

# Page Header with Project Info
st.title("Advanced Drug Discovery Pipeline")
st.markdown("""
This application provides tools for in silico protein analysis, pharmacophore modeling, 
and virtual screening for drug discovery research.

**Features:**
- Target characterization with detailed pocket analysis
- Advanced 3D visualization of protein structure
- Pharmacophore generation and optimization
- Compound screening with multiple druglikeness filters
- Structure-activity relationship analysis
""")

# Main Tabbed Interface
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Target Characterization", 
    "Pharmacophore Modeling", 
    "Virtual Screening",
    "SAR Analysis",
    "Results Dashboard"
])

# --- Target Characterization Tab ---
with tab1:
    st.header("Target Characterization")
    st.write("Analyze protein structure and identify binding pockets.")

    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.subheader("Load Protein Structure")
        
        load_option = st.radio(
            "Load from:",
            ["Upload File", "Fetch from PDB"],
            horizontal=True
        )
        
        if load_option == "Upload File":
            uploaded_file = st.file_uploader("Upload Protein Structure (PDB or PDBQT format)", type=["pdb", "pdbqt"])
            if uploaded_file:
                ppdb, file_content = load_pdb_or_pdbqt(uploaded_file)
                if ppdb and file_content:
                    file_type = "pdb" if uploaded_file.name.endswith('.pdb') else "pdbqt"
        else:
            pdb_id = st.text_input("Enter PDB ID (e.g., 1AQ1)", max_chars=4)
            fetch_button = st.button("Fetch Structure")
            if fetch_button and pdb_id:
                ppdb, file_content = fetch_protein_from_pdb(pdb_id)
                if ppdb and file_content:
                    file_type = "pdb"
        
        # Show protein info if loaded
        if 'ppdb' in locals() and ppdb:
            st.subheader("Protein Information")
            atom_count = len(ppdb.df['ATOM'])
            hetatm_count = len(ppdb.df['HETATM'])
            chains = ppdb.df['ATOM']['chain_id'].unique()
            
            st.info(f"""
            **Structure Summary:**
            - Atoms: {atom_count}
            - HET atoms: {hetatm_count}
            - Chains: {', '.join(chains)}
            """)
            
            # Show chains and allow selection
            selected_chains = st.multiselect(
                "Analyze specific chains:",
                chains,
                default=chains[0] if len(chains) > 0 else None
            )
    
    with col2:
        # Visualize Structure if loaded
        if 'ppdb' in locals() and ppdb and 'file_content' in locals() and file_content:
            st.subheader("Protein Structure")
            
            # Visualization options
            viz_options = st.expander("Visualization Options", expanded=False)
            with viz_options:
                viz_style = st.selectbox(
                    "Display Style",
                    ["Cartoon", "Cartoon+Ligands", "Sticks", "Spheres", "Surface"]
                )
                color_scheme = st.selectbox(
                    "Color Scheme",
                    ["Spectrum", "Chain", "Secondary Structure", "B-factor"]
                )
                show_surface = st.checkbox("Show Surface", value=False)
                
                # Map selections to style dictionary
                style_dict = {}
                if viz_style == "Cartoon":
                    style_dict = {'cartoon': {'colorscheme': color_scheme.lower(), 'arrows': True}}
                elif viz_style == "Cartoon+Ligands":
                    style_dict = {'cartoon': {'colorscheme': color_scheme.lower()}}
                    # Will add ligand style separately
                elif viz_style == "Sticks":
                    style_dict = {'stick': {}}
                elif viz_style == "Spheres":
                    style_dict = {'sphere': {}}
                elif viz_style == "Surface":
                    style_dict = {'cartoon': {'opacity': 0.5}}
                    show_surface = True
            
            # Visualize with selected options
            visualize_3d_structure(
                file_content, 
                file_type, 
                style=style_dict,
                surface=show_surface,
                spin=False
            )
            
            # Add the option to save the structure
            st.download_button(
                "Download Structure",
                file_content,
                file_name=f"{st.session_state.current_protein}.{file_type}",
                mime=f"chemical/{file_type}"
            )
    
    # Pocket Analysis Section
    if 'ppdb' in locals() and ppdb:
        st.markdown("---")
        st.header("Binding Pocket Analysis")
        
        with st.expander("Pocket Detection Parameters", expanded=False):
            distance_cutoff = st.slider(
                "Distance Cutoff (Å)",
                min_value=3.0,
                max_value=10.0,
                value=5.0,
                step=0.5
            )
            
            min_pocket_volume = st.slider(
                "Minimum Pocket Volume (Å³)",
                min_value=50,
                max_value=500,
                value=100,
                step=50
            )
            
            binding_site_residue = st.text_input(
                "Reference Binding Site Residue (format: ALA_101_A)",
                help="Optional: Specify a known binding site residue to identify nearby pockets"
            )
        
        # Analyze pockets
        @st.cache_data
        def analyze_pockets(_ppdb: PandasPdb, distance_cutoff: float = 5.0, 
                           min_volume: float = 100, reference_residue: str = None,
                           chains: List[str] = None) -> List[Dict]:
            """
            Analyze binding pockets with enhanced detection and characterization.
            """
            try:
                # Filter by chains if specified
                if chains and len(chains) > 0:
                    atom_data = _ppdb.df['ATOM'][_ppdb.df['ATOM']['chain_id'].isin(chains)]
                    hetatm_data = _ppdb.df['HETATM'][_ppdb.df['HETATM']['chain_id'].isin(chains)]
                else:
                    atom_data = _ppdb.df['ATOM']
                    hetatm_data = _ppdb.df['HETATM']
                
                pockets = []
                
                # Method 1: Detect pockets based on HET groups (ligands, cofactors)
                for (residue, chain), group in hetatm_data.groupby(['residue_name', 'chain_id']):
                    if residue == 'HOH':  # Skip water molecules
                        continue
                        
                    # Calculate pocket center and coordinates
                    coords = group[['x_coord', 'y_coord', 'z_coord']].values
                    center = coords.mean(axis=0)
                    min_coords = coords.min(axis=0)
                    max_coords = coords.max(axis=0)
                    
                    # Calculate volume (approximation)
                    dimensions = max_coords - min_coords
                    volume = np.prod(dimensions)
                    
                    if volume < min_volume:
                        continue
                    
                    # Find nearby protein residues
                    distances = np.sqrt(((atom_data[['x_coord', 'y_coord', 'z_coord']].values - center)**2).sum(axis=1))
                    nearby_atoms = atom_data[distances <= distance_cutoff]
                    
                    # Group by residue
                    nearby_residues = set()
                    for _, res_atom in nearby_atoms.iterrows():
                        res_id = (res_atom['residue_name'], res_atom['residue_number'], res_atom['chain_id'])
                        nearby_residues.add(res_id)
                    
                    # Calculate properties
                    hydrophobic_residues = ['ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP', 'PRO', 'TYR']
                    polar_residues = ['SER', 'THR', 'CYS', 'ASN', 'GLN', 'HIS']
                    charged_residues = ['LYS', 'ARG', 'ASP', 'GLU']
                    
                    residue_composition = {
                        'hydrophobic': sum(1 for res in nearby_residues if res[0] in hydrophobic_residues),
                        'polar': sum(1 for res in nearby_residues if res[0] in polar_residues),
                        'charged': sum(1 for res in nearby_residues if res[0] in charged_residues),
                        'total': len(nearby_residues)
                    }
                    
                    # Calculate hydrophobicity score (% of hydrophobic residues)
                    hydrophobicity = (residue_composition['hydrophobic'] / residue_composition['total']) * 100 if residue_composition['total'] > 0 else 0
                    
                    pocket = {
                        'id': f"P_{residue}_{chain}",
                        'residue': residue,
                        'chain': chain,
                        'center': center.tolist(),
                        'volume': volume,
                        'nearby_residues': [{'name': r[0], 'number': r[1], 'chain': r[2]} for r in nearby_residues],
                        'residue_count': len(nearby_residues),
                        'hydrophobicity': hydrophobicity,
                        'residue_composition': residue_composition,
                        'method': 'ligand-based'
                    }
                    pockets.append(pocket)
                
                # Method 2: Detect pockets based on protein surface geometry
                if pocket_detection_method == "Geometric" and len(pockets) < 3:
                    # Simplified geometric approach - find cavities by analyzing atom density
                    # Grid the protein space
                    min_coords = atom_data[['x_coord', 'y_coord', 'z_coord']].min().values
                    max_coords = atom_data[['x_coord', 'y_coord', 'z_coord']].max().values
                    
                    grid_step = 2.0  # Å
                    grid_margin = 4.0  # Å
                    
                    grid_min = min_coords - grid_margin
                    grid_max = max_coords + grid_margin
                    
                    # Create grid points
                    x_points = np.arange(grid_min[0], grid_max[0], grid_step)
                    y_points = np.arange(grid_min[1], grid_max[1], grid_step)
                    z_points = np.arange(grid_min[2], grid_max[2], grid_step)
                    
                    # For each atom, mark grid points within vdW radius
                    grid_occupancy = np.zeros((len(x_points), len(y_points), len(z_points)))
                    atom_coords = atom_data[['x_coord', 'y_coord', 'z_coord']].values
                    
                    # Sample a subset of grid points for efficiency
                    sample_size = min(5000, len(x_points) * len(y_points) * len(z_points))
                    grid_indices = []
                    
                    for _ in range(sample_size):
                        i = np.random.randint(0, len(x_points))
                        j = np.random.randint(0, len(y_points))
                        k = np.random.randint(0, len(z_points))
                        grid_indices.append((i, j, k))
                    
                    # Mark points close to protein surface but not inside
                    pocket_candidates = []
                    
                    for i, j, k in grid_indices:
                        grid_point = np.array([x_points[i], y_points[j], z_points[k]])
                        
                        # Calculate distances to all atoms
                        distances = np.sqrt(((atom_coords - grid_point)**2).sum(axis=1))
                        
                        # Check if point is near surface (not too close, not too far)
                        min_dist = distances.min()
                        if 3.0 < min_dist < 8.0:  # Near surface but not inside
                            # Count nearby atoms (should be several for a pocket)
                            nearby_count = (distances < 10.0).sum()
                            if nearby_count > 15:  # Enough surrounding atoms for a pocket
                                pocket_candidates.append((grid_point, nearby_count))
                    
                    # Cluster candidate points to identify distinct pockets
                    if pocket_candidates:
                        from sklearn.cluster import DBSCAN
                        points = np.array([p[0] for p in pocket_candidates])
                        scores = np.array([p[1] for p in pocket_candidates])
                        
                        # Cluster points
                        clustering = DBSCAN(eps=5.0, min_samples=5).fit(points)
                        labels = clustering.labels_
                        
                        # Process clusters
                        for label in set(labels):
                            if label == -1:  # Noise points
                                continue
                                
                            cluster_points = points[labels == label]
                            cluster_scores = scores[labels == label]
                            
                            # Calculate cluster center (weighted by scores)
                            center = np.average(cluster_points, axis=0, weights=cluster_scores)
                            
                            # Estimate volume from point density
                            hull_points = cluster_points
                            if len(hull_points) >= 4:  # Need at least 4 points for convex hull
                                try:
                                    from scipy.spatial import ConvexHull
                                    hull = ConvexHull(hull_points)
                                    volume = hull.volume
                                except Exception:
                                    # Fallback if hull calculation fails
                                    volume = len(cluster_points) * (grid_step**3)
                            else:
                                volume = len(cluster_points) * (grid_step**3)
                            
                            if volume < min_volume:
                                continue
                            
                            # Find nearby residues
                            distances = np.sqrt(((atom_data[['x_coord', 'y_coord', 'z_coord']].values - center)**2).sum(axis=1))
                            nearby_atoms = atom_data[distances <= distance_cutoff]
                            
                            # Group by residue
                            nearby_residues = set()
                            for _, res_atom in nearby_atoms.iterrows():
                                res_id = (res_atom['residue_name'], res_atom['residue_number'], res_atom['chain_id'])
                                nearby_residues.add(res_id)
                            
                            # Skip if too few residues
                            if len(nearby_residues) < 5:
                                continue
                            
                            # Calculate properties (same as above)
                            hydrophobic_residues = ['ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP', 'PRO', 'TYR']
                            polar_residues = ['SER', 'THR', 'CYS', 'ASN', 'GLN', 'HIS']
                            charged_residues = ['LYS', 'ARG', 'ASP', 'GLU']
                            
                            residue_composition = {
                                'hydrophobic': sum(1 for res in nearby_residues if res[0] in hydrophobic_residues),
                                'polar': sum(1 for res in nearby_residues if res[0] in polar_residues),
                                'charged': sum(1 for res in nearby_residues if res[0] in charged_residues),
                                'total': len(nearby_residues)
                            }
                            
                            hydrophobicity = (residue_composition['hydrophobic'] / residue_composition['total']) * 100 if residue_composition['total'] > 0 else 0
                            
                            pocket = {
                                'id': f"G_{label}",
                                'residue': f"GP_{label}",  # Geometric Pocket
                                'chain': "X",  # Not specific to a chain
                                'center': center.tolist(),
                                'volume': volume,
                                'nearby_residues': [{'name': r[0], 'number': r[1], 'chain': r[2]} for r in nearby_residues],
                                'residue_count': len(nearby_residues),
                                'hydrophobicity': hydrophobicity,
                                'residue_composition': residue_composition,
                                'method': 'geometric'
                            }
                            pockets.append(pocket)
                
                # Method 3: Reference-based pocket detection
                if reference_residue:
                    try:
                        # Parse reference residue format
                        ref_parts = reference_residue.split('_')
                        if len(ref_parts) == 3:
                            ref_name, ref_num, ref_chain = ref_parts
                            ref_num = int(ref_num)
                            
                            # Find reference residue atoms
                            ref_atoms = atom_data[
                                (atom_data['residue_name'] == ref_name) & 
                                (atom_data['residue_number'] == ref_num) & 
                                (atom_data['chain_id'] == ref_chain)
                            ]
                            
                            if not ref_atoms.empty:
                                # Calculate center of reference residue
                                ref_center = ref_atoms[['x_coord', 'y_coord', 'z_coord']].mean().values
                                
                                # Find nearby residues
                                distances = np.sqrt(((atom_data[['x_coord', 'y_coord', 'z_coord']].values - ref_center)**2).sum(axis=1))
                                nearby_atoms = atom_data[distances <= distance_cutoff * 2]  # Wider radius
                                
                                # Group by residue
                                nearby_residues = set()
                                for _, res_atom in nearby_atoms.iterrows():
                                    res_id = (res_atom['residue_name'], res_atom['residue_number'], res_atom['chain_id'])
                                    nearby_residues.add(res_id)
                                
                                # Calculate properties
                                hydrophobic_residues = ['ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP', 'PRO', 'TYR']
                                polar_residues = ['SER', 'THR', 'CYS', 'ASN', 'GLN', 'HIS']
                                charged_residues = ['LYS', 'ARG', 'ASP', 'GLU']
                                
                                residue_composition = {
                                    'hydrophobic': sum(1 for res in nearby_residues if res[0] in hydrophobic_residues),
                                    'polar': sum(1 for res in nearby_residues if res[0] in polar_residues),
                                    'charged': sum(1 for res in nearby_residues if res[0] in charged_residues),
                                    'total': len(nearby_residues)
                                }
                                
                                hydrophobicity = (residue_composition['hydrophobic'] / residue_composition['total']) * 100 if residue_composition['total'] > 0 else 0
                                
                                # Calculate approximate volume from residue count
                                volume = len(nearby_residues) * 100  # Very rough approximation
                                
                                pocket = {
                                    'id': f"R_{ref_name}_{ref_num}_{ref_chain}",
                                    'residue': f"{ref_name}",
                                    'chain': ref_chain,
                                    'center': ref_center.tolist(),
                                    'volume': volume,
                                    'nearby_residues': [{'name': r[0], 'number': r[1], 'chain': r[2]} for r in nearby_residues],
                                    'residue_count': len(nearby_residues),
                                    'hydrophobicity': hydrophobicity,
                                    'residue_composition': residue_composition,
                                    'method': 'reference-based'
                                }
                                pockets.append(pocket)
                    except Exception as e:
                        logger.error(f"Error in reference-based pocket detection: {str(e)}")
                
                # Sort pockets by volume
                pockets = sorted(pockets, key=lambda x: x['volume'], reverse=True)
                
                return pockets
            except Exception as e:
                logger.error(f"Pocket analysis failed: {str(e)}", exc_info=True)
                return []
                
        if st.button("Analyze Binding Pockets"):
            with st.spinner("Analyzing binding pockets... This may take a moment."):
                reference_res = None
                if binding_site_residue:
                    reference_res = binding_site_residue
                
                pockets = analyze_pockets(
                    ppdb, 
                    distance_cutoff=distance_cutoff,
                    min_volume=min_pocket_volume,
                    reference_residue=reference_res,
                    chains=selected_chains if 'selected_chains' in locals() else None
                )
                
                if pockets:
                    st.session_state.analyzed_proteins[st.session_state.current_protein] = {
                        'pockets': pockets,
                        'file_content': file_content,
                        'file_type': file_type
                    }
                    
                    st.success(f"Found {len(pockets)} potential binding pockets!")
                else:
                    st.warning("No significant binding pockets were detected. Try adjusting the parameters.")
        
        # Display pocket results if available
        if st.session_state.current_protein in st.session_state.analyzed_proteins:
            protein_data = st.session_state.analyzed_proteins[st.session_state.current_protein]
            pockets = protein_data['pockets']
            
            if pockets:
                # Create Pocket Table
                pocket_table = []
                for i, pocket in enumerate(pockets):
                    pocket_table.append({
                        'ID': pocket['id'],
                        'Method': pocket['method'],
                        'Volume (Å³)': f"{pocket['volume']:.1f}",
                        'Residues': pocket['residue_count'],
                        'Hydrophobicity (%)': f"{pocket['hydrophobicity']:.1f}",
                        'Polar (%)': f"{(pocket['residue_composition']['polar'] / pocket['residue_composition']['total'] * 100):.1f}" if pocket['residue_composition']['total'] > 0 else "0.0",
                        'Charged (%)': f"{(pocket['residue_composition']['charged'] / pocket['residue_composition']['total'] * 100):.1f}" if pocket['residue_composition']['total'] > 0 else "0.0"
                    })
                
                # Display table
                st.markdown("### Identified Binding Pockets")
                pocket_df = pd.DataFrame(pocket_table)
                st.dataframe(pocket_df, use_container_width=True)
                
                # Save to CSV option
                save_to_file(pocket_df, f"{st.session_state.current_protein}_pockets.{export_format.lower()}")
                
                # Visualize Pockets
                st.subheader("Pocket Visualization")
                
                # Select pocket to highlight
                selected_pocket_idx = st.selectbox(
                    "Select pocket to highlight", 
                    range(len(pockets)),
                    format_func=lambda i: f"{pockets[i]['id']} - Volume: {pockets[i]['volume']:.1f} Å³"
                )
                
                selected_pocket = pockets[selected_pocket_idx]
                
                # Visualize protein with highlighted pocket
                view = py3Dmol.view(width=800, height=500)
                view.addModel(protein_data['file_content'], protein_data['file_type'])
                view.setStyle({'cartoon': {'color': 'lightgray', 'opacity': 0.7}})
                
                # Add sphere for pocket center
                center = selected_pocket['center']
                view.addSphere({
                    'center': {'x': center[0], 'y': center[1], 'z': center[2]},
                    'radius': 2.0, 
                    'color': 'magenta', 
                    'opacity': 0.7
                })
                
                # Highlight pocket residues
                for residue in selected_pocket['nearby_residues']:
                    view.addStyle(
                        {'chain': residue['chain'], 'resi': residue['number']},
                        {'stick': {'color': 'green', 'radius': 0.3}}
                    )
                
                view.zoomTo({'chain': selected_pocket['chain']})
                view.zoom(0.8)
                
                # Create HTML representation directly
                view_html = view._make_html()
                st.components.v1.html(view_html, width=800, height=500)
                
                # Plot properties
                st.subheader("Pocket Properties")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Volume comparison
                    volumes = [p['volume'] for p in pockets]
                    fig = px.bar(
                        x=[p['id'] for p in pockets],
                        y=volumes,
                        labels={'x': 'Pocket ID', 'y': 'Volume (Å³)'},
                        title='Pocket Volumes',
                        color=volumes,
                        color_continuous_scale=selected_theme
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Residue composition
                    fig = px.bar(
                        x=[p['id'] for p in pockets],
                        y=[p['residue_count'] for p in pockets],
                        labels={'x': 'Pocket ID', 'y': 'Number of Residues'},
                        title='Pocket Residue Count',
                        color=[p['hydrophobicity'] for p in pockets],
                        color_continuous_scale='Viridis',
                        color_continuous_midpoint=50
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detailed residue composition
                st.subheader("Residue Composition")
                
                selected_pocket = pockets[selected_pocket_idx]
                
                # Create pie chart of amino acid types
                labels = ['Hydrophobic', 'Polar', 'Charged']
                values = [
                    selected_pocket['residue_composition']['hydrophobic'],
                    selected_pocket['residue_composition']['polar'],
                    selected_pocket['residue_composition']['charged']
                ]
                
                fig = px.pie(
                    values=values, 
                    names=labels,
                    title=f"Residue Composition of {selected_pocket['id']}",
                    color_discrete_sequence=px.colors.qualitative.Safe
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Residue table
                residue_counts = {}
                for residue in selected_pocket['nearby_residues']:
                    if residue['name'] not in residue_counts:
                        residue_counts[residue['name']] = 0
                    residue_counts[residue['name']] += 1
                
                residue_df = pd.DataFrame([
                    {'Residue': res, 'Count': count, 'Percentage': (count / len(selected_pocket['nearby_residues'])) * 100}
                    for res, count in residue_counts.items()
                ])
                
                # Sort by count
                residue_df = residue_df.sort_values('Count', ascending=False)
                
                st.dataframe(residue_df, use_container_width=True)
                
                # Download option for residue data
                save_to_file(residue_df, f"{st.session_state.current_protein}_pocket_{selected_pocket['id']}_residues.{export_format.lower()}")

# --- Pharmacophore Modeling Tab ---
with tab2:
    st.header("Pharmacophore Modeling")
    st.write("Generate a pharmacophore model based on pocket features and known ligands.")
    
    # Check if pockets have been analyzed
    if not any(st.session_state.analyzed_proteins):
        st.warning("Please analyze protein pockets in the Target Characterization tab first.")
    else:
        col1, col2 = st.columns([2, 3])
        
        with col1:
            # Select protein and pocket
            available_proteins = list(st.session_state.analyzed_proteins.keys())
            selected_protein = st.selectbox("Select Protein", available_proteins)
            
            if selected_protein:
                protein_data = st.session_state.analyzed_proteins[selected_protein]
                pockets = protein_data['pockets']
                
                pocket_idx = st.selectbox(
                    "Select Pocket", 
                    range(len(pockets)),
                    format_func=lambda i: f"{pockets[i]['id']} - Volume: {pockets[i]['volume']:.1f} Å³"
                )
                
                selected_pocket = pockets[pocket_idx]
                
                st.info(f"""
                **Selected Pocket:**
                - ID: {selected_pocket['id']}
                - Volume: {selected_pocket['volume']:.1f} Å³
                - Residues: {selected_pocket['residue_count']}
                - Hydrophobicity: {selected_pocket['hydrophobicity']:.1f}%
                """)
                
                # Pharmacophore Generation Parameters
                st.subheader("Pharmacophore Parameters")
                
                # Method selection
                pharma_method = st.radio(
                    "Generation Method",
                    ["Structure-based", "Ligand-based", "Hybrid"],
                    index=0
                )
                
                # Feature settings
                feature_settings = st.expander("Feature Settings", expanded=True)
                with feature_settings:
                    include_hydrophobic = st.checkbox("Include Hydrophobic Features", value=True)
                    include_hbond = st.checkbox("Include H-Bond Features", value=True)
                    include_ionic = st.checkbox("Include Ionic Features", value=True)
                    include_aromatic = st.checkbox("Include Aromatic Features", value=True)
                    
                    max_features = st.slider(
                        "Maximum Features",
                        min_value=3,
                        max_value=10,
                        value=5
                    )
                    
                    feature_tolerance = st.slider(
                        "Feature Tolerance (Å)",
                        min_value=1.0,
                        max_value=3.0,
                        value=1.5,
                        step=0.1
                    )
                
                # Optional: Upload reference ligand
                ref_ligand_file = st.file_uploader(
                    "Upload Reference Ligand (SDF or MOL format)",
                    type=["sdf", "mol"]
                )
                
                if ref_ligand_file and pharma_method in ["Ligand-based", "Hybrid"]:
                    try:
                        # Read ligand file
                        ligand_bytes = ref_ligand_file.getvalue()
                        
                        # Save temporarily to use RDKit
                        temp_file = "temp_ligand.sdf"
                        with open(temp_file, "wb") as f:
                            f.write(ligand_bytes)
                        
                        # Load molecule
                        ref_mol = Chem.SDMolSupplier(temp_file)[0]
                        
                        if ref_mol:
                            # Get basic properties
                            mol_formula = rdMolDescriptors.CalcMolFormula(ref_mol)
                            mol_weight = Descriptors.MolWt(ref_mol)
                            
                            st.success(f"Loaded reference ligand: {mol_formula} (MW: {mol_weight:.1f})")
                            
                            # Show 2D structure
                            mol_img = Draw.MolToImage(ref_mol, size=(300, 300))
                            st.image(mol_img, caption="Reference Ligand Structure")
                        else:
                            st.error("Could not parse the uploaded ligand file.")
                    except Exception as e:
                        st.error(f"Error loading ligand: {str(e)}")
        
        with col2:
            # Pharmacophore Generation
            class PharmacophoreGenerator:
                def __init__(self, method="Structure-based", pocket=None, ref_ligand=None):
                    self.method = method
                    self.pocket = pocket
                    self.ref_ligand = ref_ligand
                    self.features = {
                        'hydrophobic': {'include': include_hydrophobic, 'smarts': ['[#6D3]', '[#6D4]']},
                        'hbond_donor': {'include': include_hbond, 'smarts': ['[#7!H0]', '[#8!H0]']},
                        'hbond_acceptor': {'include': include_hbond, 'smarts': ['[#7]', '[#8]']},
                        'positive': {'include': include_ionic, 'smarts': ['[+]', '[#7H2]']},
                        'negative': {'include': include_ionic, 'smarts': ['[-]', '[#8D1]']},
                        'aromatic': {'include': include_aromatic, 'smarts': ['a5', 'a6']}
                    }
                    self.feature_tolerance = feature_tolerance
                    self.max_features = max_features
                
                def generate_features(self) -> List[Dict]:
                    features = []
                    
                    if self.method == "Structure-based" and self.pocket:
                        # Generate features from pocket properties
                        
                        # Get pocket center
                        pocket_center = self.pocket['center']
                        
                        # Add hydrophobic feature if pocket is hydrophobic
                        if self.features['hydrophobic']['include'] and self.pocket['hydrophobicity'] > 40:
                            features.append({
                                'type': 'hydrophobic',
                                'center': pocket_center,
                                'radius': self.feature_tolerance,
                                'smarts': self.features['hydrophobic']['smarts'][0]
                            })
                        
                        # Process pocket residues for other features
                        hbond_donors = 0
                        hbond_acceptors = 0
                        positive_charges = 0
                        negative_charges = 0
                        aromatic_groups = 0
                        
                        # Analyze residues
                        residue_types = {
                            'hbond_donor': ['SER', 'THR', 'TYR', 'ASN', 'GLN', 'LYS', 'ARG', 'TRP'],
                            'hbond_acceptor': ['ASP', 'GLU', 'ASN', 'GLN', 'HIS', 'SER', 'THR', 'TYR'],
                            'positive': ['LYS', 'ARG', 'HIS'],
                            'negative': ['ASP', 'GLU'],
                            'aromatic': ['PHE', 'TYR', 'TRP', 'HIS']
                        }
                        
                        for residue in self.pocket['nearby_residues']:
                            res_name = residue['name']
                            
                            if res_name in residue_types['hbond_donor']:
                                hbond_donors += 1
                            if res_name in residue_types['hbond_acceptor']:
                                hbond_acceptors += 1
                            if res_name in residue_types['positive']:
                                positive_charges += 1
                            if res_name in residue_types['negative']:
                                negative_charges += 1
                            if res_name in residue_types['aromatic']:
                                aromatic_groups += 1
                        
                        # Normalize by total residues
                        total_residues = len(self.pocket['nearby_residues'])
                        
                        # Add features based on residue composition
                        offsets = {
                            'hbond_donor': np.array([2.0, 0.0, 0.0]),
                            'hbond_acceptor': np.array([0.0, 2.0, 0.0]),
                            'positive': np.array([2.0, 2.0, 0.0]),
                            'negative': np.array([-2.0, -2.0, 0.0]),
                            'aromatic': np.array([0.0, 0.0, 2.0])
                        }
                        
                        # H-bond donor
                        if self.features['hbond_donor']['include']:
                            if total_residues == 0:
                                # Handle the case where there are no residues, e.g., set a default value,
                                # raise a custom error, or log a warning.
                                print("Warning: No residues found.")
                            elif hbond_donors / total_residues > 0.1:
                                # Proceed with your logic

                        # if self.features['hbond_donor']['include'] and hbond_donors / total_residues > 0.1:
                                donor_center = np.array(pocket_center) + offsets['hbond_donor']
                                features.append({
                                    'type': 'hbond_donor',
                                    'center': donor_center.tolist(),
                                    'radius': self.feature_tolerance,
                                    'smarts': self.features['hbond_donor']['smarts'][0]
                                })
                        
                        # H-bond acceptor
                        if self.features['hbond_acceptor']['include'] and hbond_acceptors / total_residues > 0.1:
                            acceptor_center = np.array(pocket_center) + offsets['hbond_acceptor']
                            features.append({
                                'type': 'hbond_acceptor',
                                'center': acceptor_center.tolist(),
                                'radius': self.feature_tolerance,
                                'smarts': self.features['hbond_acceptor']['smarts'][0]
                            })
                        
                        # Positive charge
                        if self.features['positive']['include'] and positive_charges / total_residues > 0.05:
                            pos_center = np.array(pocket_center) + offsets['positive']
                            features.append({
                                'type': 'positive',
                                'center': pos_center.tolist(),
                                'radius': self.feature_tolerance,
                                'smarts': self.features['positive']['smarts'][0]
                            })
                        
                        # Negative charge
                        if self.features['negative']['include'] and negative_charges / total_residues > 0.05:
                            neg_center = np.array(pocket_center) + offsets['negative']
                            features.append({
                                'type': 'negative',
                                'center': neg_center.tolist(),
                                'radius': self.feature_tolerance,
                                'smarts': self.features['negative']['smarts'][0]
                            })
                        
                        # Aromatic
                        if self.features['aromatic']['include'] and aromatic_groups / total_residues > 0.1:
                            arom_center = np.array(pocket_center) + offsets['aromatic']
                            features.append({
                                'type': 'aromatic',
                                'center': arom_center.tolist(),
                                'radius': self.feature_tolerance,
                                'smarts': self.features['aromatic']['smarts'][0]
                            })
                    
                    # For ligand-based or hybrid approaches
                    elif (self.method in ["Ligand-based", "Hybrid"]) and self.ref_ligand:
                        # Extract features from reference ligand
                        
                        # Calculate 3D coordinates if not already done
                        ref_mol = Chem.AddHs(self.ref_ligand)
                        AllChem.EmbedMolecule(ref_mol, randomSeed=42)
                        AllChem.UFFOptimizeMolecule(ref_mol)
                        
                        # Define SMARTS patterns for features
                        feature_patterns = {
                            'hydrophobic': ['[#6D3]', '[#6D4]', '[#6;X3,X4]'],
                            'hbond_donor': ['[#7!H0]', '[#8!H0]', '[N,O;!H0]'],
                            'hbond_acceptor': ['[#7]', '[#8]', '[N,O]'],
                            'positive': ['[+]', '[#7H2]', '[NH2]'],
                            'negative': ['[-]', '[#8D1]', '[COO-]'],
                            'aromatic': ['a5', 'a6', 'c1ccccc1']
                        }
                        
                        # Identify features by matching SMARTS patterns
                        for feature_type, patterns in feature_patterns.items():
                            if not self.features[feature_type]['include']:
                                continue
                                
                            feature_atoms = []
                            for pattern in patterns:
                                patt = Chem.MolFromSmarts(pattern)
                                if patt:
                                    matches = ref_mol.GetSubstructMatches(patt)
                                    for match in matches:
                                        feature_atoms.extend(match)
                            
                            # Get coordinates for matched atoms
                            if feature_atoms:
                                conf = ref_mol.GetConformer()
                                coords = [conf.GetAtomPosition(atom_idx) for atom_idx in feature_atoms]
                                
                                # Calculate centroid
                                centroid = np.mean([[pos.x, pos.y, pos.z] for pos in coords], axis=0)
                                
                                features.append({
                                    'type': feature_type,
                                    'center': centroid.tolist(),
                                    'radius': self.feature_tolerance,
                                    'smarts': self.features[feature_type]['smarts'][0]
                                })
                    
                    # For hybrid approach, combine both sets of features
                    elif self.method == "Hybrid" and self.pocket and self.ref_ligand:
                        # Get both sets of features
                        self.method = "Structure-based"
                        structure_features = self.generate_features()
                        
                        self.method = "Ligand-based"
                        ligand_features = self.generate_features()
                        
                        # Combine features, avoiding duplicates (by type and proximity)
                        for feat in structure_features:
                            features.append(feat)
                        
                        for lig_feat in ligand_features:
                            # Check if this feature is close to any existing one of the same type
                            duplicate = False
                            for struct_feat in features:
                                if struct_feat['type'] == lig_feat['type']:
                                    # Calculate distance between centers
                                    dist = np.linalg.norm(np.array(struct_feat['center']) - np.array(lig_feat['center']))
                                    if dist < 3.0:  # If centers are within 3Å, consider them duplicates
                                        duplicate = True
                                        break
                            
                            if not duplicate:
                                features.append(lig_feat)
                    
                    # Limit number of features if needed
                    if len(features) > self.max_features:
                        # Sort by feature importance (hydrophobic < ionic < hydrogen bond < aromatic)
                        feature_priority = {
                            'hydrophobic': 1,
                            'positive': 2,
                            'negative': 2,
                            'hbond_donor': 3,
                            'hbond_acceptor': 3,
                            'aromatic': 4
                        }
                        
                        features.sort(key=lambda x: feature_priority[x['type']], reverse=True)
                        features = features[:self.max_features]
                    
                    return features
                
                def create_pharmacophore_model(self, features: List[Dict]) -> Tuple[Dict, Chem.Mol]:
                    """Create a pharmacophore model from the features."""
                    # Create model info
                    model = {
                        'features': features,
                        'method': self.method,
                        'max_features': self.max_features,
                        'tolerance': self.feature_tolerance
                    }
                    
                    # Create a representative molecule for visualization
                    # Start with a central carbon atom
                    mol = Chem.MolFromSmiles('C')
                    mol = Chem.AddHs(mol)
                    
                    # Generate 3D coordinates
                    AllChem.EmbedMolecule(mol, randomSeed=42)
                    
                    # Create feature molecules
                    feature_mols = []
                    for i, feature in enumerate(features):
                        # Create appropriate atom for feature type
                        feature_mol = None
                        if feature['type'] == 'hydrophobic':
                            feature_mol = Chem.MolFromSmiles('C')
                        elif feature['type'] == 'hbond_donor':
                            feature_mol = Chem.MolFromSmiles('O')
                        elif feature['type'] == 'hbond_acceptor':
                            feature_mol = Chem.MolFromSmiles('N')
                        elif feature['type'] == 'positive':
                            feature_mol = Chem.MolFromSmiles('[NH4+]')
                        elif feature['type'] == 'negative':
                            feature_mol = Chem.MolFromSmiles('[O-]')
                        elif feature['type'] == 'aromatic':
                            feature_mol = Chem.MolFromSmiles('c1ccccc1')
                        
                        if feature_mol:
                            feature_mol = Chem.AddHs(feature_mol)
                            AllChem.EmbedMolecule(feature_mol, randomSeed=42)
                            
                            # Position at feature center
                            conf = feature_mol.GetConformer()
                            center = feature['center']
                            
                            # Adjust coordinates to position feature
                            for atom_idx in range(feature_mol.GetNumAtoms()):
                                conf.SetAtomPosition(atom_idx, 
                                                    Point3D(
                                                        conf.GetAtomPosition(atom_idx).x + center[0],
                                                        conf.GetAtomPosition(atom_idx).y + center[1],
                                                        conf.GetAtomPosition(atom_idx).z + center[2]
                                                    ))
                            
                            feature_mols.append(feature_mol)
                    
                    # Combine all molecules
                    if feature_mols:
                        combined_mol = feature_mols[0]
                        for fm in feature_mols[1:]:
                            combined_mol = Chem.CombineMols(combined_mol, fm)
                        
                        return model, combined_mol
                    
                    return model, mol
                    
                def visualize_pharmacophore(self, features: List[Dict], pocket=None):
                    """Create a 3D visualization of the pharmacophore model."""
                    # Colors for different feature types
                    feature_colors = {
                        'hydrophobic': 'yellow',
                        'hbond_donor': 'blue',
                        'hbond_acceptor': 'red',
                        'positive': 'green',
                        'negative': 'orange',
                        'aromatic': 'purple'
                    }
                    
                    # Create py3Dmol view
                    view = py3Dmol.view(width=800, height=500)
                    
                    # Add protein pocket if available
                    if pocket and selected_protein in st.session_state.analyzed_proteins:
                        protein_data = st.session_state.analyzed_proteins[selected_protein]
                        view.addModel(protein_data['file_content'], protein_data['file_type'])
                        view.setStyle({'cartoon': {'color': 'lightgray', 'opacity': 0.5}})
                        
                        # Highlight pocket residues
                        for residue in pocket['nearby_residues']:
                            view.addStyle(
                                {'chain': residue['chain'], 'resi': residue['number']},
                                {'stick': {'color': 'gray', 'opacity': 0.7}}
                            )
                    
                    # Add each feature as a sphere
                    for i, feature in enumerate(features):
                        center = feature['center']
                        color = feature_colors.get(feature['type'], 'gray')
                        
                        view.addSphere({
                            'center': {'x': center[0], 'y': center[1], 'z': center[2]},
                            'radius': feature['radius'],
                            'color': color,
                            'opacity': 0.7
                        })
                        
                        # Add label for feature type
                        view.addLabel(
                            feature['type'], 
                            {'position': {'x': center[0], 'y': center[1], 'z': center[2]}, 
                             'backgroundColor': color,
                             'fontColor': 'white',
                             'fontSize': 12,
                             'alignment': 'center'}
                        )
                    
                    view.zoomTo()
                    
                    # Create HTML representation directly
                    view_html = view._make_html()
                    st.components.v1.html(view_html, width=800, height=500)
            
            # Generate pharmacophore button
            if st.button("Generate Pharmacophore Model"):
                with st.spinner("Generating pharmacophore model..."):
                    # Initialize generator
                    generator = PharmacophoreGenerator(
                        method=pharma_method,
                        pocket=selected_pocket,
                        ref_ligand=ref_mol if 'ref_mol' in locals() else None
                    )
                    
                    # Generate features
                    features = generator.generate_features()
                    
                    if features:
                        # Create model
                        model, pharma_mol = generator.create_pharmacophore_model(features)
                        
                        # Store in session state
                        model_id = f"{selected_protein}_{selected_pocket['id']}"
                        st.session_state.pharmacophores[model_id] = {
                            'model': model,
                            'mol': pharma_mol,
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                        st.success(f"Generated pharmacophore model with {len(features)} features")
                        log_action("Pharmacophore_Generation", f"Created model {model_id}")
                        
                        # Show the model
                        st.subheader("Pharmacophore Model")
                        
                        # Display features table
                        feature_df = pd.DataFrame([
                            {
                                'Type': f.get('type', ''),
                                'Center X': f.get('center', [0, 0, 0])[0],
                                'Center Y': f.get('center', [0, 0, 0])[1],
                                'Center Z': f.get('center', [0, 0, 0])[2],
                                'Radius': f.get('radius', 0.0),
                                'SMARTS': f.get('smarts', '')
                            }
                            for f in features
                        ])
                        st.dataframe(feature_df, use_container_width=True)
                        
                        # Visualize the model
                        st.subheader("3D Pharmacophore Visualization")
                        generator.visualize_pharmacophore(features, selected_pocket)
                        
                        # Show SMARTS string for the pharmacophore
                        st.subheader("Pharmacophore Definition")
                        smarts_strings = []
                        for f in features:
                            smarts_strings.append(f"{f['type']}[{f['smarts']}]")
                        
                        pharma_def = " & ".join(smarts_strings)
                        st.code(pharma_def)
                        
                        # Download options
                        st.subheader("Export Options")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Export as JSON
                            save_to_file(model, f"{model_id}_pharmacophore.json", format="json")
                        
                        with col2:
                            # Export as SDF
                            if pharma_mol:
                                save_to_file(pharma_mol, f"{model_id}_pharmacophore.sdf", format="sdf")
                    else:
                        st.warning("Could not generate pharmacophore features. Try adjusting the parameters or selecting a different pocket.")
            
            # Show existing models
            if st.session_state.pharmacophores:
                st.markdown("---")
                st.subheader("Saved Pharmacophore Models")
                
                for model_id, model_data in st.session_state.pharmacophores.items():
                    st.markdown(f"**{model_id}** - Created: {model_data['timestamp']}")
                    
                    # Show mini stats
                    features = model_data['model']['features']
                    feature_counts = {}
                    for f in features:
                        if f['type'] not in feature_counts:
                            feature_counts[f['type']] = 0
                        feature_counts[f['type']] += 1
                    
                    feature_str = ", ".join([f"{count} {ftype}" for ftype, count in feature_counts.items()])
                    st.markdown(f"{len(features)} features: {feature_str}")
                    
                    # Option to view this model
                    if st.button(f"View {model_id}", key=f"view_{model_id}"):
                        generator = PharmacophoreGenerator()
                        generator.visualize_pharmacophore(features)
                        
                        # Show features table
                        feature_df = pd.DataFrame([
                            {
                                'Type': f.get('type', ''),
                                'Center X': f.get('center', [0, 0, 0])[0],
                                'Center Y': f.get('center', [0, 0, 0])[1],
                                'Center Z': f.get('center', [0, 0, 0])[2],
                                'Radius': f.get('radius', 0.0)
                            }
                            for f in features
                        ])
                        st.dataframe(feature_df, use_container_width=True)
                    
                    st.markdown("---")

# --- Virtual Screening Tab ---
with tab3:
    st.header("Virtual Screening")
    st.write("Screen compounds against pharmacophore models and druglikeness criteria.")
    
    col1, col2 = st.columns([2, 3])
    
    with col1:
        # Select pharmacophore model
        if not st.session_state.pharmacophores:
            st.warning("Please generate pharmacophore models in the Pharmacophore Modeling tab first.")
        else:
            available_models = list(st.session_state.pharmacophores.keys())
            selected_model = st.selectbox("Select Pharmacophore Model", available_models)
            
            if selected_model:
                model_data = st.session_state.pharmacophores[selected_model]
                features = model_data['model']['features']
                
                # Show model info
                st.info(f"""
                **Model:** {selected_model}
                **Features:** {len(features)}
                **Method:** {model_data['model']['method']}
                """)
                
                # Feature types present
                feature_types = [f['type'] for f in features]
                st.write("Feature types: " + ", ".join(set(feature_types)))
        
        # Upload compound library
        st.subheader("Compound Library")
        library_format = st.radio(
            "Library Format",
            ["SMILES File", "SDF File", "CSV File", "Custom SMILES Input"],
            index=0
        )
        
        if library_format == "SMILES File":
            smiles_file = st.file_uploader("Upload SMILES File", type=["smi", "txt"])
            
            if smiles_file:
                smiles_data = smiles_file.getvalue().decode("utf-8").splitlines()
                smiles_count = len([s for s in smiles_data if s.strip()])
                st.success(f"Loaded {smiles_count} SMILES strings")
                
                # Preview
                if smiles_count > 0:
                    preview_df = pd.DataFrame([
                        {"SMILES": s.strip().split()[0] if ' ' in s.strip() else s.strip()}
                        for s in smiles_data[:5] if s.strip()
                    ])
                    st.dataframe(preview_df, use_container_width=True)
        
        elif library_format == "SDF File":
            sdf_file = st.file_uploader("Upload SDF File", type=["sdf", "sd"])
            
            if sdf_file:
                # Save temporarily
                sdf_bytes = sdf_file.getvalue()
                temp_file = "temp_compounds.sdf"
                with open(temp_file, "wb") as f:
                    f.write(sdf_bytes)
                
                # Load compounds
                try:
                    sdf_supplier = Chem.SDMolSupplier(temp_file)
                    sdf_count = len([m for m in sdf_supplier if m is not None])
                    st.success(f"Loaded {sdf_count} compounds from SDF file")
                    
                    # Preview first few molecules
                    if sdf_count > 0:
                        preview_mols = [m for m in Chem.SDMolSupplier(temp_file) if m is not None][:3]
                        if preview_mols:
                            img = Draw.MolsToGridImage(
                                preview_mols,
                                molsPerRow=3,
                                subImgSize=(200, 200),
                                legends=[m.GetProp('_Name') if m.HasProp('_Name') else f"Mol_{i}" for i, m in enumerate(preview_mols)]
                            )
                            st.image(img, caption="Sample Compounds")
                except Exception as e:
                    st.error(f"Error loading SDF file: {str(e)}")
        
        elif library_format == "CSV File":
            csv_file = st.file_uploader("Upload CSV File", type=["csv"])
            
            if csv_file:
                try:
                    csv_df = pd.read_csv(csv_file)
                    
                    # Try to find SMILES column
                    smiles_col = None
                    potential_cols = ['SMILES', 'smiles', 'Smiles', 'SMILE', 'smile', 'Smile', 'Structure', 'structure']
                    
                    for col in potential_cols:
                        if col in csv_df.columns:
                            smiles_col = col
                            break
                    
                    if not smiles_col:
                        # Ask user to select column
                        smiles_col = st.selectbox(
                            "Select SMILES column",
                            csv_df.columns
                        )
                    
                    st.success(f"Loaded {len(csv_df)} compounds from CSV file")
                    st.dataframe(csv_df.head(5), use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error loading CSV file: {str(e)}")
        
        elif library_format == "Custom SMILES Input":
            custom_smiles = st.text_area(
                "Enter SMILES strings (one per line)",
                height=200,
                placeholder="CC(=O)OC1=CC=CC=C1C(=O)O\nCCN(CC)CCOC(=O)C1=CC=CC=C1NC1=C(C)C=CC=C1C"
            )
            
            if custom_smiles:
                smiles_data = custom_smiles.strip().split('\n')
                smiles_count = len([s for s in smiles_data if s.strip()])
                st.success(f"Entered {smiles_count} SMILES strings")
        
        # Screening parameters
        st.subheader("Screening Parameters")
        
        # Pharmacophore matching settings
        with st.expander("Pharmacophore Matching", expanded=True):
            feature_matching = st.slider(
                "Required feature matches",
                min_value=1,
                max_value=len(features) if 'features' in locals() else 5,
                value=min(len(features) if 'features' in locals() else 5, 3)
            )
            
            feature_tolerance = st.slider(
                "Feature matching tolerance (Å)",
                min_value=0.5,
                max_value=3.0,
                value=1.5,
                step=0.1
            )
            
            ignore_feature_types = st.multiselect(
                "Ignore feature types",
                ['hydrophobic', 'hbond_donor', 'hbond_acceptor', 'positive', 'negative', 'aromatic'],
                default=[]
            )
        
        # Druglikeness filters
        with st.expander("Druglikeness Filters", expanded=True):
            apply_filters = st.checkbox("Apply druglikeness filters", value=True)
            
            if apply_filters:
                selected_filters = st.multiselect(
                    "Select filters",
                    ["Lipinski's Rule of 5", "Veber Rules", "PAINS Filter", "Ghose Filter"],
                    default=["Lipinski's Rule of 5"]
                )
                
                # Custom property ranges
                use_custom_ranges = st.checkbox("Use custom property ranges", value=False)
                
                if use_custom_ranges:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        mw_min = st.number_input("MW min", value=0.0, step=10.0)
                        logp_min = st.number_input("LogP min", value=-2.0, step=0.5)
                        hbd_min = st.number_input("HBD min", value=0, step=1)
                        hba_min = st.number_input("HBA min", value=0, step=1)
                    
                    with col2:
                        mw_max = st.number_input("MW max", value=500.0, step=10.0)
                        logp_max = st.number_input("LogP max", value=5.0, step=0.5)
                        hbd_max = st.number_input("HBD max", value=5, step=1)
                        hba_max = st.number_input("HBA max", value=10, step=1)
                    
                    custom_ranges = {
                        'MW': (mw_min, mw_max),
                        'LogP': (logp_min, logp_max),
                        'HBD': (hbd_min, hbd_max),
                        'HBA': (hba_min, hba_max)
                    }
        
        # Maximum number of hits
        max_hits = st.number_input("Maximum hits to return", min_value=1, value=50, step=10)
        
        # Run screening button
        screen_button = st.button("Run Virtual Screening")
    
    with col2:
        # Screening results section
        if screen_button:
            if not 'selected_model' in locals() or not selected_model:
                st.error("Please select a pharmacophore model first.")
            elif library_format == "SMILES File" and 'smiles_data' not in locals():
                st.error("Please upload a SMILES file.")
            elif library_format == "SDF File" and 'sdf_supplier' not in locals():
                st.error("Please upload an SDF file.")
            elif library_format == "CSV File" and ('csv_df' not in locals() or 'smiles_col' not in locals()):
                st.error("Please upload a CSV file and select the SMILES column.")
            elif library_format == "Custom SMILES Input" and 'smiles_data' not in locals():
                st.error("Please enter SMILES strings.")
            else:
                st.subheader("Screening Results")
                
                # Function to screen molecules
                def screen_molecules(model_data, mols, feature_matching, feature_tolerance, ignore_features,
                                    apply_filters, selected_filters, custom_ranges=None, max_hits=50):
                    """Screen molecules against pharmacophore and druglikeness filters."""
                    results = []
                    features = model_data['model']['features']
                    
                    # Filter out ignored feature types
                    filtered_features = [f for f in features if f['type'] not in ignore_features]
                    
                    # Progress bar
                    progress_bar = st.progress(0)
                    
                    # Process each molecule
                    for i, mol in enumerate(mols):
                        if mol is None:
                            continue
                        
                        # Update progress
                        progress_bar.progress(min(1.0, (i+1) / len(mols)))
                        
                        # Skip molecules that don't contain required features
                        if not has_required_features(mol, filtered_features, feature_matching):
                            continue
                        
                        # Calculate 3D coordinates if not present
                        if mol.GetNumConformers() == 0:
                            mol = Chem.AddHs(mol)
                            try:
                                AllChem.EmbedMolecule(mol, randomSeed=42)
                                AllChem.UFFOptimizeMolecule(mol)
                            except:
                                continue  # Skip if 3D embedding fails
                        
                        # Check pharmacophore match
                        match_result = match_pharmacophore(mol, filtered_features, feature_matching, feature_tolerance)
                        
                        if match_result['match']:
                            # Calculate descriptors
                            descriptors = calculate_molecule_descriptors(mol)
                            
                            # Check druglikeness if applied
                            druglike = True
                            if apply_filters:
                                filter_results = check_druglikeness(mol, selected_filters)
                                druglike = filter_results.get('overall_pass', True)
                                
                                # Check custom ranges if provided
                                if custom_ranges and druglike:
                                    for prop, (min_val, max_val) in custom_ranges.items():
                                        if prop in descriptors and (descriptors[prop] < min_val or descriptors[prop] > max_val):
                                            druglike = False
                                            break
                            
                            if druglike:
                                # Get molecule name or SMILES
                                mol_name = mol.GetProp('_Name') if mol.HasProp('_Name') else Chem.MolToSmiles(mol)
                                
                                # Add to results
                                results.append({
                                    'mol': mol,
                                    'name': mol_name,
                                    'match_score': match_result['score'],
                                    'features_matched': match_result['matched_count'],
                                    'descriptors': descriptors
                                })
                    
                    # Sort by match score
                    results.sort(key=lambda x: x['match_score'], reverse=True)
                    
                    # Limit to max hits
                    return results[:max_hits]
                
                def has_required_features(mol, features, min_required):
                    """Quick check if molecule has potential to match required features."""
                    feature_patterns = {
                        'hydrophobic': '[#6]',
                        'hbond_donor': '[#7!H0,#8!H0]',
                        'hbond_acceptor': '[#7,#8]',
                        'positive': '[+,#7]',
                        'negative': '[-,#8]',
                        'aromatic': 'a'
                    }
                    
                    # Count feature types in the pharmacophore
                    feature_types = {}
                    for f in features:
                        if f['type'] not in feature_types:
                            feature_types[f['type']] = 0
                        feature_types[f['type']] += 1
                    
                    # Check if molecule has the required feature types
                    potential_matches = 0
                    for ftype, count in feature_types.items():
                        if ftype in feature_patterns:
                            pattern = Chem.MolFromSmarts(feature_patterns[ftype])
                            if pattern and mol.HasSubstructMatch(pattern):
                                potential_matches += 1
                    
                    return potential_matches >= min_required
                
                def match_pharmacophore(mol, features, min_required, tolerance):
                    """Check if molecule matches pharmacophore features."""
                    if mol.GetNumConformers() == 0:
                        return {'match': False, 'score': 0, 'matched_count': 0}
                    
                    conf = mol.GetConformer()
                    matched_features = []
                    
                    # Feature-specific SMARTS patterns
                    feature_patterns = {
                        'hydrophobic': ['[#6D3]', '[#6D4]', '[#6;X3,X4]'],
                        'hbond_donor': ['[#7!H0]', '[#8!H0]', '[N,O;!H0]'],
                        'hbond_acceptor': ['[#7]', '[#8]', '[N,O]'],
                        'positive': ['[+]', '[#7H2]', '[NH2]'],
                        'negative': ['[-]', '[#8D1]', '[COO-]'],
                        'aromatic': ['a5', 'a6', 'c1ccccc1']
                    }
                    
                    # Check each pharmacophore feature
                    for feature in features:
                        feature_type = feature['type']
                        feature_center = np.array(feature['center'])
                        feature_radius = feature['radius'] + tolerance  # Add tolerance
                        
                        # Get matching atoms for this feature type
                        matching_atoms = []
                        
                        if feature_type in feature_patterns:
                            for pattern in feature_patterns[feature_type]:
                                patt = Chem.MolFromSmarts(pattern)
                                if patt:
                                    matches = mol.GetSubstructMatches(patt)
                                    for match in matches:
                                        matching_atoms.extend(match)
                        
                        # Check distances to feature center
                        for atom_idx in matching_atoms:
                            atom_pos = conf.GetAtomPosition(atom_idx)
                            atom_coords = np.array([atom_pos.x, atom_pos.y, atom_pos.z])
                            
                            distance = np.linalg.norm(atom_coords - feature_center)
                            
                            if distance <= feature_radius:
                                matched_features.append(feature)
                                break  # Match found for this feature
                    
                    # Calculate matching score
                    match_count = len(matched_features)
                    match_score = match_count / len(features) * 100
                    
                    return {
                        'match': match_count >= min_required,
                        'score': match_score,
                        'matched_count': match_count
                    }
                
                # Prepare molecules based on input format
                with st.spinner("Preparing molecules for screening..."):
                    mols = []
                    
                    if library_format == "SMILES File" or library_format == "Custom SMILES Input":
                        for smiles in smiles_data:
                            if not smiles.strip():
                                continue
                                
                            # Parse SMILES - handle case where there might be a name after the SMILES
                            parts = smiles.strip().split()
                            smiles_str = parts[0]
                            name = parts[1] if len(parts) > 1 else smiles_str
                            
                            mol = Chem.MolFromSmiles(smiles_str)
                            if mol:
                                mol.SetProp("_Name", name)
                                mol = Chem.AddHs(mol)
                                try:
                                    AllChem.EmbedMolecule(mol, randomSeed=42)
                                    mols.append(mol)
                                except:
                                    # If 3D embedding fails, still keep the molecule for screening
                                    # It will be embedded later
                                    mols.append(mol)
                    
                    elif library_format == "SDF File":
                        for mol in Chem.SDMolSupplier(temp_file):
                            if mol:
                                mols.append(mol)
                    
                    elif library_format == "CSV File":
                        for _, row in csv_df.iterrows():
                            smiles_str = row[smiles_col]
                            if pd.isna(smiles_str) or not isinstance(smiles_str, str):
                                continue
                                
                            mol = Chem.MolFromSmiles(smiles_str)
                            if mol:
                                # Try to get a name from another column
                                name_cols = ['Name', 'ID', 'Compound_ID', 'CompoundID', 'Compound ID']
                                mol_name = smiles_str
                                
                                for col in name_cols:
                                    if col in csv_df.columns and not pd.isna(row[col]):
                                        mol_name = str(row[col])
                                        break
                                
                                mol.SetProp("_Name", mol_name)
                                
                                # Add properties from the CSV if present
                                for col in csv_df.columns:
                                    if col != smiles_col and not pd.isna(row[col]):
                                        mol.SetProp(col, str(row[col]))
                                
                                mol = Chem.AddHs(mol)
                                try:
                                    AllChem.EmbedMolecule(mol, randomSeed=42)
                                except:
                                    pass
                                
                                mols.append(mol)
                
                # Run screening
                with st.spinner(f"Screening {len(mols)} compounds..."):
                    results = screen_molecules(
                        model_data,
                        mols,
                        feature_matching,
                        feature_tolerance,
                        ignore_feature_types,
                        apply_filters,
                        selected_filters,
                        custom_ranges if 'use_custom_ranges' in locals() and use_custom_ranges else None,
                        max_hits
                    )
                
                # Display results
                if results:
                    st.success(f"Found {len(results)} matching compounds!")
                    
                    # Save to session state
                    screen_id = f"screen_{selected_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    st.session_state.screened_compounds[screen_id] = {
                        'results': results,
                        'model': selected_model,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    # Create results table
                    results_table = []
                    for i, hit in enumerate(results):
                        desc = hit['descriptors']
                        results_table.append({
                            'Rank': i+1,
                            'Name': hit['name'],
                            'Match Score': f"{hit['match_score']:.1f}%",
                            'MW': f"{desc['MW']:.1f}",
                            'LogP': f"{desc['LogP']:.2f}",
                            'TPSA': f"{desc['TPSA']:.1f}",
                            'HBD': desc['HBD'],
                            'HBA': desc['HBA'],
                            'RotBonds': desc['RotBonds'],
                            'QED': f"{desc['QED']:.3f}"
                        })
                    
                    results_df = pd.DataFrame(results_table)
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Save results
                    save_to_file(results_df, f"screening_results_{screen_id}.{export_format.lower()}")
                    
                    # Visualization
                    st.subheader("Hit Visualization")
                    
                    # View top hits
                    top_n = min(10, len(results))
                    top_mols = [hit['mol'] for hit in results[:top_n]]
                    top_names = [hit['name'] if len(hit['name']) < 20 else hit['name'][:17]+"..." for hit in results[:top_n]]
                    
                    # Generate images
                    img = Draw.MolsToGridImage(
                        top_mols,
                        molsPerRow=5,
                        subImgSize=(300, 300),
                        legends=top_names
                    )
                    st.image(img, caption="Top Hits")
                    
                    # Property distribution plots
                    st.subheader("Property Distribution")
                    
                    # Prepare data for plots
                    plot_data = pd.DataFrame([{
                        'MW': hit['descriptors']['MW'],
                        'LogP': hit['descriptors']['LogP'],
                        'TPSA': hit['descriptors']['TPSA'],
                        'HBD': hit['descriptors']['HBD'],
                        'HBA': hit['descriptors']['HBA'],
                        'QED': hit['descriptors']['QED'],
                        'Rank': i+1,
                        'MatchScore': hit['match_score']
                    } for i, hit in enumerate(results)])
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # MW vs LogP
                        fig = px.scatter(
                            plot_data,
                            x='MW',
                            y='LogP',
                            color='MatchScore',
                            size='QED',
                            hover_data=['Rank', 'HBD', 'HBA'],
                            title='Molecular Weight vs LogP',
                            color_continuous_scale=selected_theme
                        )
                        
                        # Add Lipinski boundaries
                        fig.add_shape(
                            type="rect",
                            x0=0, y0=-0.4, x1=500, y1=5,
                            line=dict(color="rgba(0,255,0,0.1)"),
                            fillcolor="rgba(0,255,0,0.1)"
                        )
                        
                        fig.update_layout(
                            xaxis_title="Molecular Weight",
                            yaxis_title="LogP"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # TPSA vs QED
                        fig = px.scatter(
                            plot_data,
                            x='TPSA',
                            y='QED',
                            color='MatchScore',
                            size='MW',
                            hover_data=['Rank', 'LogP'],
                            title='TPSA vs Drug-likeness (QED)',
                            color_continuous_scale=selected_theme
                        )
                        
                        fig.update_layout(
                            xaxis_title="TPSA",
                            yaxis_title="QED"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # 3D plot of top compound with pharmacophore
                    st.subheader("Top Hit with Pharmacophore")
                    
                    if len(results) > 0:
                        top_hit = results[0]
                        top_mol = top_hit['mol']
                        
                        # Create py3Dmol view
                        view = py3Dmol.view(width=800, height=500)
                        
                        # Add molecule
                        mol_block = Chem.MolToMolBlock(top_mol)
                        view.addModel(mol_block, "mol")
                        view.setStyle({'stick': {}})
                        
                        # Add pharmacophore features
                        feature_colors = {
                            'hydrophobic': 'yellow',
                            'hbond_donor': 'blue',
                            'hbond_acceptor': 'red',
                            'positive': 'green',
                            'negative': 'orange',
                            'aromatic': 'purple'
                        }
                        
                        for feature in features:
                            center = feature['center']
                            color = feature_colors.get(feature['type'], 'gray')
                            
                            view.addSphere({
                                'center': {'x': center[0], 'y': center[1], 'z': center[2]},
                                'radius': feature['radius'],
                                'color': color,
                                'opacity': 0.4
                            })
                        
                        view.zoomTo()
                        # Create HTML representation directly with py3Dmol
                        view_html = view._make_html()
                        st.components.v1.html(view_html, width=800, height=500)
                        
                        # Compound details
                        st.subheader(f"Details for Top Hit: {top_hit['name']}")
                        
                        desc = top_hit['descriptors']
                        detail_cols = st.columns(3)
                        
                        with detail_cols[0]:
                            st.metric("Molecular Weight", f"{desc['MW']:.1f}")
                            st.metric("LogP", f"{desc['LogP']:.2f}")
                            st.metric("TPSA", f"{desc['TPSA']:.1f}")
                        
                        with detail_cols[1]:
                            st.metric("H-Bond Donors", desc['HBD'])
                            st.metric("H-Bond Acceptors", desc['HBA'])
                            st.metric("Rotatable Bonds", desc['RotBonds'])
                        
                        with detail_cols[2]:
                            st.metric("QED Score", f"{desc['QED']:.3f}")
                            st.metric("Aromatic Rings", desc['AromaticRings'])
                            st.metric("Match Score", f"{top_hit['match_score']:.1f}%")
                            
                        # Save individual compound
                        st.download_button(
                            "Download Structure (SDF)",
                            Chem.MolToMolBlock(top_mol),
                            file_name=f"{top_hit['name'].replace('/', '_')}.sdf",
                            mime="chemical/x-mdl-sdfile"
                        )
                else:
                    st.warning("No compounds matched the criteria. Try adjusting the parameters or using a different library.")
        
        # Show previous screening results
        if st.session_state.screened_compounds:
            st.markdown("---")
            st.subheader("Previous Screening Results")
            
            screens = list(st.session_state.screened_compounds.keys())
            selected_screen = st.selectbox("Select screening run", screens, format_func=lambda x: f"{x} - {st.session_state.screened_compounds[x]['timestamp']}")
            
            if selected_screen:
                screen_data = st.session_state.screened_compounds[selected_screen]
                st.write(f"Model: {screen_data['model']} - Hits: {len(screen_data['results'])}")
                
                # Show top 3 hits
                top_hits = screen_data['results'][:3]
                
                if top_hits:
                    top_mols = [hit['mol'] for hit in top_hits]
                    top_names = [hit['name'] if len(hit['name']) < 20 else hit['name'][:17]+"..." for hit in top_hits]
                    
                    img = Draw.MolsToGridImage(
                        top_mols,
                        molsPerRow=3,
                        subImgSize=(200, 200),
                        legends=top_names
                    )
                    st.image(img, caption="Top 3 Hits")

# --- SAR Analysis Tab ---
with tab4:
    st.header("Structure-Activity Relationship Analysis")
    st.write("Analyze chemical features that correlate with activity against targets.")
    
    # Check if we have screening results
    if not st.session_state.screened_compounds:
        st.warning("Please run virtual screening first to generate compounds for SAR analysis.")
    else:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Select screening results to analyze
            screens = list(st.session_state.screened_compounds.keys())
            selected_screen = st.selectbox(
                "Select screening results", 
                screens, 
                format_func=lambda x: f"{x} - {st.session_state.screened_compounds[x]['timestamp']}",
                key="sar_screen_select"
            )
            
            if selected_screen:
                screen_data = st.session_state.screened_compounds[selected_screen]
                results = screen_data['results']
                
                st.success(f"Analyzing {len(results)} compounds")
                
                # Activity threshold
                st.subheader("Activity Definition")
                
                activity_source = st.radio(
                    "Activity data source",
                    ["Match Score", "Custom Property", "Manual Entry"],
                    index=0
                )
                
                if activity_source == "Match Score":
                    # Use match score as activity
                    activity_threshold = st.slider(
                        "Activity threshold (match score %)",
                        min_value=0.0,
                        max_value=100.0,
                        value=70.0,
                        step=5.0
                    )
                    
                    # Mark compounds as active/inactive
                    for hit in results:
                        hit['active'] = hit['match_score'] >= activity_threshold
                
                elif activity_source == "Custom Property":
                    # Check if molecules have properties
                    available_props = set()
                    for hit in results:
                        mol = hit['mol']
                        for prop_name in mol.GetPropNames():
                            available_props.add(prop_name)
                    
                    if available_props:
                        selected_prop = st.selectbox(
                            "Select property for activity",
                            sorted(list(available_props))
                        )
                        
                        prop_threshold = st.number_input(
                            f"{selected_prop} threshold",
                            value=0.5,
                            step=0.1
                        )
                        
                        higher_is_active = st.checkbox("Higher values are active", value=True)
                        
                        # Mark compounds as active/inactive
                        for hit in results:
                            mol = hit['mol']
                            if mol.HasProp(selected_prop):
                                try:
                                    prop_val = float(mol.GetProp(selected_prop))
                                    if higher_is_active:
                                        hit['active'] = prop_val >= prop_threshold
                                    else:
                                        hit['active'] = prop_val <= prop_threshold
                                except ValueError:
                                    hit['active'] = False
                            else:
                                hit['active'] = False
                    else:
                        st.warning("No properties found in the molecules.")
                        for hit in results:
                            hit['active'] = False
                
                elif activity_source == "Manual Entry":
                    st.write("Select active compounds:")
                    
                    # Create a dataframe for easier selection
                    df_select = pd.DataFrame([
                        {"ID": i, "Name": hit['name']}
                        for i, hit in enumerate(results)
                    ])
                    
                    edited_df = st.data_editor(
                        df_select,
                        hide_index=True,
                        column_config={
                            "ID": st.column_config.NumberColumn(
                                "ID",
                                help="Compound ID",
                                disabled=True
                            ),
                            "Name": st.column_config.TextColumn(
                                "Name",
                                help="Compound name",
                                disabled=True
                            ),
                            "Active": st.column_config.CheckboxColumn(
                                "Active",
                                help="Select active compounds",
                                default=False
                            )
                        }
                    )
                    
                    # Get selected active compounds
                    active_ids = set()
                    for _, row in edited_df.iterrows():
                        if 'Active' in row and row['Active']:
                            active_ids.add(row['ID'])
                    
                    # Mark compounds as active/inactive
                    for i, hit in enumerate(results):
                        hit['active'] = i in active_ids
                
                # Analysis options
                st.subheader("Analysis Options")
                
                analysis_type = st.multiselect(
                    "Select analyses to run",
                    ["Scaffold Analysis", "Property Distribution", "MCS Analysis", "Feature Importance",
                     "Pharmacophore Refinement"],
                    default=["Scaffold Analysis", "Property Distribution"]
                )
                
                # Run analysis button
                run_analysis = st.button("Run SAR Analysis")
        
        with col2:
            # Display analysis results
            if 'run_analysis' in locals() and run_analysis and 'results' in locals():
                st.subheader("SAR Analysis Results")
                
                # Count active/inactive compounds
                active_count = sum(1 for hit in results if hit.get('active', False))
                inactive_count = len(results) - active_count
                
                st.write(f"Active compounds: {active_count}, Inactive compounds: {inactive_count}")
                
                # Split compounds for analysis
                active_mols = [hit['mol'] for hit in results if hit.get('active', False)]
                inactive_mols = [hit['mol'] for hit in results if not hit.get('active', False)]
                
                if not active_mols:
                    st.warning("No active compounds found. Please adjust activity definition.")
                else:
                    # Run selected analyses
                    if "Scaffold Analysis" in analysis_type:
                        st.markdown("### Scaffold Analysis")
                        
                        # Extract scaffolds from active and inactive compounds
                        active_scaffolds = extract_scaffolds(active_mols)
                        inactive_scaffolds = extract_scaffolds(inactive_mols)
                        
                        # Find enriched scaffolds (present more in active than inactive)
                        enriched_scaffolds = {}
                        
                        for scaffold, indices in active_scaffolds.items():
                            active_count = len(indices)
                            inactive_count = len(inactive_scaffolds.get(scaffold, []))
                            
                            active_fraction = active_count / len(active_mols) if active_mols else 0
                            inactive_fraction = inactive_count / len(inactive_mols) if inactive_mols else 0
                            
                            if active_fraction > inactive_fraction and active_count >= 2:
                                enrichment = active_fraction / max(inactive_fraction, 0.01)  # Avoid division by zero
                                enriched_scaffolds[scaffold] = {
                                    'smiles': scaffold,
                                    'active_count': active_count,
                                    'inactive_count': inactive_count,
                                    'enrichment': enrichment
                                }
                        
                        # Sort enriched scaffolds by enrichment factor
                        sorted_scaffolds = sorted(
                            enriched_scaffolds.values(),
                            key=lambda x: x['enrichment'],
                            reverse=True
                        )
                        
                        if sorted_scaffolds:
                            # Create table of enriched scaffolds
                            scaffold_df = pd.DataFrame(sorted_scaffolds)
                            st.dataframe(scaffold_df, use_container_width=True)
                            
                            # Visualize top enriched scaffolds
                            top_scaffolds = sorted_scaffolds[:min(5, len(sorted_scaffolds))]
                            scaffold_mols = [Chem.MolFromSmiles(s['smiles']) for s in top_scaffolds]
                            scaffold_labels = [f"Enrichment: {s['enrichment']:.1f}x\nActive: {s['active_count']}" for s in top_scaffolds]
                            
                            img = Draw.MolsToGridImage(
                                scaffold_mols,
                                molsPerRow=3,
                                subImgSize=(200, 200),
                                legends=scaffold_labels
                            )
                            st.image(img, caption="Top Enriched Scaffolds")
                        else:
                            st.info("No enriched scaffolds found.")
                    
                    if "Property Distribution" in analysis_type:
                        st.markdown("### Property Distribution")
                        
                        # Calculate properties for all compounds
                        active_props = [calculate_molecule_descriptors(mol) for mol in active_mols]
                        inactive_props = [calculate_molecule_descriptors(mol) for mol in inactive_mols]
                        
                        # Create dataframes for plotting
                        active_df = pd.DataFrame(active_props)
                        active_df['Activity'] = 'Active'
                        
                        inactive_df = pd.DataFrame(inactive_props)
                        inactive_df['Activity'] = 'Inactive'
                        
                        combined_df = pd.concat([active_df, inactive_df])
                        
                        # Plot distributions for key properties
                        properties = ['MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotBonds', 'Fsp3', 'QED']
                        
                        # Create distribution plots
                        for i in range(0, len(properties), 2):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                if i < len(properties):
                                    prop = properties[i]
                                    fig = px.histogram(
                                        combined_df, 
                                        x=prop, 
                                        color='Activity',
                                        barmode='overlay',
                                        histnorm='percent',
                                        title=f"{prop} Distribution",
                                        color_discrete_map={'Active': 'green', 'Inactive': 'red'}
                                    )
                                    fig.update_layout(bargap=0.1)
                                    st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                if i+1 < len(properties):
                                    prop = properties[i+1]
                                    fig = px.histogram(
                                        combined_df, 
                                        x=prop, 
                                        color='Activity',
                                        barmode='overlay',
                                        histnorm='percent',
                                        title=f"{prop} Distribution",
                                        color_discrete_map={'Active': 'green', 'Inactive': 'red'}
                                    )
                                    fig.update_layout(bargap=0.1)
                                    st.plotly_chart(fig, use_container_width=True)
                        
                        # Property statistics
                        st.markdown("#### Property Statistics")
                        
                        property_stats = []
                        for prop in properties:
                            if active_df[prop].count() > 0 and inactive_df[prop].count() > 0:
                                active_mean = active_df[prop].mean()
                                inactive_mean = inactive_df[prop].mean()
                                
                                property_stats.append({
                                    'Property': prop,
                                    'Active Mean': f"{active_mean:.2f}",
                                    'Inactive Mean': f"{inactive_mean:.2f}",
                                    'Difference': f"{active_mean - inactive_mean:.2f}",
                                    'Ratio': f"{active_mean / inactive_mean:.2f}" if inactive_mean != 0 else "N/A"
                                })
                        
                        stats_df = pd.DataFrame(property_stats)
                        st.dataframe(stats_df, use_container_width=True)
                    
                    if "MCS Analysis" in analysis_type:
                        st.markdown("### Maximum Common Substructure Analysis")
                        
                        # Find MCS among active compounds
                        if len(active_mols) >= 2:
                            try:
                                from rdkit.Chem import rdFMCS
                                
                                mcs = rdFMCS.FindMCS(
                                    active_mols,
                                    threshold=0.8,
                                    timeout=60,
                                    matchValences=True,
                                    ringMatchesRingOnly=True,
                                    completeRingsOnly=True
                                )
                                
                                if mcs and mcs.numAtoms > 0:
                                    mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
                                    mcs_mol_template = Chem.MolFromSmiles(Chem.MolToSmiles(mcs_mol, isomericSmiles=True)) if mcs_mol else None
                                    
                                    if mcs_mol_template:
                                        st.write(f"MCS contains {mcs.numAtoms} atoms and {mcs.numBonds} bonds")
                                        
                                        # Display MCS structure
                                        img = Draw.MolToImage(mcs_mol_template, size=(400, 400))
                                        st.image(img, caption="Maximum Common Substructure")
                                        
                                        # Highlight MCS in active compounds
                                        top_active = active_mols[:min(5, len(active_mols))]
                                        highlighted_mols = []
                                        
                                        for mol in top_active:
                                            matches = mol.GetSubstructMatches(mcs_mol)
                                            if matches:
                                                match_atoms = matches[0]
                                                highlight_mol = Chem.Mol(mol)
                                                
                                                for atom_idx in range(highlight_mol.GetNumAtoms()):
                                                    highlight_mol.GetAtomWithIdx(atom_idx).SetProp(
                                                        'atomNote', 'MCS' if atom_idx in match_atoms else ''
                                                    )
                                                
                                                highlighted_mols.append(highlight_mol)
                                            else:
                                                highlighted_mols.append(mol)
                                        
                                        # Draw with highlighting
                                        img = Draw.MolsToGridImage(
                                            highlighted_mols,
                                            molsPerRow=3,
                                            subImgSize=(250, 250),
                                            highlightAtomLists=[mol.GetSubstructMatch(mcs_mol) for mol in highlighted_mols],
                                            useSVG=False
                                        )
                                        st.image(img, caption="MCS Highlighted in Active Compounds")
                                    else:
                                        st.warning("Could not convert MCS SMARTS to molecule.")
                                else:
                                    st.warning("No significant MCS found among active compounds.")
                            except Exception as e:
                                st.error(f"Error in MCS analysis: {str(e)}")
                        else:
                            st.warning("Need at least 2 active compounds for MCS analysis.")
                    
                    if "Feature Importance" in analysis_type:
                        st.markdown("### Feature Importance Analysis")
                        
                        # Calculate features for all compounds
                        from rdkit.Chem import AllChem
                        
                        # Generate Morgan fingerprints for active and inactive compounds
                        active_fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in active_mols]
                        inactive_fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in inactive_mols]
                        
                        # Convert to binary arrays
                        from rdkit import DataStructs
                        import numpy as np
                        
                        active_np = []
                        for fp in active_fps:
                            arr = np.zeros((1024,))
                            DataStructs.ConvertToNumpyArray(fp, arr)
                            active_np.append(arr)
                        
                        inactive_np = []
                        for fp in inactive_fps:
                            arr = np.zeros((1024,))
                            DataStructs.ConvertToNumpyArray(fp, arr)
                            inactive_np.append(arr)
                        
                        if active_np and inactive_np:
                            # Calculate feature frequency in active vs inactive
                            active_counts = np.sum(active_np, axis=0) / len(active_np)
                            inactive_counts = np.sum(inactive_np, axis=0) / len(inactive_np) if len(inactive_np) > 0 else np.zeros((1024,))
                            
                            # Calculate enrichment ratio
                            eps = 1e-6  # Avoid division by zero
                            enrichment = active_counts / (inactive_counts + eps)
                            
                            # Get top enriched features
                            top_indices = np.argsort(enrichment)[-20:]
                            top_features = []
                            
                            for idx in top_indices:
                                if active_counts[idx] > 0.2:  # Present in at least 20% of active compounds
                                    top_features.append({
                                        'Feature': idx,
                                        'Active %': f"{active_counts[idx]*100:.1f}%",
                                        'Inactive %': f"{inactive_counts[idx]*100:.1f}%",
                                        'Enrichment': f"{enrichment[idx]:.1f}x"
                                    })
                            
                            # Show table of enriched features
                            if top_features:
                                df = pd.DataFrame(top_features)
                                st.dataframe(df, use_container_width=True)
                                
                                # Visualize top enriched features in compounds
                                feature_mol_examples = []
                                feature_legends = []
                                
                                # Find example molecules for top features
                                for idx in top_indices[-5:]:
                                    # Find actives with this feature
                                    for i, fp_array in enumerate(active_np):
                                        if fp_array[idx] == 1:
                                            mol = active_mols[i]
                                            info = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024, bitInfo={})
                                            
                                            # Get atom indices for this feature
                                            bit_info = {}
                                            AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024, bitInfo=bit_info)
                                            
                                            if idx in bit_info:
                                                feature_mol_examples.append(mol)
                                                feature_legends.append(f"Feature {idx}")
                                                break
                                
                                if feature_mol_examples:
                                    img = Draw.MolsToGridImage(
                                        feature_mol_examples,
                                        molsPerRow=3,
                                        subImgSize=(250, 250),
                                        legends=feature_legends
                                    )
                                    st.image(img, caption="Example Compounds with Top Features")
                            else:
                                st.info("No significantly enriched features found.")
                        else:
                            st.warning("Insufficient data for feature importance analysis.")
                    
                    if "Pharmacophore Refinement" in analysis_type:
                        st.markdown("### Pharmacophore Refinement")
                        
                        # Get original pharmacophore
                        model_data = st.session_state.pharmacophores[screen_data['model']]
                        original_features = model_data['model']['features']
                        
                        # Analyze feature matching in active compounds
                        feature_match_rates = []
                        
                        for i, feature in enumerate(original_features):
                            feature_type = feature['type']
                            feature_center = np.array(feature['center'])
                            feature_radius = feature['radius']
                            
                            # Count matches in active compounds
                            active_matches = 0
                            for mol in active_mols:
                                if mol.GetNumConformers() == 0:
                                    mol = Chem.AddHs(mol)
                                    try:
                                        AllChem.EmbedMolecule(mol, randomSeed=42)
                                        AllChem.UFFOptimizeMolecule(mol)
                                    except:
                                        continue
                                
                                # Check if molecule matches this feature
                                conf = mol.GetConformer()
                                
                                # Feature-specific SMARTS patterns
                                feature_patterns = {
                                    'hydrophobic': ['[#6D3]', '[#6D4]'],
                                    'hbond_donor': ['[#7!H0]', '[#8!H0]'],
                                    'hbond_acceptor': ['[#7]', '[#8]'],
                                    'positive': ['[+]', '[#7H2]'],
                                    'negative': ['[-]', '[#8D1]'],
                                    'aromatic': ['a5', 'a6']
                                }
                                
                                matched = False
                                if feature_type in feature_patterns:
                                    for pattern in feature_patterns[feature_type]:
                                        patt = Chem.MolFromSmarts(pattern)
                                        if patt:
                                            matches = mol.GetSubstructMatches(patt)
                                            for match in matches:
                                                # Check distances to feature center
                                                for atom_idx in match:
                                                    atom_pos = conf.GetAtomPosition(atom_idx)
                                                    atom_coords = np.array([atom_pos.x, atom_pos.y, atom_pos.z])
                                                    
                                                    distance = np.linalg.norm(atom_coords - feature_center)
                                                    
                                                    if distance <= feature_radius + 1.5:  # Add tolerance
                                                        matched = True
                                                        break
                                                
                                                if matched:
                                                    break
                                        
                                        if matched:
                                            break
                                
                                if matched:
                                    active_matches += 1
                            
                            # Calculate match rate
                            active_match_rate = active_matches / len(active_mols) if active_mols else 0
                            
                            feature_match_rates.append({
                                'Feature': i,
                                'Type': feature_type,
                                'Center': feature_center,
                                'Radius': feature_radius,
                                'Match Rate': active_match_rate,
                                'Keep': active_match_rate >= 0.5  # Keep if matches at least 50% of actives
                            })
                        
                        # Show feature match rates
                        match_df = pd.DataFrame([
                            {
                                'Feature': f["Feature"],
                                'Type': f["Type"], 
                                'Match Rate': f"{f['Match Rate']*100:.1f}%",
                                'Keep': f["Keep"]
                            } 
                            for f in feature_match_rates
                        ])
                        st.dataframe(match_df, use_container_width=True)
                        
                        # Create refined pharmacophore
                        refined_features = [
                            original_features[f['Feature']] for f in feature_match_rates if f['Keep']
                        ]
                        
                        if refined_features:
                            st.write(f"Refined pharmacophore has {len(refined_features)} features (originally {len(original_features)})")
                            
                            # Visualize refined pharmacophore
                            # Add visualization of refined pharmacophore
                            feature_colors = {
                                'hydrophobic': 'yellow',
                                'hbond_donor': 'blue',
                                'hbond_acceptor': 'red',
                                'positive': 'green',
                                'negative': 'orange',
                                'aromatic': 'purple'
                            }
                            
                            # Create py3Dmol view
                            view = py3Dmol.view(width=800, height=500)
                            
                            # Add pharmacophore features
                            for feature in refined_features:
                                center = feature['center']
                                color = feature_colors.get(feature['type'], 'gray')
                                
                                view.addSphere({
                                    'center': {'x': center[0], 'y': center[1], 'z': center[2]},
                                    'radius': feature['radius'],
                                    'color': color,
                                    'opacity': 0.7
                                })
                                
                                # Add label for feature type
                                view.addLabel(
                                    feature['type'], 
                                    {'position': {'x': center[0], 'y': center[1], 'z': center[2]}, 
                                     'backgroundColor': color,
                                     'fontColor': 'white',
                                     'fontSize': 12,
                                     'alignment': 'center'}
                                )
                            
                            view.zoomTo()
                            
                            # Create HTML representation directly
                            view_html = view._make_html()
                            st.components.v1.html(view_html, width=800, height=500)
                            
                            # Option to save refined pharmacophore
                            if st.button("Save Refined Pharmacophore"):
                                # Create model info
                                refined_model = {
                                    'features': refined_features,
                                    'method': "Refined",
                                    'parent_model': screen_data['model'],
                                    'max_features': len(refined_features),
                                    'tolerance': model_data['model']['tolerance'] if 'tolerance' in model_data['model'] else 1.5
                                }
                                
                                # Save to session state
                                model_id = f"{screen_data['model']}_refined_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                                st.session_state.pharmacophores[model_id] = {
                                    'model': refined_model,
                                    'mol': None,  # Would need to recreate
                                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                }
                                
                                st.success(f"Saved refined pharmacophore as {model_id}")
                        else:
                            st.warning("No features met the refinement criteria.")

# --- Results Dashboard Tab ---
with tab5:
    st.header("Results Dashboard")
    st.write("Overview of all analysis results and project summary.")
    
    # Summary of loaded data
    st.subheader("Project Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        protein_count = len(st.session_state.analyzed_proteins)
        st.metric("Analyzed Proteins", protein_count)
        
        if protein_count > 0:
            st.write("**Proteins:**")
            for protein in st.session_state.analyzed_proteins:
                st.write(f"• {protein}")
    
    with col2:
        pharmacophore_count = len(st.session_state.pharmacophores)
        st.metric("Pharmacophore Models", pharmacophore_count)
        
        if pharmacophore_count > 0:
            st.write("**Models:**")
            for model_id in st.session_state.pharmacophores:
                st.write(f"• {model_id}")
    
    with col3:
        screening_count = len(st.session_state.screened_compounds)
        total_hits = sum(len(screen['results']) for screen in st.session_state.screened_compounds.values())
        st.metric("Screening Runs", screening_count)
        st.metric("Total Hits", total_hits)
    
    # Timeline of actions
    if st.session_state.history:
        st.subheader("Analysis Timeline")
        
        history_df = pd.DataFrame(st.session_state.history)
        
        # Format for display
        history_df = history_df[['timestamp', 'action', 'details']]
        history_df.columns = ['Timestamp', 'Action', 'Details']
        
        st.dataframe(history_df, use_container_width=True)
        
        # Timeline visualization - use a bar chart instead of timeline
        # Convert timestamps to datetime objects for proper sorting
        history_df['Timestamp'] = pd.to_datetime(history_df['Timestamp'])
        history_df = history_df.sort_values('Timestamp')
        
        # Plotting a more compatible visualization - bar chart by time
        fig = px.bar(
            history_df,
            x='Timestamp',
            y='Action',
            color='Action',
            hover_data=['Details'],
            title="Project Timeline"
        )
        
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(xaxis_title="Time", yaxis_title="Action")
        st.plotly_chart(fig, use_container_width=True)
    
    # Data export options
    st.subheader("Export Project Data")
    
    export_type = st.radio(
        "Select export format",
        ["JSON", "CSV", "SDF"],
        horizontal=True
    )
    
    export_content = st.multiselect(
        "Select content to export",
        ["Proteins", "Pockets", "Pharmacophores", "Screening Results", "All"],
        default=["All"]
    )
    
    if st.button("Export Data"):
        if "All" in export_content:
            export_content = ["Proteins", "Pockets", "Pharmacophores", "Screening Results"]
        
        export_data = {}
        
        if "Proteins" in export_content:
            # Can't easily serialize the protein data, so just export metadata
            protein_meta = {}
            for protein_id, protein_data in st.session_state.analyzed_proteins.items():
                protein_meta[protein_id] = {
                    "pockets_count": len(protein_data.get('pockets', [])),
                    "file_type": protein_data.get('file_type', '')
                }
            export_data["proteins"] = protein_meta
        
        if "Pockets" in export_content:
            pockets_data = {}
            for protein_id, protein_data in st.session_state.analyzed_proteins.items():
                if 'pockets' in protein_data:
                    pockets_data[protein_id] = [
                        {k: v for k, v in pocket.items() if k != 'nearby_residues'}  # Exclude large nearby_residues list
                        for pocket in protein_data['pockets']
                    ]
            export_data["pockets"] = pockets_data
        
        if "Pharmacophores" in export_content:
            pharmacophore_data = {}
            for model_id, model_data in st.session_state.pharmacophores.items():
                pharmacophore_data[model_id] = {
                    "model": model_data['model'],
                    "timestamp": model_data['timestamp']
                }
            export_data["pharmacophores"] = pharmacophore_data
        
        if "Screening Results" in export_content:
            # Export screening results metadata (not molecules)
            screening_data = {}
            for screen_id, screen_data in st.session_state.screened_compounds.items():
                results_meta = []
                for hit in screen_data['results']:
                    results_meta.append({
                        "name": hit['name'],
                        "match_score": hit['match_score'],
                        "descriptors": hit['descriptors']
                    })
                
                screening_data[screen_id] = {
                    "model": screen_data['model'],
                    "timestamp": screen_data['timestamp'],
                    "hit_count": len(screen_data['results']),
                    "hits": results_meta
                }
            export_data["screening"] = screening_data
        
        # Export based on selected format
        if export_type == "JSON":
            export_json = json.dumps(export_data, indent=2)
            st.download_button(
                "Download JSON",
                export_json,
                file_name=f"drug_discovery_export_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
        
        elif export_type == "CSV":
            # Create separate CSV files for different data types
            for data_type, data in export_data.items():
                if data_type == "screening" and "Screening Results" in export_content:
                    # Flatten screening results
                    rows = []
                    for screen_id, screen_info in data.items():
                        for hit in screen_info['hits']:
                            row = {
                                "screen_id": screen_id,
                                "model": screen_info['model'],
                                "timestamp": screen_info['timestamp'],
                                "compound_name": hit['name'],
                                "match_score": hit['match_score']
                            }
                            
                            # Add descriptors
                            for desc_name, desc_val in hit['descriptors'].items():
                                row[f"desc_{desc_name}"] = desc_val
                            
                            rows.append(row)
                    
                    if rows:
                        df = pd.DataFrame(rows)
                        csv = df.to_csv(index=False)
                        st.download_button(
                            f"Download {data_type}.csv",
                            csv,
                            file_name=f"{data_type}_export_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
                
                elif data_type == "pockets" and "Pockets" in export_content:
                    # Flatten pocket data
                    rows = []
                    for protein_id, pockets in data.items():
                        for pocket in pockets:
                            row = {
                                "protein_id": protein_id,
                                "pocket_id": pocket['id'],
                                "method": pocket['method'],
                                "volume": pocket['volume'],
                                "residue_count": pocket['residue_count'],
                                "hydrophobicity": pocket['hydrophobicity']
                            }
                            rows.append(row)
                    
                    if rows:
                        df = pd.DataFrame(rows)
                        csv = df.to_csv(index=False)
                        st.download_button(
                            f"Download {data_type}.csv",
                            csv,
                            file_name=f"{data_type}_export_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
        
        elif export_type == "SDF" and "Screening Results" in export_content:
            # Export hit compounds from screening
            all_hit_mols = []
            
            for screen_id, screen_data in st.session_state.screened_compounds.items():
                for hit in screen_data['results']:
                    mol = hit['mol']
                    if mol:
                        # Add properties
                        mol.SetProp("ScreenID", screen_id)
                        mol.SetProp("MatchScore", str(hit['match_score']))
                        
                        for desc_name, desc_val in hit['descriptors'].items():
                            mol.SetProp(f"Desc_{desc_name}", str(desc_val))
                        
                        all_hit_mols.append(mol)
            
            if all_hit_mols:
                # Write to temporary SDF file
                temp_file = "temp_export.sdf"
                with Chem.SDWriter(temp_file) as writer:
                    for mol in all_hit_mols:
                        writer.write(mol)
                
                with open(temp_file, "rb") as f:
                    sdf_bytes = f.read()
                    st.download_button(
                        "Download Compounds SDF",
                        sdf_bytes,
                        file_name=f"screening_hits_{datetime.now().strftime('%Y%m%d')}.sdf",
                        mime="chemical/x-mdl-sdfile"
                    )
                
                # Clean up temp file
                os.remove(temp_file)
    
    # Visualization of top compounds
    if st.session_state.screened_compounds:
        st.subheader("Top Compounds Across All Screenings")
        
        # Collect top hits from all screenings
        all_hits = []
        for screen_id, screen_data in st.session_state.screened_compounds.items():
            for hit in screen_data['results'][:5]:  # Get top 5 from each screening
                if hit['match_score'] > 60:  # Only consider good matches
                    all_hits.append({
                        'screen_id': screen_id,
                        'model': screen_data['model'],
                        'mol': hit['mol'],
                        'name': hit['name'],
                        'score': hit['match_score'],
                        'descriptors': hit['descriptors']
                    })
        
        # Sort by match score
        all_hits.sort(key=lambda x: x['score'], reverse=True)
        
        # Take top 10
        top_hits = all_hits[:10]
        
        if top_hits:
            # Create a grid of molecules
            top_mols = [hit['mol'] for hit in top_hits]
            top_legends = [f"{hit['name']}\nScore: {hit['score']:.1f}%" for hit in top_hits]
            
            img = Draw.MolsToGridImage(
                top_mols,
                molsPerRow=5,
                subImgSize=(250, 250),
                legends=top_legends
            )
            st.image(img, caption="Top Compounds Across All Screenings")
            
            # Property space of all hits
            st.subheader("Chemical Space of Hits")
            
            # Collect descriptor data for all hits
            space_data = []
            for hit in all_hits:
                desc = hit['descriptors']
                space_data.append({
                    'Name': hit['name'],
                    'Model': hit['model'],
                    'Score': hit['score'],
                    'MW': desc['MW'],
                    'LogP': desc['LogP'],
                    'TPSA': desc['TPSA'],
                    'HBD': desc['HBD'],
                    'HBA': desc['HBA'],
                    'QED': desc['QED'],
                    'Fsp3': desc['Fsp3']
                })
            
            space_df = pd.DataFrame(space_data)
            
            # Create 3D scatter plot
            fig = px.scatter_3d(
                space_df,
                x='MW',
                y='LogP',
                z='TPSA',
                color='Score',
                size='QED',
                hover_name='Name',
                hover_data=['HBD', 'HBA', 'Fsp3'],
                color_continuous_scale=selected_theme,
                title="Chemical Space of Hits"
            )
            
            fig.update_layout(
                scene=dict(
                    xaxis_title="Molecular Weight",
                    yaxis_title="LogP",
                    zaxis_title="TPSA"
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center;">
    <h3>Advanced Drug Discovery Pipeline</h3>
    <p>Built with Streamlit and RDKit for computational drug discovery research</p>
    <p>Version 2.0.0 | © 2025</p>
</div>
""", unsafe_allow_html=True)

# Add a button to download the full application code
st.download_button(
    "Download Application Code",
    __file__,
    file_name="drug_discovery_pipeline.py",
    mime="text/plain"
)