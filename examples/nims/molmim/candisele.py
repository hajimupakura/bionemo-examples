import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from io import BytesIO
import json
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, Descriptors, Lipinski, QED
from rdkit.Chem.Draw import rdMolDraw2D
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage

# Set page configuration
st.set_page_config(
    page_title="Drug Candidate Prioritization",
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
        font-size: 1.8rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .molecule-card {
        background-color: white;
        border-radius: 0.5rem;
        padding: 1rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        color: #6c757d;
        font-size: 0.8rem;
    }
    .highlight {
        background-color: #ffffcc;
        padding: 0.2rem;
        border-radius: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def mol_to_img(mol, molSize=(300, 200)):
    """Convert RDKit molecule to SVG image string"""
    if mol is None:
        return None
    
    # Generate 2D coordinates if needed
    if mol.GetNumConformers() == 0:
        AllChem.Compute2DCoords(mol)
    
    # Create drawer and set options
    drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0], molSize[1])
    drawer.SetFontSize(0.8)
    
    # Draw molecule
    opts = rdMolDraw2D.MolDrawOptions()
    opts.addAtomIndices = False
    opts.addStereoAnnotation = True
    drawer.SetDrawOptions(opts)
    
    # Draw molecule and get SVG
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    
    return svg

def svg_to_html(svg_str):
    """Convert SVG string to HTML for display"""
    return f'<div style="text-align: center;">{svg_str}</div>'

def calculate_molecular_descriptors(mol):
    """Calculate key molecular descriptors using RDKit"""
    if mol is None:
        return None
    
    descriptors = {
        'MW': round(Descriptors.MolWt(mol), 2),
        'LogP': round(Descriptors.MolLogP(mol), 2),
        'TPSA': round(Descriptors.TPSA(mol), 2),
        'HBA': Descriptors.NumHAcceptors(mol),
        'HBD': Descriptors.NumHDonors(mol),
        'RotBonds': Descriptors.NumRotatableBonds(mol),
        'ArRings': Chem.Lipinski.NumAromaticRings(mol),
        'QED': round(QED.qed(mol), 3),
        'HeavyAtoms': mol.GetNumHeavyAtoms(),
        'Rings': Chem.Lipinski.RingCount(mol),
    }
    
    return descriptors

def lipinski_violations(mol):
    """Count Lipinski's Rule of Five violations"""
    if mol is None:
        return 0
    
    violations = 0
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hba = Descriptors.NumHAcceptors(mol)
    hbd = Descriptors.NumHDonors(mol)
    
    if mw > 500: violations += 1
    if logp > 5: violations += 1
    if hba > 10: violations += 1
    if hbd > 5: violations += 1
    
    return violations

def calculate_fingerprints(mol):
    """Calculate Morgan fingerprints for a molecule"""
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

def calculate_fingerprint_similarity(fp1, fp2):
    """Calculate Tanimoto similarity between fingerprints"""
    if fp1 is None or fp2 is None:
        return 0.0
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def calculate_synthetic_accessibility(mol):
    """Calculate synthetic accessibility score (1-10, lower is better)"""
    # This is a simplified implementation - in practice you'd use SA_Score or similar
    try:
        # Count complexity factors
        num_rings = Chem.Lipinski.RingCount(mol)
        num_hetero_atoms = Descriptors.NumHeteroatoms(mol)
        num_stereo_centers = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        
        # Calculate base score (higher for more complex molecules)
        base_score = 1.0 + 0.25 * num_rings + 0.15 * num_hetero_atoms + 0.4 * num_stereo_centers
        
        # Scale to 1-10 range
        score = min(10, max(1, base_score))
        
        return score
    except:
        return 5.0  # Default mid-range score if calculation fails

def calculate_admet_risk(mol_descriptors):
    """Estimate ADMET risk based on molecular descriptors"""
    # Simplified method - real implementation would use QSAR models
    risk_score = 0
    
    # Extract properties
    mw = mol_descriptors['MW']
    logp = mol_descriptors['LogP']
    tpsa = mol_descriptors['TPSA']
    rot_bonds = mol_descriptors['RotBonds']
    
    # Property risk assessment
    if mw > 500:
        risk_score += (mw - 500) / 100
    
    if logp > 5:
        risk_score += (logp - 5) * 0.5
    elif logp < 1:
        risk_score += (1 - logp) * 0.5
    
    if tpsa > 140:
        risk_score += (tpsa - 140) / 20
    elif tpsa < 50:
        risk_score += (50 - tpsa) / 10
    
    if rot_bonds > 10:
        risk_score += (rot_bonds - 10) * 0.2
    
    # Scale to 0-10
    risk_score = min(10, max(0, risk_score))
    
    # Risk categories
    if risk_score < 3:
        risk_category = "Low"
    elif risk_score < 7:
        risk_category = "Medium"
    else:
        risk_category = "High"
    
    return risk_score, risk_category

def calculate_mpo_score(props, weights, target_ranges):
    """
    Calculate Multi-Parameter Optimization (MPO) score
    
    Parameters:
    props (dict): Dictionary of property values
    weights (dict): Dictionary of weights for each property
    target_ranges (dict): Dictionary with ideal ranges for each property
    
    Returns:
    float: Weighted MPO score (0-10, higher is better)
    """
    mpo_score = 0
    total_weight = sum(weights.values())
    
    for prop, value in props.items():
        if prop in weights and prop in target_ranges:
            # Get target range for this property
            t_min, t_max, hard_min, hard_max = target_ranges[prop]
            weight = weights[prop] / total_weight
            
            # Calculate property score (0-10)
            if value < hard_min or value > hard_max:
                # Outside hard limits - zero score
                prop_score = 0
            elif t_min <= value <= t_max:
                # In ideal range - full score
                prop_score = 10
            elif value < t_min:
                # Between hard minimum and ideal minimum
                prop_score = 10 * (value - hard_min) / (t_min - hard_min)
            else:
                # Between ideal maximum and hard maximum
                prop_score = 10 * (hard_max - value) / (hard_max - t_max)
            
            # Add weighted property score
            mpo_score += weight * prop_score
    
    return mpo_score

def calculate_scores_for_dataframe(df, weights, target_ranges):
    """Calculate MPO scores for all compounds in a DataFrame"""
    mpo_scores = []
    
    for idx, row in df.iterrows():
        # Create properties dictionary
        props = {
            'MW': row['MW'],
            'LogP': row['LogP'],
            'TPSA': row['TPSA'],
            'HBA': row['HBA'],
            'HBD': row['HBD'],
            'RotBonds': row['RotBonds'],
            'QED': row['QED'],
            'Efficacy': row['Efficacy'],
            'Safety': row['Safety'],
            'ADMET': row['ADMET'],
            'Synthetic_Feasibility': row['Synthetic_Feasibility'],
            'Selectivity': row['Selectivity'],
            'Synthetic_Accessibility': row['Synthetic_Accessibility'],
        }
        
        # Calculate MPO score
        score = calculate_mpo_score(props, weights, target_ranges)
        mpo_scores.append(score)
    
    return mpo_scores

def create_radar_chart(categories, values, max_values, title="Property Profile"):
    """Create a radar chart for molecule properties"""
    # Normalize values
    norm_values = [v/m if m > 0 else 0 for v, m in zip(values, max_values)]
    
    # Complete the loop for the radar chart
    categories = categories + [categories[0]]
    norm_values = norm_values + [norm_values[0]]
    
    # Create figure
    fig = go.Figure()
    
    # Add trace
    fig.add_trace(go.Scatterpolar(
        r=norm_values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(67, 147, 195, 0.3)',
        line=dict(color='rgb(67, 147, 195)', width=2)
    ))
    
    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False,
        title=title
    )
    
    return fig

def create_property_distribution_plot(df, property_name, highlight_compounds=None):
    """Create histogram of property distribution with optional highlighting"""
    fig = px.histogram(
        df, 
        x=property_name, 
        nbins=20,
        opacity=0.7,
        title=f"Distribution of {property_name}",
        color_discrete_sequence=['rgba(67, 147, 195, 0.7)']
    )
    
    # Add vertical lines for top compounds if provided
    if highlight_compounds is not None:
        for i, (_, compound) in enumerate(highlight_compounds.iterrows()):
            fig.add_vline(
                x=compound[property_name],
                line_dash="dash",
                line_color=["red", "blue", "green"][i % 3],
                annotation_text=compound["Name"],
                annotation_position="top right"
            )
    
    return fig

def create_parallel_coordinates_plot(df, params):
    """Create parallel coordinates plot for multi-parameter visualization"""
    # Create plot
    fig = px.parallel_coordinates(
        df,
        dimensions=params,
        color='MPO_Score',
        color_continuous_scale=px.colors.sequential.Viridis,
        title="Multi-Parameter Visualization of Drug Candidates",
        range_color=[df['MPO_Score'].min(), df['MPO_Score'].max()]
    )
    
    # Update layout
    fig.update_layout(
        font=dict(size=12),
        margin=dict(l=80, r=80, t=50, b=50)
    )
    
    return fig

def create_heatmap(df, properties):
    """Create property heatmap of compounds"""
    # Prepare data for heatmap
    heatmap_data = df[properties].copy()
    
    # Normalize each column to 0-1 range for better visualization
    scaler = MinMaxScaler()
    heatmap_data_scaled = pd.DataFrame(
        scaler.fit_transform(heatmap_data),
        columns=heatmap_data.columns,
        index=heatmap_data.index
    )
    
    # Add names for index
    heatmap_data_scaled.index = df['Name']
    
    # Create heatmap
    fig = px.imshow(
        heatmap_data_scaled.T,
        color_continuous_scale='Viridis',
        labels=dict(x="Compound", y="Property", color="Value"),
        x=heatmap_data_scaled.index,
        y=heatmap_data_scaled.columns,
        title="Compound-Property Heatmap"
    )
    
    fig.update_layout(
        height=600,
        xaxis={'side': 'top'}
    )
    
    # Rotate x-axis labels
    fig.update_xaxes(tickangle=45)
    
    return fig

def create_scatter_plot(df, x_property, y_property, color_property=None):
    """Create scatter plot for visualizing relationships between properties"""
    if color_property:
        fig = px.scatter(
            df,
            x=x_property,
            y=y_property,
            color=color_property,
            hover_name="Name",
            title=f"{y_property} vs {x_property}, colored by {color_property}",
            color_continuous_scale=px.colors.sequential.Viridis
        )
    else:
        fig = px.scatter(
            df,
            x=x_property,
            y=y_property,
            hover_name="Name",
            title=f"{y_property} vs {x_property}"
        )
    
    fig.update_traces(marker=dict(size=10))
    
    return fig

def load_sample_data():
    """Load sample molecular data for demonstration"""
    # SMILES strings for some approved drugs and drug-like compounds
    smiles_list = [
        # Approved drugs
        ("Ibuprofen", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", 8.2, 9.3, 7.5, 8.9, 6.4),
        ("Aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O", 7.5, 8.1, 8.9, 9.2, 9.5),
        ("Atorvastatin", "CC(C)C1=C(C=C(C=C1)C(C)C)C(=O)NC(CC(C)C)C(=O)NC(CC(O)=O)C(O)CC(O)CC(O)=O", 8.0, 7.5, 6.8, 5.2, 4.9),
        ("Loratadine", "CCOC(=O)N1CCC(=C2C3=CC=CC=C3CCN2CC)CC1", 8.6, 7.0, 7.4, 7.9, 8.3),
        ("Metformin", "CN(C)C(=N)NC(=N)N", 6.5, 8.0, 8.5, 9.0, 7.5),
        
        # Drug candidates (fictional)
        ("Candidate-1", "COC1=CC=C(CC2=NC=CS2)C=C1OC", 7.2, 7.8, 8.1, 8.5, 8.0),
        ("Candidate-2", "CN(C)C(=O)C1=CC(=C(O)C(=C1)I)C(=O)NCCN", 6.8, 7.5, 8.2, 7.9, 6.5),
        ("Candidate-3", "CC1=C(C=CC(=C1)S(=O)(=O)N)CC(C(=O)O)N", 7.5, 8.0, 7.3, 7.7, 7.9),
        ("Candidate-4", "FC1=CC=C(CN2C=NC=N2)C=C1C1=CC=C(Cl)C=C1", 8.3, 7.9, 6.8, 7.1, 8.5),
        ("Candidate-5", "CC1(C)CC(NC(=O)COC2=CC=C(Cl)C=C2)CC(C)(C)N1", 7.7, 8.3, 7.6, 8.2, 7.9),
        ("Candidate-6", "FC(F)(F)C1=CC=C(NC(=O)C2CCCN2)C=C1", 8.9, 7.8, 6.5, 8.0, 9.2),
        ("Candidate-7", "CC1=NC=C(C=C1C)C(=O)NC(C)C1=NN=C(C)S1", 6.9, 7.5, 8.0, 8.5, 7.2),
        ("Candidate-8", "CN1C(=O)CC(C1=O)N1CCN(CC1)C1=CC=CC=C1", 7.8, 8.4, 7.9, 6.7, 7.3),
        ("Candidate-9", "CC(C)N1C=NC2=C1N=C(NCC1=CC=C(F)C=C1)N=C2", 8.5, 7.1, 7.8, 8.3, 6.9),
        ("Candidate-10", "CN1CCN(CC1)C1=CC=C(NC(=O)C2=CC=C(F)C=C2)C=C1", 8.2, 8.7, 7.5, 7.8, 8.1),
        ("Candidate-11", "COC1=CC=C(CNC(=O)C2=CC(=NN2C)C2=CC=CC=C2)C=C1", 7.3, 6.8, 7.9, 8.4, 7.6),
        ("Candidate-12", "CC1=C(C(=O)NC2=CC=CC=C2)C(C)=NN1C1=CC=C(C)C=C1", 6.5, 7.2, 8.3, 7.7, 6.8),
        ("Candidate-13", "CC(=O)NC1=NC2=C(C=C1)N=C(NC1=CC=C(F)C=C1)N2", 8.7, 7.6, 6.9, 7.4, 8.5),
        ("Candidate-14", "CCC1=NN=C(NC2=CC=C(C=C2)S(=O)(=O)NC)S1", 7.0, 7.9, 8.5, 8.2, 7.4),
        ("Candidate-15", "CCOC(=O)C1=CN=C(NC2=CC=C(OC)C=C2)S1", 8.4, 8.8, 7.6, 7.9, 8.3),
    ]
    
    # Convert to list of dictionaries with RDKit molecules
    molecules = []
    for name, smiles, efficacy, safety, admet, synth, select in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            # Calculate descriptors
            descriptors = calculate_molecular_descriptors(mol)
            
            # Synthetic accessibility and ADMET risk
            synth_score = calculate_synthetic_accessibility(mol)
            admet_risk, risk_category = calculate_admet_risk(descriptors)
            
            # Create molecule dict
            molecule = {
                'Name': name,
                'SMILES': smiles,
                'ROMol': mol,
                'MW': descriptors['MW'],
                'LogP': descriptors['LogP'],
                'TPSA': descriptors['TPSA'],
                'HBA': descriptors['HBA'],
                'HBD': descriptors['HBD'],
                'RotBonds': descriptors['RotBonds'],
                'QED': descriptors['QED'],
                'Lipinski_Violations': lipinski_violations(mol),
                'ArRings': descriptors['ArRings'],
                'Efficacy': efficacy,  # 0-10 scale, higher is better
                'Safety': safety,      # 0-10 scale, higher is better
                'ADMET': admet,        # 0-10 scale, higher is better
                'Synthetic_Feasibility': synth,  # 0-10 scale, higher is better
                'Selectivity': select,  # 0-10 scale, higher is better
                'ADMET_Risk': admet_risk,
                'ADMET_Risk_Category': risk_category,
                'Synthetic_Accessibility': synth_score,  # 1-10 scale, lower is better
            }
            molecules.append(molecule)
    
    # Create DataFrame
    df = pd.DataFrame(molecules)
    
    # Calculate fingerprints
    fps = [calculate_fingerprints(mol) for mol in df['ROMol']]
    
    # Add fingerprint to dataframe
    df['Fingerprint'] = fps
    
    return df

def export_candidate_report(df, top_n=5, format="csv"):
    """Export prioritized candidates report"""
    # Get top candidates
    top_candidates = df.sort_values('MPO_Score', ascending=False).head(top_n)
    
    if format == "csv":
        # Prepare CSV export
        csv_buffer = BytesIO()
        
        # Export candidate data without ROMol and Fingerprint columns
        export_df = top_candidates.drop(columns=['ROMol', 'Fingerprint'])
        export_df.to_csv(csv_buffer, index=False)
        
        return csv_buffer.getvalue()
    
    elif format == "excel":
        # Prepare Excel export
        excel_buffer = BytesIO()
        
        # Export without ROMol and Fingerprint
        export_df = top_candidates.drop(columns=['ROMol', 'Fingerprint'])
        export_df.to_excel(excel_buffer, index=False, engine='openpyxl')
        
        excel_buffer.seek(0)
        return excel_buffer.getvalue()
    
    elif format == "json":
        # Prepare JSON export (without ROMol and Fingerprint)
        export_df = top_candidates.drop(columns=['ROMol', 'Fingerprint'])
        
        # Convert to records format
        records = export_df.to_dict(orient='records')
        
        # Add metadata
        export_data = {
            "report_type": "Prioritized Drug Candidates",
            "generated_on": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            "candidates": records
        }
        
        return json.dumps(export_data, indent=2)
    
    else:
        return None

# Function to create detailed candidate card
def create_candidate_card(compound):
    """Create a detailed card view for a candidate molecule"""
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Display molecule structure
        mol = compound['ROMol']
        svg = mol_to_img(mol)
        st.markdown(svg_to_html(svg), unsafe_allow_html=True)
    
    with col2:
        # Display basic info
        st.markdown(f"### {compound['Name']}")
        st.markdown(f"**SMILES:** `{compound['SMILES']}`")
        st.markdown(f"**MPO Score:** {compound['MPO_Score']:.2f}/10")
        
        # Create three columns for properties
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.markdown("**Physical Properties:**")
            st.markdown(f"MW: {compound['MW']:.1f}")
            st.markdown(f"LogP: {compound['LogP']:.2f}")
            st.markdown(f"TPSA: {compound['TPSA']:.1f} Å²")
            st.markdown(f"HBA/HBD: {int(compound['HBA'])}/{int(compound['HBD'])}")
        
        with col_b:
            st.markdown("**Drug-likeness:**")
            st.markdown(f"QED: {compound['QED']:.2f}")
            st.markdown(f"Lipinski Violations: {int(compound['Lipinski_Violations'])}")
            st.markdown(f"Rot. Bonds: {int(compound['RotBonds'])}")
            st.markdown(f"Arom. Rings: {int(compound['ArRings'])}")
        
        with col_c:
            st.markdown("**Performance Metrics:**")
            st.markdown(f"Efficacy: {compound['Efficacy']:.1f}/10")
            st.markdown(f"Safety: {compound['Safety']:.1f}/10")
            st.markdown(f"ADMET: {compound['ADMET']:.1f}/10")
            st.markdown(f"Selectivity: {compound['Selectivity']:.1f}/10")
    
    # Create expandable sections for more details
    with st.expander("Detailed Assessment"):
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("**Synthetic Feasibility:**")
            st.progress(compound['Synthetic_Feasibility']/10)
            st.markdown(f"{compound['Synthetic_Feasibility']:.1f}/10 (higher is better)")
            
            st.markdown("**Synthetic Accessibility Score:**")
            st.progress((10 - compound['Synthetic_Accessibility'])/10)
            st.markdown(f"{compound['Synthetic_Accessibility']:.1f}/10 (lower is better)")
        
        with col_right:
            st.markdown("**ADMET Risk Assessment:**")
            risk_color = {"Low": "green", "Medium": "orange", "High": "red"}[compound['ADMET_Risk_Category']]
            st.markdown(f"Risk Category: <span style='color:{risk_color};font-weight:bold;'>{compound['ADMET_Risk_Category']}</span>", unsafe_allow_html=True)
            st.progress(1 - compound['ADMET_Risk']/10)
            st.markdown(f"Risk Score: {compound['ADMET_Risk']:.1f}/10 (lower is better)")
    
    st.markdown("---")

# Initialize session state for persistent data
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

if 'compounds_df' not in st.session_state:
    st.session_state.compounds_df = None

if 'mpo_weights' not in st.session_state:
    st.session_state.mpo_weights = {
        'Efficacy': 5.0,
        'Safety': 5.0,
        'ADMET': 3.0,
        'Synthetic_Feasibility': 2.0,
        'Selectivity': 4.0,
        'MW': 1.0,
        'LogP': 1.0,
        'TPSA': 1.0,
        'QED': 2.0,
        'HBA': 0.5,
        'HBD': 0.5,
        'RotBonds': 0.5,
        'Synthetic_Accessibility': 2.0,
    }

if 'target_ranges' not in st.session_state:
    st.session_state.target_ranges = {
        # Format: (ideal_min, ideal_max, hard_min, hard_max)
        'MW': (300, 500, 150, 700),
        'LogP': (1, 4, -1, 6),
        'TPSA': (60, 140, 20, 200),
        'HBA': (2, 7, 0, 15),
        'HBD': (1, 4, 0, 8),
        'RotBonds': (2, 8, 0, 15),
        'QED': (0.5, 1.0, 0, 1.0),
        'Efficacy': (7, 10, 5, 10),
        'Safety': (7, 10, 5, 10),
        'ADMET': (7, 10, 5, 10),
        'Synthetic_Feasibility': (6, 10, 3, 10),
        'Selectivity': (7, 10, 5, 10),
        'Synthetic_Accessibility': (1, 5, 1, 10),
    }

if 'calculated_scores' not in st.session_state:
    st.session_state.calculated_scores = False

# App title and description
st.markdown("<h1 class='main-header'>Drug Candidate Selection & Prioritization</h1>", unsafe_allow_html=True)

st.markdown("""
This application helps you identify and prioritize the most promising drug candidates for experimental validation.
It uses multi-parameter optimization (MPO) to balance potency, selectivity, ADMET properties, and synthetic feasibility.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Select Application Mode", 
    ["Data Import", "Parameter Setup", "Candidate Analysis", "Visualization", "Report Generation"])

# Data Import Mode
if app_mode == "Data Import":
    st.markdown("<h2 class='sub-header'>Data Import</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    Import your drug candidate data in various formats or use our example dataset.
    Make sure your data contains SMILES structures and relevant molecular properties.
    """)
    
    import_method = st.radio(
        "Select import method",
        ["Upload file", "Use example data", "Paste SMILES"]
    )
    
    if import_method == "Upload file":
        uploaded_file = st.file_uploader("Upload your file", type=["csv", "sdf", "xlsx"])
        
        if uploaded_file is not None:
            st.info("File upload support would be implemented here for CSV, SDF, or Excel files.")
            st.warning("For this demonstration, please use the 'Example Data' option.")
    
    elif import_method == "Use example data":
        # Load sample data
        if st.button("Load Example Data"):
            df = load_sample_data()
            st.success(f"Successfully loaded {len(df)} example compounds.")
            
            # Store in session state
            st.session_state.compounds_df = df
            st.session_state.data_loaded = True
            
            # Display sample of the data
            st.markdown("### Example Data Preview:")
            display_cols = ['Name', 'SMILES', 'MW', 'LogP', 'QED', 'Efficacy', 'Safety', 'ADMET', 'Selectivity']
            st.dataframe(df[display_cols].head())
    
    elif import_method == "Paste SMILES":
        st.markdown("Enter SMILES strings, one per line:")
        smiles_text = st.text_area("SMILES Input", 
            """CC(C)CC1=CC=C(C=C1)C(C)C(=O)O Ibuprofen
CC(=O)OC1=CC=CC=C1C(=O)O Aspirin
CN1C(=O)N(C)C(=O)C(N(C)C=N2)=C12 Caffeine
CC12CCC3C(CCC4=CC(=O)CCC34C)C1CCC2O Testosterone""")
        
        if st.button("Process SMILES"):
            st.info("SMILES processing would be implemented here.")
            st.warning("For this demonstration, please use the 'Example Data' option.")
    
    # Data inspection if loaded
    if st.session_state.data_loaded and st.session_state.compounds_df is not None:
        df = st.session_state.compounds_df
        
        st.markdown("<h3 class='sub-header'>Data Summary</h3>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Compounds", len(df))
        with col2:
            unique_smiles = df['SMILES'].nunique()
            st.metric("Unique Structures", unique_smiles)
        with col3:
            mw_range = f"{df['MW'].min():.1f} - {df['MW'].max():.1f}"
            st.metric("MW Range", mw_range)
        
        # Structure visualization
        with st.expander("View Structure Gallery"):
            # Show first few compounds
            n_cols = 4
            n_rows = 2
            n_mols = min(n_cols * n_rows, len(df))
            
            rows = []
            for i in range(0, n_mols, n_cols):
                cols = st.columns(n_cols)
                for j in range(n_cols):
                    if i + j < n_mols:
                        mol = df['ROMol'].iloc[i + j]
                        name = df['Name'].iloc[i + j]
                        with cols[j]:
                            svg = mol_to_img(mol)
                            st.markdown(f"<strong style='font-size:0.9em'>{name}</strong>", unsafe_allow_html=True)
                            st.markdown(svg_to_html(svg), unsafe_allow_html=True)
        
        # Property distribution visualization
        with st.expander("View Property Distributions"):
            prop_to_vis = st.selectbox(
                "Select property to visualize:", 
                ['MW', 'LogP', 'TPSA', 'QED', 'Efficacy', 'Safety', 'ADMET']
            )
            
            fig = create_property_distribution_plot(df, prop_to_vis)
            st.plotly_chart(fig, use_container_width=True)

# Parameter Setup Mode
elif app_mode == "Parameter Setup":
    st.markdown("<h2 class='sub-header'>Parameter Setup</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    Configure the multi-parameter optimization (MPO) settings to balance different criteria:
    
    - **Primary Activity Parameters:** Efficacy, selectivity, etc.
    - **ADMET Properties:** Absorption, distribution, metabolism, excretion, toxicity
    - **Physicochemical Properties:** MW, LogP, TPSA, etc.
    - **Synthetic Feasibility:** Ease of synthesis and scale-up
    """)
    
    if not st.session_state.data_loaded or st.session_state.compounds_df is None:
        st.warning("Please import your compound data first in the 'Data Import' tab.")
    else:
        df = st.session_state.compounds_df
        
        st.markdown("<h3 class='sub-header'>Parameter Weights</h3>", unsafe_allow_html=True)
        st.markdown("""
        Adjust the relative importance of each parameter (0-10 scale, higher is more important).
        These weights will be used to calculate the overall MPO score.
        """)
        
        # Create parameter weight adjustment interface
        st.markdown("#### Activity Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            efficacy_weight = st.slider(
                "Efficacy Weight", 
                min_value=0.0, 
                max_value=10.0, 
                value=st.session_state.mpo_weights['Efficacy'],
                step=0.5,
                help="Importance of target efficacy (e.g., binding affinity, IC50, EC50)"
            )
            st.session_state.mpo_weights['Efficacy'] = efficacy_weight
            
            safety_weight = st.slider(
                "Safety Weight", 
                min_value=0.0, 
                max_value=10.0, 
                value=st.session_state.mpo_weights['Safety'],
                step=0.5,
                help="Importance of safety profile and low toxicity potential"
            )
            st.session_state.mpo_weights['Safety'] = safety_weight
        
        with col2:
            admet_weight = st.slider(
                "ADMET Weight", 
                min_value=0.0, 
                max_value=10.0, 
                value=st.session_state.mpo_weights['ADMET'],
                step=0.5,
                help="Importance of good absorption, distribution, metabolism, excretion, toxicity"
            )
            st.session_state.mpo_weights['ADMET'] = admet_weight
            
            selectivity_weight = st.slider(
                "Selectivity Weight", 
                min_value=0.0, 
                max_value=10.0, 
                value=st.session_state.mpo_weights['Selectivity'],
                step=0.5,
                help="Importance of target selectivity (avoiding off-target effects)"
            )
            st.session_state.mpo_weights['Selectivity'] = selectivity_weight
        
        st.markdown("#### Physicochemical Parameters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            mw_weight = st.slider(
                "Molecular Weight Weight", 
                min_value=0.0, 
                max_value=10.0, 
                value=st.session_state.mpo_weights['MW'],
                step=0.5
            )
            st.session_state.mpo_weights['MW'] = mw_weight
            
            hba_weight = st.slider(
                "H-Bond Acceptors Weight", 
                min_value=0.0, 
                max_value=10.0, 
                value=st.session_state.mpo_weights['HBA'],
                step=0.5
            )
            st.session_state.mpo_weights['HBA'] = hba_weight
        
        with col2:
            logp_weight = st.slider(
                "LogP Weight", 
                min_value=0.0, 
                max_value=10.0, 
                value=st.session_state.mpo_weights['LogP'],
                step=0.5
            )
            st.session_state.mpo_weights['LogP'] = logp_weight
            
            hbd_weight = st.slider(
                "H-Bond Donors Weight", 
                min_value=0.0, 
                max_value=10.0, 
                value=st.session_state.mpo_weights['HBD'],
                step=0.5
            )
            st.session_state.mpo_weights['HBD'] = hbd_weight
        
        with col3:
            tpsa_weight = st.slider(
                "TPSA Weight", 
                min_value=0.0, 
                max_value=10.0, 
                value=st.session_state.mpo_weights['TPSA'],
                step=0.5
            )
            st.session_state.mpo_weights['TPSA'] = tpsa_weight
            
            rotb_weight = st.slider(
                "Rotatable Bonds Weight", 
                min_value=0.0, 
                max_value=10.0, 
                value=st.session_state.mpo_weights['RotBonds'],
                step=0.5
            )
            st.session_state.mpo_weights['RotBonds'] = rotb_weight
        
        st.markdown("#### Synthetic & Drug-likeness Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            synth_feas_weight = st.slider(
                "Synthetic Feasibility Weight", 
                min_value=0.0, 
                max_value=10.0, 
                value=st.session_state.mpo_weights['Synthetic_Feasibility'],
                step=0.5,
                help="Importance of ease of synthesis (higher score = easier synthesis)"
            )
            st.session_state.mpo_weights['Synthetic_Feasibility'] = synth_feas_weight
            
            synth_acc_weight = st.slider(
                "Synthetic Accessibility Weight", 
                min_value=0.0, 
                max_value=10.0, 
                value=st.session_state.mpo_weights['Synthetic_Accessibility'],
                step=0.5,
                help="Importance of synthetic accessibility score (1-10 scale, lower is better)"
            )
            st.session_state.mpo_weights['Synthetic_Accessibility'] = synth_acc_weight
        
        with col2:
            qed_weight = st.slider(
                "QED Drug-likeness Weight", 
                min_value=0.0, 
                max_value=10.0, 
                value=st.session_state.mpo_weights['QED'],
                step=0.5,
                help="Importance of Quantitative Estimate of Drug-likeness (0-1 scale, higher is better)"
            )
            st.session_state.mpo_weights['QED'] = qed_weight
        
        # Display total weight and normalized weights
        total_weight = sum(st.session_state.mpo_weights.values())
        st.info(f"Total weight sum: {total_weight:.1f}")
        
        # Option to normalize weights
        if st.button("Normalize Weights"):
            # Normalize to sum to 10
            norm_factor = 10.0 / total_weight
            for param in st.session_state.mpo_weights:
                st.session_state.mpo_weights[param] *= norm_factor
            st.success("Weights normalized to sum to 10!")
        
        # Preset profiles
        st.markdown("<h3 class='sub-header'>Preset Profiles</h3>", unsafe_allow_html=True)
        
        preset_col1, preset_col2 = st.columns([1, 2])
        
        with preset_col1:
            preset_profile = st.selectbox(
                "Select a preset profile",
                ["Custom", "Oral Drug", "Blood-Brain Barrier", "Metabolic Stability"]
            )
        
        with preset_col2:
            if st.button("Apply Preset"):
                if preset_profile == "Oral Drug":
                    # Update weights for oral drug profile
                    st.session_state.mpo_weights = {
                        'Efficacy': 5.0,
                        'Safety': 5.0,
                        'ADMET': 4.0,
                        'Synthetic_Feasibility': 2.0,
                        'Selectivity': 3.0,
                        'MW': 1.0,
                        'LogP': 1.5,
                        'TPSA': 1.5,
                        'QED': 2.0,
                        'HBA': 1.0,
                        'HBD': 1.0,
                        'RotBonds': 1.0,
                        'Synthetic_Accessibility': 2.0,
                    }
                    st.success("Applied 'Oral Drug' preset profile!")
                
                elif preset_profile == "Blood-Brain Barrier":
                    # Update weights for BBB profile
                    st.session_state.mpo_weights = {
                        'Efficacy': 4.0,
                        'Safety': 5.0,
                        'ADMET': 5.0,
                        'Synthetic_Feasibility': 2.0,
                        'Selectivity': 3.0,
                        'MW': 2.0,
                        'LogP': 2.0,
                        'TPSA': 3.0,
                        'QED': 1.0,
                        'HBA': 1.5,
                        'HBD': 1.5,
                        'RotBonds': 1.0,
                        'Synthetic_Accessibility': 2.0,
                    }
                    st.success("Applied 'Blood-Brain Barrier' preset profile!")
                
                elif preset_profile == "Metabolic Stability":
                    # Update weights for metabolic stability profile
                    st.session_state.mpo_weights = {
                        'Efficacy': 4.0,
                        'Safety': 3.0,
                        'ADMET': 6.0,
                        'Synthetic_Feasibility': 2.0,
                        'Selectivity': 3.0,
                        'MW': 1.0,
                        'LogP': 2.0,
                        'TPSA': 1.0,
                        'QED': 1.0,
                        'HBA': 0.5,
                        'HBD': 0.5,
                        'RotBonds': 2.0,
                        'Synthetic_Accessibility': 1.0,
                    }
                    st.success("Applied 'Metabolic Stability' preset profile!")
        
        # Calculate MPO scores
        st.markdown("<h3 class='sub-header'>Calculate Multi-Parameter Optimization Scores</h3>", unsafe_allow_html=True)
        
        if st.button("Calculate MPO Scores"):
            # Calculate scores
            df = st.session_state.compounds_df
            mpo_scores = calculate_scores_for_dataframe(df, st.session_state.mpo_weights, st.session_state.target_ranges)
            
            # Add scores to DataFrame
            df['MPO_Score'] = mpo_scores
            
            # Store in session state
            st.session_state.compounds_df = df
            st.session_state.calculated_scores = True
            
            st.success(f"MPO scores calculated for {len(df)} compounds!")
            
            # Show top compounds
            st.markdown("### Top 5 Compounds by MPO Score")
            top_compounds = df.sort_values('MPO_Score', ascending=False).head(5)
            
            # Display as table
            display_cols = ['Name', 'MPO_Score', 'Efficacy', 'Safety', 'ADMET', 'Selectivity', 'Synthetic_Feasibility']
            st.dataframe(top_compounds[display_cols].style.highlight_max(axis=0, subset=['MPO_Score']))

# Candidate Analysis Mode
elif app_mode == "Candidate Analysis":
    st.markdown("<h2 class='sub-header'>Candidate Analysis</h2>", unsafe_allow_html=True)
    
    if not st.session_state.data_loaded or st.session_state.compounds_df is None:
        st.warning("Please import your compound data first in the 'Data Import' tab.")
    elif not st.session_state.calculated_scores:
        st.warning("Please calculate MPO scores first in the 'Parameter Setup' tab.")
    else:
        df = st.session_state.compounds_df
        
        # Sort by MPO score
        df_sorted = df.sort_values('MPO_Score', ascending=False).reset_index(drop=True)
        
        st.markdown("### Prioritized Candidates")
        
        # Select number of top candidates to display
        top_n = st.slider("Number of top candidates to display", 1, min(10, len(df)), 3)
        
        # Get top compounds
        top_compounds = df_sorted.head(top_n)
        
        # Display top compounds with detailed cards
        for i, (_, compound) in enumerate(top_compounds.iterrows()):
            st.markdown(f"## {i+1}. {compound['Name']} (Score: {compound['MPO_Score']:.2f}/10)")
            create_candidate_card(compound)
        
        # Property comparison
        st.markdown("### Comparative Analysis")
        
        # Select specific compounds for comparison
        st.markdown("#### Select Compounds for Detailed Comparison")
        
        selected_compounds = st.multiselect(
            "Select compounds to compare",
            options=df['Name'].tolist(),
            default=top_compounds['Name'].tolist()[:3]
        )
        
        if selected_compounds:
            selected_df = df[df['Name'].isin(selected_compounds)]
            
            # Create radar chart for selected parameters
            st.markdown("#### Property Radar Chart")
            
            # Select parameters for radar chart
            radar_params = st.multiselect(
                "Select parameters for radar chart",
                options=['Efficacy', 'Safety', 'ADMET', 'Selectivity', 'Synthetic_Feasibility', 
                        'MW', 'LogP', 'TPSA', 'QED', 'Synthetic_Accessibility'],
                default=['Efficacy', 'Safety', 'ADMET', 'Selectivity', 'Synthetic_Feasibility']
            )
            
            if radar_params:
                # Create radar chart for each compound
                for i, (_, compound) in enumerate(selected_df.iterrows()):
                    # Get values for selected parameters
                    values = [compound[param] for param in radar_params]
                    
                    # For Synthetic_Accessibility, invert the scale (lower is better)
                    for j, param in enumerate(radar_params):
                        if param == 'Synthetic_Accessibility':
                            values[j] = 10 - values[j]  # Invert scale
                    
                    # Max values for normalization
                    max_values = [10] * len(radar_params)  # All parameters on 0-10 scale
                    
                    # Create radar chart
                    fig = create_radar_chart(
                        radar_params, 
                        values, 
                        max_values, 
                        title=f"Property Profile: {compound['Name']}"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # Property table comparison
            st.markdown("#### Property Comparison Table")
            
            # Select columns to display
            display_cols = ['Name', 'MPO_Score', 'Efficacy', 'Safety', 'ADMET', 'Selectivity', 
                           'Synthetic_Feasibility', 'MW', 'LogP', 'TPSA', 'QED', 'Synthetic_Accessibility']
            
            # Show comparison table
            st.dataframe(selected_df[display_cols].style.highlight_max(axis=0, subset=['MPO_Score']))
            
            # Export comparison
            export_format = st.selectbox("Export format", ["CSV", "Excel", "JSON"])
            
            if st.button("Export Comparison"):
                export_data = export_candidate_report(selected_df, len(selected_df), export_format)
                
                if export_format == "CSV":
                    st.download_button(
                        "Download CSV",
                        export_data,
                        file_name="compound_comparison.csv",
                        mime="text/csv"
                    )
                elif export_format == "Excel":
                    st.download_button(
                        "Download Excel",
                        export_data,
                        file_name="compound_comparison.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                elif export_format == "JSON":
                    st.download_button(
                        "Download JSON",
                        export_data,
                        file_name="compound_comparison.json",
                        mime="application/json"
                    )

# Visualization Mode
elif app_mode == "Visualization":
    st.markdown("<h2 class='sub-header'>Visualization</h2>", unsafe_allow_html=True)
    
    if not st.session_state.data_loaded or st.session_state.compounds_df is None:
        st.warning("Please import your compound data first in the 'Data Import' tab.")
    elif not st.session_state.calculated_scores:
        st.warning("Please calculate MPO scores first in the 'Parameter Setup' tab.")
    else:
        df = st.session_state.compounds_df
        
        # Create tabs for different visualization types
        viz_tabs = st.tabs(["Property Distribution", "Multi-Parameter", "Structure-Property", "Heatmap"])
        
        with viz_tabs[0]:
            st.markdown("### Property Distribution Visualization")
            
            # Select property to visualize
            prop_to_viz = st.selectbox(
                "Select property to visualize:",
                ['MPO_Score', 'Efficacy', 'Safety', 'ADMET', 'Selectivity', 'Synthetic_Feasibility',
                 'MW', 'LogP', 'TPSA', 'QED', 'HBA', 'HBD', 'RotBonds', 'Synthetic_Accessibility']
            )
            
            # Get top compounds for highlighting
            top_compounds = df.sort_values('MPO_Score', ascending=False).head(3)
            
            # Create distribution plot
            fig = create_property_distribution_plot(df, prop_to_viz, top_compounds)
            st.plotly_chart(fig, use_container_width=True)
            
            # Add descriptive statistics
            st.markdown("#### Descriptive Statistics")
            stats_df = df[prop_to_viz].describe().reset_index()
            stats_df.columns = ['Statistic', 'Value']
            st.dataframe(stats_df.style.format({'Value': '{:.2f}'}))
        
        with viz_tabs[1]:
            st.markdown("### Multi-Parameter Visualization")
            
            # Select parameters for parallel coordinates plot
            st.markdown("#### Parallel Coordinates Plot")
            
            param_options = ['MPO_Score', 'Efficacy', 'Safety', 'ADMET', 'Selectivity', 'Synthetic_Feasibility',
                          'MW', 'LogP', 'TPSA', 'QED', 'Synthetic_Accessibility']
            
            selected_params = st.multiselect(
                "Select parameters to include:",
                param_options,
                default=['MPO_Score', 'Efficacy', 'Safety', 'ADMET', 'Selectivity']
            )
            
            if selected_params:
                fig = create_parallel_coordinates_plot(df, selected_params)
                st.plotly_chart(fig, use_container_width=True)
            
            # Create scatter plot matrix
            st.markdown("#### Scatter Plot")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                x_param = st.selectbox("X-axis parameter:", param_options, index=6)  # Default to MW
            
            with col2:
                y_param = st.selectbox("Y-axis parameter:", param_options, index=7)  # Default to LogP
            
            with col3:
                color_param = st.selectbox("Color by:", ['MPO_Score'] + param_options, index=0)
            
            fig = create_scatter_plot(df, x_param, y_param, color_param)
            st.plotly_chart(fig, use_container_width=True)
        
        with viz_tabs[2]:
            st.markdown("### Structure-Property Analysis")
            
            # Select compounds for display
            selected_compounds = st.multiselect(
                "Select compounds to analyze:",
                options=df['Name'].tolist(),
                default=df.sort_values('MPO_Score', ascending=False)['Name'].tolist()[:3]
            )
            
            if selected_compounds:
                selected_df = df[df['Name'].isin(selected_compounds)]
                
                # Show structures with properties
                for i, (_, compound) in enumerate(selected_df.iterrows()):
                    st.markdown(f"### {compound['Name']}")
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        # Display molecule
                        svg = mol_to_img(compound['ROMol'])
                        st.markdown(svg_to_html(svg), unsafe_allow_html=True)
                    
                    with col2:
                        # Display properties
                        props_to_display = ['MPO_Score', 'Efficacy', 'Safety', 'ADMET', 'Selectivity', 
                                          'MW', 'LogP', 'TPSA', 'QED']
                        
                        for prop in props_to_display:
                            st.metric(prop, f"{compound[prop]:.2f}")
        
        with viz_tabs[3]:
            st.markdown("### Property Heatmap Visualization")
            
            # Select properties for heatmap
            heatmap_props = st.multiselect(
                "Select properties for heatmap:",
                ['MPO_Score', 'Efficacy', 'Safety', 'ADMET', 'Selectivity', 'Synthetic_Feasibility',
                 'MW', 'LogP', 'TPSA', 'QED', 'HBA', 'HBD', 'RotBonds', 'Synthetic_Accessibility'],
                default=['MPO_Score', 'Efficacy', 'Safety', 'ADMET', 'Selectivity', 'MW', 'LogP']
            )
            
            if heatmap_props:
                # Number of compounds to include
                n_compounds = st.slider("Number of compounds to include", 5, min(20, len(df)), 10)
                
                # Get top compounds
                top_df = df.sort_values('MPO_Score', ascending=False).head(n_compounds)
                
                # Create heatmap
                fig = create_heatmap(top_df, heatmap_props)
                st.plotly_chart(fig, use_container_width=True)

# Report Generation Mode
elif app_mode == "Report Generation":
    st.markdown("<h2 class='sub-header'>Report Generation</h2>", unsafe_allow_html=True)
    
    if not st.session_state.data_loaded or st.session_state.compounds_df is None:
        st.warning("Please import your compound data first in the 'Data Import' tab.")
    elif not st.session_state.calculated_scores:
        st.warning("Please calculate MPO scores first in the 'Parameter Setup' tab.")
    else:
        df = st.session_state.compounds_df
        
        st.markdown("""
        Generate a comprehensive report of prioritized drug candidates for experimental validation.
        The report includes detailed analysis of compound properties, scores, and visualizations.
        """)
        
        # Report configuration
        st.markdown("### Report Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            report_title = st.text_input("Report Title", "Drug Candidate Prioritization Report")
            export_format = st.selectbox("Export Format", ["PDF", "Excel", "CSV", "JSON"])
        
        with col2:
            n_compounds = st.slider("Number of Candidates to Include", 3, min(20, len(df)), 5)
            include_viz = st.checkbox("Include Visualizations", True)
        
        # Report sections
        st.markdown("### Report Sections")
        
        sections = {
            "Executive Summary": st.checkbox("Executive Summary", True),
            "Prioritized Candidates": st.checkbox("Prioritized Candidates", True),
            "Candidate Properties": st.checkbox("Candidate Properties", True),
            "Comparative Analysis": st.checkbox("Comparative Analysis", True),
            "MPO Parameter Details": st.checkbox("MPO Parameter Details", True),
            "Methodology": st.checkbox("Methodology", False)
        }
        
        # Generate report
        if st.button("Generate Report"):
            # Get top compounds
            top_compounds = df.sort_values('MPO_Score', ascending=False).head(n_compounds)
            
            st.success(f"Report generated for top {n_compounds} compounds!")
            
            # Display report preview
            st.markdown("### Report Preview")
            
            # Executive Summary
            if sections["Executive Summary"]:
                st.markdown("#### Executive Summary")
                st.markdown(f"""
                This report presents the prioritized list of {n_compounds} drug candidates for experimental validation.
                The compounds were ranked using multi-parameter optimization (MPO) to balance efficacy, safety, ADMET properties,
                and synthetic feasibility.
                
                **Top candidate:** {top_compounds.iloc[0]['Name']} (MPO Score: {top_compounds.iloc[0]['MPO_Score']:.2f}/10)
                
                **Score range:** {top_compounds['MPO_Score'].min():.2f} - {top_compounds['MPO_Score'].max():.2f}
                
                **Key findings:**
                - The top candidates show excellent balance of potency and ADMET properties
                - {len(top_compounds[top_compounds['Synthetic_Feasibility'] > 7])} compounds have high synthetic feasibility (>7/10)
                - {len(top_compounds[top_compounds['Safety'] > 8])} compounds have excellent safety profiles (>8/10)
                """)
            
            # Prioritized Candidates
            if sections["Prioritized Candidates"]:
                st.markdown("#### Prioritized Candidates")
                
                # Display top compounds table
                display_cols = ['Name', 'MPO_Score', 'Efficacy', 'Safety', 'ADMET', 'Selectivity', 'Synthetic_Feasibility']
                st.dataframe(top_compounds[display_cols].style.highlight_max(axis=0, subset=['MPO_Score']))
                
                # Show top 3 structures
                st.markdown("**Top 3 Chemical Structures:**")
                cols = st.columns(3)
                for i in range(min(3, len(top_compounds))):
                    with cols[i]:
                        mol = top_compounds.iloc[i]['ROMol']
                        name = top_compounds.iloc[i]['Name']
                        score = top_compounds.iloc[i]['MPO_Score']
                        
                        svg = mol_to_img(mol)
                        st.markdown(f"**{name}** (Score: {score:.2f})")
                        st.markdown(svg_to_html(svg), unsafe_allow_html=True)
            
            # Candidate Properties
            if sections["Candidate Properties"]:
                st.markdown("#### Candidate Properties")
                
                # Create property heatmap
                properties = ['Efficacy', 'Safety', 'ADMET', 'Selectivity', 'Synthetic_Feasibility', 
                             'MW', 'LogP', 'TPSA', 'QED', 'Synthetic_Accessibility']
                
                fig = create_heatmap(top_compounds, properties)
                st.plotly_chart(fig, use_container_width=True)
            
            # Comparative Analysis
            if sections["Comparative Analysis"]:
                st.markdown("#### Comparative Analysis")
                
                # Create parallel coordinates plot
                params = ['MPO_Score', 'Efficacy', 'Safety', 'ADMET', 'Selectivity', 'MW', 'LogP']
                fig = create_parallel_coordinates_plot(top_compounds, params)
                st.plotly_chart(fig, use_container_width=True)
            
            # MPO Parameter Details
            if sections["MPO Parameter Details"]:
                st.markdown("#### MPO Parameter Details")
                
                st.markdown("**Parameter Weights:**")
                weights_df = pd.DataFrame({
                    'Parameter': list(st.session_state.mpo_weights.keys()),
                    'Weight': list(st.session_state.mpo_weights.values())
                })
                weights_df = weights_df.sort_values('Weight', ascending=False)
                
                st.dataframe(weights_df)
                
                # Display weight distribution in a chart
                fig = px.bar(
                    weights_df,
                    x='Parameter',
                    y='Weight',
                    title="MPO Parameter Weights",
                    color='Weight',
                    color_continuous_scale=px.colors.sequential.Viridis
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Methodology
            if sections["Methodology"]:
                st.markdown("#### Methodology")
                
                st.markdown("""
                **Multi-Parameter Optimization (MPO) Approach:**
                
                The compounds were evaluated using a weighted multi-parameter optimization approach.
                Each parameter was assigned a weight based on its relative importance, and scored
                against target ranges. The overall MPO score is calculated as:
                
                MPO Score = Σ (parameter_weight × parameter_score) / Σ weights
                
                Where parameter_score is a value between 0-10 based on how well the parameter
                value fits within the specified ideal and hard limit ranges.
                
                **Parameter Ranges:**
                - **Ideal Range:** Values within this range receive full score (10/10)
                - **Hard Limits:** Values outside these limits receive zero score (0/10)
                - **In-between:** Values between ideal range and hard limits receive partial scores
                
                **Key Parameters:**
                - **Efficacy:** Potency against the primary target
                - **Safety:** Absence of concerning toxicity alerts
                - **ADMET:** Favorable absorption, distribution, metabolism, excretion, and toxicity
                - **Selectivity:** Specificity for the target vs. off-targets
                - **Synthetic Feasibility:** Ease of chemical synthesis
                """)
            
            # Export options
            st.markdown("### Export Report")
            
            export_data = export_candidate_report(top_compounds, n_compounds, "json" if export_format == "JSON" else "csv")
            
            if export_format == "CSV":
                st.download_button(
                    "Download CSV Report",
                    export_data,
                    file_name="prioritized_candidates.csv",
                    mime="text/csv"
                )
            elif export_format == "JSON":
                st.download_button(
                    "Download JSON Report",
                    export_data,
                    file_name="prioritized_candidates.json",
                    mime="application/json"
                )
            elif export_format == "Excel":
                st.info("Excel export would be implemented here with more complete formatting.")
            elif export_format == "PDF":
                st.info("PDF export would be implemented here with full report formatting.")

# App footer
st.markdown("""
<div class='footer'>
    <p>Drug Candidate Selection & Prioritization Platform | Developed with Streamlit</p>
    <p><small>For research and educational purposes only.</small></p>
</div>
""", unsafe_allow_html=True)