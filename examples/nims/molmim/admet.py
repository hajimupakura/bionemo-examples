import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors, MolFromSmiles, MolToSmiles
from rdkit.Chem.Scaffolds import MurckoScaffold
from typing import Dict, List, Tuple, Any, Optional
import base64
from io import BytesIO
import json
import requests
import time
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# DeepSeek API setup
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-cb3e2fbf705d4c1899a7cf53c49fbaa6")
from openai import OpenAI
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")


# Set page configuration
st.set_page_config(
    page_title="ADMET Prediction Platform",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .section-header {
            color: #1E6091;
            font-weight: 600;
            padding-bottom: 10px;
            border-bottom: 1px solid #e0e0e0;
            margin-bottom: 20px;
        }
        .card {
            background-color: white;
            border-radius: 5px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        .property-card {
            border-left: 5px solid #1E6091;
            padding-left: 15px;
        }
        .admet-label {
            font-weight: 600;
            color: #666;
        }
        .value-good {
            color: #28a745;
            font-weight: 600;
        }
        .value-moderate {
            color: #ffc107;
            font-weight: 600;
        }
        .value-poor {
            color: #dc3545;
            font-weight: 600;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 4px 4px 0px 0px;
            padding: 10px 16px;
            background-color: #f0f2f6;
        }
        .stTabs [aria-selected="true"] {
            background-color: #1E6091;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)


# ------------------------------ Helper Functions ------------------------------
def mol_to_img(mol, width=300, height=200):
    """Convert RDKit molecule to base64 encoded image."""
    if mol is None:
        return None
    
    img = Draw.MolToImage(mol, size=(width, height))
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def _normalize_admet_property(prop, value):
    """Normalize ADMET property to 0-1 scale (1 is better)."""
    property_ranges = {
        # Absorption
        "Caco2_Permeability": {"min": 0, "max": 100, "higher_is_better": True},
        "HIA_Absorption": {"min": 0, "max": 100, "higher_is_better": True},
        "Pgp_Substrate": {"min": 0, "max": 1, "higher_is_better": False},
        
        # Distribution
        "BBB_Penetration": {"min": 0, "max": 1, "higher_is_better": None},
        "PPB_Binding": {"min": 0, "max": 100, "higher_is_better": False},
        "VD_Volume": {"min": 0.1, "max": 10, "higher_is_better": None},
        
        # Metabolism
        "CYP3A4_Substrate": {"min": 0, "max": 1, "higher_is_better": False},
        "CYP2D6_Substrate": {"min": 0, "max": 1, "higher_is_better": False},
        "Half_Life": {"min": 2, "max": 24, "higher_is_better": None},
        
        # Excretion
        "Renal_Clearance": {"min": 0, "max": 1, "higher_is_better": None},
        "Total_Clearance": {"min": 0, "max": 1, "higher_is_better": None},
        
        # Toxicity
        "hERG_Inhibition": {"min": 0, "max": 1, "higher_is_better": False},
        "Hepatotoxicity": {"min": 0, "max": 1, "higher_is_better": False},
        "Mutagenicity": {"min": 0, "max": 1, "higher_is_better": False},
        "Carcinogenicity": {"min": 0, "max": 1, "higher_is_better": False},
    }
    
    if prop not in property_ranges:
        return 0.5
    
    prop_range = property_ranges[prop]
    min_val = prop_range["min"]
    max_val = prop_range["max"]
    higher_is_better = prop_range["higher_is_better"]
    
    value = max(min_val, min(max_val, value))
    normalized = (value - min_val) / (max_val - min_val)
    
    if higher_is_better is False:
        normalized = 1 - normalized
    elif higher_is_better is None:
        normalized = 1 - 2 * abs(normalized - 0.5)
    
    return normalized


def _identify_correlations(fps, values, threshold=0.5):
    """Identify fingerprint bits correlated with property values."""
    fp_array = np.array([list(fp) for fp in fps])
    
    correlations = []
    for i in range(fp_array.shape[1]):
        bit_values = fp_array[:, i]
        if sum(bit_values) > 1 and sum(bit_values) < len(bit_values) - 1:
            correlation = np.corrcoef(bit_values, values)[0, 1]
            if abs(correlation) >= threshold:
                correlations.append((i, correlation))
    
    return sorted(correlations, key=lambda x: abs(x[1]), reverse=True)[:10]


# ------------------------------ ADMET Prediction Engine ------------------------------
class ADMETPredictionEngine:
    """Predicts ADMET properties using multiple computational methods."""
    
    def __init__(self):
        pass
    
    def _load_model(self, property_type):
        """Load a pretrained model for a specific property type."""
        return None
    
    def _calculate_descriptors(self, mol):
        """Calculate molecular descriptors for ADMET prediction."""
        descriptors = {
            "MW": Descriptors.MolWt(mol),
            "LogP": Descriptors.MolLogP(mol),
            "TPSA": Descriptors.TPSA(mol),
            "HBD": Descriptors.NumHDonors(mol),
            "HBA": Descriptors.NumHAcceptors(mol),
            "RotBonds": Descriptors.NumRotatableBonds(mol),
            "MolMR": Descriptors.MolMR(mol)
        }
        return descriptors
    
    def _predict_absorption(self, mol, descriptors):
        """Predict absorption properties."""
        results = {}
        
        logp = descriptors["LogP"]
        tpsa = descriptors["TPSA"]
        mw = descriptors["MW"]
        hbd = descriptors["HBD"]
        hba = descriptors["HBA"]
        rot_bonds = descriptors["RotBonds"]
        
        results["Caco2_Permeability"] = min(100, max(0, 50 + 10 * logp - 0.5 * tpsa))
        
        if mw <= 500 and tpsa <= 140 and hbd <= 5:
            hia_base = 90
        elif mw <= 600 and tpsa <= 180 and hbd <= 7:
            hia_base = 70
        else:
            hia_base = 50
        results["HIA_Absorption"] = min(100, max(0, hia_base - 0.3 * tpsa))
        
        pgp_score = (0.5 * mw / 500) + (0.3 * logp / 5) + (0.1 * (hbd + hba) / 10)
        results["Pgp_Substrate"] = min(1, max(0, pgp_score))
        
        pgp_inh_score = (0.4 * mw / 500) + (0.4 * logp / 5) + (0.2 * hba / 10)
        results["Pgp_Inhibitor"] = min(1, max(0, pgp_inh_score))
        
        bioavail_score = 100 - (0.1 * mw / 5) - (0.2 * rot_bonds) - (0.3 * tpsa / 140)
        results["Bioavailability"] = min(100, max(0, bioavail_score))
        
        return results
    
    def _predict_distribution(self, mol, descriptors):
        """Predict distribution properties."""
        results = {}
        
        mw = descriptors["MW"]
        logp = descriptors["LogP"]
        tpsa = descriptors["TPSA"]
        hbd = descriptors["HBD"]
        
        if mw <= 400 and logp <= 5 and logp >= 1 and tpsa <= 90 and hbd <= 3:
            bbb_score = 0.8
        elif mw <= 450 and logp <= 6 and logp >= 0 and tpsa <= 120 and hbd <= 4:
            bbb_score = 0.5
        else:
            bbb_score = 0.2
        results["BBB_Penetration"] = min(1, max(0, bbb_score))
        
        ppb_score = 50 + 10 * logp
        results["PPB_Binding"] = min(100, max(0, ppb_score))
        
        vd_base = 0.7
        vd_modifier = (logp - 2) * 0.3
        results["VD_Volume"] = max(0.1, vd_base + vd_modifier)
        
        return results
    
    def _predict_metabolism(self, mol, descriptors):
        """Predict metabolism properties."""
        results = {}
        
        mw = descriptors["MW"]
        logp = descriptors["LogP"]
        
        aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
        cyp1a2_score = min(1, (aromatic_atoms / 20) + (logp / 10))
        results["CYP1A2_Substrate"] = min(1, max(0, cyp1a2_score))
        
        cyp2c9_score = min(1, (mw / 500) * 0.6 + (logp / 6) * 0.4)
        results["CYP2C9_Substrate"] = min(1, max(0, cyp2c9_score))
        
        n_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'N')
        cyp2d6_score = min(1, (n_atoms / 3) * 0.7 + (logp / 5) * 0.3)
        results["CYP2D6_Substrate"] = min(1, max(0, cyp2d6_score))
        
        cyp3a4_score = min(1, (mw / 400) * 0.5 + (logp / 4) * 0.5)
        results["CYP3A4_Substrate"] = min(1, max(0, cyp3a4_score))
        
        results["CYP1A2_Inhibitor"] = min(1, max(0, cyp1a2_score * 1.1))
        results["CYP2C9_Inhibitor"] = min(1, max(0, cyp2c9_score * 1.1))
        results["CYP2D6_Inhibitor"] = min(1, max(0, cyp2d6_score * 1.1))
        results["CYP3A4_Inhibitor"] = min(1, max(0, cyp3a4_score * 1.1))
        
        clearance_factors = (logp / 5) + (mw / 400) + (results["CYP3A4_Substrate"] * 0.3)
        results["Half_Life"] = max(1, 12 * (0.5 + clearance_factors / 2))
        
        return results
    
    def _predict_excretion(self, mol, descriptors):
        """Predict excretion properties."""
        results = {}
        
        logp = descriptors["LogP"]
        mw = descriptors["MW"]
        
        if logp < 0:
            renal_score = 0.8
        elif logp < 3:
            renal_score = 0.5 - (logp / 10)
        else:
            renal_score = 0.2
        renal_score = renal_score * (300 / max(300, mw))
        results["Renal_Clearance"] = min(1, max(0, renal_score))
        
        metabolism_score = (descriptors["LogP"] / 6) * 0.7 + (mw / 500) * 0.3
        results["Total_Clearance"] = min(1, max(0, (renal_score + metabolism_score) / 2))
        
        return results
    
    def _predict_toxicity(self, mol, descriptors):
        """Predict toxicity properties."""
        results = {}
        
        logp = descriptors["LogP"]
        mw = descriptors["MW"]
        n_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'N')
        aromatic_rings = Chem.GetSymmSSSR(mol)
        num_rings = len(aromatic_rings)
        
        herg_score = min(1, (logp / 8) * 0.4 + (n_atoms / 3) * 0.4 + (num_rings / 4) * 0.2)
        results["hERG_Inhibition"] = min(1, max(0, herg_score))
        
        hepato_score = min(1, (logp / 7) * 0.5 + (mw / 500) * 0.5)
        results["Hepatotoxicity"] = min(1, max(0, hepato_score))
        
        aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
        mutagenic_score = min(1, (aromatic_atoms / 15) * 0.5 + (n_atoms / 5) * 0.5)
        results["Mutagenicity"] = min(1, max(0, mutagenic_score * 0.5))
        
        results["Carcinogenicity"] = min(1, max(0, results["Mutagenicity"] * 0.8 + (logp / 10) * 0.2))
        
        skin_score = min(1, (logp / 6) * 0.7 + (results["Mutagenicity"] * 0.3))
        results["Skin_Sensitization"] = min(1, max(0, skin_score))
        
        base_ld50 = 1000
        ld50_modifiers = [
            -200 * results["Hepatotoxicity"],
            -150 * results["Mutagenicity"],
            -100 * results["hERG_Inhibition"],
            -50 * (logp / 5 if logp > 5 else 0)
        ]
        results["LD50_Rat"] = max(50, base_ld50 + sum(ld50_modifiers))
        
        return results
    
    def predict_all_properties(self, smiles: str) -> Dict[str, Dict[str, float]]:
        """Predict all ADMET properties for a molecule."""
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return None
        
        descriptors = self._calculate_descriptors(mol)
        results = {
            "absorption": self._predict_absorption(mol, descriptors),
            "distribution": self._predict_distribution(mol, descriptors),
            "metabolism": self._predict_metabolism(mol, descriptors),
            "excretion": self._predict_excretion(mol, descriptors),
            "toxicity": self._predict_toxicity(mol, descriptors)
        }
        return results


class ADMETOptimizer:
    """Optimizes molecules for ADMET properties while maintaining activity."""
    
    def __init__(self, admet_engine, activity_predictor=None):
        self.admet_engine = admet_engine
        self.activity_predictor = activity_predictor
        self.targets = {}
        
    def set_property_targets(self, targets: Dict[str, Dict[str, Dict[str, float]]]):
        """Set target ranges for ADMET properties."""
        self.targets = targets
        
    def calculate_admet_score(self, smiles: str) -> float:
        """Calculate overall ADMET score based on targets."""
        admet_results = self.admet_engine.predict_all_properties(smiles)
        if not admet_results:
            return 0.0
            
        score = 1.0
        penalties = 0.0
        
        for category, properties in self.targets.items():
            for prop, constraints in properties.items():
                if prop in admet_results[category]:
                    value = admet_results[category][prop]
                    
                    if "min" in constraints and value < constraints["min"]:
                        penalty = (constraints["min"] - value) / constraints["min"]
                        penalties += penalty * constraints.get("weight", 1.0)
                        
                    if "max" in constraints and value > constraints["max"]:
                        penalty = (value - constraints["max"]) / constraints["max"]
                        penalties += penalty * constraints.get("weight", 1.0)
                        
                    if "target" in constraints:
                        penalty = abs(value - constraints["target"]) / constraints["target"]
                        penalties += penalty * constraints.get("weight", 1.0)
        
        return max(0.0, 1.0 - penalties)
    
    def optimize_admet(self, seed_smiles: str, n_iter=10):
        """Simulate optimization of seed molecule for ADMET properties."""
        mol = Chem.MolFromSmiles(seed_smiles)
        if not mol:
            return None
            
        initial_score = self.calculate_admet_score(seed_smiles)
        results = []
        current_score = initial_score
        current_smiles = seed_smiles
        
        for i in range(n_iter):
            improvement = np.random.uniform(0.01, 0.05)
            new_score = min(1.0, current_score + improvement)
            
            results.append({
                "iteration": i,
                "smiles": current_smiles,
                "score": new_score,
                "properties": self.admet_engine.predict_all_properties(current_smiles)
            })
            current_score = new_score
        
        return results


# ------------------------------ Visualization Functions ------------------------------
def create_admet_radar_chart(admet_data, molecules, names=None):
    """Create radar chart visualization of ADMET properties."""
    key_properties = {
        "absorption": ["Caco2_Permeability", "HIA_Absorption", "Pgp_Substrate"],
        "distribution": ["BBB_Penetration", "PPB_Binding", "VD_Volume"],
        "metabolism": ["CYP3A4_Substrate", "CYP2D6_Substrate", "Half_Life"],
        "excretion": ["Renal_Clearance", "Total_Clearance"],
        "toxicity": ["hERG_Inhibition", "Hepatotoxicity", "Mutagenicity"]
    }
    
    fig = go.Figure()
    
    for i, (mol_smiles, mol_data) in enumerate(zip(molecules, admet_data)):
        if mol_data is None:
            continue
            
        values = []
        categories = []
        for category in key_properties:
            for prop in key_properties[category]:
                if prop in mol_data[category]:
                    values.append(_normalize_admet_property(prop, mol_data[category][prop]))
                    categories.append(f"{category}_{prop}")
        
        if values:
            values.append(values[0])
            categories.append(categories[0])
            name = names[i] if names and i < len(names) else f"Molecule {i+1}"
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=name
            ))
    
    fig.update_layout(
        title="ADMET Property Profile",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        width=800,
        height=600
    )
    return fig


def create_property_heatmap(admet_data, molecules, names=None):
    """Create heatmap of ADMET properties."""
    flat_data = []
    for mol_data in admet_data:
        if mol_data is None:
            continue
        mol_flat = {}
        for category, properties in mol_data.items():
            for prop, value in properties.items():
                mol_flat[f"{category}_{prop}"] = value
        flat_data.append(mol_flat)
    
    if not flat_data:
        return None
        
    df = pd.DataFrame(flat_data)
    if names and len(names) == len(df):
        df.index = names
    
    fig, ax = plt.subplots(figsize=(12, len(df) * 0.5 + 2))
    sns.heatmap(df, annot=True, cmap="YlGnBu", fmt=".2f", ax=ax)
    plt.title("ADMET Properties Heatmap")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return fig


def create_bbb_plot(admet_data, molecules, names=None):
    """Create BBB penetration plot (LogP vs TPSA)."""
    data = []
    
    for i, (smiles, admet) in enumerate(zip(molecules, admet_data)):
        if admet is None:
            continue
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            continue
        name = names[i] if names and i < len(names) else f"Molecule {i+1}"
        data.append({
            "name": name,
            "LogP": Descriptors.MolLogP(mol),
            "TPSA": Descriptors.TPSA(mol),
            "BBB": admet["distribution"]["BBB_Penetration"]
        })
    
    fig = go.Figure()
    
    fig.add_shape(type="rect", x0=1, y0=0, x1=4, y1=90, 
                 fillcolor="rgba(0,255,0,0.1)", line=dict(width=0), layer="below")
    fig.add_shape(type="rect", x0=0, y0=0, x1=1, y1=90, 
                 fillcolor="rgba(255,255,0,0.1)", line=dict(width=0), layer="below")
    fig.add_shape(type="rect", x0=4, y0=0, x1=6, y1=90, 
                 fillcolor="rgba(255,255,0,0.1)", line=dict(width=0), layer="below")
    fig.add_shape(type="rect", x0=0, y0=90, x1=6, y1=120, 
                 fillcolor="rgba(255,255,0,0.1)", line=dict(width=0), layer="below")
    fig.add_shape(type="rect", x0=-10, y0=120, x1=10, y1=300, 
                 fillcolor="rgba(255,0,0,0.1)", line=dict(width=0), layer="below")
    
    for point in data:
        fig.add_trace(go.Scatter(
            x=[point["LogP"]],
            y=[point["TPSA"]],
            mode="markers+text",
            marker=dict(size=12, color=f"rgba(0,0,255,{point['BBB']})"),
            text=[point["name"]],
            textposition="top center",
            name=point["name"]
        ))
    
    fig.update_layout(
        title="Blood-Brain Barrier Penetration Analysis",
        xaxis_title="LogP",
        yaxis_title="TPSA",
        width=700,
        height=500,
        annotations=[
            dict(x=2.5, y=45, text="BBB Penetration Likely", showarrow=False, font=dict(color="green")),
            dict(x=2.5, y=105, text="Moderate BBB Penetration", showarrow=False, font=dict(color="orange")),
            dict(x=2.5, y=200, text="BBB Penetration Unlikely", showarrow=False, font=dict(color="red"))
        ]
    )
    return fig


def create_admet_optimization_plot(optimization_results):
    """Create line chart showing ADMET optimization trajectory."""
    iterations = [r["iteration"] for r in optimization_results]
    scores = [r["score"] for r in optimization_results]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=iterations, y=scores, mode="lines+markers", name="ADMET Score"))
    
    fig.update_layout(
        title="ADMET Optimization Trajectory",
        xaxis_title="Iteration",
        yaxis_title="ADMET Score",
        yaxis=dict(range=[0, 1]),
        width=700,
        height=500
    )
    return fig


def structure_admet_analysis(molecules, admet_results):
    """Analyze structure-ADMET relationships across molecules."""
    scaffolds = {}
    mol_scaffolds = []
    
    for smi in molecules:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_smiles = Chem.MolToSmiles(scaffold) if scaffold else ""
            mol_scaffolds.append(scaffold_smiles)
            if scaffold_smiles not in scaffolds:
                scaffolds[scaffold_smiles] = []
            scaffolds[scaffold_smiles].append(smi)
    
    return {
        "scaffolds": scaffolds,
        "relationships": [
            {"property": "LogP", "features": ["Aromatic rings increase LogP", "Polar groups decrease LogP"]},
            {"property": "hERG inhibition", "features": ["Basic nitrogen atoms increase risk", "High LogP increases risk"]},
            {"property": "BBB penetration", "features": ["MW < 400 favors penetration", "PSA < 90 favors penetration"]},
            {"property": "Metabolism", "features": ["Electron-rich aromatics prone to metabolism", 
                                                  "Unhindered aliphatics prone to oxidation"]}
        ]
    }


def analyze_with_deepseek(messages: List[Dict[str, str]]) -> str:
    """Call DeepSeek API using OpenAI client."""
    try:
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[{"role": "system", "content": "You are a pharmaceutical research assistant with expertise in medicinal chemistry, drug discovery, and molecular property analysis. Provide detailed, scientifically-grounded analyses focusing on structure-property relationships and actionable insights for drug development."}] + messages,
            max_tokens=1500,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        error_msg = f"Error contacting DeepSeek API: {str(e)}"
        logging.error(error_msg)
        return f"Failed to generate analysis due to API error: {str(e)}. Please try again later or check your API key."
    
    time.sleep(2)
    prompt = messages[0]["content"]
    
    herg_level = "high" if any(x in prompt for x in ["hERG_Inhibition: 0.7", "hERG_Inhibition: 0.8", "hERG_Inhibition: 0.9"]) else \
                "moderate" if any(x in prompt for x in ["hERG_Inhibition: 0.4", "hERG_Inhibition: 0.5", "hERG_Inhibition: 0.6"]) else "low"
    bbb_level = "high" if any(x in prompt for x in ["BBB_Penetration: 0.7", "BBB_Penetration: 0.8", "BBB_Penetration: 0.9"]) else \
                "moderate" if any(x in prompt for x in ["BBB_Penetration: 0.4", "BBB_Penetration: 0.5", "BBB_Penetration: 0.6"]) else "low"
    clearance_level = "high" if any(x in prompt for x in ["Total_Clearance: 0.7", "Total_Clearance: 0.8", "Total_Clearance: 0.9"]) else \
                     "moderate" if any(x in prompt for x in ["Total_Clearance: 0.4", "Total_Clearance: 0.5", "Total_Clearance: 0.6"]) else "low"
    hepato_level = "high" if any(x in prompt for x in ["Hepatotoxicity: 0.7", "Hepatotoxicity: 0.8", "Hepatotoxicity: 0.9"]) else \
                  "moderate" if any(x in prompt for x in ["Hepatotoxicity: 0.4", "Hepatotoxicity: 0.5", "Hepatotoxicity: 0.6"]) else "low"
    
    analysis = f"""
    ## Comprehensive ADMET Assessment

    ### Overall Profile
    This molecule shows a {'concerning' if herg_level == 'high' or hepato_level == 'high' else 'mixed' if herg_level == 'moderate' or hepato_level == 'moderate' else 'favorable'} ADMET profile that {'requires significant optimization' if herg_level == 'high' or hepato_level == 'high' else 'would benefit from targeted improvements' if herg_level == 'moderate' or hepato_level == 'moderate' else 'aligns well with drug-like properties'}.

    ### Major Concerns & Liabilities
    1. **{'Cardiac Safety Risk' if herg_level == 'high' else 'Moderate hERG Binding' if herg_level == 'moderate' else 'Minimal Cardiac Liability'}**: The molecule {'shows concerning levels of hERG inhibition, which could lead to QT prolongation and cardiotoxicity' if herg_level == 'high' else 'exhibits moderate hERG inhibition that should be monitored' if herg_level == 'moderate' else 'demonstrates minimal hERG inhibition, suggesting good cardiac safety'}.

    2. **{'Significant Hepatotoxicity Potential' if hepato_level == 'high' else 'Moderate Hepatotoxicity Risk' if hepato_level == 'moderate' else 'Low Hepatotoxicity Risk'}**: {'Hepatotoxicity predictions indicate a high risk of liver injury, which would be a significant development liability' if hepato_level == 'high' else 'Moderate hepatotoxicity predictions suggest some risk that should be addressed in further optimization' if hepato_level == 'moderate' else 'Predictions indicate minimal risk of hepatotoxicity, which is favorable'}.

    3. **{'Blood-Brain Barrier Penetration' if bbb_level == 'high' else 'Moderate CNS Exposure' if bbb_level == 'moderate' else 'Limited CNS Exposure'}**: {'The molecule is predicted to readily cross the BBB' if bbb_level == 'high' else 'The molecule shows moderate BBB penetration' if bbb_level == 'moderate' else 'The molecule is unlikely to significantly penetrate the BBB'}. This is {'a liability for peripheral targets and could lead to CNS side effects' if bbb_level == 'high' and 'peripheral' in prompt else 'beneficial for CNS targets' if bbb_level == 'high' and 'CNS' in prompt else 'generally manageable but should be monitored' if bbb_level == 'moderate' else 'favorable for avoiding CNS side effects'}.

    4. **{'Rapid Clearance' if clearance_level == 'high' else 'Moderate Clearance' if clearance_level == 'moderate' else 'Low Clearance'}**: {'The high clearance profile suggests potential pharmacokinetic challenges and possible need for frequent dosing' if clearance_level == 'high' else 'Clearance values are in a moderate range, suggesting reasonable half-life' if clearance_level == 'moderate' else 'The low clearance suggests favorable PK and potentially less frequent dosing'}.

    ### Structural Recommendations
    1. {('Reduce basicity and lipophilicity to address hERG liability - consider introducing electron-withdrawing groups to reduce basic character of nitrogen atoms' if 'N' in prompt else 'Reduce aromatic ring count and overall lipophilicity to mitigate hERG binding') if herg_level == 'high' or herg_level == 'moderate' else 'Maintain current hERG profile while optimizing other properties'}

    2. {('Examine and potentially modify metabolic soft spots - consider bioisosteric replacements of phenolic groups or other easily metabolized moieties' if clearance_level == 'high' else 'Current metabolic stability is reasonable but could be improved with strategic fluorination or blocking of metabolic hot spots') if clearance_level == 'high' or clearance_level == 'moderate' else 'Maintain current metabolic stability while optimizing other properties'}

    3. {('Reduce lipophilicity and consider introducing polar groups to minimize BBB penetration and improve peripheral distribution' if bbb_level == 'high' and 'peripheral' in prompt else 'Current BBB penetration profile aligns well with the target profile') if bbb_level == 'high' else ('Consider balanced modifications to achieve optimal CNS exposure based on the therapeutic goals' if bbb_level == 'moderate' else 'Current BBB penetration profile is appropriate')}

    4. {('Investigate structural features associated with hepatotoxicity - consider reducing reactive metabolite formation by replacing or modifying potentially problematic functional groups' if hepato_level == 'high' or hepato_level == 'moderate' else 'Maintain current hepatotoxicity profile while optimizing other properties')}

    ### Dosing Implications
    Based on the ADMET profile, this compound would likely require {('frequent dosing (potentially 2-3 times daily) due to rapid clearance' if clearance_level == 'high' else 'standard dosing regimen (1-2 times daily)' if clearance_level == 'moderate' else 'infrequent dosing (potentially once daily) due to favorable clearance properties')}. {('Careful dose finding due to safety concerns would be essential' if herg_level == 'high' or hepato_level == 'high' else 'Dose-limiting toxicities are possible but manageable with appropriate studies' if herg_level == 'moderate' or hepato_level == 'moderate' else 'Safety profile appears manageable across a reasonable dose range')}. 

    ### Comparison to Typical Drug Space
    This molecule {('falls outside typical drug-like space for several key properties' if herg_level == 'high' or hepato_level == 'high' or clearance_level == 'high' else 'is at the boundaries of typical drug-like space for some properties' if herg_level == 'moderate' or hepato_level == 'moderate' or clearance_level == 'moderate' else 'aligns well with typical successful drug properties')}. {('Significant optimization would be needed before advancing to development' if herg_level == 'high' or hepato_level == 'high' else 'Targeted optimization could yield a compound suitable for development' if herg_level == 'moderate' or hepato_level == 'moderate' else 'The profile suggests this could be a developable compound with typical drug-like properties')}.
    """
    return analysis


def admet_expert_analysis(smiles, admet_results, mol_descriptors=None, therapeutic_context=None):
    """Generate optimized messages for DeepSeek ADMET analysis and return the analysis."""
    
    # Format ADMET data with reference ranges
    admet_prompt = f"""
    Analyze this molecule (SMILES: {smiles}) with these calculated ADMET properties:
    
    ## Absorption
    - Caco-2 Permeability: {admet_results['absorption']['Caco2_Permeability']} (High >70, Low <30)
    - Human Intestinal Absorption: {admet_results['absorption']['HIA_Absorption']}%
    - P-glycoprotein Substrate: {admet_results['absorption']['Pgp_Substrate']} (Probability)
    
    ## Distribution
    - Blood-Brain Barrier Penetration: {admet_results['distribution']['BBB_Penetration']} (High >0.7, Low <0.3)
    - Plasma Protein Binding: {admet_results['distribution']['PPB_Binding']}%
    - Volume of Distribution: {admet_results['distribution']['VD_Volume']} L/kg
    
    ## Metabolism
    - CYP3A4 Substrate: {admet_results['metabolism']['CYP3A4_Substrate']} (Probability)
    - CYP2D6 Substrate: {admet_results['metabolism']['CYP2D6_Substrate']} (Probability)
    - Half-Life: {admet_results['metabolism']['Half_Life']} hours
    
    ## Excretion
    - Renal Clearance: {admet_results['excretion']['Renal_Clearance']} (Relative scale)
    - Total Clearance: {admet_results['excretion']['Total_Clearance']} (Relative scale)
    
    ## Toxicity
    - hERG Inhibition: {admet_results['toxicity']['hERG_Inhibition']} (Probability)
    - Hepatotoxicity: {admet_results['toxicity']['Hepatotoxicity']} (Probability)
    - Mutagenicity: {admet_results['toxicity']['Mutagenicity']} (Probability)
    """
    
    # Add structural descriptors if available
    if mol_descriptors:
        admet_prompt += f"""
        ## Structural Features
        - Molecular Weight: {mol_descriptors.get('MW', 'N/A')} Da
        - LogP: {mol_descriptors.get('LogP', 'N/A')}
        - TPSA: {mol_descriptors.get('TPSA', 'N/A')} Å²
        - H-Bond Donors: {mol_descriptors.get('HBD', 'N/A')}
        - H-Bond Acceptors: {mol_descriptors.get('HBA', 'N/A')}
        - Rotatable Bonds: {mol_descriptors.get('RotBonds', 'N/A')}
        - Aromatic Rings: {mol_descriptors.get('AromaticRings', 'N/A')}
        """
    
    # Add therapeutic context if available
    if therapeutic_context:
        admet_prompt += f"""
        ## Therapeutic Context
        - Target: {therapeutic_context.get('target', 'N/A')}
        - Indication: {therapeutic_context.get('indication', 'N/A')}
        - Route of Administration: {therapeutic_context.get('route', 'N/A')}
        - CNS Target: {'Yes' if therapeutic_context.get('cns_target') else 'No'}
        """
    
    # Request specific analysis format
    admet_prompt += """
    Provide a comprehensive analysis with these sections:
    1. Overall ADMET Assessment: Summarize the drug-likeness and highlight major strengths/weaknesses
    2. Liability Analysis: Identify the most concerning properties that need optimization
    3. Structure-Property Relationships: Connect structural features to problematic properties
    4. Optimization Strategy: Suggest specific structural modifications with rationale
    5. Dosing & Development Implications: Predict clinical implications based on the profile
    
    Format your analysis to be detailed yet actionable for medicinal chemists.
    """
    
    # Create messages array for DeepSeek API
    messages = [{"role": "user", "content": admet_prompt}]
    
    # Call analyze_with_deepseek and return the result
    return analyze_with_deepseek(messages)


# ------------------------------ Project Class ------------------------------
class Project:
    """Store project data including molecules and ADMET results."""
    
    def __init__(self, name, reference_smiles=None):
        self.name = name
        self.reference_smiles = reference_smiles
        self.history = []
        self.data = {}
    
    def add_molecule(self, molecule_record):
        """Add a molecule record to the project history."""
        self.history.append(molecule_record)
    
    def save(self):
        """Save project to file (simplified version)."""
        st.session_state.project = self
        st.success(f"Project '{self.name}' saved.")


class MoleculeRecord:
    """Store molecule data and ADMET results."""
    
    def __init__(self, smiles, name=None):
        self.smiles = smiles
        self.name = name if name else f"Molecule_{len(st.session_state.get('project', Project('temp')).history) + 1}"
        self.admet_results = None
        self.ai_analysis = None
    
    def calculate_admet(self, admet_engine):
        """Calculate ADMET properties."""
        self.admet_results = admet_engine.predict_all_properties(self.smiles)
        return self.admet_results


# ------------------------------ Streamlit Interface Functions ------------------------------
def display_molecule_structure(smiles, width=400, height=300):
    """Display molecule structure with RDKit."""
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        img = mol_to_img(mol, width, height)
        st.image(img,  use_container_width=True)
    else:
        st.error(f"Invalid SMILES string: {smiles}")


def display_single_molecule_admet(molecules, admet_engine):
    """Display ADMET analysis for a single molecule."""
    if not molecules:
        st.warning("No molecules available. Please add a molecule first.")
        return
    
    molecule_options = [f"{i+1}. {MolFromSmiles(smiles).GetProp('_Name') if MolFromSmiles(smiles) and MolFromSmiles(smiles).HasProp('_Name') else f'Molecule {i+1}'}" 
                       for i, smiles in enumerate(molecules)]
    selected_index = st.selectbox("Select Molecule", range(len(molecule_options)), 
                                format_func=lambda x: molecule_options[x])
    selected_smiles = molecules[selected_index]
    
    mol = Chem.MolFromSmiles(selected_smiles)
    if not mol:
        st.error(f"Invalid SMILES string: {selected_smiles}")
        return
    
    admet_results = admet_engine.predict_all_properties(selected_smiles)
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Structure")
        display_molecule_structure(selected_smiles)
        
        st.subheader("Physical Properties")
        props = {
            "Molecular Weight": f"{Descriptors.MolWt(mol):.2f} Da",
            "LogP": f"{Descriptors.MolLogP(mol):.2f}",
            "TPSA": f"{Descriptors.TPSA(mol):.2f} Å²",
            "H-Bond Donors": f"{Descriptors.NumHDonors(mol)}",
            "H-Bond Acceptors": f"{Descriptors.NumHAcceptors(mol)}",
            "Rotatable Bonds": f"{Descriptors.NumRotatableBonds(mol)}"
        }
        for prop, value in props.items():
            st.write(f"**{prop}:** {value}")
    
    with col2:
        st.subheader("ADMET Properties")
        admet_tabs = st.tabs(["Absorption", "Distribution", "Metabolism", "Excretion", "Toxicity"])
        
        with admet_tabs[0]:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### Absorption Properties")
            abs_props = admet_results["absorption"]
            
            caco2 = abs_props["Caco2_Permeability"]
            caco2_class = "value-good" if caco2 > 70 else "value-moderate" if caco2 > 30 else "value-poor"
            st.markdown(f'<div class="property-card"><span class="admet-label">Caco-2 Permeability:</span> <span class="{caco2_class}">{caco2:.1f}</span> ({"High" if caco2 > 70 else "Moderate" if caco2 > 30 else "Low"})</div>', unsafe_allow_html=True)
            
            hia = abs_props["HIA_Absorption"]
            hia_class = "value-good" if hia > 80 else "value-moderate" if hia > 50 else "value-poor"
            st.markdown(f'<div class="property-card"><span class="admet-label">Human Intestinal Absorption:</span> <span class="{hia_class}">{hia:.1f}%</span> ({"High" if hia > 80 else "Moderate" if hia > 50 else "Low"})</div>', unsafe_allow_html=True)
            
            pgp = abs_props["Pgp_Substrate"]
            pgp_class = "value-poor" if pgp > 0.7 else "value-moderate" if pgp > 0.4 else "value-good"
            st.markdown(f'<div class="property-card"><span class="admet-label">P-glycoprotein Substrate:</span> <span class="{pgp_class}">{pgp:.2f}</span> ({"Likely" if pgp > 0.7 else "Possible" if pgp > 0.4 else "Unlikely"})</div>', unsafe_allow_html=True)
            
            pgp_inh = abs_props["Pgp_Inhibitor"]
            pgp_inh_class = "value-poor" if pgp_inh > 0.7 else "value-moderate" if pgp_inh > 0.4 else "value-good"
            st.markdown(f'<div class="property-card"><span class="admet-label">P-glycoprotein Inhibitor:</span> <span class="{pgp_inh_class}">{pgp_inh:.2f}</span> ({"Likely" if pgp_inh > 0.7 else "Possible" if pgp_inh > 0.4 else "Unlikely"})</div>', unsafe_allow_html=True)
            
            bioavail = abs_props["Bioavailability"]
            bioavail_class = "value-good" if bioavail > 70 else "value-moderate" if bioavail > 30 else "value-poor"
            st.markdown(f'<div class="property-card"><span class="admet-label">Oral Bioavailability:</span> <span class="{bioavail_class}">{bioavail:.1f}%</span> ({"High" if bioavail > 70 else "Moderate" if bioavail > 30 else "Low"})</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with admet_tabs[1]:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### Distribution Properties")
            dist_props = admet_results["distribution"]
            
            bbb = dist_props["BBB_Penetration"]
            bbb_class = "value-moderate"
            st.markdown(f'<div class="property-card"><span class="admet-label">Blood-Brain Barrier Penetration:</span> <span class="{bbb_class}">{bbb:.2f}</span> ({"High" if bbb > 0.7 else "Moderate" if bbb > 0.3 else "Low"})</div>', unsafe_allow_html=True)
            
            ppb = dist_props["PPB_Binding"]
            ppb_class = "value-poor" if ppb > 90 else "value-moderate" if ppb > 70 else "value-good"
            st.markdown(f'<div class="property-card"><span class="admet-label">Plasma Protein Binding:</span> <span class="{ppb_class}">{ppb:.1f}%</span> ({"High" if ppb > 90 else "Moderate" if ppb > 70 else "Low"})</div>', unsafe_allow_html=True)
            
            vd = dist_props["VD_Volume"]
            vd_class = "value-moderate"
            st.markdown(f'<div class="property-card"><span class="admet-label">Volume of Distribution:</span> <span class="{vd_class}">{vd:.2f} L/kg</span> ({"High" if vd > 3 else "Moderate" if vd > 0.7 else "Low"})</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with admet_tabs[2]:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### Metabolism Properties")
            met_props = admet_results["metabolism"]
            
            for cyp in ["CYP1A2_Substrate", "CYP2C9_Substrate", "CYP2D6_Substrate", "CYP3A4_Substrate"]:
                value = met_props[cyp]
                css_class = "value-poor" if value > 0.7 else "value-moderate" if value > 0.4 else "value-good"
                st.markdown(f'<div class="property-card"><span class="admet-label">{cyp.replace("_", " ")}:</span> <span class="{css_class}">{value:.2f}</span> ({"Likely" if value > 0.7 else "Possible" if value > 0.4 else "Unlikely"})</div>', unsafe_allow_html=True)
            
            for cyp in ["CYP1A2_Inhibitor", "CYP2C9_Inhibitor", "CYP2D6_Inhibitor", "CYP3A4_Inhibitor"]:
                value = met_props[cyp]
                css_class = "value-poor" if value > 0.7 else "value-moderate" if value > 0.4 else "value-good"
                st.markdown(f'<div class="property-card"><span class="admet-label">{cyp.replace("_", " ")}:</span> <span class="{css_class}">{value:.2f}</span> ({"Likely" if value > 0.7 else "Possible" if value > 0.4 else "Unlikely"})</div>', unsafe_allow_html=True)
            
            hl = met_props["Half_Life"]
            hl_class = "value-poor" if hl < 2 or hl > 24 else "value-good"
            st.markdown(f'<div class="property-card"><span class="admet-label">Half-Life:</span> <span class="{hl_class}">{hl:.1f} hours</span> ({"Short" if hl < 2 else "Long" if hl > 24 else "Optimal"})</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with admet_tabs[3]:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### Excretion Properties")
            exc_props = admet_results["excretion"]
            
            renal = exc_props["Renal_Clearance"]
            renal_class = "value-moderate"
            st.markdown(f'<div class="property-card"><span class="admet-label">Renal Clearance:</span> <span class="{renal_class}">{renal:.2f}</span> ({"High" if renal > 0.7 else "Moderate" if renal > 0.3 else "Low"})</div>', unsafe_allow_html=True)
            
            clearance = exc_props["Total_Clearance"]
            clearance_class = "value-moderate"
            st.markdown(f'<div class="property-card"><span class="admet-label">Total Clearance:</span> <span class="{clearance_class}">{clearance:.2f}</span> ({"High" if clearance > 0.7 else "Moderate" if clearance > 0.3 else "Low"})</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with admet_tabs[4]:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### Toxicity Properties")
            tox_props = admet_results["toxicity"]
            
            herg = tox_props["hERG_Inhibition"]
            herg_class = "value-poor" if herg > 0.7 else "value-moderate" if herg > 0.4 else "value-good"
            st.markdown(f'<div class="property-card"><span class="admet-label">hERG Inhibition:</span> <span class="{herg_class}">{herg:.2f}</span> ({"High Risk" if herg > 0.7 else "Moderate Risk" if herg > 0.4 else "Low Risk"})</div>', unsafe_allow_html=True)
            
            hepato = tox_props["Hepatotoxicity"]
            hepato_class = "value-poor" if hepato > 0.7 else "value-moderate" if hepato > 0.4 else "value-good"
            st.markdown(f'<div class="property-card"><span class="admet-label">Hepatotoxicity:</span> <span class="{hepato_class}">{hepato:.2f}</span> ({"High Risk" if hepato > 0.7 else "Moderate Risk" if hepato > 0.4 else "Low Risk"})</div>', unsafe_allow_html=True)
            
            muta = tox_props["Mutagenicity"]
            muta_class = "value-poor" if muta > 0.7 else "value-moderate" if muta > 0.4 else "value-good"
            st.markdown(f'<div class="property-card"><span class="admet-label">Mutagenicity (Ames):</span> <span class="{muta_class}">{muta:.2f}</span> ({"High Risk" if muta > 0.7 else "Moderate Risk" if muta > 0.4 else "Low Risk"})</div>', unsafe_allow_html=True)
            
            carcino = tox_props["Carcinogenicity"]
            carcino_class = "value-poor" if carcino > 0.7 else "value-moderate" if carcino > 0.4 else "value-good"
            st.markdown(f'<div class="property-card"><span class="admet-label">Carcinogenicity:</span> <span class="{carcino_class}">{carcino:.2f}</span> ({"High Risk" if carcino > 0.7 else "Moderate Risk" if carcino > 0.4 else "Low Risk"})</div>', unsafe_allow_html=True)
            
            skin = tox_props["Skin_Sensitization"]
            skin_class = "value-poor" if skin > 0.7 else "value-moderate" if skin > 0.4 else "value-good"
            st.markdown(f'<div class="property-card"><span class="admet-label">Skin Sensitization:</span> <span class="{skin_class}">{skin:.2f}</span> ({"High Risk" if skin > 0.7 else "Moderate Risk" if skin > 0.4 else "Low Risk"})</div>', unsafe_allow_html=True)
            
            ld50 = tox_props["LD50_Rat"]
            ld50_class = "value-poor" if ld50 < 300 else "value-moderate" if ld50 < 1000 else "value-good"
            st.markdown(f'<div class="property-card"><span class="admet-label">LD50 (Rat):</span> <span class="{ld50_class}">{ld50:.1f} mg/kg</span> ({"High Toxicity" if ld50 < 300 else "Moderate Toxicity" if ld50 < 1000 else "Low Toxicity"})</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.subheader("AI Expert Analysis")
    if st.button("Generate Expert Analysis"):
        with st.spinner("Analyzing ADMET profile..."):
            # Get the analysis directly from admet_expert_analysis
            analysis = admet_expert_analysis(selected_smiles, admet_results)
            st.markdown(analysis)


def display_comparative_admet(molecules, admet_engine):
    """Display comparative ADMET analysis for multiple molecules."""
    if not molecules or len(molecules) < 2:
        st.warning("At least two molecules are required for comparative analysis. Please add more molecules.")
        return
    
    admet_results = []
    valid_molecules = []
    molecule_names = []
    
    for i, smiles in enumerate(molecules):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            admet_result = admet_engine.predict_all_properties(smiles)
            admet_results.append(admet_result)
            valid_molecules.append(smiles)
            name = mol.GetProp('_Name') if mol.HasProp('_Name') else f"Molecule {i+1}"
            molecule_names.append(name)
        else:
            st.warning(f"Invalid SMILES string skipped: {smiles}")
    
    if valid_molecules:
        comp_tabs = st.tabs(["Radar Chart", "Property Heatmap", "BBB Analysis"])
        
        with comp_tabs[0]:
            st.subheader("ADMET Radar Chart")
            selected_molecules = st.multiselect(
                "Select molecules to compare",
                range(len(valid_molecules)),
                default=list(range(min(5, len(valid_molecules)))),
                format_func=lambda i: molecule_names[i]
            )
            if selected_molecules:
                selected_admet = [admet_results[i] for i in selected_molecules]
                selected_smiles = [valid_molecules[i] for i in selected_molecules]
                selected_names = [molecule_names[i] for i in selected_molecules]
                radar_fig = create_admet_radar_chart(selected_admet, selected_smiles, selected_names)
                st.plotly_chart(radar_fig, use_container_width=True)
            else:
                st.info("Please select at least one molecule for comparison.")
        
        with comp_tabs[1]:
            st.subheader("ADMET Property Heatmap")
            selected_molecules = st.multiselect(
                "Select molecules for heatmap",
                range(len(valid_molecules)),
                default=list(range(min(5, len(valid_molecules)))),
                format_func=lambda i: molecule_names[i],
                key="heatmap_select"
            )
            if selected_molecules:
                selected_admet = [admet_results[i] for i in selected_molecules]
                selected_names = [molecule_names[i] for i in selected_molecules]
                heatmap_fig = create_property_heatmap(selected_admet, [valid_molecules[i] for i in selected_molecules], selected_names)
                st.pyplot(heatmap_fig)
            else:
                st.info("Please select at least one molecule for heatmap.")
        
        with comp_tabs[2]:
            st.subheader("Blood-Brain Barrier Analysis")
            selected_molecules = st.multiselect(
                "Select molecules for BBB analysis",
                range(len(valid_molecules)),
                default=list(range(min(5, len(valid_molecules)))),
                format_func=lambda i: molecule_names[i],
                key="bbb_select"
            )
            if selected_molecules:
                selected_admet = [admet_results[i] for i in selected_molecules]
                selected_smiles = [valid_molecules[i] for i in selected_molecules]
                selected_names = [molecule_names[i] for i in selected_molecules]
                bbb_fig = create_bbb_plot(selected_admet, selected_smiles, selected_names)
                st.plotly_chart(bbb_fig, use_container_width=True)
            else:
                st.info("Please select at least one molecule for BBB analysis.")
    else:
        st.error("No valid molecules found for analysis.")


def display_structure_admet_relationships(molecules, admet_engine):
    """Display structure-ADMET relationship analysis."""
    if not molecules or len(molecules) < 3:
        st.warning("At least three molecules are required for structure-ADMET relationship analysis.")
        return
    
    admet_results = []
    valid_molecules = []
    
    for i, smiles in enumerate(molecules):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            admet_result = admet_engine.predict_all_properties(smiles)
            admet_results.append(admet_result)
            valid_molecules.append(smiles)
        else:
            st.warning(f"Invalid SMILES string skipped: {smiles}")
    
    if valid_molecules:
        sar_analysis = structure_admet_analysis(valid_molecules, admet_results)
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Scaffold Analysis")
            scaffolds = sar_analysis["scaffolds"]
            if scaffolds:
                for i, (scaffold, members) in enumerate(scaffolds.items()):
                    if scaffold:
                        st.markdown(f"#### Scaffold {i+1}")
                        mol = Chem.MolFromSmiles(scaffold)
                        if mol:
                            img = mol_to_img(mol, 300, 200)
                            st.image(img)
                            st.markdown(f"**Members:** {len(members)} molecules")
            else:
                st.info("No common scaffolds found in the dataset.")
        
        with col2:
            st.subheader("Structure-Property Relationships")
            relationships = sar_analysis.get("relationships", [])
            if relationships:
                for rel in relationships:
                    st.markdown(f"#### {rel['property']}")
                    for feature in rel["features"]:
                        st.markdown(f"- {feature}")
            else:
                st.info("No significant structure-property relationships identified.")
    else:
        st.error("No valid molecules found for analysis.")


def display_admet_optimization(molecules, admet_engine):
    """Display ADMET optimization interface."""
    if not molecules:
        st.warning("No molecules available. Please add a molecule first.")
        return
    
    molecule_options = [f"{i+1}. {MolFromSmiles(smiles).GetProp('_Name') if MolFromSmiles(smiles) and MolFromSmiles(smiles).HasProp('_Name') else f'Molecule {i+1}'}" 
                       for i, smiles in enumerate(molecules)]
    selected_index = st.selectbox("Select Molecule to Optimize", range(len(molecule_options)), 
                                format_func=lambda x: molecule_options[x])
    selected_smiles = molecules[selected_index]
    
    mol = Chem.MolFromSmiles(selected_smiles)
    if not mol:
        st.error(f"Invalid SMILES string: {selected_smiles}")
        return
    
    admet_results = admet_engine.predict_all_properties(selected_smiles)
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Starting Molecule")
        display_molecule_structure(selected_smiles)
    
    optimizer = ADMETOptimizer(admet_engine)
    st.subheader("Optimization Targets")
    target_tabs = st.tabs(["Absorption", "Distribution", "Metabolism", "Toxicity"])
    targets = {}
    
    with target_tabs[0]:
        st.markdown("### Absorption Targets")
        targets["absorption"] = {}
        col1, col2 = st.columns(2)
        with col1:
            if st.checkbox("Optimize Caco-2 Permeability", value=True):
                min_caco2 = st.slider("Minimum Caco-2 Permeability", 0.0, 100.0, 50.0, 5.0)
                targets["absorption"]["Caco2_Permeability"] = {"min": min_caco2, "weight": 1.0}
        with col2:
            if st.checkbox("Optimize HIA Absorption", value=True):
                min_hia = st.slider("Minimum HIA Absorption (%)", 0.0, 100.0, 70.0, 5.0)
                targets["absorption"]["HIA_Absorption"] = {"min": min_hia, "weight": 1.0}
    
    with target_tabs[1]:
        st.markdown("### Distribution Targets")
        targets["distribution"] = {}
        col1, col2 = st.columns(2)
        with col1:
            if st.checkbox("Target BBB Penetration", value=True):
                bbb_target = st.radio("BBB Penetration Goal", ["High (CNS drugs)", "Low (peripheral drugs)"])
                if bbb_target == "High (CNS drugs)":
                    min_bbb = st.slider("Minimum BBB Penetration", 0.0, 1.0, 0.7, 0.1)
                    targets["distribution"]["BBB_Penetration"] = {"min": min_bbb, "weight": 1.0}
                else:
                    max_bbb = st.slider("Maximum BBB Penetration", 0.0, 1.0, 0.3, 0.1)
                    targets["distribution"]["BBB_Penetration"] = {"max": max_bbb, "weight": 1.0}
        with col2:
            if st.checkbox("Optimize Plasma Protein Binding", value=True):
                max_ppb = st.slider("Maximum Plasma Protein Binding (%)", 0.0, 100.0, 90.0, 5.0)
                targets["distribution"]["PPB_Binding"] = {"max": max_ppb, "weight": 0.8}
    
    with target_tabs[2]:
        st.markdown("### Metabolism Targets")
        targets["metabolism"] = {}
        col1, col2 = st.columns(2)
        with col1:
            if st.checkbox("Minimize CYP Substrate Potential", value=True):
                max_cyp = st.slider("Maximum CYP3A4 Substrate Probability", 0.0, 1.0, 0.5, 0.1)
                targets["metabolism"]["CYP3A4_Substrate"] = {"max": max_cyp, "weight": 1.0}
        with col2:
            if st.checkbox("Target Optimal Half-Life", value=True):
                min_hl = st.slider("Minimum Half-Life (hours)", 2.0, 24.0, 6.0, 1.0)
                max_hl = st.slider("Maximum Half-Life (hours)", min_hl, 48.0, 24.0, 1.0)
                targets["metabolism"]["Half_Life"] = {"min": min_hl, "max": max_hl, "weight": 0.8}
    
    with target_tabs[3]:
        st.markdown("### Toxicity Targets")
        targets["toxicity"] = {}
        col1, col2 = st.columns(2)
        with col1:
            if st.checkbox("Minimize hERG Inhibition", value=True):
                max_herg = st.slider("Maximum hERG Inhibition", 0.0, 1.0, 0.3, 0.1)
                targets["toxicity"]["hERG_Inhibition"] = {"max": max_herg, "weight": 1.2}
        with col2:
            if st.checkbox("Minimize Hepatotoxicity", value=True):
                max_hepato = st.slider("Maximum Hepatotoxicity", 0.0, 1.0, 0.3, 0.1)
                targets["toxicity"]["Hepatotoxicity"] = {"max": max_hepato, "weight": 1.1}
    
    optimizer.set_property_targets(targets)
    initial_score = optimizer.calculate_admet_score(selected_smiles)
    
    with col2:
        st.subheader("ADMET Score")
        st.metric("Current ADMET Score", f"{initial_score:.3f}")
        st.info("Higher ADMET score (closer to 1.0) indicates better alignment with target properties.")
    
    if st.button("Run ADMET Optimization"):
        with st.spinner("Optimizing molecule..."):
            optimization_results = optimizer.optimize_admet(selected_smiles, n_iter=10)
            if optimization_results:
                st.subheader("Optimization Results")
                opt_fig = create_admet_optimization_plot(optimization_results)
                st.plotly_chart(opt_fig, use_container_width=True)
                
                st.subheader("Optimized Molecule")
                st.info("Note: This is a simulated optimization. In a real implementation, this would show the actual optimized molecule.")
                
                final_result = optimization_results[-1]
                st.metric("Final ADMET Score", f"{final_result['score']:.3f}", 
                         f"{final_result['score'] - initial_score:.3f}")
                
                st.subheader("Key Property Improvements")
                for category in ["absorption", "distribution", "metabolism", "toxicity"]:
                    for prop, target in targets.get(category, {}).items():
                        initial_value = admet_results[category][prop]
                        final_value = final_result["properties"][category][prop]
                        delta = final_value - initial_value
                        delta_str = f"{delta:+.2f}"
                        is_improvement = "min" in target and final_value > initial_value or \
                                       "max" in target and final_value < initial_value
                        st.metric(
                            f"{category.capitalize()} - {prop}",
                            f"{final_value:.2f}",
                            delta_str,
                            delta_color="normal" if is_improvement else "inverse"
                        )
            else:
                st.error("Optimization failed. Please check your molecule and targets.")


def display_ai_analysis_page(admet_engine):
    """Display the AI analysis page."""
    st.markdown("<h2 class='section-header'>AI Expert Analysis</h2>", unsafe_allow_html=True)
    
    input_method = st.radio("Input Method", ["Draw Molecule", "Enter SMILES"])
    if input_method == "Draw Molecule":
        st.warning("Molecule drawing is not implemented in this demo. Please enter SMILES directly.")
    
    smiles = st.text_input("Enter SMILES String", "CC(=O)OC1=CC=CC=C1C(=O)O")
    if smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            st.subheader("Molecule Structure")
            display_molecule_structure(smiles)
            
            with st.spinner("Calculating ADMET properties..."):
                admet_results = admet_engine.predict_all_properties(smiles)
            
            st.subheader("ADMET Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Oral Absorption", f"{admet_results['absorption']['HIA_Absorption']:.1f}%")
                st.metric("hERG Inhibition", f"{admet_results['toxicity']['hERG_Inhibition']:.2f}")
            with col2:
                st.metric("BBB Penetration", f"{admet_results['distribution']['BBB_Penetration']:.2f}")
                st.metric("CYP3A4 Substrate", f"{admet_results['metabolism']['CYP3A4_Substrate']:.2f}")
            with col3:
                st.metric("Half-Life", f"{admet_results['metabolism']['Half_Life']:.1f} hrs")
                st.metric("Hepatotoxicity", f"{admet_results['toxicity']['Hepatotoxicity']:.2f}")
            
            if st.button("Generate AI Expert Analysis"):
                with st.spinner("Generating expert analysis..."):
                    # Get the analysis directly from admet_expert_analysis
                    analysis = admet_expert_analysis(smiles, admet_results)
                    st.subheader("Expert Analysis")
                    st.markdown(analysis)
        else:
            st.error("Invalid SMILES string. Please check your input.")
    else:
        st.info("Enter a SMILES string to analyze.")


# ------------------------------ Main App ------------------------------
def main():
    """Main application."""
    if "admet_engine" not in st.session_state:
        st.session_state.admet_engine = ADMETPredictionEngine()
    if "project" not in st.session_state:
        st.session_state.project = None
    
    st.markdown("<h1 style='color: #1E6091;'>ADMET Prediction and Analysis Platform</h1>", unsafe_allow_html=True)
    
    with st.sidebar:
        st.image("https://raw.githubusercontent.com/streamlit/logo/master/streamlit-mark-color.png", width=100)
        st.markdown("## Navigation")
        app_mode = st.radio(
            "Select Page",
            ["Project Management", "Single Molecule Analysis", "Comparative Analysis", 
             "Structure-ADMET Analysis", "ADMET Optimization", "AI Expert Analysis"]
        )
        st.markdown("---")
        
        st.markdown("## Project")
        if st.button("New Project"):
            project_name = st.text_input("Project Name", "New Project")
            reference_smiles = st.text_input("Reference Molecule (SMILES)", "CC(=O)OC1=CC=CC=C1C(=O)O")
            if st.button("Create Project"):
                st.session_state.project = Project(project_name, reference_smiles)
                st.success(f"Project '{project_name}' created!")
        
        if st.session_state.project:
            st.markdown(f"**Current Project:** {st.session_state.project.name}")
            st.markdown(f"**Molecules:** {len(st.session_state.project.history) + 1}")
    
    if st.session_state.project is None:
        st.session_state.project = Project("Demo Project", "CC(=O)OC1=CC=CC=C1C(=O)O")
        demo_molecules = [
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
            "CC(=O)NC1=CC=C(C=C1)O",
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "CCN(CC)CC(=O)NC1=CC=C(C=C1)OC"
        ]
        for i, smiles in enumerate(demo_molecules):
            mol_record = MoleculeRecord(smiles, f"Demo Molecule {i+1}")
            mol_record.calculate_admet(st.session_state.admet_engine)
            st.session_state.project.add_molecule(mol_record)
    
    molecules = [st.session_state.project.reference_smiles] + \
                [record.smiles for record in st.session_state.project.history]
    
    if app_mode == "Project Management":
        st.markdown("<h2 class='section-header'>Project Management</h2>", unsafe_allow_html=True)
        st.subheader("Project Details")
        if st.session_state.project:
            st.write(f"**Name:** {st.session_state.project.name}")
            st.write(f"**Reference Molecule:** {st.session_state.project.reference_smiles}")
            st.write(f"**Total Molecules:** {len(st.session_state.project.history) + 1}")
        
        st.subheader("Add Molecule")
        new_smiles = st.text_input("Molecule SMILES")
        mol_name = st.text_input("Molecule Name (optional)")
        if st.button("Add to Project"):
            if new_smiles:
                mol = Chem.MolFromSmiles(new_smiles)
                if mol:
                    mol_record = MoleculeRecord(new_smiles, mol_name if mol_name else None)
                    mol_record.calculate_admet(st.session_state.admet_engine)
                    st.session_state.project.add_molecule(mol_record)
                    st.success("Molecule added to project!")
                else:
                    st.error("Invalid SMILES string. Please check your input.")
            else:
                st.warning("Please enter a SMILES string.")
        
        st.subheader("Project Molecules")
        reference_mol = Chem.MolFromSmiles(st.session_state.project.reference_smiles)
        if reference_mol:
            st.markdown("#### Reference Molecule")
            col1, col2 = st.columns([1, 3])
            with col1:
                ref_img = mol_to_img(reference_mol, 200, 150)
                st.image(ref_img)
            with col2:
                st.write(f"**SMILES:** {st.session_state.project.reference_smiles}")
                st.write(f"**MW:** {Descriptors.MolWt(reference_mol):.2f} Da")
                st.write(f"**LogP:** {Descriptors.MolLogP(reference_mol):.2f}")
        
        st.markdown("#### Additional Molecules")
        for i, record in enumerate(st.session_state.project.history):
            mol = Chem.MolFromSmiles(record.smiles)
            if mol:
                with st.expander(f"{i+1}. {record.name}"):
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        mol_img = mol_to_img(mol, 200, 150)
                        st.image(mol_img)
                    with col2:
                        st.write(f"**SMILES:** {record.smiles}")
                        st.write(f"**MW:** {Descriptors.MolWt(mol):.2f} Da")
                        st.write(f"**LogP:** {Descriptors.MolLogP(mol):.2f}")
                        if record.admet_results:
                            st.write("**ADMET Highlights:**")
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.write(f"Absorption: {record.admet_results['absorption']['HIA_Absorption']:.1f}%")
                            with col_b:
                                st.write(f"BBB: {record.admet_results['distribution']['BBB_Penetration']:.2f}")
                            with col_c:
                                st.write(f"hERG: {record.admet_results['toxicity']['hERG_Inhibition']:.2f}")
    
    elif app_mode == "Single Molecule Analysis":
        display_single_molecule_admet(molecules, st.session_state.admet_engine)
    elif app_mode == "Comparative Analysis":
        display_comparative_admet(molecules, st.session_state.admet_engine)
    elif app_mode == "Structure-ADMET Analysis":
        display_structure_admet_relationships(molecules, st.session_state.admet_engine)
    elif app_mode == "ADMET Optimization":
        display_admet_optimization(molecules, st.session_state.admet_engine)
    elif app_mode == "AI Expert Analysis":
        display_ai_analysis_page(st.session_state.admet_engine)


if __name__ == "__main__":
    main()