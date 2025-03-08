# """Advanced Drug Discovery Optimization Platform with integrated scoring and analysis."""

import streamlit as st
import numpy as np
import pandas as pd
import cma
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.QED import qed
from rdkit.Chem import rdFingerprintGenerator, GetSymmSSSR 
from rdkit.Chem.Descriptors import MolLogP as rdkit_logp  # noqa
from rdkit.DataStructs import TanimotoSimilarity, BulkTanimotoSimilarity
from rdkit.Chem.Scaffolds import MurckoScaffold
import logging
from multiprocessing import Pool, cpu_count
from typing import List, Callable, Dict, Tuple, Optional, Union, Set
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from dataclasses import dataclass, field
from datetime import datetime
import pickle
from pathlib import Path
import tempfile
import json
import os
import sys
import requests
import io
import base64
from tqdm import tqdm

# DeepSeek API setup
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-cb3e2fbf705d4c1899a7cf53c49fbaa6")
from openai import OpenAI
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

# RDKit Contrib for Synthetic Accessibility
try:
    from rdkit.Chem import RDConfig
    import sys
    sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
    from sascorer import calculateScore
    SA_SCORE_AVAILABLE = True
except ImportError:
    SA_SCORE_AVAILABLE = False
    logging.warning("SA_Score unavailable. Install RDKit Contrib for synthetic accessibility.")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# === Data Classes and Container Objects === #

@dataclass
class OptimizationResult:
    """Stores results from molecule optimization runs."""
    smiles: str
    qed_score: float
    tanimoto_score: float
    sa_score: float
    timestamp: datetime
    parameters: dict
    properties: Dict[str, float] = field(default_factory=dict)
    pareto_front: bool = False
    cluster_id: Optional[int] = None
    
    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            'smiles': self.smiles,
            'qed_score': self.qed_score,
            'tanimoto_score': self.tanimoto_score,
            'sa_score': self.sa_score,
            'timestamp': self.timestamp.isoformat(),
            'parameters': self.parameters,
            'properties': self.properties,
            'pareto_front': self.pareto_front,
            'cluster_id': self.cluster_id
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create instance from dictionary."""
        result = cls(
            smiles=data['smiles'],
            qed_score=data['qed_score'],
            tanimoto_score=data['tanimoto_score'],
            sa_score=data['sa_score'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            parameters=data['parameters']
        )
        if 'properties' in data:
            result.properties = data['properties']
        if 'pareto_front' in data:
            result.pareto_front = data['pareto_front']
        if 'cluster_id' in data:
            result.cluster_id = data['cluster_id']
        return result


@dataclass
class OptimizationStrategy:
    """Configuration for molecule optimization strategy."""
    algorithm: str = "cmaes"  # Options: cmaes, genetic, bayesian
    property_weights: Dict[str, float] = field(default_factory=lambda: {
        "qed": 1.0, 
        "tanimoto": 1.0, 
        "sa_score": 0.1,
        "novelty": 0.5
    })
    constraints: Dict[str, Dict[str, float]] = field(default_factory=dict)
    exploration_rate: float = 1.0
    population_size: int = 50
    n_iterations: int = 50
    mutation_rate: float = 0.1  # For genetic algorithm
    elitism_rate: float = 0.1   # For genetic algorithm
    
    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            'algorithm': self.algorithm,
            'property_weights': self.property_weights,
            'constraints': self.constraints,
            'exploration_rate': self.exploration_rate,
            'population_size': self.population_size,
            'n_iterations': self.n_iterations,
            'mutation_rate': self.mutation_rate,
            'elitism_rate': self.elitism_rate
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create instance from dictionary."""
        return cls(
            algorithm=data['algorithm'],
            property_weights=data['property_weights'],
            constraints=data.get('constraints', {}),
            exploration_rate=data['exploration_rate'],
            population_size=data['population_size'],
            n_iterations=data['n_iterations'],
            mutation_rate=data.get('mutation_rate', 0.1),
            elitism_rate=data.get('elitism_rate', 0.1)
        )


@dataclass
class ProjectState:
    """State container for project data and configuration."""
    reference_smiles: str
    optimization_strategy: OptimizationStrategy
    history: List[OptimizationResult] = field(default_factory=list)
    name: str = "Untitled Project"
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    version: int = 1
    
    def save(self, filepath):
        """Save project state to file."""
        self.updated_at = datetime.now()
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath):
        """Load project state from file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
            
    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            'reference_smiles': self.reference_smiles,
            'optimization_strategy': self.optimization_strategy.to_dict(),
            'history': [result.to_dict() for result in self.history],
            'name': self.name,
            'description': self.description,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'version': self.version
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create instance from dictionary."""
        project = cls(
            reference_smiles=data['reference_smiles'],
            optimization_strategy=OptimizationStrategy.from_dict(data['optimization_strategy']),
            name=data['name'],
            description=data['description'],
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            version=data['version']
        )
        project.history = [OptimizationResult.from_dict(result) for result in data['history']]
        return project


# === Core Classes === #

class MolMIMClient:
    """Client for interacting with a molecule generative model server."""
    
    def __init__(self, host: str = "localhost", port: str = "8000"):
        self.base_url = f"http://{host}:{port}"
        self.available = False
        self.check_health()
        
    def check_health(self) -> bool:
        """Check if MolMIM server is accessible."""
        try:
            response = requests.get(f"{self.base_url}/v1/health/ready", timeout=2)
            self.available = response.status_code == 200
            return self.available
        except Exception as e:
            logging.warning(f"Unable to connect to MolMIM server: {str(e)}")
            self.available = False
            return False
            
    def get_hidden_state(self, smiles: str) -> np.ndarray:
        """Get latent space representation of molecule."""
        if not self.available:
            raise ConnectionError("MolMIM server not available")
            
        try:
            response = requests.post(
                f"{self.base_url}/hidden",
                headers={'Content-Type': 'application/json'},
                json={"sequences": [smiles]},
                timeout=10
            )
            response.raise_for_status()
            return np.squeeze(np.array(response.json()["hiddens"]))
        except Exception as e:
            logging.error(f"Error getting hidden state: {str(e)}")
            raise
        
    def decode_hidden(self, hidden_states: np.ndarray) -> List[str]:
        """Decode latent vectors to SMILES strings."""
        if not self.available:
            raise ConnectionError("MolMIM server not available")
            
        try:
            # Ensure hidden_states is properly shaped for the API
            if len(hidden_states.shape) == 1:
                hidden_states = np.expand_dims(hidden_states, axis=0)
                
            hiddens_array = hidden_states if len(hidden_states.shape) > 1 else np.expand_dims(hidden_states, axis=1)
            mask = [[True] for _ in range(hiddens_array.shape[0])]
            
            response = requests.post(
                f"{self.base_url}/decode",
                headers={'Content-Type': 'application/json'},
                json={"hiddens": hiddens_array.tolist(), "mask": mask},
                timeout=30
            )
            response.raise_for_status()
            # Remove duplicates while preserving order
            return list(dict.fromkeys(response.json()['generated']))
        except Exception as e:
            logging.error(f"Error decoding hidden state: {str(e)}")
            raise
            
    def decode_batch(self, hidden_states_batch: List[np.ndarray], batch_size: int = 10) -> List[str]:
        """Decode multiple latent vectors in batches."""
        all_molecules = []
        
        for i in range(0, len(hidden_states_batch), batch_size):
            batch = hidden_states_batch[i:i+batch_size]
            try:
                molecules = self.decode_hidden(np.array(batch))
                all_molecules.extend(molecules)
            except Exception as e:
                logging.error(f"Error in batch {i//batch_size}: {str(e)}")
                
        return all_molecules


def _score_mol(args):
    """Worker function for parallel molecule scoring."""
    smi, scorer, default_val = args
    mol = Chem.MolFromSmiles(smi)
    return scorer(mol) if mol else default_val


def _iterate_and_score_smiles(
    smis: List[str], scorer: Callable[[Chem.Mol], float], default_val: float = 0.0, parallel: bool = False, n_jobs: int = None
) -> np.ndarray:
    """Batch-process SMILES with a scoring function, optionally in parallel."""
    if not smis:
        return np.array([])
        
    if parallel and len(smis) > 10:
        n_jobs = n_jobs or max(1, min(cpu_count() - 1, len(smis) // 5))
        with Pool(processes=n_jobs) as pool:
            results = pool.map(_score_mol, [(smi, scorer, default_val) for smi in smis])
        return np.array(results)
        
    results = np.zeros((len(smis),)) + default_val
    for i, smi in enumerate(smis):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            logging.warning(f"Failed to parse SMILES at index {i}: {smi}")
            continue
        results[i] = scorer(mol)
    return results


class MoleculeOptimizer:
    """Core engine for optimizing molecules using generative models."""
    
    def __init__(self, reference_smiles: str, radius: int = 2, fp_size: int = 2048):
        self.reference_smiles = reference_smiles
        self.reference_mol = Chem.MolFromSmiles(reference_smiles)
        if not self.reference_mol:
            raise ValueError(f"Invalid reference SMILES: {reference_smiles}")
            
        self.reference_qed = qed(self.reference_mol)
        self.history: List[OptimizationResult] = []
        self.radius = radius
        self.fp_size = fp_size
        self.mfpgen = AllChem.GetMorganGenerator(radius=radius, fpSize=fp_size)
        self.reference_fp = self.mfpgen.GetFingerprint(self.reference_mol)
        self.seen_molecules: Set[str] = {reference_smiles}
        
    def tanimoto_similarity(self, smis: List[str], parallel: bool = False) -> np.ndarray:
        """Compute Tanimoto similarity to the reference molecule."""
        def scorer(mol): 
            return TanimotoSimilarity(self.mfpgen.GetFingerprint(mol), self.reference_fp)
        return _iterate_and_score_smiles(smis, scorer, default_val=0.0, parallel=parallel)

    def qed(self, smis: List[str], parallel: bool = False) -> np.ndarray:
        """Compute QED (drug-likeness) scores."""
        return _iterate_and_score_smiles(smis, qed, default_val=0.0, parallel=parallel)

    def synthetic_accessibility(self, smis: List[str], parallel: bool = False) -> np.ndarray:
        """Compute synthetic accessibility scores (lower is better)."""
        if not SA_SCORE_AVAILABLE:
            logging.error("SA_Score not available. Returning zeros.")
            return np.zeros((len(smis),))
        return _iterate_and_score_smiles(smis, calculateScore, default_val=10.0, parallel=parallel)
        
    def novelty(self, smis: List[str], parallel: bool = False) -> np.ndarray:
        """Compute novelty scores based on distance from previously seen molecules."""
        # Handle empty list
        if not smis:
            return np.array([])
        
        # Filter out invalid molecules
        valid_mols = [(i, Chem.MolFromSmiles(smi)) for i, smi in enumerate(smis)]
        valid_indices = [i for i, mol in valid_mols if mol is not None]
        valid_mols = [mol for _, mol in valid_mols if mol is not None]
        
        if not valid_mols:
            return np.zeros((len(smis),))
        
        # Generate fingerprints for all valid molecules
        fps = [self.mfpgen.GetFingerprint(mol) for mol in valid_mols]
        
        # Get fingerprints for all previously seen molecules
        seen_mols = [Chem.MolFromSmiles(smi) for smi in self.seen_molecules if Chem.MolFromSmiles(smi)]
        seen_fps = [self.mfpgen.GetFingerprint(mol) for mol in seen_mols]
        
        # Calculate maximum similarity to any previously seen molecule
        novelty_scores = np.zeros(len(smis))
        
        for idx, fp in zip(valid_indices, fps):
            if seen_fps:
                # Calculate similarity to all seen molecules
                similarities = BulkTanimotoSimilarity(fp, seen_fps)
                # Novelty is 1 minus maximum similarity
                max_similarity = max(similarities) if similarities else 0
                novelty_scores[idx] = 1.0 - max_similarity
            else:
                # If no previously seen molecules, novelty is maximum
                novelty_scores[idx] = 1.0
                
        return novelty_scores

    def scoring_function(
        self, 
        smis: List[str], 
        weights: Dict[str, float] = None, 
        constraints: Dict[str, Dict[str, float]] = None,
        parallel: bool = False
    ) -> np.ndarray:
        """Combined score based on multiple properties with configurable weights."""
        if not smis:
            return np.array([])
            
        # Default weights
        default_weights = {
            "qed": 1.0, 
            "tanimoto": 1.0, 
            "sa_score": 0.1,
            "novelty": 0.5
        }
        weights = weights or default_weights
        
        # Compute property scores
        qeds = self.qed(smis, parallel)
        similarities = self.tanimoto_similarity(smis, parallel)
        sa_scores = self.synthetic_accessibility(smis, parallel)
        novelty_scores = self.novelty(smis, parallel)
        
        # Normalize scores
        normalized_qed = np.clip(qeds / 0.9, 0, 1) * weights.get("qed", default_weights["qed"])
        normalized_similarity = np.clip(similarities / 0.4, 0, 1) * weights.get("tanimoto", default_weights["tanimoto"])
        normalized_sa = np.clip((10 - sa_scores) / 9, 0, 1) * weights.get("sa_score", default_weights["sa_score"])
        normalized_novelty = novelty_scores * weights.get("novelty", default_weights["novelty"])
        
        # Apply constraints if provided
        penalty = np.zeros(len(smis))
        if constraints:
            # Get molecules
            mols = [Chem.MolFromSmiles(smi) for smi in smis]
            
            for prop, constraint in constraints.items():
                if prop == "molecular_weight":
                    values = np.array([Descriptors.ExactMolWt(mol) if mol else 0 for mol in mols])
                elif prop == "logP":
                    values = np.array([Descriptors.MolLogP(mol) if mol else 0 for mol in mols])
                elif prop == "hbd":
                    values = np.array([Descriptors.NumHDonors(mol) if mol else 0 for mol in mols])
                elif prop == "hba":
                    values = np.array([Descriptors.NumHAcceptors(mol) if mol else 0 for mol in mols])
                elif prop == "rotatable_bonds":
                    values = np.array([Descriptors.NumRotatableBonds(mol) if mol else 0 for mol in mols])
                else:
                    continue
                
                # Apply min/max constraints with smooth penalty
                if "min" in constraint:
                    min_val = constraint["min"]
                    penalty += np.maximum(0, (min_val - values) / min_val) * 5
                if "max" in constraint:
                    max_val = constraint["max"]
                    penalty += np.maximum(0, (values - max_val) / max_val) * 5
        
        # Combined score (negative because optimization algorithms minimize)
        combined_score = -1.0 * (normalized_qed + normalized_similarity + normalized_sa + normalized_novelty)
        
        # Add penalties from constraints
        combined_score += penalty
        
        # Update seen molecules set for novelty calculation
        self.seen_molecules.update(smis)
        
        return combined_score

    def compute_pareto_front(self, results: List[OptimizationResult], objectives=None):
        """Identify Pareto-optimal solutions for multi-objective optimization."""
        if not objectives:
            objectives = ["qed_score", "tanimoto_score", "sa_score"]
        
        # Reset all pareto flags
        for result in results:
            result.pareto_front = False
        
        # Check if there are enough results to compute a Pareto front
        if len(results) < 2:
            if len(results) == 1:
                results[0].pareto_front = True  # Single result is trivially Pareto-optimal
            return results  # Return early if fewer than 2 results
        
        # Extract values for each objective, ensuring 2D array
        values = np.array([[getattr(r, obj) for obj in objectives] for r in results], dtype=float)
        if values.ndim == 1:  # If only one result was coerced to 1D
            values = values.reshape(1, -1)  # Reshape to (1, n_objectives)
        
        # For each solution, check if it's dominated by any other
        n = len(results)
        is_efficient = np.ones(n, dtype=bool)
        
        # Adjust for objectives that should be minimized (e.g., SA score)
        minimize = [False, False, True]  # QED and Tanimoto: maximize, SA score: minimize
        values_adjusted = values.copy()
        for i, min_obj in enumerate(minimize):
            if min_obj:
                values_adjusted[:, i] = -values_adjusted[:, i]
        
        for i in range(n):
            if not is_efficient[i]:
                continue
            for j in range(n):
                if i == j or not is_efficient[j]:
                    continue
                if np.all(values_adjusted[j] >= values_adjusted[i]) and np.any(values_adjusted[j] > values_adjusted[i]):
                    is_efficient[i] = False
                    break
        
        # Update pareto front flags
        for i, result in enumerate(results):
            result.pareto_front = is_efficient[i]
        
        return [r for r, is_pareto in zip(results, is_efficient) if is_pareto]

    def extract_scaffolds(self, smiles_list: List[str]) -> Dict[str, List[str]]:
        """Group molecules by their Murcko scaffolds."""
        scaffolds = {}
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if not mol:
                continue
                
            # Get scaffold
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_smiles = Chem.MolToSmiles(scaffold) if scaffold else ""
            
            if scaffold_smiles not in scaffolds:
                scaffolds[scaffold_smiles] = []
            scaffolds[scaffold_smiles].append(smi)
            
        return scaffolds
        
    def cluster_molecules(self, smiles_list: List[str], n_clusters: int = 5) -> Dict[str, int]:
        """Cluster molecules based on fingerprint similarity."""
        # Generate fingerprints
        mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
        valid_indices = [i for i, mol in enumerate(mols) if mol is not None]
        valid_mols = [mol for mol in mols if mol is not None]
        valid_smiles = [smiles_list[i] for i in valid_indices]
        
        if len(valid_mols) < n_clusters:
            return {smi: 0 for smi in valid_smiles}
        
        # Calculate fingerprints
        fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048) for mol in valid_mols]
        fp_array = []
        for fp in fps:
            arr = np.zeros((0,))
            DataStructs.ConvertToNumpyArray(fp, arr)
            fp_array.append(arr)
        
        # Cluster fingerprints
        X = np.array(fp_array)
        kmeans = KMeans(n_clusters=min(n_clusters, len(X)), random_state=42).fit(X)
        clusters = kmeans.labels_
        
        # Map SMILES to cluster IDs
        return {smi: int(cluster) for smi, cluster in zip(valid_smiles, clusters)}

    def save_history(self, filepath: str):
        """Save optimization history to a file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.history, f)
            
    def load_history(self, filepath: str):
        """Load optimization history from a file."""
        with open(filepath, 'rb') as f:
            self.history = pickle.load(f)

    def save_result(self, smiles: str, qed_score: float, tanimoto_score: float, sa_score: float, params: dict):
        """Save a result to the optimization history."""
        self.history.append(OptimizationResult(
            smiles=smiles, 
            qed_score=qed_score, 
            tanimoto_score=tanimoto_score,
            sa_score=sa_score, 
            timestamp=datetime.now(), 
            parameters=params
        ))


class ConditionalMoleculeGenerator:
    """Generate molecules with specific property constraints."""
    
    def __init__(self, client: MolMIMClient, reference_smiles: str = None):
        self.client = client
        self.reference_smiles = reference_smiles
        self.reference_mol = Chem.MolFromSmiles(reference_smiles) if reference_smiles else None
        self.property_constraints = {}
        self.optimizer = MoleculeOptimizer(reference_smiles) if reference_smiles else None
        
    def set_reference(self, smiles: str):
        """Set or update reference molecule."""
        self.reference_smiles = smiles
        self.reference_mol = Chem.MolFromSmiles(smiles)
        self.optimizer = MoleculeOptimizer(smiles)
        
    def set_constraint(self, property_name: str, min_value: float = None, max_value: float = None, target_value: float = None):
        """Set a property constraint for generation."""
        self.property_constraints[property_name] = {
            'min': min_value,
            'max': max_value,
            'target': target_value
        }
        
    def clear_constraints(self):
        """Clear all property constraints."""
        self.property_constraints = {}
        
    def generate_with_constraints(self, seed_smiles: str = None, num_molecules: int = 10, n_iter: int = 5):
        """Generate molecules meeting specified property constraints."""
        seed_smiles = seed_smiles or self.reference_smiles
        if not seed_smiles:
            raise ValueError("No seed SMILES provided")
            
        if not self.client.available:
            raise ConnectionError("MolMIM server not available")
            
        hidden_state = self.client.get_hidden_state(seed_smiles)
        
        # Use CMA-ES to optimize hidden state to meet constraints
        sigma = 1.0
        es = cma.CMAEvolutionStrategy(hidden_state, sigma, {'popsize': 20})
        
        best_molecules = []
        scores = []
        
        for _ in range(n_iter):
            # Generate candidates
            solutions = es.ask()
            molecules = self.client.decode_batch(solutions)
            
            # Filter valid molecules
            valid_mols = [(i, smi) for i, smi in enumerate(molecules) if Chem.MolFromSmiles(smi)]
            
            if not valid_mols:
                continue
                
            # Score candidates based on constraints
            indices = [i for i, _ in valid_mols]
            valid_smiles = [smi for _, smi in valid_mols]
            
            # Calculate property scores
            if self.optimizer:
                mol_properties = {}
                mol_properties["qed"] = self.optimizer.qed(valid_smiles)
                mol_properties["tanimoto"] = self.optimizer.tanimoto_similarity(valid_smiles)
                mol_properties["sa_score"] = self.optimizer.synthetic_accessibility(valid_smiles)
                
                # Calculate constraint scores
                constraint_scores = np.zeros(len(valid_smiles))
                
                for prop_name, constraint in self.property_constraints.items():
                    # Calculate property values
                    values = self._calculate_property(valid_smiles, prop_name)
                    
                    # Calculate penalty for constraint violations
                    penalty = np.zeros(len(valid_smiles))
                    
                    if "min" in constraint and constraint["min"] is not None:
                        min_val = constraint["min"]
                        penalty += np.maximum(0, min_val - values) * 5.0
                        
                    if "max" in constraint and constraint["max"] is not None:
                        max_val = constraint["max"]
                        penalty += np.maximum(0, values - max_val) * 5.0
                        
                    if "target" in constraint and constraint["target"] is not None:
                        target = constraint["target"]
                        penalty += np.abs(values - target) * 3.0
                        
                    constraint_scores += penalty
                
                # Update CMA-ES with constraints
                objective_scores = np.zeros(len(solutions))
                for i, idx in enumerate(indices):
                    objective_scores[idx] = constraint_scores[i]
                
                es.tell(solutions, objective_scores)
                
                # Save results
                for i, smiles in enumerate(valid_smiles):
                    score = constraint_scores[i]
                    best_molecules.append(smiles)
                    scores.append(score)
                    
                    # Save for debugging
                    properties = {
                        "qed": mol_properties["qed"][i],
                        "tanimoto": mol_properties["tanimoto"][i],
                        "sa_score": mol_properties["sa_score"][i],
                        "constraint_score": score
                    }
                    
                    for prop_name in self.property_constraints:
                        value = self._calculate_property([smiles], prop_name)[0]
                        properties[prop_name] = value
            else:
                # If no optimizer, just return valid molecules
                best_molecules.extend(valid_smiles)
        
        # Sort by constraint score (lower is better)
        if scores:
            sorted_molecules = [x for _, x in sorted(zip(scores, best_molecules))]
            return sorted_molecules[:num_molecules]
        
        # Deduplicate and return
        return list(dict.fromkeys(best_molecules))[:num_molecules]
        
    def fragment_based_generation(self, 
                                core_fragment: str, 
                                num_molecules: int = 10, 
                                attachment_points: List[int] = None,
                                decorations_library: List[str] = None):
        """Generate molecules by attaching fragments to a core structure."""
        # Parse core fragment
        core_mol = Chem.MolFromSmiles(core_fragment)
        if not core_mol:
            raise ValueError(f"Invalid core fragment SMILES: {core_fragment}")
            
        # If no attachment points specified, try to identify them
        if attachment_points is None:
            attachment_points = self._identify_attachment_points(core_mol)
            
        # If no decoration library provided, use a default one
        if decorations_library is None:
            decorations_library = self._get_default_fragments()
            
        # Generate molecules by attaching fragments
        generated_molecules = []
        
        for attachment in attachment_points:
            # Create attachment point molecule
            attachment_mol = Chem.RWMol(core_mol)
            
            # Add attachment point marker (*) 
            if attachment < attachment_mol.GetNumAtoms():
                # Add dummy atom with atomic number 0 connected to attachment point
                atom_idx = attachment_mol.AddAtom(Chem.Atom(0))
                attachment_mol.AddBond(attachment, atom_idx, Chem.BondType.SINGLE)
                
                attachment_smiles = Chem.MolToSmiles(attachment_mol)
                
                # Try to generate molecules using the model
                if self.client.available:
                    try:
                        hidden_state = self.client.get_hidden_state(attachment_smiles)
                        sigma = 1.0
                        es = cma.CMAEvolutionStrategy(hidden_state, sigma, {'popsize': 20})
                        
                        for _ in range(3):  # 3 iterations for diversity
                            solutions = es.ask()
                            molecules = self.client.decode_batch(solutions)
                            valid_mols = [smi for smi in molecules if Chem.MolFromSmiles(smi)]
                            generated_molecules.extend(valid_mols)
                    except Exception as e:
                        logging.error(f"Error in generative model-based fragment generation: {str(e)}")
        
        # If we didn't generate enough molecules or model failed, use direct fragment attachment
        if len(generated_molecules) < num_molecules:
            for attachment in attachment_points:
                for fragment in decorations_library:
                    try:
                        frag_mol = Chem.MolFromSmiles(fragment)
                        if not frag_mol:
                            continue
                            
                        # Combine fragment with core
                        combined = self._attach_fragment(core_mol, frag_mol, attachment)
                        if combined:
                            combined_smiles = Chem.MolToSmiles(combined)
                            generated_molecules.append(combined_smiles)
                    except Exception as e:
                        logging.warning(f"Error attaching fragment {fragment}: {str(e)}")
        
        # Deduplicate
        unique_molecules = list(dict.fromkeys(generated_molecules))
        
        # Score and rank if optimizer is available
        if self.optimizer:
            valid_mols = [smi for smi in unique_molecules if Chem.MolFromSmiles(smi)]
            
            if valid_mols:
                # Calculate property scores
                qed_scores = self.optimizer.qed(valid_mols)
                sa_scores = self.optimizer.synthetic_accessibility(valid_mols)
                
                # Combined score (higher is better)
                combined_scores = qed_scores - 0.1 * sa_scores
                
                # Sort by score
                sorted_molecules = [x for _, x in sorted(
                    zip(combined_scores, valid_mols), 
                    key=lambda pair: pair[0], 
                    reverse=True
                )]
                
                return sorted_molecules[:num_molecules]
        
        # If no scoring or no optimizer, just return unique molecules
        return unique_molecules[:num_molecules]
    
    def _identify_attachment_points(self, mol: Chem.Mol) -> List[int]:
        """Identify potential attachment points on a molecule."""
        attachment_points = []
        
        for atom in mol.GetAtoms():
            # Consider atoms with available valence
            if atom.GetImplicitValence() > 0:
                attachment_points.append(atom.GetIdx())
                
        # If no atoms with available valence, use hydrogen-bearing atoms
        if not attachment_points:
            for atom in mol.GetAtoms():
                if atom.GetNumImplicitHs() > 0:
                    attachment_points.append(atom.GetIdx())
                    
        return attachment_points
    
    def _get_default_fragments(self) -> List[str]:
        """Get a default library of fragments for decoration."""
        # Common fragments for medicinal chemistry
        return [
            "C", "CC", "CCC", "CCCC", "c1ccccc1", "c1ccncc1", "c1ccnnc1", 
            "C(=O)O", "C(=O)N", "CN", "CO", "CS", "CF", "CCl", "CBr", 
            "C#N", "N", "NC", "NC=O", "NS(=O)(=O)C", "NOC", "N1CCOCC1",
            "c1ccccc1C", "c1ccccc1N", "c1ccccc1O", "c1ccccc1F", "c1ccccc1Cl",
            "C1CCNCC1", "C1CCOCC1", "C1CCSCC1"
        ]
    
    def _attach_fragment(self, core_mol: Chem.Mol, frag_mol: Chem.Mol, attach_idx: int) -> Optional[Chem.Mol]:
        """Attach a fragment to a core molecule at specified attachment point."""
        if attach_idx >= core_mol.GetNumAtoms():
            return None
            
        # Make sure molecules are editable
        core_rw = Chem.RWMol(core_mol)
        frag_rw = Chem.RWMol(frag_mol)
        
        # Find an attachment point on the fragment (first H or first atom)
        frag_attach_idx = -1
        for atom in frag_rw.GetAtoms():
            if atom.GetNumImplicitHs() > 0:
                frag_attach_idx = atom.GetIdx()
                break
                
        if frag_attach_idx == -1 and frag_rw.GetNumAtoms() > 0:
            frag_attach_idx = 0
        
        if frag_attach_idx == -1:
            return None
            
        # Combine molecules
        combo = Chem.CombineMols(core_rw, frag_rw)
        combo_rw = Chem.RWMol(combo)
        
        # Add bond between attachment points
        combo_rw.AddBond(
            attach_idx, 
            frag_attach_idx + core_rw.GetNumAtoms(), 
            Chem.BondType.SINGLE
        )
        
        try:
            # Cleanup and sanitize
            Chem.SanitizeMol(combo_rw)
            return combo_rw
        except Exception:
            return None
    
    def scaffold_hopping(self, seed_smiles: str, num_hops: int = 10, scaffold_library: List[str] = None):
        """Generate molecules with similar properties but different scaffolds."""
        # Get seed molecule
        seed_mol = Chem.MolFromSmiles(seed_smiles)
        if not seed_mol:
            raise ValueError(f"Invalid seed SMILES: {seed_smiles}")
            
        # Default scaffold library if none provided
        if scaffold_library is None:
            # Load basic scaffolds
            scaffold_library = [
                "c1ccccc1", "c1cnccc1", "c1ccnnc1", "c1cnccn1", "c1ccncc1", 
                "C1CCCCC1", "C1CCNCC1", "C1CCOCC1", "C1CCSCC1", "C1CCNC1", 
                "c1ccc2ccccc2c1", "c1ccc2ncccc2c1", "c1ccc2[nH]ccc2c1",
                "C1CCC2CCCCC2C1", "C1CCC2NCCCC2C1", "C1CCC2OCCCC2C1",
                "C1=CC=CC=C1", "C1=CC=CN=C1", "c1nc2ccccc2[nH]1", "c1nc2ccccc2o1", 
                "c1nc2ccccc2s1", "c1cc(cc2c1)ccc2", "c1cc2cc(ccc2cc1)C"
            ]
            
        # Extract Murcko scaffold from seed
        seed_scaffold = MurckoScaffold.GetScaffoldForMol(seed_mol)
        seed_scaffold_smiles = Chem.MolToSmiles(seed_scaffold)
        
        # Generate new molecules
        hopped_molecules = []
        
        for scaffold in scaffold_library:
            # Skip if this is the same as the seed scaffold
            if scaffold == seed_scaffold_smiles:
                continue
                
            scaffold_mol = Chem.MolFromSmiles(scaffold)
            if not scaffold_mol:
                continue
                
            # Try model-based generation if available
            if self.client.available:
                try:
                    # Create a hybrid SMILES combining the seed and new scaffold
                    hybrid_smiles = self._create_hybrid_smiles(seed_smiles, scaffold)
                    
                    # Generate molecules using the hybrid as seed
                    hidden_state = self.client.get_hidden_state(hybrid_smiles)
                    
                    # Sample around this state
                    sigma = 0.5
                    solutions = []
                    for _ in range(5):  # Generate 5 samples per scaffold
                        perturbed = hidden_state + np.random.normal(0, sigma, size=hidden_state.shape)
                        solutions.append(perturbed)
                        
                    molecules = self.client.decode_batch(solutions)
                    valid_mols = [smi for smi in molecules if Chem.MolFromSmiles(smi)]
                    
                    # Verify that generated molecules contain the new scaffold
                    for smi in valid_mols:
                        mol = Chem.MolFromSmiles(smi)
                        mol_scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                        if mol_scaffold and Chem.MolToSmiles(mol_scaffold) == scaffold:
                            hopped_molecules.append(smi)
                except Exception as e:
                    logging.error(f"Error in generative scaffold hopping: {str(e)}")
            
        # Fill remaining slots with direct scaffold replacements if needed
        if len(hopped_molecules) < num_hops:
            # Try direct replacement for remaining scaffolds
            for scaffold in scaffold_library:
                if scaffold == seed_scaffold_smiles:
                    continue
                    
                scaffold_mol = Chem.MolFromSmiles(scaffold)
                if not scaffold_mol:
                    continue
                    
                try:
                    # Replace scaffold directly
                    new_mol = self._replace_scaffold(seed_mol, scaffold_mol)
                    if new_mol:
                        new_smiles = Chem.MolToSmiles(new_mol)
                        if new_smiles not in hopped_molecules:
                            hopped_molecules.append(new_smiles)
                except Exception as e:
                    logging.warning(f"Error in direct scaffold replacement: {str(e)}")
                    
                if len(hopped_molecules) >= num_hops:
                    break
        
        # Deduplicate and return
        return list(dict.fromkeys(hopped_molecules))[:num_hops]
    
    def _create_hybrid_smiles(self, seed_smiles: str, scaffold_smiles: str) -> str:
        """Create a hybrid SMILES combining properties of seed and scaffold."""
        # Simple concatenation - the model will learn to merge them
        return f"{seed_smiles}.{scaffold_smiles}"
    
    def _replace_scaffold(self, mol: Chem.Mol, scaffold: Chem.Mol) -> Optional[Chem.Mol]:
        """Replace the scaffold of a molecule with a new one."""
        try:
            # Get original scaffold
            orig_scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            if not orig_scaffold:
                return None
                
            # This is a simplified approach - production code would use more
            # sophisticated scaffold replacement algorithms
            return scaffold
        except:
            return None
    
    def _calculate_property(self, smiles_list: List[str], property_name: str) -> np.ndarray:
        """Calculate property values for a list of SMILES."""
        mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
        values = np.zeros(len(smiles_list))
        
        for i, mol in enumerate(mols):
            if not mol:
                continue
                
            if property_name == "qed":
                values[i] = qed(mol)
            elif property_name == "logP":
                values[i] = Descriptors.MolLogP(mol)
            elif property_name == "molecular_weight":
                values[i] = Descriptors.MolWt(mol)
            elif property_name == "hbd":
                values[i] = Descriptors.NumHDonors(mol)
            elif property_name == "hba":
                values[i] = Descriptors.NumHAcceptors(mol)
            elif property_name == "rotatable_bonds":
                values[i] = Descriptors.NumRotatableBonds(mol)
            elif property_name == "tpsa":
                values[i] = Descriptors.TPSA(mol)
            elif property_name == "aromatic_rings":
                values[i] = len(mol.GetAromaticAtoms())
            elif property_name == "tanimoto_similarity" and self.reference_mol:
                ref_fp = AllChem.GetMorganFingerprintAsBitVect(self.reference_mol, 2, 2048)
                mol_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
                values[i] = TanimotoSimilarity(ref_fp, mol_fp)
            elif property_name == "sa_score" and SA_SCORE_AVAILABLE:
                values[i] = calculateScore(mol)
                
        return values


class MoleculeValidator:
    """Validates molecules and computes drug-likeness properties."""
    
    def __init__(self):
        self.lipinski_rules = {'MW': 500, 'LogP': 5, 'HBD': 5, 'HBA': 10}
        self.veber_rules = {'RotBonds': 10, 'TPSA': 140}
        self.ghose_rules = {'MW': [160, 480], 'LogP': [-0.4, 5.6], 'MR': [40, 130], 'Atoms': [20, 70]}
        
    def validate_smiles(self, smiles: List[str], parallel: bool = False) -> Dict[str, dict]:
        """Validate and compute properties for a list of SMILES."""
        if parallel and len(smiles) > 10:
            with Pool(processes=max(1, cpu_count() - 1)) as pool:
                results = pool.map(self._get_properties, smiles)
            return {smi: result for smi, result in zip(smiles, results)}
        else:
            return {smi: self._get_properties(smi) for smi in smiles}
            
    def _get_properties(self, smi: str) -> dict:
        """Calculate properties for a single SMILES string."""
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            return {'valid': False, 'error': 'Invalid SMILES'}
            
        # Basic properties
        mw = Descriptors.ExactMolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        tpsa = Descriptors.TPSA(mol)
        rotatable_bonds = Descriptors.NumRotatableBonds(mol)
        aromatic_rings = len([ring for ring in GetSymmSSSR(mol) if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring)])
        heavy_atoms = mol.GetNumHeavyAtoms()
        mr = Descriptors.MolMR(mol)
        
        # Lipinski Rule of 5 violations
        lipinski_violations = sum([
            mw > self.lipinski_rules['MW'],
            logp > self.lipinski_rules['LogP'],
            hbd > self.lipinski_rules['HBD'],
            hba > self.lipinski_rules['HBA']
        ])
        
        # Veber rule violations
        veber_violations = sum([
            rotatable_bonds > self.veber_rules['RotBonds'],
            tpsa > self.veber_rules['TPSA']
        ])
        
        # Ghose rule violations
        ghose_violations = sum([
            mw < self.ghose_rules['MW'][0] or mw > self.ghose_rules['MW'][1],
            logp < self.ghose_rules['LogP'][0] or logp > self.ghose_rules['LogP'][1],
            mr < self.ghose_rules['MR'][0] or mr > self.ghose_rules['MR'][1],
            heavy_atoms < self.ghose_rules['Atoms'][0] or heavy_atoms > self.ghose_rules['Atoms'][1]
        ])
        
        # QED and SA score
        qed_value = qed(mol)
        sa_value = calculateScore(mol) if SA_SCORE_AVAILABLE else 0.0
        
        # Detailed property report
        props = {
            'molecular_weight': mw,
            'logP': logp,
            'hbd': hbd,
            'hba': hba,
            'tpsa': tpsa,
            'rotatable_bonds': rotatable_bonds,
            'aromatic_rings': aromatic_rings,
            'heavy_atoms': heavy_atoms,
            'molecular_refractivity': mr,
            'formal_charge': Chem.GetFormalCharge(mol),
            'qed': qed_value,
            'lipinski_violations': lipinski_violations,
            'veber_violations': veber_violations,
            'ghose_violations': ghose_violations,
            'synthetic_accessibility': sa_value,
            'passes_lipinski': lipinski_violations <= 1,  # Allowing one violation
            'passes_veber': veber_violations == 0,
            'passes_ghose': ghose_violations == 0
        }
        
        # Extended structural information
        try:
            props['ring_count'] = rdMolDescriptors.CalcNumRings(mol)
            props['fraction_sp3'] = rdMolDescriptors.CalcFractionCSP3(mol)
            props['stereo_centers'] = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
            props['murcko_scaffold'] = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
        except Exception as e:
            logging.warning(f"Error calculating extended properties: {str(e)}")
        
        return {'valid': True, 'properties': props}
    
    def calculate_diversity(self, smiles_list: List[str], method: str = "tanimoto") -> float:
        """Calculate diversity of a set of molecules (0-1 scale)."""
        if len(smiles_list) <= 1:
            return 0.0
            
        mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
        valid_mols = [mol for mol in mols if mol is not None]
        
        if len(valid_mols) <= 1:
            return 0.0
            
        # Calculate fingerprints
        fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048) for mol in valid_mols]
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(fps)):
            for j in range(i+1, len(fps)):
                similarities.append(TanimotoSimilarity(fps[i], fps[j]))
                
        # Diversity is 1 - average similarity
        return 1.0 - (sum(similarities) / len(similarities)) if similarities else 0.0
        
    def calculate_scaffold_diversity(self, smiles_list: List[str]) -> float:
        """Calculate scaffold diversity as fraction of unique scaffolds."""
        if not smiles_list:
            return 0.0
            
        # Extract scaffolds
        scaffolds = set()
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                if scaffold:
                    scaffolds.add(Chem.MolToSmiles(scaffold))
                    
        # Scaffold diversity is ratio of unique scaffolds to molecules
        return len(scaffolds) / len(smiles_list) if smiles_list else 0.0


class ConversationalMoleculeAnalyzer:
    """AI-powered analyzer for drug candidates with conversational interface."""
    
    def __init__(self):
        self.validator = MoleculeValidator()
        self.conversation_history = []
        
    def add_to_history(self, role: str, content: str):
        """Add message to conversation history."""
        self.conversation_history.append({"role": role, "content": content})
        
    def analyze_input_molecule(self, smiles: str) -> str:
        """Provide detailed AI analysis of a molecule."""
        validation = self.validator.validate_smiles([smiles])[smiles]
        if not validation['valid']:
            return f"Invalid molecule: {validation['error']}"
            
        props = validation['properties']
        
        # Get SVG representation for visualization reference
        mol = Chem.MolFromSmiles(smiles)
        svg = self._get_molecule_svg(mol)
        
        # Pre-compute fraction_sp3 display value
        fraction_sp3_display = f"{props['fraction_sp3']:.2f}" if 'fraction_sp3' in props else 'N/A'
        
        prompt = f"""
        As an expert pharmaceutical researcher, provide a detailed analysis of this molecule (SMILES: {smiles}):

        **Chemical Properties:**
        - Molecular Weight: {props['molecular_weight']:.1f}
        - LogP: {props['logP']:.1f}
        - TPSA: {props['tpsa']:.1f}
        - H-Bond Donors: {props['hbd']}
        - H-Bond Acceptors: {props['hba']}
        - Rotatable Bonds: {props['rotatable_bonds']}
        - Aromatic Rings: {props.get('aromatic_rings', 'N/A')}
        - QED (Drug-likeness): {props['qed']:.3f}
        - Synthetic Accessibility (SA): {props['synthetic_accessibility']:.1f}
        - Lipinski Violations: {props['lipinski_violations']}
        - Veber Rule Violations: {props.get('veber_violations', 'N/A')}
        - Fraction SP3: {fraction_sp3_display}
        - Stereo Centers: {props.get('stereo_centers', 'N/A')}
        - Murcko Scaffold: {props.get('murcko_scaffold', 'N/A')}

        **Instructions:**
        1. Assess drug-likeness using multiple criteria (Lipinski's Rule of Five, Veber rules, QED interpretation).
        2. Discuss synthetic feasibility based on SA score (1-10 scale, lower is easier) and structural complexity.
        3. Evaluate potential therapeutic applications based on physicochemical properties.
        4. Analyze absorption potential (oral bioavailability) and blood-brain barrier penetration potential.
        5. Identify potential metabolic liabilities and stability concerns.
        6. Highlight development challenges (solubility, stability, toxicity risks).
        7. Provide specific, actionable structural modifications to improve:
        - Drug-likeness
        - Synthetic accessibility 
        - Target binding (if binding is likely limited by current structure)
        - Metabolic stability
        8. Suggest 3-5 follow-up questions that would help refine the molecule further.

        Use clear, concise language suitable for a medicinal chemist, and ensure recommendations are specific and actionable.
        """
        
        messages = [{"role": "user", "content": prompt}]
        response = analyze_with_deepseek(messages)
        self.add_to_history("user", prompt)
        self.add_to_history("assistant", response)
        return response
        
    def compare_molecules(self, smiles_list: List[str]) -> str:
        """Compare multiple molecules and highlight key differences."""
        if len(smiles_list) < 2:
            return "Please provide at least two molecules to compare."
            
        # Validate molecules
        validations = self.validator.validate_smiles(smiles_list)
        valid_smiles = [smi for smi in smiles_list if validations[smi]['valid']]
        
        if len(valid_smiles) < 2:
            return "At least two valid molecules are required for comparison."
            
        # Gather properties for all molecules
        all_props = {smi: validations[smi]['properties'] for smi in valid_smiles}
        
        # Generate SVG representations
        svg_dict = {}
        for smi in valid_smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                svg_dict[smi] = self._get_molecule_svg(mol)
        
        # Format the properties table
        props_table = "| Property | " + " | ".join([f"Molecule {i+1}" for i in range(len(valid_smiles))]) + " |\n"
        props_table += "|" + "-|"*len(valid_smiles) + "-|\n"
        
        key_properties = [
            'molecular_weight', 'logP', 'tpsa', 'hbd', 'hba', 
            'rotatable_bonds', 'qed', 'synthetic_accessibility', 
            'lipinski_violations'
        ]
        
        for prop in key_properties:
            formatted_prop = prop.replace('_', ' ').title()
            props_table += f"| {formatted_prop} | "
            
            for smi in valid_smiles:
                value = all_props[smi].get(prop, "N/A")
                if isinstance(value, float):
                    props_table += f"{value:.2f} | "
                else:
                    props_table += f"{value} | "
            props_table += "\n"
        
        # Create prompt for AI analysis
        prompt = f"""
        As a medicinal chemistry expert, compare these {len(valid_smiles)} molecules:
        
        **SMILES Structures:**
        {', '.join([f"Molecule {i+1}: {smi}" for i, smi in enumerate(valid_smiles)])}
        
        **Property Comparison:**
        {props_table}
        
        Please provide:
        
        1. A detailed comparison highlighting key structural differences between these molecules
        2. Analysis of how these structural differences affect their drug-like properties
        3. Comparative assessment of their potential advantages/disadvantages for drug development
        4. Identification of which molecule(s) appear most promising and why
        5. Specific recommendations for further optimization based on the comparative analysis
        
        Focus particularly on structure-property relationships and medicinal chemistry insights.
        """
        
        messages = [{"role": "user", "content": prompt}]
        response = analyze_with_deepseek(messages)
        self.add_to_history("user", prompt)
        self.add_to_history("assistant", response)
        return response

    def analyze_optimization_results(self, results: List[OptimizationResult], reference_smiles: str) -> str:
        """Analyze the results of a molecule optimization run."""
        if not results:
            return "No optimization results to analyze."
            
        # Get reference molecule properties
        ref_mol = Chem.MolFromSmiles(reference_smiles)
        if not ref_mol:
            return "Invalid reference molecule."
            
        ref_props = self.validator.validate_smiles([reference_smiles])[reference_smiles]['properties']
        
        # Get properties for top molecules
        top_smiles = [r.smiles for r in results[:5]]  # Analyze top 5
        validations = self.validator.validate_smiles(top_smiles)
        
        # Calculate property changes
        property_changes = []
        
        for i, result in enumerate(results[:5]):
            if validations[result.smiles]['valid']:
                props = validations[result.smiles]['properties']
                
                changes = {
                    'smiles': result.smiles,
                    'qed_change': props['qed'] - ref_props['qed'],
                    'logp_change': props['logP'] - ref_props['logP'],
                    'mw_change': props['molecular_weight'] - ref_props['molecular_weight'],
                    'tpsa_change': props['tpsa'] - ref_props['tpsa'],
                    'sa_change': props['synthetic_accessibility'] - ref_props['synthetic_accessibility']
                }
                
                property_changes.append(changes)
        
        # Build prompt for analysis
        prompt = f"""
        As a medicinal chemistry expert, analyze the results of this molecular optimization run. 
        
        **Reference Molecule:**
        SMILES: {reference_smiles}
        QED: {ref_props['qed']:.3f}
        LogP: {ref_props['logP']:.2f}
        MW: {ref_props['molecular_weight']:.1f}
        TPSA: {ref_props['tpsa']:.1f}
        SA Score: {ref_props['synthetic_accessibility']:.1f}
        
        **Top Generated Molecules:**
        """
        
        for i, changes in enumerate(property_changes):
            prompt += f"\nMolecule {i+1}: {changes['smiles']}\n"
            prompt += f"QED Change: {changes['qed_change']:.3f}\n"
            prompt += f"LogP Change: {changes['logp_change']:.2f}\n"
            prompt += f"MW Change: {changes['mw_change']:.1f}\n"
            prompt += f"TPSA Change: {changes['tpsa_change']:.1f}\n"
            prompt += f"SA Score Change: {changes['sa_change']:.1f}\n"
        
        prompt += """
        Please provide:
        
        1. Analysis of the key structural modifications introduced during optimization
        2. Assessment of how these modifications affected drug-like properties
        3. Evaluation of synthetic feasibility changes
        4. Identification of the most promising candidate(s) and why
        5. Suggestions for further refinement in the next optimization round
        6. Potential concerns or liabilities introduced during optimization
        
        Focus on actionable insights for medicinal chemistry decision-making.
        """
        
        messages = [{"role": "user", "content": prompt}]
        response = analyze_with_deepseek(messages)
        self.add_to_history("user", prompt)
        self.add_to_history("assistant", response)
        return response
    
    def answer_follow_up(self, question: str) -> str:
        """Answer a follow-up question based on conversation history."""
        messages = self.conversation_history + [{"role": "user", "content": question}]
        response = analyze_with_deepseek(messages)
        self.add_to_history("user", question)
        self.add_to_history("assistant", response)
        return response
        
    def _get_molecule_svg(self, mol: Chem.Mol, width: int = 400, height: int = 300) -> str:
        """Generate SVG representation of molecule."""
        if not mol:
            return ""
            
        try:
            from rdkit.Chem import rdDepictor
            from rdkit.Chem.Draw import rdMolDraw2D
            
            # Generate 2D coordinates if needed
            if not mol.GetNumConformers():
                rdDepictor.Compute2DCoords(mol)
                
            # Create drawing object
            drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            
            # Return SVG as string
            return drawer.GetDrawingText()
        except Exception as e:
            logging.error(f"Error generating SVG: {str(e)}")
            return ""


@st.cache_data
def analyze_with_deepseek(messages: List[Dict[str, str]]) -> str:
    """Call DeepSeek API using OpenAI client."""
    try:
        response = client.chat.completions.create(
            model="deepseek-reasoner",  # or "deepseek-reasoner" if preferred
            messages=[{"role": "system", "content": "You are a pharmaceutical research assistant with expertise in medicinal chemistry, drug discovery, and molecular property analysis. Provide detailed, scientifically-grounded analyses focusing on structure-property relationships and actionable insights for drug development."}] + messages,
            max_tokens=1500,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        error_msg = f"Error contacting DeepSeek API: {str(e)}"
        logging.error(error_msg)
        # Fallback to local analysis if API fails
        return _fallback_analysis(messages)
    
def _fallback_analysis(messages: List[Dict[str, str]]) -> str:
    """Provide basic analysis when API is unavailable."""
    user_message = messages[-1]["content"]
    
    if "SMILES" in user_message and "Chemical Properties" in user_message:
        # This is a molecule analysis request
        return """
        ## Molecule Analysis (Offline Mode)
        
        The DeepSeek API is currently unavailable. Here's a basic analysis:
        
        ### Drug-likeness Assessment
        The provided molecular properties can be evaluated against Lipinski's Rule of Five:
        - Molecular Weight: Should be ≤ 500 Da
        - LogP: Should be ≤ 5
        - H-Bond Donors: Should be ≤ 5
        - H-Bond Acceptors: Should be ≤ 10
        
        QED values range from 0-1, with higher values indicating better drug-likeness.
        
        ### Synthetic Accessibility
        SA scores range from 1-10, with lower scores indicating easier synthesis.
        
        ### Next Steps
        Please try again later for a more comprehensive AI-powered analysis. For now, compare the properties against standard drug-likeness criteria and consider how structural modifications might affect these properties.
        """
    else:
        # Generic fallback
        return """
        ## Analysis (Offline Mode)
        
        The DeepSeek API is currently unavailable. Please try again later for a comprehensive AI-powered analysis of your molecule or comparison request.
        
        In the meantime, you can:
        - Manually compare the properties against drug-likeness criteria
        - Review structural features that might affect potency, selectivity, and ADMET properties
        - Consider standard medicinal chemistry principles for further optimization
        """


# === Visualization Functions === #

@st.cache_data
def create_comparison_table(smiles_list: List[str], include_columns: List[str] = None) -> pd.DataFrame:
    """Create a detailed comparison table of molecular properties."""
    validator = MoleculeValidator()
    results = validator.validate_smiles(smiles_list)
    
    # Extract properties
    df = pd.DataFrame({smi: res['properties'] for smi, res in results.items() if res['valid']}).T
    
    # Filter columns if specified
    if include_columns:
        available_columns = set(df.columns)
        filtered_columns = [col for col in include_columns if col in available_columns]
        df = df[filtered_columns]
    
    # Rename columns for readability
    column_mapping = {
        'molecular_weight': 'MW',
        'logP': 'LogP',
        'hbd': 'H-Donors',
        'hba': 'H-Acceptors',
        'tpsa': 'TPSA',
        'rotatable_bonds': 'Rot. Bonds',
        'aromatic_rings': 'Arom. Rings',
        'qed': 'QED',
        'synthetic_accessibility': 'SA Score',
        'lipinski_violations': 'Lipinski Viol.',
        'veber_violations': 'Veber Viol.',
        'ghose_violations': 'Ghose Viol.',
        'fraction_sp3': 'Frac. SP3',
        'ring_count': 'Ring Count',
        'stereo_centers': 'Stereo Centers'
    }
    
    # Apply column renaming
    df = df.rename(columns={col: column_mapping.get(col, col) for col in df.columns})
    
    return df

def create_similarity_heatmap(smiles_list: List[str], names: List[str] = None, title: str = "Tanimoto Similarity Heatmap") -> go.Figure:
    """Create a heatmap of molecular similarities."""
    # Convert SMILES to molecules
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    valid_indices = [i for i, mol in enumerate(mols) if mol is not None]
    
    if not valid_indices:
        # Return empty figure if no valid molecules
        fig = go.Figure()
        fig.update_layout(
            title="No valid molecules for similarity calculation",
            width=800, height=400
        )
        return fig
    
    # Filter valid molecules and corresponding names
    valid_mols = [mols[i] for i in valid_indices]
    valid_smiles = [smiles_list[i] for i in valid_indices]
    
    if names:
        valid_names = [names[i] for i in valid_indices]
    else:
        valid_names = [f"Mol {i+1}" for i in range(len(valid_mols))]
    
    # Generate fingerprints
    fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048) for mol in valid_mols]
    
    # Calculate similarity matrix
    sim_matrix = np.array([[TanimotoSimilarity(f1, f2) for f2 in fps] for f1 in fps])
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=sim_matrix, 
        x=valid_names, 
        y=valid_names, 
        colorscale='Viridis',
        text=[[f"{val:.2f}" for val in row] for row in sim_matrix],
        hoverinfo="text",
        colorbar=dict(title="Similarity")
    ))
    
    fig.update_layout(
        title=title,
        width=800, 
        height=800,
        xaxis=dict(title=""),
        yaxis=dict(title="")
    )
    
    return fig

def create_property_box_plot(properties_df: pd.DataFrame, properties: List[str] = None) -> go.Figure:
    """Create boxplots for selected molecular properties."""
    if properties is None:
        properties = ['QED', 'LogP', 'SA Score', 'TPSA', 'MW']
    
    # Map standard property names to DataFrame columns
    property_mapping = {
        'QED': 'QED',
        'LogP': 'LogP',
        'SA Score': 'SA Score',
        'TPSA': 'TPSA',
        'MW': 'MW',
        'H-Donors': 'H-Donors',
        'H-Acceptors': 'H-Acceptors',
        'Rotatable Bonds': 'Rot. Bonds'
    }
    
    # Get available columns
    available_props = [p for p in properties if property_mapping.get(p, p) in properties_df.columns]
    
    if not available_props:
        # Return empty figure if no properties available
        fig = go.Figure()
        fig.update_layout(
            title="No properties available for boxplot",
            width=800, height=400
        )
        return fig
    
    # Create figure
    fig = go.Figure()
    
    for prop in available_props:
        col = property_mapping.get(prop, prop)
        if col in properties_df.columns:
            fig.add_trace(go.Box(
                y=properties_df[col], 
                name=prop, 
                boxpoints='outliers',
                jitter=0.3,
                pointpos=-1.8,
                marker=dict(size=6)
            ))
    
    fig.update_layout(
        title="Property Distribution",
        yaxis_title="Value",
        width=800,
        height=600,
        boxmode='group'
    )
    
    return fig

def create_property_radar_chart(properties_df: pd.DataFrame, molecule_names: List[str] = None, normalize: bool = True) -> go.Figure:
    """Create a radar chart comparing properties of multiple molecules."""
    # Define properties to include
    properties = ['QED', 'LogP', 'MW', 'TPSA', 'H-Donors', 'H-Acceptors', 'Rot. Bonds']
    
    # Map to actual column names
    available_props = [p for p in properties if p in properties_df.columns]
    
    if not available_props:
        # Return empty figure if no properties available
        fig = go.Figure()
        fig.update_layout(
            title="No properties available for radar chart",
            width=800, height=600
        )
        return fig
    
    # Normalize properties if requested
    if normalize:
        # For each property, determine normalization approach
        norm_df = properties_df.copy()
        
        for prop in available_props:
            # Different normalization strategies based on property
            if prop == 'QED':
                # QED is already 0-1, higher is better
                norm_df[prop] = properties_df[prop]
            elif prop == 'SA Score':
                # SA Score is 1-10, lower is better, so invert
                norm_df[prop] = 1 - (properties_df[prop] - 1) / 9
            elif prop == 'LogP':
                # LogP optimal range is typically 1-3
                norm_df[prop] = 1 - np.abs(properties_df[prop] - 2) / 5
            elif prop == 'MW':
                # MW optimal is typically 200-500
                norm_df[prop] = 1 - np.abs(properties_df[prop] - 350) / 350
            elif prop == 'TPSA':
                # TPSA optimal is typically 40-140
                norm_df[prop] = 1 - np.abs(properties_df[prop] - 90) / 100
            else:
                # Min-max normalization for other properties
                min_val = properties_df[prop].min()
                max_val = properties_df[prop].max()
                if max_val > min_val:
                    norm_df[prop] = (properties_df[prop] - min_val) / (max_val - min_val)
                else:
                    norm_df[prop] = properties_df[prop] / properties_df[prop].max()
        
        plotting_df = norm_df
    else:
        plotting_df = properties_df
    
    # Use provided names or generate default ones
    if molecule_names is None or len(molecule_names) != len(properties_df):
        molecule_names = [f"Molecule {i+1}" for i in range(len(properties_df))]
    
    # Create radar chart
    fig = go.Figure()
    
    for i, (idx, row) in enumerate(plotting_df.iterrows()):
        values = [row[prop] for prop in available_props]
        # Close the loop by repeating first value
        values.append(values[0])
        props_list = available_props + [available_props[0]]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=props_list,
            fill='toself',
            name=molecule_names[i]
        ))
    
    fig.update_layout(
        title="Molecular Property Comparison",
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1] if normalize else None
            )
        ),
        width=700,
        height=600,
        showlegend=True
    )
    
    return fig

def create_qed_vs_sa_scatter(properties_df: pd.DataFrame, color_by: str = None, names: List[str] = None) -> go.Figure:
    """Create a scatter plot of QED vs Synthetic Accessibility."""
    # Check if required columns exist
    if 'QED' not in properties_df.columns or 'SA Score' not in properties_df.columns:
        # Try alternative column names
        qed_col = next((col for col in properties_df.columns if col.lower() == 'qed'), None)
        sa_col = next((col for col in properties_df.columns if 'sa' in col.lower() and 'score' in col.lower()), None)
        
        if not qed_col or not sa_col:
            # Return empty figure if columns not found
            fig = go.Figure()
            fig.update_layout(
                title="QED or SA Score data not available",
                width=800, height=600
            )
            return fig
    else:
        qed_col = 'QED'
        sa_col = 'SA Score'
    
    # Prepare point labels
    if names is None or len(names) != len(properties_df):
        names = [f"Mol {i+1}" for i in range(len(properties_df))]
    
    # Create scatter plot
    if color_by and color_by in properties_df.columns:
        fig = px.scatter(
            properties_df, 
            x=qed_col, 
            y=sa_col,
            color=color_by,
            text=names,
            labels={qed_col: 'QED Score', sa_col: 'Synthetic Accessibility'},
            title="QED vs. Synthetic Accessibility",
            hover_data=properties_df.columns
        )
    else:
        fig = px.scatter(
            properties_df, 
            x=qed_col, 
            y=sa_col,
            text=names,
            labels={qed_col: 'QED Score', sa_col: 'Synthetic Accessibility'},
            title="QED vs. Synthetic Accessibility",
            hover_data=properties_df.columns
        )
    
    # Add reference lines for good drug-likeness
    fig.add_shape(
        type="rect",
        x0=0.6, y0=1, x1=1, y1=4,
        line=dict(color="rgba(0,200,0,0.1)", width=1),
        fillcolor="rgba(0,200,0,0.1)"
    )
    
    fig.add_annotation(
        x=0.8, y=2.5,
        text="Preferred<br>Region",
        showarrow=False,
        font=dict(color="green")
    )
    
    fig.update_traces(
        textposition='top center',
        marker=dict(size=12, opacity=0.7, line=dict(width=1, color='black'))
    )
    
    fig.update_layout(
        width=800,
        height=600,
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[1, 10], autorange="reversed")  # Lower SA is better
    )
    
    return fig

def create_3d_pca_scatter(smiles_list: List[str], properties_df: pd.DataFrame = None, names: List[str] = None) -> go.Figure:
    """Create a 3D PCA visualization of molecular similarity."""
    # Convert SMILES to molecules
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    valid_indices = [i for i, mol in enumerate(mols) if mol is not None]
    
    if not valid_indices:
        # Return empty figure if no valid molecules
        fig = go.Figure()
        fig.update_layout(
            title="No valid molecules for PCA visualization",
            width=800, height=600
        )
        return fig
    
    # Filter valid molecules
    valid_mols = [mols[i] for i in valid_indices]
    
    # Generate fingerprints
    fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048) for mol in valid_mols]
    fps_array = np.zeros((len(fps), fps[0].GetNumBits()))
    
    for i, fp in enumerate(fps):
        DataStructs.ConvertToNumpyArray(fp, fps_array[i])
    
    # Perform PCA
    pca = PCA(n_components=3)
    reduced = pca.fit_transform(fps_array)
    
    # Prepare point labels
    if names is None or len(names) != len(valid_indices):
        names = [f"Mol {i+1}" for i in range(len(valid_indices))]
    else:
        names = [names[i] for i in valid_indices]
    
    # Prepare marker colors based on property if available
    marker_color = None
    color_scale = 'Viridis'
    
    if properties_df is not None and len(properties_df) == len(valid_indices):
        if 'QED' in properties_df.columns:
            marker_color = properties_df['QED'].values
            colorbar_title = 'QED'
        elif 'SA Score' in properties_df.columns:
            marker_color = properties_df['SA Score'].values
            colorbar_title = 'SA Score'
            color_scale = 'Viridis_r'  # Reverse so better (lower) values are blue
    
    # Create 3D scatter plot
    if marker_color is not None:
        fig = go.Figure(data=[go.Scatter3d(
            x=reduced[:, 0],
            y=reduced[:, 1],
            z=reduced[:, 2],
            mode='markers',
            marker=dict(
                size=8,
                color=marker_color,
                colorscale=color_scale,
                opacity=0.8,
                colorbar=dict(title=colorbar_title)
            ),
            text=names,
            hoverinfo='text'
        )])
    else:
        fig = go.Figure(data=[go.Scatter3d(
            x=reduced[:, 0],
            y=reduced[:, 1],
            z=reduced[:, 2],
            mode='markers',
            marker=dict(
                size=8,
                color='blue',
                opacity=0.7
            ),
            text=names,
            hoverinfo='text'
        )])
    
    # Calculate explained variance
    explained_variance = pca.explained_variance_ratio_ * 100
    
    fig.update_layout(
        title="3D PCA of Molecular Diversity",
        scene=dict(
            xaxis_title=f"PC1 ({explained_variance[0]:.1f}%)",
            yaxis_title=f"PC2 ({explained_variance[1]:.1f}%)",
            zaxis_title=f"PC3 ({explained_variance[2]:.1f}%)"
        ),
        width=800,
        height=600
    )
    
    return fig

def create_pareto_front_visualization(results: List[OptimizationResult], 
                                     prop1: str = "qed_score", 
                                     prop2: str = "tanimoto_score", 
                                     highlight_pareto: bool = True) -> go.Figure:
    """Create a visualization of the Pareto front for multi-objective optimization."""
    if not results:
        # Return empty figure if no results
        fig = go.Figure()
        fig.update_layout(
            title="No optimization results available",
            width=800, height=600
        )
        return fig
    
    # Map property names to readable labels
    prop_labels = {
        "qed_score": "QED (Drug-likeness)",
        "tanimoto_score": "Similarity to Reference",
        "sa_score": "Synthetic Accessibility"
    }
    
    # Extract data
    x_vals = [getattr(r, prop1) for r in results]
    y_vals = [getattr(r, prop2) for r in results]
    smiles = [r.smiles for r in results]
    
    # If prop2 is sa_score, invert values since lower is better
    invert_y = prop2 == "sa_score"
    if invert_y:
        y_vals = [10 - val for val in y_vals]  # Assuming SA score is in 0-10 range
        prop_labels[prop2] = "Synthetic Accessibility (inverted)"
    
    # Create dataframe for plotting
    df = pd.DataFrame({
        'x': x_vals,
        'y': y_vals,
        'smiles': smiles,
        'pareto': [r.pareto_front for r in results] if highlight_pareto else [False] * len(results),
        'cluster': [r.cluster_id if r.cluster_id is not None else 0 for r in results]
    })
    
    # Create scatter plot
    if highlight_pareto:
        # Create two traces - one for regular points, one for Pareto front
        fig = go.Figure()
        
        # Non-Pareto points
        non_pareto = df[~df['pareto']]
        fig.add_trace(go.Scatter(
            x=non_pareto['x'],
            y=non_pareto['y'],
            mode='markers',
            marker=dict(
                size=8, 
                color=non_pareto['cluster'],
                colorscale='Viridis',
                opacity=0.6,
                line=dict(width=1, color='DarkSlateGrey')
            ),
            text=non_pareto['smiles'],
            name='Generated Molecules'
        ))
        
        # Pareto front points
        pareto = df[df['pareto']]
        fig.add_trace(go.Scatter(
            x=pareto['x'],
            y=pareto['y'],
            mode='markers',
            marker=dict(
                size=12, 
                color='rgba(255,0,0,0.8)',
                symbol='star',
                line=dict(width=1, color='DarkSlateGrey')
            ),
            text=pareto['smiles'],
            name='Pareto Front'
        ))
    else:
        fig = px.scatter(
            df, 
            x='x', 
            y='y',
            color='cluster',
            text='smiles',
            labels={'x': prop_labels.get(prop1, prop1), 'y': prop_labels.get(prop2, prop2)},
            title=f"{prop_labels.get(prop1, prop1)} vs. {prop_labels.get(prop2, prop2)}",
            hover_data=['x', 'y', 'smiles']
        )
    
    # Update layout
    fig.update_layout(
        xaxis=dict(title=prop_labels.get(prop1, prop1)),
        yaxis=dict(title=prop_labels.get(prop2, prop2)),
        width=800,
        height=600,
        hovermode='closest'
    )
    
    # Add optimal region shading
    fig.add_shape(
        type="rect",
        x0=0.8 if prop1 == "qed_score" else 0.7,
        y0=0.8 if not invert_y else 7,
        x1=1.0 if prop1 == "qed_score" else 1.0,
        y1=1.0 if not invert_y else 10,
        line=dict(color="rgba(0,255,0,0.1)", width=1),
        fillcolor="rgba(0,255,0,0.1)"
    )
    
    fig.add_annotation(
        x=0.9 if prop1 == "qed_score" else 0.85,
        y=0.9 if not invert_y else 8.5,
        text="Optimal<br>Region",
        showarrow=False,
        font=dict(color="green")
    )
    
    return fig

def create_optimization_progress_chart(history: List[OptimizationResult]) -> go.Figure:
    """Create a chart showing optimization progress over iterations."""
    if not history:
        # Return empty figure if no history
        fig = go.Figure()
        fig.update_layout(
            title="No optimization history available",
            width=800, height=400
        )
        return fig
    
    # Extract data
    iterations = []
    qed_scores = []
    tanimoto_scores = []
    sa_scores = []
    
    for result in history:
        iter_num = result.parameters.get('iter', 0)
        iterations.append(iter_num)
        qed_scores.append(result.qed_score)
        tanimoto_scores.append(result.tanimoto_score)
        sa_scores.append(result.sa_score)
    
    # Create dataframe for plotting
    df = pd.DataFrame({
        'Iteration': iterations,
        'QED': qed_scores,
        'Tanimoto': tanimoto_scores,
        'SA Score': sa_scores
    })
    
    # Group by iteration and calculate statistics
    grouped = df.groupby('Iteration').agg({
        'QED': ['mean', 'max', 'min', 'std'],
        'Tanimoto': ['mean', 'max', 'min', 'std'],
        'SA Score': ['mean', 'max', 'min', 'std']
    })
    
    # Flatten multi-index
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    grouped = grouped.reset_index()
    
    # Create figure with multiple traces
    fig = go.Figure()
    
    # QED score progress
    fig.add_trace(go.Scatter(
        x=grouped['Iteration'],
        y=grouped['QED_mean'],
        mode='lines+markers',
        name='QED (mean)',
        line=dict(color='blue'),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=grouped['Iteration'],
        y=grouped['QED_max'],
        mode='lines',
        name='QED (max)',
        line=dict(color='blue', dash='dash')
    ))
    
    # Tanimoto score progress
    fig.add_trace(go.Scatter(
        x=grouped['Iteration'],
        y=grouped['Tanimoto_mean'],
        mode='lines+markers',
        name='Tanimoto (mean)',
        line=dict(color='green'),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=grouped['Iteration'],
        y=grouped['Tanimoto_max'],
        mode='lines',
        name='Tanimoto (max)',
        line=dict(color='green', dash='dash')
    ))
    
    # SA score progress (lower is better, so show min instead of max)
    fig.add_trace(go.Scatter(
        x=grouped['Iteration'],
        y=grouped['SA Score_mean'],
        mode='lines+markers',
        name='SA Score (mean)',
        line=dict(color='red'),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=grouped['Iteration'],
        y=grouped['SA Score_min'],
        mode='lines',
        name='SA Score (min)',
        line=dict(color='red', dash='dash')
    ))
    
    # Update layout
    fig.update_layout(
        title="Optimization Progress",
        xaxis=dict(title="Iteration"),
        yaxis=dict(title="Score"),
        width=900,
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def create_scaffold_summary(smiles_list: List[str]) -> Tuple[go.Figure, dict]:
    """Create a summary of scaffolds present in molecules."""
    # Extract scaffolds
    scaffolds = {}
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            continue
            
        # Get Murcko scaffold
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if not scaffold:
            continue
            
        scaffold_smiles = Chem.MolToSmiles(scaffold)
        if scaffold_smiles not in scaffolds:
            scaffolds[scaffold_smiles] = []
        scaffolds[scaffold_smiles].append(smi)
    
    if not scaffolds:
        # Return empty figure if no valid scaffolds
        fig = go.Figure()
        fig.update_layout(
            title="No valid scaffolds found",
            width=700, height=400
        )
        return fig, {}
    
    # Sort scaffolds by frequency
    sorted_scaffolds = sorted(scaffolds.items(), key=lambda x: len(x[1]), reverse=True)
    
    # Prepare data for visualization
    scaffold_smiles = [s[0] for s in sorted_scaffolds]
    counts = [len(s[1]) for s in sorted_scaffolds]
    
    # Create bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=[f"Scaffold {i+1}" for i in range(len(scaffold_smiles))],
        y=counts,
        text=counts,
        textposition='auto',
        hovertext=scaffold_smiles,
        marker_color='royalblue'
    ))
    
    fig.update_layout(
        title="Scaffold Distribution",
        xaxis=dict(title="Scaffold"),
        yaxis=dict(title="Count"),
        width=700,
        height=400
    )
    
    # Return figure and scaffold mapping
    return fig, {f"Scaffold {i+1}": {"smiles": s[0], "molecules": s[1]} for i, s in enumerate(sorted_scaffolds)}


def smiles_to_image(smiles: str, width: int = 300, height: int = 200) -> str:
    """Convert SMILES to base64-encoded PNG image."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return ""
        
    try:
        img = Draw.MolToImage(mol, size=(width, height))
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        logging.error(f"Error generating molecule image: {str(e)}")
        return ""


# === Application Pages === #

def display_home_page(server_available: bool):
    """Display the home page with overview and instructions."""
    st.markdown("<h2 class='section-header'>Welcome to the Advanced Drug Discovery Platform</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        This platform combines state-of-the-art generative chemistry, cheminformatics, and AI-driven analysis
        to accelerate drug discovery research. The system enables:
        
        - **Molecule Optimization**: Iteratively improve molecules using advanced algorithms
        - **Generative Chemistry**: Create novel molecules using AI-based generative models
        - **Comprehensive Analysis**: Visualize and analyze molecular properties
        - **AI-Powered Insights**: Get medicinal chemistry insights from AI assistants
        """)
        
        st.markdown("<h3 class='subsection-header'>Getting Started</h3>", unsafe_allow_html=True)
        st.markdown("""
        1. Create a new project in the sidebar with your reference molecule
        2. Navigate to "Molecule Optimization" to generate improved variants
        3. Use "Generative Methods" for targeted molecule design approaches
        4. Explore results in "Analysis Dashboard" with interactive visualizations
        5. Get expert AI feedback in the "AI Analysis" section
        """)
        
        if not server_available:
            st.warning("""
            ⚠️ The MolMIM generative model server is not accessible. 
            Some generative features will be disabled until connection is established.
            
            Check the server settings in the sidebar and ensure the server is running.
            """)
    
    with col2:
        if st.session_state.project is not None:
            st.markdown("<h3 class='subsection-header'>Current Molecule</h3>", unsafe_allow_html=True)
            mol = Chem.MolFromSmiles(st.session_state.current_smiles)
            if mol:
                img = smiles_to_image(st.session_state.current_smiles)
                st.image(img, caption="Reference Molecule")
                
                # Basic properties
                validator = MoleculeValidator()
                props = validator.validate_smiles([st.session_state.current_smiles])[st.session_state.current_smiles]['properties']
                
                st.markdown("**Basic Properties:**")
                st.markdown(f"- MW: {props['molecular_weight']:.1f}")
                st.markdown(f"- LogP: {props['logP']:.1f}")
                st.markdown(f"- QED: {props['qed']:.3f}")
                st.markdown(f"- Lipinski Violations: {props['lipinski_violations']}")
        else:
            st.info("No project loaded. Create a new project in the sidebar to get started.")
    
    # Feature overview
    st.markdown("<h3 class='subsection-header'>Platform Features</h3>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Optimization Methods**")
        st.markdown("- CMA-ES Optimization")
        st.markdown("- Multi-objective Pareto Front")
        st.markdown("- Property-based Constraints")
        st.markdown("- Custom Scoring Functions")
    
    with col2:
        st.markdown("**Generative Approaches**")
        st.markdown("- Conditional Generation")
        st.markdown("- Fragment-based Design")
        st.markdown("- Scaffold Hopping")
        st.markdown("- Focused Libraries")
    
    with col3:
        st.markdown("**Analysis Tools**")
        st.markdown("- Interactive Visualizations")
        st.markdown("- Property Analysis")
        st.markdown("- Structural Diversity")
        st.markdown("- AI-driven Insights")


def display_optimization_page(client: MolMIMClient, radius: int, fp_size: int, parallel: bool, n_jobs: int, server_available: bool):
    st.markdown("<h2 class='section-header'>Molecule Optimization</h2>", unsafe_allow_html=True)
    
    if st.session_state.project is None:
        st.warning("Please create or load a project first (use the sidebar).")
        return
    
    # Reference molecule display
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("<h3 class='subsection-header'>Reference Molecule</h3>", unsafe_allow_html=True)
        seed_smiles = st.session_state.project.reference_smiles
        mol = Chem.MolFromSmiles(seed_smiles)
        if mol:
            img = smiles_to_image(seed_smiles)
            st.image(img, caption="Reference Structure")
            st.text_area("Reference SMILES", seed_smiles, height=80)
    
    with col2:
        st.markdown("<h3 class='subsection-header'>Optimization Strategy</h3>", unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["Basic Settings", "Advanced Settings"])
        
        with tab1:
            with st.container():
                pop_size = st.number_input("Population Size", 10, 100, st.session_state.project.optimization_strategy.population_size)
                n_iter = st.number_input("Iterations", 10, 200, st.session_state.project.optimization_strategy.n_iterations)
                sigma = st.number_input("CMA-ES Sigma", 0.1, 5.0, 1.0)
            st.session_state.project.optimization_strategy.population_size = pop_size
            st.session_state.project.optimization_strategy.n_iterations = n_iter
        
        with tab2:
            # Property weights
            st.subheader("Property Weights")
            with st.container():
                qed_weight = st.slider("QED Weight", 0.0, 2.0, st.session_state.project.optimization_strategy.property_weights.get("qed", 1.0), 0.1, key="qed_weight_slider")
                tanimoto_weight = st.slider("Similarity Weight", 0.0, 2.0, st.session_state.project.optimization_strategy.property_weights.get("tanimoto", 1.0), 0.1, key="tanimoto_weight_slider")
                sa_weight = st.slider("SA Score Weight", 0.0, 1.0, st.session_state.project.optimization_strategy.property_weights.get("sa_score", 0.1), 0.1, key="sa_weight_slider")
                novelty_weight = st.slider("Novelty Weight", 0.0, 2.0, st.session_state.project.optimization_strategy.property_weights.get("novelty", 0.5), 0.1, key="novelty_weight_slider")
            
            # Update strategy
            st.session_state.project.optimization_strategy.property_weights = {
                "qed": qed_weight,
                "tanimoto": tanimoto_weight,
                "sa_score": sa_weight,
                "novelty": novelty_weight
            }
            
            # Property constraints
            st.subheader("Property Constraints")
            constraint_cols = st.columns(3)
            properties = ["molecular_weight", "logP", "hbd", "hba", "rotatable_bonds"]
            property_labels = ["Molecular Weight", "LogP", "H-Bond Donors", "H-Bond Acceptors", "Rotatable Bonds"]
            constraints = st.session_state.project.optimization_strategy.constraints
            
            for i, (prop, label) in enumerate(zip(properties, property_labels)):
                with constraint_cols[i % 3]:
                    st.markdown(f"**{label}**")
                    with st.container():  # Replace nested st.columns(2) with a container
                        min_val = constraints.get(prop, {}).get("min", None)
                        if prop == "molecular_weight":
                            min_input = st.number_input(f"Min {label}", 0.0, 1000.0, min_val if min_val is not None else 0.0, 10.0, key=f"min_{prop}_input")
                        elif prop == "logP":
                            min_input = st.number_input(f"Min {label}", -5.0, 10.0, min_val if min_val is not None else -1.0, 0.5, key=f"min_{prop}_input")
                        elif prop in ["hbd", "hba", "rotatable_bonds"]:
                            min_input = st.number_input(f"Min {label}", 0, 20, min_val if min_val is not None else 0, 1, key=f"min_{prop}_input")
                        min_val = min_input if min_input > 0 else None
                        
                        max_val = constraints.get(prop, {}).get("max", None)
                        if prop == "molecular_weight":
                            max_input = st.number_input(f"Max {label}", 0.0, 1000.0, max_val if max_val is not None else 500.0, 10.0, key=f"max_{prop}_input")
                        elif prop == "logP":
                            max_input = st.number_input(f"Max {label}", -5.0, 10.0, max_val if max_val is not None else 5.0, 0.5, key=f"max_{prop}_input")
                        elif prop in ["hbd", "hba", "rotatable_bonds"]:
                            max_input = st.number_input(f"Max {label}", 0, 20, max_val if max_val is not None else 10, 1, key=f"max_{prop}_input")
                        max_val = max_input
                    
                        # Update constraints
                        if min_val is not None or max_val is not None:
                            constraints[prop] = {}
                            if min_val is not None:
                                constraints[prop]["min"] = min_val
                            if max_val is not None:
                                constraints[prop]["max"] = max_val
                        else:
                            if prop in constraints:
                                del constraints[prop]
            
            # Save updated constraints
            st.session_state.project.optimization_strategy.constraints = constraints
    
    # Run optimization
    if server_available:
        if st.button("Start Optimization", key="run_optimization"):
            if not client.available:
                st.error("MolMIM server is not available. Check connection settings.")
                return
                
            # Initialize optimizer with current settings
            optimizer = MoleculeOptimizer(seed_smiles, radius, fp_size)
            
            # Convert constraints
            constraints = st.session_state.project.optimization_strategy.constraints
            
            try:
                # Get the latent representation
                with st.spinner("Getting latent representation..."):
                    hidden_state = client.get_hidden_state(seed_smiles)
                
                # Initialize CMA-ES
                es = cma.CMAEvolutionStrategy(hidden_state, sigma, {'popsize': pop_size})
                
                # Create progress bar
                progress_bar = st.progress(0)
                results_container = st.empty()
                
                # Perform optimization
                with st.spinner(f"Running optimization ({n_iter} iterations)..."):
                    for i in range(n_iter):
                        # Sample solutions
                        trials = es.ask(pop_size)
                        
                        # Decode solutions
                        molecules = client.decode_batch(trials)
                        
                        # Filter valid molecules
                        valid_mols = [(idx, smi) for idx, smi in enumerate(molecules) if Chem.MolFromSmiles(smi)]
                        if not valid_mols:
                            continue
                            
                        # Extract valid data
                        valid_indices = [idx for idx, _ in valid_mols]
                        valid_smiles = [smi for _, smi in valid_mols]
                        
                        # Score molecules
                        qeds = optimizer.qed(valid_smiles, parallel)
                        tanimotos = optimizer.tanimoto_similarity(valid_smiles, parallel)
                        sa_scores = optimizer.synthetic_accessibility(valid_smiles, parallel)
                        
                        # Combined score with weights
                        scores = optimizer.scoring_function(
                            valid_smiles, 
                            weights=st.session_state.project.optimization_strategy.property_weights,
                            constraints=constraints,
                            parallel=parallel
                        )
                        
                        # Update CMA-ES
                        objective_values = np.zeros(len(trials))
                        for j, idx in enumerate(valid_indices):
                            objective_values[idx] = scores[j]
                        es.tell(trials, objective_values)
                        
                        # Save results
                        for j, smi in enumerate(valid_smiles):
                            optimizer.save_result(
                                smi, 
                                qeds[j], 
                                tanimotos[j], 
                                sa_scores[j], 
                                {'pop_size': pop_size, 'iter': i}
                            )
                        
                        # Update progress
                        progress_bar.progress((i + 1) / n_iter)
                        
                        # Show intermediate results
                        if i % 5 == 0 or i == n_iter - 1:
                            with results_container.container():
                                st.markdown(f"**Iteration {i+1}/{n_iter}:** Found {len(valid_smiles)} valid molecules")
                                
                                # Show the best molecule so far
                                if optimizer.history:
                                    best_qed_idx = np.argmax([r.qed_score for r in optimizer.history])
                                    best_qed = optimizer.history[best_qed_idx]
                                    
                                    st.markdown(f"**Best QED:** {best_qed.qed_score:.3f} (vs reference: {optimizer.reference_qed:.3f})")
                                    best_mol = Chem.MolFromSmiles(best_qed.smiles)
                                    if best_mol:
                                        st.image(smiles_to_image(best_qed.smiles, width=300, height=200), caption="Best molecule so far")
                
                # Compute Pareto front
                optimizer.compute_pareto_front(optimizer.history)
                
                # Cluster molecules
                if len(optimizer.history) > 5:
                    smiles_list = [r.smiles for r in optimizer.history]
                    clusters = optimizer.cluster_molecules(smiles_list, n_clusters=min(5, len(smiles_list)))
                    for result in optimizer.history:
                        result.cluster_id = clusters.get(result.smiles, 0)
                
                # Store results
                st.session_state.optimization_results = optimizer.history
                st.session_state.project.history.extend(optimizer.history)
                
                # Show success message
                st.success(f"Optimization completed! Generated {len(optimizer.history)} molecules.")
                
                # Display top molecules
                st.subheader("Top Molecules")
                
                # Get top molecules by QED
                top_qed_indices = np.argsort([-r.qed_score for r in optimizer.history])[:5]
                top_molecules = [optimizer.history[i] for i in top_qed_indices]
                
                # Display molecules in columns
                cols = st.columns(5)
                for idx, col in enumerate(cols):
                    if idx < len(top_molecules):
                        mol_data = top_molecules[idx]
                        with col:
                            mol_img = smiles_to_image(mol_data.smiles, width=150, height=100)
                            st.image(mol_img, caption=f"Rank {idx+1}")
                            st.text_area(f"SMILES {idx+1}", mol_data.smiles, height=70)
                            st.metric("QED", f"{mol_data.qed_score:.3f}")
                            st.metric("Similarity", f"{mol_data.tanimoto_score:.3f}")
                            st.metric("SA Score", f"{mol_data.sa_score:.1f}")
                            if mol_data.pareto_front:
                                st.markdown("✅ Pareto Optimal")
                
                # Show pareto front
                if len(optimizer.history) > 5:
                    st.subheader("Pareto Front Visualization")
                    fig = create_pareto_front_visualization(optimizer.history)
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error during optimization: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    else:
        st.warning("MolMIM server is not available. Optimization requires a running server.")
    
    # History section
    if st.session_state.project.history:
        st.markdown("<h3 class='subsection-header'>Optimization History</h3>", unsafe_allow_html=True)
        
        # Show optimization progress
        fig = create_optimization_progress_chart(st.session_state.project.history)
        st.plotly_chart(fig, use_container_width=True)
        
        # Add export button
        history_df = pd.DataFrame([
            {
                'SMILES': r.smiles,
                'QED': r.qed_score,
                'Similarity': r.tanimoto_score,
                'SA_Score': r.sa_score,
                'Pareto_Front': r.pareto_front,
                'Cluster': r.cluster_id,
                'Iteration': r.parameters.get('iter', 0)
            }
            for r in st.session_state.project.history
        ])
        
        csv = history_df.to_csv(index=False)
        st.download_button(
            "Download Results as CSV",
            data=csv,
            file_name="optimization_results.csv",
            mime="text/csv"
        )

def display_generative_page(client: MolMIMClient, server_available: bool):
    """Display the generative methods page."""
    st.markdown("<h2 class='section-header'>Generative Chemistry Methods</h2>", unsafe_allow_html=True)
    
    if st.session_state.project is None:
        st.warning("Please create or load a project first (use the sidebar).")
        return
    
    if not server_available:
        st.warning("MolMIM server is not available. Some generative features may be disabled.")
    
    # Initialize generator if needed
    if server_available and (st.session_state.generator is None or 
                            st.session_state.generator.reference_smiles != st.session_state.current_smiles):
        st.session_state.generator = ConditionalMoleculeGenerator(client, st.session_state.current_smiles)
    
    # Tabs for different generative methods
    method_tab = st.radio(
        "Select Generative Method",
        ["Conditional Generation", "Fragment-Based Design", "Scaffold Hopping"],
        horizontal=True
    )
    
    if method_tab == "Conditional Generation":
        display_conditional_generation(client, server_available)
    elif method_tab == "Fragment-Based Design":
        display_fragment_based_design(client, server_available)
    elif method_tab == "Scaffold Hopping":
        display_scaffold_hopping(client, server_available)


def display_conditional_generation(client: MolMIMClient, server_available: bool):
    """Display the conditional generation interface."""
    st.markdown("<h3 class='subsection-header'>Conditional Molecule Generation</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    Generate molecules with specific property constraints. This approach uses the generative model to create
    molecules that meet desired property criteria while maintaining similarity to the reference molecule.
    """)
    
    # Reference molecule
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Reference Molecule**")
        mol = Chem.MolFromSmiles(st.session_state.current_smiles)
        if mol:
            st.image(smiles_to_image(st.session_state.current_smiles), caption="Reference Structure")
    
    with col2:
        st.markdown("**Property Constraints**")
        
        # Property constraints
        constraint_tabs = st.tabs(["Physical Properties", "Structural Features", "Drug-likeness"])
        
        constraints = {}
        
        with constraint_tabs[0]:
            col1, col2 = st.columns(2)
            
            with col1:
                mw_min = st.slider("Min Molecular Weight", 100, 800, 200, 10)
                logp_min = st.slider("Min LogP", -2.0, 8.0, 1.0, 0.1)
            
            with col2:
                mw_max = st.slider("Max Molecular Weight", 100, 800, 500, 10)
                logp_max = st.slider("Max LogP", -2.0, 8.0, 5.0, 0.1)
            
            # Add constraints if values are valid
            if mw_min < mw_max:
                constraints["molecular_weight"] = {"min": mw_min, "max": mw_max}
            
            if logp_min < logp_max:
                constraints["logP"] = {"min": logp_min, "max": logp_max}
        
        with constraint_tabs[1]:
            col1, col2 = st.columns(2)
            
            with col1:
                hbd_max = st.slider("Max H-Bond Donors", 0, 10, 5, 1)
                hba_max = st.slider("Max H-Bond Acceptors", 0, 10, 10, 1)
            
            with col2:
                rotb_max = st.slider("Max Rotatable Bonds", 0, 15, 10, 1)
            
            constraints["hbd"] = {"max": hbd_max}
            constraints["hba"] = {"max": hba_max}
            constraints["rotatable_bonds"] = {"max": rotb_max}
        
        with constraint_tabs[2]:
            qed_min = st.slider("Min QED Score", 0.0, 1.0, 0.5, 0.05)
            sa_max = st.slider("Max Synthetic Accessibility", 1.0, 10.0, 5.0, 0.5)
            tanimoto_min = st.slider("Min Similarity to Reference", 0.0, 1.0, 0.4, 0.05)
            
            constraints["qed"] = {"min": qed_min}
            constraints["sa_score"] = {"max": sa_max}
            constraints["tanimoto_similarity"] = {"min": tanimoto_min}
    
    # Generation parameters
    st.markdown("**Generation Parameters**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        num_molecules = st.number_input("Number of Molecules", 1, 50, 10, 1)
    
    with col2:
        n_iterations = st.number_input("Optimization Iterations", 1, 10, 3, 1)
    
    with col3:
        use_seed = st.checkbox("Use Custom Seed", False)
    
    if use_seed:
        seed_smiles = st.text_area("Seed SMILES", st.session_state.current_smiles, height=80)
    else:
        seed_smiles = st.session_state.current_smiles
    
    # Run generation
    if st.button("Generate Molecules", key="run_conditional"):
        if not server_available:
            st.error("MolMIM server is not available. Cannot run generation.")
            return
            
        try:
            with st.spinner("Generating molecules with constraints..."):
                # Update generator constraints
                generator = st.session_state.generator
                generator.property_constraints = {}
                
                for prop, constraint in constraints.items():
                    for ctype, value in constraint.items():
                        generator.set_constraint(prop, 
                            min_value=value if ctype == "min" else None,
                            max_value=value if ctype == "max" else None,
                            target_value=value if ctype == "target" else None
                        )
                
                # Generate molecules
                molecules = generator.generate_with_constraints(
                    seed_smiles=seed_smiles,
                    num_molecules=num_molecules,
                    n_iter=n_iterations
                )
                
                if not molecules:
                    st.warning("No valid molecules generated. Try relaxing constraints.")
                    return
                
                # Validate and score molecules
                validator = MoleculeValidator()
                validations = validator.validate_smiles(molecules)
                
                # Display molecules
                st.subheader("Generated Molecules")
                
                # Create display grid
                num_cols = 3
                num_rows = (len(molecules) + num_cols - 1) // num_cols
                
                for row in range(num_rows):
                    cols = st.columns(num_cols)
                    for col_idx in range(num_cols):
                        mol_idx = row * num_cols + col_idx
                        if mol_idx < len(molecules):
                            mol_smiles = molecules[mol_idx]
                            with cols[col_idx]:
                                mol = Chem.MolFromSmiles(mol_smiles)
                                if mol:
                                    # Display molecule
                                    st.markdown(f"**Molecule {mol_idx+1}**")
                                    st.image(smiles_to_image(mol_smiles), use_container_width=True)
                                    
                                    # Display properties
                                    if validations[mol_smiles]['valid']:
                                        props = validations[mol_smiles]['properties']
                                        st.markdown(f"QED: {props['qed']:.3f}")
                                        st.markdown(f"LogP: {props['logP']:.2f}")
                                        st.markdown(f"MW: {props['molecular_weight']:.1f}")
                                        st.markdown(f"SA Score: {props['synthetic_accessibility']:.1f}")
                                    
                                    # Add button to add to project
                                    if st.button(f"Add to Project", key=f"add_{mol_idx}"):
                                        # Create result object
                                        if validations[mol_smiles]['valid']:
                                            props = validations[mol_smiles]['properties']
                                            result = OptimizationResult(
                                                smiles=mol_smiles,
                                                qed_score=props['qed'],
                                                tanimoto_score=props.get('tanimoto_similarity', 0.0),
                                                sa_score=props['synthetic_accessibility'],
                                                timestamp=datetime.now(),
                                                parameters={'method': 'conditional_generation'}
                                            )
                                            st.session_state.project.history.append(result)
                                            st.success(f"Added molecule {mol_idx+1} to project")
                
                # Add property visualization
                if len(molecules) > 2:
                    st.subheader("Property Analysis")
                    
                    # Create comparison table
                    comp_df = create_comparison_table(molecules)
                    
                    # Show data table
                    st.dataframe(comp_df)
                    
                    # Show property distributions
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = create_property_box_plot(comp_df)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig = create_qed_vs_sa_scatter(comp_df)
                        st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error during generation: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


def display_fragment_based_design(client: MolMIMClient, server_available: bool):
    """Display the fragment-based design interface."""
    st.markdown("<h3 class='subsection-header'>Fragment-Based Design</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    Generate molecules by attaching fragments to core structures. This approach allows for targeted 
    modification of specific parts of molecules.
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Core fragment input
        st.markdown("**Core Fragment**")
        
        use_reference = st.checkbox("Use Reference as Core", True)
        
        if use_reference:
            core_smiles = st.session_state.current_smiles
            st.image(smiles_to_image(core_smiles), caption="Core Structure")
        else:
            core_smiles = st.text_area("Core SMILES", "", height=80)
            mol = Chem.MolFromSmiles(core_smiles) if core_smiles else None
            if mol:
                st.image(smiles_to_image(core_smiles), caption="Core Structure")
            else:
                st.warning("Please enter a valid SMILES for the core structure.")
    
    with col2:
        # Fragment library
        st.markdown("**Fragment Library**")
        
        library_type = st.radio("Fragment Source", ["Default Library", "Custom Fragments"])
        
        if library_type == "Default Library":
            fragments = None  # Use default library in the generator
            
            # Display some examples from default library
            default_frags = ["C", "CC", "CO", "CN", "c1ccccc1", "c1ccncc1"]
            frag_cols = st.columns(6)
            for i, frag in enumerate(default_frags):
                with frag_cols[i]:
                    mol = Chem.MolFromSmiles(frag)
                    if mol:
                        st.image(smiles_to_image(frag, width=80, height=80), caption=frag)
            
            st.markdown("*Default library includes common medicinal chemistry fragments.*")
        else:
            fragments_text = st.text_area("Enter fragments (one SMILES per line)", height=100)
            fragments = fragments_text.strip().split('\n') if fragments_text else []
            
            # Validate fragments
            valid_fragments = []
            for frag in fragments:
                if Chem.MolFromSmiles(frag):
                    valid_fragments.append(frag)
                    
            if valid_fragments:
                st.success(f"Found {len(valid_fragments)} valid fragments.")
            else:
                st.warning("No valid fragments found.")
                fragments = None
    
    # Generation parameters  
    st.markdown("**Generation Parameters**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_molecules = st.number_input("Number of Molecules", 1, 50, 10, 1)
    
    with col2:
        manual_attachment = st.checkbox("Specify Attachment Points", False)
    
    if manual_attachment:
        attachment_text = st.text_input("Attachment Points (comma-separated indices)", "0,1,2")
        try:
            attachment_points = [int(idx.strip()) for idx in attachment_text.split(",")]
        except:
            st.error("Invalid attachment points. Please enter comma-separated integers.")
            attachment_points = None
    else:
        attachment_points = None
    
    # Run generation
    if st.button("Generate Molecules", key="run_fragment"):
        if not server_available:
            st.error("MolMIM server is not available. Cannot run generation.")
            return
            
        if not core_smiles or not Chem.MolFromSmiles(core_smiles):
            st.error("Please provide a valid core structure.")
            return
            
        try:
            with st.spinner("Generating molecules with fragment-based approach..."):
                # Run fragment-based generation
                generator = st.session_state.generator
                
                molecules = generator.fragment_based_generation(
                    core_fragment=core_smiles,
                    num_molecules=num_molecules,
                    attachment_points=attachment_points,
                    decorations_library=fragments
                )
                
                if not molecules:
                    st.warning("No valid molecules generated.")
                    return
                
                # Validate and score molecules
                validator = MoleculeValidator()
                validations = validator.validate_smiles(molecules)
                
                # Display molecules
                st.subheader("Generated Molecules")
                
                # Create display grid
                num_cols = 3
                num_rows = (len(molecules) + num_cols - 1) // num_cols
                
                for row in range(num_rows):
                    cols = st.columns(num_cols)
                    for col_idx in range(num_cols):
                        mol_idx = row * num_cols + col_idx
                        if mol_idx < len(molecules):
                            mol_smiles = molecules[mol_idx]
                            with cols[col_idx]:
                                mol = Chem.MolFromSmiles(mol_smiles)
                                if mol:
                                    # Display molecule
                                    st.markdown(f"**Molecule {mol_idx+1}**")
                                    st.image(smiles_to_image(mol_smiles), use_container_width=True)
                                    
                                    # Display properties
                                    if validations[mol_smiles]['valid']:
                                        props = validations[mol_smiles]['properties']
                                        st.markdown(f"QED: {props['qed']:.3f}")
                                        st.markdown(f"LogP: {props['logP']:.2f}")
                                        st.markdown(f"MW: {props['molecular_weight']:.1f}")
                                        st.markdown(f"SA Score: {props['synthetic_accessibility']:.1f}")
                                    
                                    # Add button to add to project
                                    if st.button(f"Add to Project", key=f"add_frag_{mol_idx}"):
                                        # Create result object
                                        if validations[mol_smiles]['valid']:
                                            props = validations[mol_smiles]['properties']
                                            # Calculate similarity to reference
                                            ref_mol = Chem.MolFromSmiles(st.session_state.current_smiles)
                                            tanimoto = TanimotoSimilarity(
                                                AllChem.GetMorganFingerprintAsBitVect(ref_mol, 2, 2048),
                                                AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
                                            ) if ref_mol else 0.0
                                            
                                            result = OptimizationResult(
                                                smiles=mol_smiles,
                                                qed_score=props['qed'],
                                                tanimoto_score=tanimoto,
                                                sa_score=props['synthetic_accessibility'],
                                                timestamp=datetime.now(),
                                                parameters={'method': 'fragment_based_design'}
                                            )
                                            st.session_state.project.history.append(result)
                                            st.success(f"Added molecule {mol_idx+1} to project")
                
                # Add property visualization
                if len(molecules) > 2:
                    st.subheader("Property Analysis")
                    
                    # Create comparison table
                    comp_df = create_comparison_table(molecules)
                    
                    # Show data table
                    st.dataframe(comp_df)
                    
                    # Show scaffolds
                    fig, scaffold_info = create_scaffold_summary(molecules)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show top scaffold
                    if scaffold_info and len(scaffold_info) > 0:
                        top_scaffold = scaffold_info["Scaffold 1"]
                        scaffold_mol = Chem.MolFromSmiles(top_scaffold["smiles"])
                        if scaffold_mol:
                            st.markdown("**Most Common Scaffold:**")
                            st.image(smiles_to_image(top_scaffold["smiles"]), width=200)
        
        except Exception as e:
            st.error(f"Error during generation: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


def display_scaffold_hopping(client: MolMIMClient, server_available: bool):
    """Display the scaffold hopping interface."""
    st.markdown("<h3 class='subsection-header'>Scaffold Hopping</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    Generate molecules with similar properties but different core scaffolds. This approach is useful for
    exploring structurally diverse alternatives to a reference molecule.
    """)
    
    # Reference molecule
    st.markdown("**Reference Molecule**")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.image(smiles_to_image(st.session_state.current_smiles), caption="Reference Structure")
    
    with col2:
        # Extract scaffold from reference
        ref_mol = Chem.MolFromSmiles(st.session_state.current_smiles)
        if ref_mol:
            scaffold = MurckoScaffold.GetScaffoldForMol(ref_mol)
            scaffold_smiles = Chem.MolToSmiles(scaffold) if scaffold else "N/A"
            
            st.markdown(f"**Current Scaffold:** `{scaffold_smiles}`")
            st.image(smiles_to_image(scaffold_smiles), width=200, caption="Reference Scaffold")
    
    # Scaffold library
    st.markdown("**Scaffold Library**")
    
    library_type = st.radio("Scaffold Source", ["Default Library", "Custom Scaffolds"])
    
    if library_type == "Default Library":
        scaffolds = None  # Use default library in the generator
        
        # Display some examples from default library
        default_scaffs = ["c1ccccc1", "c1ccncc1", "C1CCCCC1", "c1ccc2ccccc2c1", "c1nc2ccccc2[nH]1"]
        scaff_cols = st.columns(5)
        for i, scaff in enumerate(default_scaffs):
            with scaff_cols[i]:
                mol = Chem.MolFromSmiles(scaff)
                if mol:
                    st.image(smiles_to_image(scaff, width=100, height=100), caption=scaff)
        
        st.markdown("*Default library includes common scaffolds for medicinal chemistry.*")
    else:
        scaffolds_text = st.text_area("Enter scaffolds (one SMILES per line)", height=100)
        scaffolds = scaffolds_text.strip().split('\n') if scaffolds_text else []
        
        # Validate scaffolds
        valid_scaffolds = []
        for scaff in scaffolds:
            if Chem.MolFromSmiles(scaff):
                valid_scaffolds.append(scaff)
                
        if valid_scaffolds:
            st.success(f"Found {len(valid_scaffolds)} valid scaffolds.")
        else:
            st.warning("No valid scaffolds found.")
            scaffolds = None
    
    # Generation parameters  
    st.markdown("**Generation Parameters**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_hops = st.number_input("Number of Molecules", 1, 50, 10, 1)
    
    with col2:
        seed_smiles = st.session_state.current_smiles
    
    # Run generation
    if st.button("Generate Molecules", key="run_scaffold"):
        if not server_available:
            st.error("MolMIM server is not available. Cannot run generation.")
            return
            
        try:
            with st.spinner("Generating molecules with scaffold hopping..."):
                # Run scaffold hopping
                generator = st.session_state.generator
                
                molecules = generator.scaffold_hopping(
                    seed_smiles=seed_smiles,
                    num_hops=num_hops,
                    scaffold_library=scaffolds
                )
                
                if not molecules:
                    st.warning("No valid molecules generated.")
                    return
                
                # Validate and score molecules
                validator = MoleculeValidator()
                validations = validator.validate_smiles(molecules)
                
                # Display molecules
                st.subheader("Generated Molecules")
                
                # Create display grid
                num_cols = 3
                num_rows = (len(molecules) + num_cols - 1) // num_cols
                
                for row in range(num_rows):
                    cols = st.columns(num_cols)
                    for col_idx in range(num_cols):
                        mol_idx = row * num_cols + col_idx
                        if mol_idx < len(molecules):
                            mol_smiles = molecules[mol_idx]
                            with cols[col_idx]:
                                mol = Chem.MolFromSmiles(mol_smiles)
                                if mol:
                                    # Extract scaffold
                                    mol_scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                                    scaffold_smiles = Chem.MolToSmiles(mol_scaffold) if mol_scaffold else "N/A"
                                    
                                    # Display molecule
                                    st.markdown(f"**Molecule {mol_idx+1}**")
                                    st.image(smiles_to_image(mol_smiles), use_container_width=True)
                                    
                                    # Display scaffold
                                    with st.expander("Show Scaffold"):
                                        st.image(smiles_to_image(scaffold_smiles), width=150)
                                        st.text(scaffold_smiles)
                                    
                                    # Display properties
                                    if validations[mol_smiles]['valid']:
                                        props = validations[mol_smiles]['properties']
                                        st.markdown(f"QED: {props['qed']:.3f}")
                                        st.markdown(f"LogP: {props['logP']:.2f}")
                                        st.markdown(f"MW: {props['molecular_weight']:.1f}")
                                        st.markdown(f"SA Score: {props['synthetic_accessibility']:.1f}")
                                    
                                    # Add button to add to project
                                    if st.button(f"Add to Project", key=f"add_scaff_{mol_idx}"):
                                        # Create result object
                                        if validations[mol_smiles]['valid']:
                                            props = validations[mol_smiles]['properties']
                                            # Calculate similarity to reference
                                            ref_mol = Chem.MolFromSmiles(st.session_state.current_smiles)
                                            tanimoto = TanimotoSimilarity(
                                                AllChem.GetMorganFingerprintAsBitVect(ref_mol, 2, 2048),
                                                AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
                                            ) if ref_mol else 0.0
                                            
                                            result = OptimizationResult(
                                                smiles=mol_smiles,
                                                qed_score=props['qed'],
                                                tanimoto_score=tanimoto,
                                                sa_score=props['synthetic_accessibility'],
                                                timestamp=datetime.now(),
                                                parameters={'method': 'scaffold_hopping'}
                                            )
                                            st.session_state.project.history.append(result)
                                            st.success(f"Added molecule {mol_idx+1} to project")
                
                # Add property visualization
                if len(molecules) > 2:
                    st.subheader("Property Analysis")
                    
                    # Create comparison table
                    comp_df = create_comparison_table(molecules)
                    
                    # Show data table
                    st.dataframe(comp_df)
                    
                    # Show scaffolds
                    fig, scaffold_info = create_scaffold_summary(molecules)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show diversity metrics
                    diversity = validator.calculate_diversity(molecules)
                    scaffold_diversity = validator.calculate_scaffold_diversity(molecules)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Molecular Diversity", f"{diversity:.2f}")
                    with col2:
                        st.metric("Scaffold Diversity", f"{scaffold_diversity:.2f}")
        
        except Exception as e:
            st.error(f"Error during generation: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


def display_analysis_page():
    """Display the analysis dashboard page."""
    st.markdown("<h2 class='section-header'>Analysis Dashboard</h2>", unsafe_allow_html=True)
    
    if st.session_state.project is None or len(st.session_state.project.history) == 0:
        st.warning("Please run optimization or generative methods first to generate molecules for analysis.")
        return
    
    # Get molecules from project history
    molecules = [r.smiles for r in st.session_state.project.history]
    
    # Analysis options
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["Molecule Properties", "Similarity Analysis", "Diversity Analysis", "Scaffold Analysis"]
    )
    
    if analysis_type == "Molecule Properties":
        display_property_analysis(molecules)
    elif analysis_type == "Similarity Analysis":
        display_similarity_analysis(molecules)
    elif analysis_type == "Diversity Analysis":
        display_diversity_analysis(molecules)
    elif analysis_type == "Scaffold Analysis":
        display_scaffold_analysis(molecules)


def display_property_analysis(molecules: List[str]):
    """Display property-based analysis of molecules."""
    st.markdown("<h3 class='subsection-header'>Molecular Property Analysis</h3>", unsafe_allow_html=True)
    
    # Create comparison table
    property_df = create_comparison_table(molecules)
    
    # Show data table with filters
    st.markdown("**Property Table**")
    
    # Add filters
    filter_cols = st.columns(4)
    
    filters = {}
    
    with filter_cols[0]:
        if 'MW' in property_df.columns:
            min_mw = st.slider("Min MW", 
                           float(property_df['MW'].min()), 
                           float(property_df['MW'].max()), 
                           float(property_df['MW'].min()))
            max_mw = st.slider("Max MW", 
                           float(property_df['MW'].min()), 
                           float(property_df['MW'].max()), 
                           float(property_df['MW'].max()))
            filters['MW'] = (min_mw, max_mw)
    
    with filter_cols[1]:
        if 'LogP' in property_df.columns:
            min_logp = st.slider("Min LogP", 
                             float(property_df['LogP'].min()), 
                             float(property_df['LogP'].max()), 
                             float(property_df['LogP'].min()))
            max_logp = st.slider("Max LogP", 
                             float(property_df['LogP'].min()), 
                             float(property_df['LogP'].max()), 
                             float(property_df['LogP'].max()))
            filters['LogP'] = (min_logp, max_logp)
    
    with filter_cols[2]:
        if 'QED' in property_df.columns:
            min_qed = st.slider("Min QED", 
                            float(property_df['QED'].min()), 
                            float(property_df['QED'].max()), 
                            float(property_df['QED'].min()))
            filters['QED'] = (min_qed, 1.0)
    
    with filter_cols[3]:
        if 'SA Score' in property_df.columns:
            max_sa = st.slider("Max SA Score", 
                           float(property_df['SA Score'].min()), 
                           float(property_df['SA Score'].max()), 
                           float(property_df['SA Score'].max()))
            filters['SA Score'] = (1.0, max_sa)
    
    # Apply filters
    filtered_df = property_df.copy()
    for prop, (min_val, max_val) in filters.items():
        filtered_df = filtered_df[(filtered_df[prop] >= min_val) & (filtered_df[prop] <= max_val)]

    # Show filtered data
    st.dataframe(filtered_df)
    
    if len(filtered_df) > 0:
        st.success(f"Showing {len(filtered_df)} of {len(property_df)} molecules that meet criteria.")
        
        # Property visualizations
        st.markdown("**Property Distributions**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = create_property_box_plot(filtered_df)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = create_qed_vs_sa_scatter(filtered_df)
            st.plotly_chart(fig, use_container_width=True)
        
        # Radar chart for selected molecules
        st.markdown("**Multi-Property Comparison**")
        
        # Select molecules for radar chart
        if len(filtered_df) > 10:
            st.info("Select up to 5 molecules to compare in detail.")
            selected_indices = st.multiselect(
                "Select molecules for radar chart",
                filtered_df.index.tolist(),
                filtered_df.index.tolist()[:5] if len(filtered_df) > 5 else filtered_df.index.tolist()
            )
            
            if selected_indices:
                selected_df = filtered_df.loc[selected_indices]
                fig = create_property_radar_chart(selected_df, normalize=True)
                st.plotly_chart(fig, use_container_width=True)
        else:
            fig = create_property_radar_chart(filtered_df, normalize=True)
            st.plotly_chart(fig, use_container_width=True)
        
        # Show molecules
        st.markdown("**Filtered Molecules**")
        
        # Create a grid of molecules
        n_cols = 4
        n_rows = (len(filtered_df) + n_cols - 1) // n_cols
        
        for i in range(n_rows):
            cols = st.columns(n_cols)
            for j in range(n_cols):
                idx = i * n_cols + j
                if idx < len(filtered_df):
                    smiles = filtered_df.index[idx]
                    with cols[j]:
                        mol = Chem.MolFromSmiles(smiles)
                        if mol:
                            st.image(smiles_to_image(smiles, width=150, height=150), caption=f"Molecule {idx+1}")
                            with st.expander("Properties"):
                                for col in filtered_df.columns:
                                    st.markdown(f"**{col}**: {filtered_df.iloc[idx][col]}")
    else:
        st.warning("No molecules meet all criteria. Try adjusting the filters.")


def display_similarity_analysis(molecules: List[str]):
    """Display similarity-based analysis of molecules."""
    st.markdown("<h3 class='subsection-header'>Similarity Analysis</h3>", unsafe_allow_html=True)
    
    if st.session_state.project is None:
        st.warning("Please load a project first.")
        return
    
    # Get reference molecule
    reference_smiles = st.session_state.project.reference_smiles
    reference_mol = Chem.MolFromSmiles(reference_smiles)
    
    if not reference_mol:
        st.error("Invalid reference molecule.")
        return
    
    # Display reference molecule
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("**Reference Molecule**")
        st.image(smiles_to_image(reference_smiles), caption="Reference")
    
    with col2:
        # Calculate similarities to reference
        similarities = []
        valid_mols = []
        
        for smi in molecules:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                # Calculate similarity
                ref_fp = AllChem.GetMorganFingerprintAsBitVect(reference_mol, 2, 2048)
                mol_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
                similarity = TanimotoSimilarity(ref_fp, mol_fp)
                
                similarities.append(similarity)
                valid_mols.append(smi)
        
        # Create histogram
        if similarities:
            fig = go.Figure(data=[go.Histogram(x=similarities, nbinsx=20)])
            fig.update_layout(
                title="Similarity Distribution to Reference",
                xaxis_title="Tanimoto Similarity",
                yaxis_title="Count",
                bargap=0.1
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Similarity matrix
    st.markdown("**Molecule Similarity Matrix**")
    
    # Limit to a reasonable number of molecules for visualization
    max_display = 50
    if len(valid_mols) > max_display:
        st.info(f"Showing similarity matrix for the first {max_display} molecules (of {len(valid_mols)}).")
        display_mols = valid_mols[:max_display]
    else:
        display_mols = valid_mols
    
    # Create similarity heatmap
    fig = create_similarity_heatmap(display_mols)
    st.plotly_chart(fig, use_container_width=True)
    
    # MDS/PCA visualization
    st.markdown("**Molecular Space Visualization (PCA)**")
    
    # Create PCA visualization
    fig = create_3d_pca_scatter(valid_mols)
    st.plotly_chart(fig, use_container_width=True)
    
    # Show most and least similar molecules
    if similarities and valid_mols:
        st.markdown("**Most & Least Similar Molecules to Reference**")
        
        col1, col2 = st.columns(2)
        
        # Most similar molecules (excluding reference itself)
        sim_pairs = sorted(zip(valid_mols, similarities), key=lambda x: x[1], reverse=True)
        
        # Filter out the reference if it's in the list
        sim_pairs = [p for p in sim_pairs if p[0] != reference_smiles]
        
        if sim_pairs:
            with col1:
                st.markdown("**Most Similar**")
                for i, (smi, sim) in enumerate(sim_pairs[:3]):
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        st.image(smiles_to_image(smi, width=200, height=150), caption=f"Similarity: {sim:.3f}")
            
            with col2:
                st.markdown("**Least Similar**")
                for i, (smi, sim) in enumerate(sim_pairs[-3:]):
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        st.image(smiles_to_image(smi, width=200, height=150), caption=f"Similarity: {sim:.3f}")


def display_diversity_analysis(molecules: List[str]):
    """Display diversity-based analysis of molecules."""
    st.markdown("<h3 class='subsection-header'>Diversity Analysis</h3>", unsafe_allow_html=True)
    
    # Calculate diversity metrics
    validator = MoleculeValidator()
    
    # Get valid molecules
    valid_molecules = []
    for smi in molecules:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            valid_molecules.append(smi)
    
    if not valid_molecules:
        st.warning("No valid molecules to analyze.")
        return
    
    # Calculate diversity metrics
    tanimoto_diversity = validator.calculate_diversity(valid_molecules)
    scaffold_diversity = validator.calculate_scaffold_diversity(valid_molecules)
    
    # Display metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Molecular Diversity", f"{tanimoto_diversity:.3f}", 
                 help="Higher values indicate more diverse molecules (0-1 scale)")
    
    with col2:
        st.metric("Scaffold Diversity", f"{scaffold_diversity:.3f}",
                help="Ratio of unique scaffolds to molecules (0-1 scale)")
    
    # Run PCA and clustering
    st.markdown("**Chemical Space Clustering**")
    
    # Perform clustering if there are enough molecules
    if len(valid_molecules) >= 5:
        # Convert molecules to fingerprints
        mols = [Chem.MolFromSmiles(smi) for smi in valid_molecules]
        fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048) for mol in mols if mol]
        
        # Convert fingerprints to numpy array
        fp_array = np.zeros((len(fps), fps[0].GetNumBits()))
        for i, fp in enumerate(fps):
            DataStructs.ConvertToNumpyArray(fp, fp_array[i])
        
        # Perform clustering
        n_clusters = min(5, len(fps))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(fp_array)
        
        # Perform PCA for visualization
        pca = PCA(n_components=3)
        reduced = pca.fit_transform(fp_array)
        
        # Create PCA plot with clusters
        cluster_df = pd.DataFrame({
            'PC1': reduced[:, 0],
            'PC2': reduced[:, 1],
            'PC3': reduced[:, 2],
            'Cluster': clusters,
            'SMILES': valid_molecules[:len(clusters)]
        })
        
        fig = px.scatter_3d(
            cluster_df,
            x='PC1',
            y='PC2',
            z='PC3',
            color='Cluster',
            hover_data=['SMILES'],
            title="Molecular Clusters in Chemical Space"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show cluster statistics
        st.markdown("**Cluster Statistics**")
        
        # Count molecules per cluster
        cluster_counts = pd.Series(clusters).value_counts().sort_index()
        
        # Calculate within-cluster diversity
        cluster_diversities = []
        for i in range(n_clusters):
            cluster_mols = [valid_molecules[j] for j, c in enumerate(clusters) if c == i]
            if len(cluster_mols) > 1:
                diversity = validator.calculate_diversity(cluster_mols)
                cluster_diversities.append(diversity)
            else:
                cluster_diversities.append(0.0)
        
        # Create cluster statistics dataframe
        cluster_stats = pd.DataFrame({
            'Cluster': range(n_clusters),
            'Molecules': cluster_counts.values,
            'Within-Cluster Diversity': cluster_diversities
        })
        
        st.dataframe(cluster_stats)
        
        # Show representative molecules from each cluster
        st.markdown("**Representative Molecules from Each Cluster**")
        
        # Find molecules closest to cluster centers
        closest_to_center = []
        for i in range(n_clusters):
            cluster_indices = [j for j, c in enumerate(clusters) if c == i]
            if cluster_indices:
                cluster_points = fp_array[cluster_indices]
                center = kmeans.cluster_centers_[i]
                
                # Calculate distances to center
                distances = np.linalg.norm(cluster_points - center, axis=1)
                closest_idx = cluster_indices[np.argmin(distances)]
                closest_to_center.append(valid_molecules[closest_idx])
        
        # Display representative molecules
        cols = st.columns(n_clusters)
        for i, (mol_smiles, col) in enumerate(zip(closest_to_center, cols)):
            with col:
                mol = Chem.MolFromSmiles(mol_smiles)
                if mol:
                    st.image(smiles_to_image(mol_smiles), caption=f"Cluster {i}")
    else:
        st.info("At least 5 valid molecules are needed for clustering analysis.")


def display_scaffold_analysis(molecules: List[str]):
    """Display scaffold-based analysis of molecules."""
    st.markdown("<h3 class='subsection-header'>Scaffold Analysis</h3>", unsafe_allow_html=True)
    
    # Extract scaffolds
    scaffolds = {}
    for smi in molecules:
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            continue
            
        # Get Murcko scaffold
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if not scaffold:
            continue
            
        scaffold_smiles = Chem.MolToSmiles(scaffold)
        if scaffold_smiles not in scaffolds:
            scaffolds[scaffold_smiles] = []
        scaffolds[scaffold_smiles].append(smi)
    
    if not scaffolds:
        st.warning("No valid scaffolds found in the molecules.")
        return
    
    # Create scaffold summary chart
    fig, scaffold_info = create_scaffold_summary(molecules)
    st.plotly_chart(fig, use_container_width=True)
    
    # Display top scaffolds
    st.markdown("**Top Scaffolds**")
    
    # Sort scaffolds by frequency
    sorted_scaffolds = sorted(scaffolds.items(), key=lambda x: len(x[1]), reverse=True)
    
    # Display top N scaffolds
    top_n = min(5, len(sorted_scaffolds))
    
    cols = st.columns(top_n)
    for i, (scaff_smiles, mols) in enumerate(sorted_scaffolds[:top_n]):
        with cols[i]:
            scaff_mol = Chem.MolFromSmiles(scaff_smiles)
            if scaff_mol:
                st.image(smiles_to_image(scaff_smiles), caption=f"Scaffold {i+1}")
                st.markdown(f"**Count:** {len(mols)}")
                st.markdown(f"**%:** {100 * len(mols) / len(molecules):.1f}%")
    
    # Show molecules for selected scaffold
    st.markdown("**Molecules by Scaffold**")
    
    # Create a selectbox for scaffolds
    scaffold_options = [f"Scaffold {i+1} ({len(mols)} molecules)" 
                        for i, (_, mols) in enumerate(sorted_scaffolds)]
    
    selected_scaffold = st.selectbox("Select scaffold to view molecules", scaffold_options)
    
    if selected_scaffold:
        # Get the index from the selection string
        scaffold_idx = int(selected_scaffold.split(" ")[1]) - 1
        
        # Get molecules for this scaffold
        scaffold_smiles, scaffold_molecules = sorted_scaffolds[scaffold_idx]
        
        # Display the scaffold
        scaff_mol = Chem.MolFromSmiles(scaffold_smiles)
        if scaff_mol:
            st.image(smiles_to_image(scaffold_smiles, width=300, height=200), caption="Scaffold Structure")
        
        # Display molecules with this scaffold
        st.markdown(f"**Molecules containing this scaffold ({len(scaffold_molecules)})**")
        
        # Create a grid of molecules
        n_cols = 4
        n_rows = (len(scaffold_molecules) + n_cols - 1) // n_cols
        
        for i in range(min(n_rows, 3)):  # Limit to 3 rows
            cols = st.columns(n_cols)
            for j in range(n_cols):
                idx = i * n_cols + j
                if idx < len(scaffold_molecules):
                    smiles = scaffold_molecules[idx]
                    with cols[j]:
                        mol = Chem.MolFromSmiles(smiles)
                        if mol:
                            st.image(smiles_to_image(smiles, width=150, height=150))


def display_ai_analysis_page():
    """Display the AI analysis page."""
    st.markdown("<h2 class='section-header'>AI-Driven Molecule Analysis</h2>", unsafe_allow_html=True)
    
    if st.session_state.project is None:
        st.warning("Please create or load a project first (use the sidebar).")
        return
    
    # Get molecules from project or allow direct input
    analysis_source = st.radio(
        "Molecule Source",
        ["Project Molecules", "Direct Input"]
    )
    
    if analysis_source == "Project Molecules":
        if len(st.session_state.project.history) == 0:
            st.warning("No molecules in project history. Run optimization or generation first.")
            return
            
        # Create selectbox for molecules
        molecule_options = []
        for i, result in enumerate(st.session_state.project.history):
            # Truncate SMILES for display
            smiles_display = result.smiles[:30] + "..." if len(result.smiles) > 30 else result.smiles
            molecule_options.append(f"Molecule {i+1}: {smiles_display} (QED: {result.qed_score:.2f})")
        
        # Add reference molecule as option
        ref_smiles = st.session_state.project.reference_smiles
        ref_smiles_display = ref_smiles[:30] + "..." if len(ref_smiles) > 30 else ref_smiles
        molecule_options.insert(0, f"Reference: {ref_smiles_display}")
        
        selected_option = st.selectbox("Select molecule to analyze", molecule_options)
        
        # Extract molecule index from selection
        if selected_option.startswith("Reference"):
            selected_smiles = st.session_state.project.reference_smiles
        else:
            idx = int(selected_option.split(":")[0].split(" ")[1]) - 1
            selected_smiles = st.session_state.project.history[idx].smiles
    else:
        # Direct SMILES input
        selected_smiles = st.text_area("Enter SMILES to analyze", st.session_state.current_smiles, height=100)
    
    # Ensure valid molecule
    mol = Chem.MolFromSmiles(selected_smiles) if selected_smiles else None
    
    if not mol:
        st.error("Invalid SMILES string")
        return
    
    # Show molecule
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Selected Molecule**")
        st.image(smiles_to_image(selected_smiles), use_container_width=True)
    
    # Analyze button
    if st.button("Analyze Molecule", key="analyze_single"):
        with st.spinner("Running AI analysis..."):
            try:
                # Get analyzer
                if "analyzer" not in st.session_state:
                    st.session_state.analyzer = ConversationalMoleculeAnalyzer()
                
                # Run analysis
                analysis = st.session_state.analyzer.analyze_input_molecule(selected_smiles)
                
                # Show analysis
                with col2:
                    st.markdown("**AI Analysis Results**")
                    st.markdown(analysis)
                
                # Add area for follow-up questions
                st.markdown("**Follow-up Questions**")
                follow_up = st.text_input("Ask a follow-up question about this molecule:")
                
                if follow_up:
                    with st.spinner("Processing follow-up question..."):
                        response = st.session_state.analyzer.answer_follow_up(follow_up)
                        st.markdown(response)
            
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
    
    # Molecule comparison
    st.markdown("<h3 class='subsection-header'>Molecule Comparison</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    Compare multiple molecules to understand their key differences and advantages.
    """)
    
    # Allow selection of multiple molecules for comparison
    if analysis_source == "Project Molecules" and len(st.session_state.project.history) > 1:
        selected_indices = st.multiselect(
            "Select molecules to compare (2-3 recommended)",
            range(len(st.session_state.project.history)),
            format_func=lambda i: f"Molecule {i+1}: QED={st.session_state.project.history[i].qed_score:.2f}, SA={st.session_state.project.history[i].sa_score:.2f}"
        )
        
        if selected_indices:
            comparison_smiles = [st.session_state.project.history[i].smiles for i in selected_indices]
            
            # Check if reference should be included
            include_reference = st.checkbox("Include reference molecule in comparison", value=True)
            
            if include_reference:
                comparison_smiles.insert(0, st.session_state.project.reference_smiles)
            
            # Show selected molecules
            st.markdown("**Selected Molecules for Comparison**")
            
            cols = st.columns(len(comparison_smiles))
            for i, (smi, col) in enumerate(zip(comparison_smiles, cols)):
                with col:
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        st.image(smiles_to_image(smi, width=150, height=120))
                        label = "Reference" if i == 0 and include_reference else f"Molecule {selected_indices[i-1 if include_reference else i]+1}"
                        st.markdown(f"**{label}**")
            
            # Compare button
            if st.button("Compare Molecules", key="compare_molecules") and len(comparison_smiles) >= 2:
                with st.spinner("Running comparative analysis..."):
                    try:
                        # Get analyzer
                        if "analyzer" not in st.session_state:
                            st.session_state.analyzer = ConversationalMoleculeAnalyzer()
                        
                        # Run comparison
                        comparison = st.session_state.analyzer.compare_molecules(comparison_smiles)
                        
                        # Show comparison
                        st.markdown("**Comparative Analysis Results**")
                        st.markdown(comparison)
                    
                    except Exception as e:
                        st.error(f"Error during comparison: {str(e)}")
    
    else:
        st.info("Add multiple molecules to your project to enable comparison.")
    
    # Optimization results analysis
    if st.session_state.project and len(st.session_state.project.history) > 5:
        st.markdown("<h3 class='subsection-header'>Optimization Results Analysis</h3>", unsafe_allow_html=True)
        
        st.markdown("""
        Get expert insights on your optimization results and suggestions for further improvements.
        """)
        
        if st.button("Analyze Optimization Results", key="analyze_optimization"):
            with st.spinner("Analyzing optimization results..."):
                try:
                    # Get analyzer
                    if "analyzer" not in st.session_state:
                        st.session_state.analyzer = ConversationalMoleculeAnalyzer()
                    
                    # Sort results by QED score
                    sorted_results = sorted(st.session_state.project.history, key=lambda r: r.qed_score, reverse=True)
                    
                    # Take top results
                    top_results = sorted_results[:10]
                    
                    # Run analysis
                    analysis = st.session_state.analyzer.analyze_optimization_results(
                        top_results,
                        st.session_state.project.reference_smiles
                    )
                    
                    # Show analysis
                    st.markdown("**Optimization Results Analysis**")
                    st.markdown(analysis)
                
                except Exception as e:
                    st.error(f"Error during optimization analysis: {str(e)}")


def main():
    """Main Streamlit application."""
    # Page configuration
    st.set_page_config(
        page_title="Advanced Drug Discovery Optimization Platform",
        page_icon="💊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        color: #1E3A8A;
    }
    .section-header {
        font-size: 1.8rem;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        color: #1E3A8A;
    }
    .subsection-header {
        font-size: 1.4rem;
        margin-top: 0.8rem;
        margin-bottom: 0.4rem;
        color: #2563EB;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #E0FFFF;
        border: 1px solid #87CEEB;
        margin-bottom: 1rem;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #FFF3CD;
        border: 1px solid #FFECB5;
        margin-bottom: 1rem;
    }
    .molecule-card {
        border: 1px solid #E5E7EB;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        background-color: #F9FAFB;
    }
    .center-content {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Application title
    st.markdown("<h1 class='main-header'>Advanced Drug Discovery Optimization Platform</h1>", unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("Settings & Navigation")
    
    # Server connection settings
    st.sidebar.subheader("Model Server Connection")
    host = st.sidebar.text_input("MolMIM Host", "localhost")
    port = st.sidebar.text_input("MolMIM Port", "8000")
    
    # Global parameters
    st.sidebar.subheader("Global Parameters")
    radius = st.sidebar.slider("Fingerprint Radius", 1, 4, 2)
    fp_size = st.sidebar.selectbox("Fingerprint Size", [1024, 2048, 4096], index=1)
    parallel = st.sidebar.checkbox("Enable Parallel Processing", True)
    n_jobs = st.sidebar.slider("CPU Cores (if parallel)", 1, max(1, cpu_count() - 1), max(1, cpu_count() // 2))
    
    # Initialize session state if needed
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.project = None
        st.session_state.top_molecules = []
        st.session_state.current_smiles = ""
        st.session_state.analyzer = ConversationalMoleculeAnalyzer()
        st.session_state.optimization_running = False
        st.session_state.optimization_results = []
        st.session_state.generator = None
        st.session_state.client = None
    
    # Initialize MolMIM client
    if st.session_state.client is None or st.session_state.client.base_url != f"http://{host}:{port}":
        st.session_state.client = MolMIMClient(host, port)
    
    # Check MolMIM server health
    client = st.session_state.client
    if not client.check_health():
        st.sidebar.warning("⚠️ MolMIM server is not accessible. Some generative features will be disabled.")
        server_available = False
    else:
        st.sidebar.success("✅ Connected to MolMIM server")
        server_available = True
    
    # Page navigation
    page = st.sidebar.radio(
        "Navigation",
        ["Home", "Molecule Optimization", "Generative Methods", "Analysis Dashboard", "AI Analysis"]
    )
    
    # Project management
    st.sidebar.markdown("---")
    st.sidebar.subheader("Project Management")
    
    project_action = st.sidebar.radio("Project Options", ["Current Project", "New Project", "Load Project", "Save Project"])
    
    if project_action == "New Project":
        with st.sidebar.form("new_project_form"):
            project_name = st.text_input("Project Name", "Untitled Project")
            project_description = st.text_area("Description", "")
            ref_smiles = st.text_input("Reference SMILES", "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5")
            
            if st.form_submit_button("Create Project"):
                try:
                    # Validate reference SMILES
                    if not Chem.MolFromSmiles(ref_smiles):
                        st.error("Invalid reference SMILES")
                    else:
                        # Create new project
                        strategy = OptimizationStrategy()
                        st.session_state.project = ProjectState(
                            reference_smiles=ref_smiles,
                            optimization_strategy=strategy,
                            name=project_name,
                            description=project_description
                        )
                        st.session_state.current_smiles = ref_smiles
                        st.success(f"Project '{project_name}' created successfully!")
                except Exception as e:
                    st.error(f"Error creating project: {str(e)}")
    
    elif project_action == "Load Project":
        uploaded_file = st.sidebar.file_uploader("Upload Project File", type=["pkl", "json"])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.pkl'):
                    # Load pickle file
                    with tempfile.NamedTemporaryFile(delete=False) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                    
                    st.session_state.project = ProjectState.load(tmp_path)
                    os.unlink(tmp_path)
                else:
                    # Load JSON file
                    project_data = json.loads(uploaded_file.getvalue())
                    st.session_state.project = ProjectState.from_dict(project_data)
                
                st.session_state.current_smiles = st.session_state.project.reference_smiles
                st.sidebar.success(f"Project '{st.session_state.project.name}' loaded successfully!")
            except Exception as e:
                st.sidebar.error(f"Error loading project: {str(e)}")
    
    elif project_action == "Save Project" and st.session_state.project is not None:
        # Save format selection
        save_format = st.sidebar.radio("Save Format", ["Pickle (.pkl)", "JSON (.json)"])
        
        if st.sidebar.button("Save Project"):
            try:
                if save_format == "Pickle (.pkl)":
                    # Save as pickle
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
                        st.session_state.project.save(tmp.name)
                        tmp_path = tmp.name
                    
                    with open(tmp_path, 'rb') as f:
                        file_data = f.read()
                    
                    st.sidebar.download_button(
                        label="Download Project File",
                        data=file_data,
                        file_name=f"{st.session_state.project.name.replace(' ', '_')}.pkl",
                        mime="application/octet-stream"
                    )
                    
                    os.unlink(tmp_path)
                else:
                    # Save as JSON
                    project_json = json.dumps(st.session_state.project.to_dict(), indent=2, default=str)
                    
                    st.sidebar.download_button(
                        label="Download Project File",
                        data=project_json,
                        file_name=f"{st.session_state.project.name.replace(' ', '_')}.json",
                        mime="application/json"
                    )
                
                st.sidebar.success("Project ready to download!")
            except Exception as e:
                st.sidebar.error(f"Error saving project: {str(e)}")
    
    # Initialize generator if needed
    if server_available and (st.session_state.generator is None or st.session_state.generator.reference_smiles != st.session_state.current_smiles):
        st.session_state.generator = ConditionalMoleculeGenerator(client, st.session_state.current_smiles)
    
    # Display current project info
    if st.session_state.project is not None:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Current Project")
        st.sidebar.markdown(f"**Name:** {st.session_state.project.name}")
        st.sidebar.markdown(f"**Created:** {st.session_state.project.created_at.strftime('%Y-%m-%d')}")
        st.sidebar.markdown(f"**Molecules:** {len(st.session_state.project.history)}")
    
    # Main content based on selected page
    if page == "Home":
        display_home_page(server_available)
    elif page == "Molecule Optimization":
        display_optimization_page(client, radius, fp_size, parallel, n_jobs, server_available)
    elif page == "Generative Methods":
        display_generative_page(client, server_available)
    elif page == "Analysis Dashboard":
        display_analysis_page()
    elif page == "AI Analysis":
        display_ai_analysis_page()


if __name__ == "__main__":
    main()         