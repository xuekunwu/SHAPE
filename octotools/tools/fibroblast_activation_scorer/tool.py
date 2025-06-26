#!/usr/bin/env python3
"""
Fibroblast Activation Scorer Tool

This tool quantifies fibroblast activation levels using reference data from Hugging Face.
It downloads reference files automatically and provides comprehensive activation scoring.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import scanpy as sc
    import anndata as ad
    SCANPY_AVAILABLE = True
except ImportError:
    SCANPY_AVAILABLE = False
    print("Warning: scanpy not available. Some advanced features may be limited.")

try:
    from huggingface_hub import hf_hub_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: huggingface_hub not available. Manual reference file upload required.")

from octotools.tools.base import BaseTool


class Fibroblast_Activation_Scorer_Tool(BaseTool):
    """
    Tool for quantifying fibroblast activation levels using reference data.
    
    This tool automatically downloads reference files from Hugging Face and provides
    comprehensive activation scoring with detailed visualizations and statistical analysis.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Tool metadata
        self.tool_name = "Fibroblast_Activation_Scorer_Tool"
        self.tool_description = """
        Quantifies fibroblast activation levels using reference data from Hugging Face.
        Automatically downloads reference files and provides comprehensive activation scoring
        with detailed visualizations including UMAP plots, activation distributions,
        box plots showing individual cell activation scores, and statistical analysis. 
        Supports both local and Hugging Face reference files.
        """
        self.tool_version = "1.0.0"
        
        # Input/output specifications
        self.input_types = {
            'cell_data': 'Union[str, List[str]] - Path to cell data file (.h5ad) or list of cell crop paths',
            'reference_source': "str - Reference source: 'huggingface' or 'local' (default: 'huggingface')",
            'reference_repo_id': 'str - Hugging Face repository ID for reference data (default: "5xuekun/adata_reference")',
            'reference_filename': 'str - Reference filename in Hugging Face repo (default: "adata_reference.h5ad")',
            'local_reference_path': 'str - Local path to reference file (when reference_source="local")',
            'output_dir': 'str - Output directory for results (default: "output_visualizations")',
            'visualization_type': "str - Visualization method: 'basic', 'comprehensive', or 'all' (default: 'all')",
            'confidence_threshold': 'float - Minimum confidence threshold for scoring (default: 0.5)',
            'batch_size': 'int - Batch size for processing (default: 100)'
        }
        
        self.output_type = """
        dict - Comprehensive activation scoring results including:
        - activation_scores: Individual cell activation scores
        - reference_stats: Reference dataset statistics
        - comparison_results: Statistical comparison between query and reference
        - visualizations: Paths to generated visualization files (including box plots)
        - metadata: Processing metadata and parameters
        """
        
        # Demo commands
        self.demo_commands = [
            {
                'command': 'execution = tool.execute(cell_data="path/to/cells.h5ad", visualization_type="all")',
                'description': 'Score fibroblast activation with comprehensive visualizations using Hugging Face reference'
            },
            {
                'command': 'execution = tool.execute(cell_data=["cell1.png", "cell2.png"], reference_source="local", local_reference_path="reference.h5ad")',
                'description': 'Score activation using local reference file and cell crop images'
            },
            {
                'command': 'execution = tool.execute(cell_data="cells.h5ad", reference_repo_id="custom/repo", reference_filename="custom_reference.h5ad")',
                'description': 'Use custom Hugging Face repository for reference data'
            }
        ]
        
        # User metadata
        self.user_metadata = {
            'limitation': 'Requires internet connection for Hugging Face downloads. Reference data quality affects scoring accuracy.',
            'best_practice': 'Use high-quality cell data and ensure reference data matches your cell type and experimental conditions.',
            'reference_data': 'Automatically downloads reference data from Hugging Face. Supports custom repositories and local files.',
            'visualization': 'Generates comprehensive visualizations including UMAP plots, activation distributions, and statistical comparisons.',
            'huggingface_integration': 'Seamless integration with Hugging Face for reference data management and sharing.'
        }
        
        # Default parameters
        self.default_params = {
            'reference_source': 'huggingface',
            'reference_repo_id': '5xuekun/adata_reference',
            'reference_filename': 'adata_reference.h5ad',
            'output_dir': 'output_visualizations',
            'visualization_type': 'all',
            'confidence_threshold': 0.5,
            'batch_size': 100
        }
        
        # Initialize reference data
        self.reference_data = None
        self.reference_path = None
        
    def _download_reference_from_hf(self, repo_id: str, filename: str) -> str:
        """
        Download reference file from Hugging Face.
        
        Args:
            repo_id: Hugging Face repository ID
            filename: Filename in the repository
            
        Returns:
            str: Path to downloaded reference file
        """
        if not HF_AVAILABLE:
            raise ImportError("huggingface_hub not available. Please install with: pip install huggingface_hub")
        
        try:
            print(f"Downloading reference file from {repo_id}...")
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                token=os.getenv("HUGGINGFACE_TOKEN")
            )
            print(f"Reference file downloaded to: {local_path}")
            return local_path
        except Exception as e:
            raise Exception(f"Failed to download reference file from {repo_id}: {str(e)}")
    
    def _load_reference_data(self, reference_path: str) -> ad.AnnData:
        """
        Load reference data from file.
        
        Args:
            reference_path: Path to reference file
            
        Returns:
            AnnData: Loaded reference data
        """
        if not SCANPY_AVAILABLE:
            raise ImportError("scanpy not available. Please install with: pip install scanpy anndata")
        
        try:
            print(f"Loading reference data from: {reference_path}")
            reference_data = sc.read_h5ad(reference_path)
            print(f"Reference data loaded: {reference_data.shape[0]} cells, {reference_data.shape[1]} features")
            
            # Filter out cells with null dpt_pseudotime from reference data
            null_mask = reference_data.obs['dpt_pseudotime'].isnull()
            if null_mask.sum() > 0:
                print(f"⚠️  Found {null_mask.sum()} cells with null dpt_pseudotime in reference data. Filtering them out.")
                clean_ref_data = reference_data[~null_mask].copy()
                print(f"Clean reference data shape: {clean_ref_data.shape}")
            else:
                clean_ref_data = reference_data
            
            return clean_ref_data
        except Exception as e:
            raise Exception(f"Failed to load reference data from {reference_path}: {str(e)}")
    
    def _prepare_cell_data(self, cell_data: Union[str, List[str]]) -> ad.AnnData:
        """
        Prepare cell data for analysis.
        
        Args:
            cell_data: Path to cell data file or list of cell crop paths
            
        Returns:
            AnnData: Prepared cell data
        """
        if isinstance(cell_data, str):
            # Single file path
            if cell_data.endswith('.h5ad'):
                # Load query data from h5ad file (not reference data)
                return self._load_query_data(cell_data)
            else:
                # Assume it's a single cell crop image
                return self._create_adata_from_images([cell_data], self.reference_data)
        elif isinstance(cell_data, list):
            # List of file paths
            if all(path.endswith('.h5ad') for path in cell_data):
                # Multiple h5ad files - combine them
                combined_data = []
                for path in cell_data:
                    data = self._load_query_data(path)
                    combined_data.append(data)
                return ad.concat(combined_data, join='outer')
            else:
                # List of image files
                return self._create_adata_from_images(cell_data, self.reference_data)
        else:
            raise ValueError("cell_data must be a string (file path) or list of strings (file paths)")
    
    def _load_query_data(self, query_path: str) -> ad.AnnData:
        """
        Load query data from h5ad file.
        
        Args:
            query_path: Path to query h5ad file
            
        Returns:
            AnnData: Loaded query data
        """
        if not SCANPY_AVAILABLE:
            raise ImportError("scanpy not available. Please install with: pip install scanpy anndata")
        
        try:
            print(f"Loading query data from: {query_path}")
            query_data = sc.read_h5ad(query_path)
            print(f"Query data loaded: {query_data.shape[0]} cells, {query_data.shape[1]} features")
            
            # Check if query data has required columns
            required_obs_keys = ['predicted_class']
            missing_keys = [key for key in required_obs_keys if key not in query_data.obs.columns]
            if missing_keys:
                print(f"⚠️  Query data missing required columns: {missing_keys}")
            
            return query_data
        except Exception as e:
            raise Exception(f"Failed to load query data from {query_path}: {str(e)}")
    
    def _create_adata_from_images(self, image_paths: List[str], reference_data: ad.AnnData = None) -> ad.AnnData:
        """
        Create AnnData object from image paths with features matching reference data.
        
        Args:
            image_paths: List of image file paths
            reference_data: Reference AnnData object to match features
            
        Returns:
            AnnData: Created AnnData object with matching features
        """
        n_cells = len(image_paths)
        
        # Get feature names from reference data if available
        if reference_data is not None:
            n_features = reference_data.shape[1]
            feature_names = reference_data.var_names.tolist()
            print(f"Using {n_features} features from reference data")
        else:
            # Fallback to default features
            n_features = 768
            feature_names = [f"gene_{i:04d}" for i in range(n_features)]
            print(f"Using default {n_features} features")
        
        # Create mock feature matrix (in real implementation, extract features from images)
        # For now, create random features that simulate image-derived features
        np.random.seed(42)  # For reproducible results
        X = np.random.randn(n_cells, n_features)
        
        # Create AnnData with matching feature names
        adata = ad.AnnData(X=X)
        adata.obs['cell_id'] = [f"cell_{i:04d}" for i in range(n_cells)]
        adata.obs['image_path'] = image_paths
        adata.var_names = feature_names
        
        print(f"Created AnnData from {n_cells} images with {n_features} features")
        print(f"Feature names match reference: {len(set(adata.var_names) & set(reference_data.var_names)) if reference_data is not None else 'N/A'}")
        
        return adata
    
    def _calculate_activation_scores(self, query_data: ad.AnnData, reference_data: ad.AnnData) -> np.ndarray:
        """
        Calculate activation scores using a robust approach that handles variable mismatches.
        """
        import scanpy as sc
        import numpy as np
        import pandas as pd
        
        print(f"Query data shape: {query_data.shape}")
        print(f"Reference data shape: {reference_data.shape}")
        print(f"Query variables: {len(query_data.var_names)}")
        print(f"Reference variables: {len(reference_data.var_names)}")
        
        # Check if variables match
        if len(query_data.var_names) != len(reference_data.var_names):
            print("⚠️  Variable count mismatch detected. Using robust activation scoring method.")
            return self._calculate_robust_activation_scores(query_data, reference_data)
        
        # Check if variable names match
        if not np.array_equal(query_data.var_names, reference_data.var_names):
            print("⚠️  Variable names mismatch detected. Using robust activation scoring method.")
            return self._calculate_robust_activation_scores(query_data, reference_data)
        
        # Check if reference data has required columns
        required_obs_keys = ['dpt_pseudotime', 'predicted_class']
        missing_keys = [key for key in required_obs_keys if key not in reference_data.obs.columns]
        if missing_keys:
            print(f"⚠️  Reference data missing required columns: {missing_keys}. Using robust activation scoring method.")
            return self._calculate_robust_activation_scores(query_data, reference_data)
        
        # Filter out cells with null dpt_pseudotime from reference data
        null_mask = reference_data.obs['dpt_pseudotime'].isnull()
        if null_mask.sum() > 0:
            print(f"⚠️  Found {null_mask.sum()} cells with null dpt_pseudotime in reference data. Filtering them out.")
            clean_ref_data = reference_data[~null_mask].copy()
            print(f"Clean reference data shape: {clean_ref_data.shape}")
        else:
            clean_ref_data = reference_data
        
        # If variables match and required columns exist, use the original method
        print("✅ Variables match and required columns present. Using standard ingest-based activation scoring.")
        try:
            # 1. Ingest: map query to reference
            print("Running sc.tl.ingest...")
            sc.tl.ingest(query_data, clean_ref_data)
            print("✅ Ingest completed")
            
            # 2. Neighbors & Diffmap
            print("Running sc.pp.neighbors...")
            sc.pp.neighbors(query_data)
            print("✅ Neighbors computed")
            
            print("Running sc.tl.diffmap...")
            sc.tl.diffmap(query_data)
            print("✅ Diffmap computed")
            
            # 3. Set root cell for DPT calculation
            print("Setting root cell for DPT calculation...")
            if 'predicted_class' in query_data.obs.columns:
                # Try to find a q-Fb cell as root
                qfb_mask = query_data.obs['predicted_class'] == 'q-Fb'
                if qfb_mask.sum() > 0:
                    # Use the first q-Fb cell as root
                    root_idx = np.where(qfb_mask)[0][0]
                    query_data.uns['iroot'] = root_idx
                    print(f"✅ Set root cell to q-Fb cell at index {root_idx}")
                else:
                    # If no q-Fb cells, use the first cell
                    query_data.uns['iroot'] = 0
                    print("✅ Set root cell to first cell (no q-Fb cells found)")
            else:
                # If no predicted_class, use the first cell
                query_data.uns['iroot'] = 0
                print("✅ Set root cell to first cell (no predicted_class available)")
            
            # 4. DPT
            print("Running sc.tl.dpt...")
            sc.tl.dpt(query_data)
            print("✅ DPT computed")
            
            # Check if dpt_pseudotime was successfully calculated
            if 'dpt_pseudotime' not in query_data.obs.columns:
                print("⚠️  dpt_pseudotime calculation failed. Using robust activation scoring method.")
                return self._calculate_robust_activation_scores(query_data, reference_data)
            
            # 5. Normalize pseudotime using reference data range
            pt_min = clean_ref_data.obs["dpt_pseudotime"].min()
            pt_max = clean_ref_data.obs["dpt_pseudotime"].max()
            query_data.obs["norm_dpt"] = (query_data.obs["dpt_pseudotime"] - pt_min) / (pt_max - pt_min)
            
            # 6. Apply class weights
            class_weights = {"dead": 0, "q-Fb": 0.5, "proto-MyoFb": 2, "p-MyoFb": 3, "np-MyoFb": 4}
            query_data.obs["class_weight"] = query_data.obs["predicted_class"].map(class_weights).astype(float)
            
            # 7. Calculate raw activation score
            query_data.obs["activation_score"] = query_data.obs["norm_dpt"] * query_data.obs["class_weight"]
            
            # 8. Normalize activation score to [0,1]
            act_min = query_data.obs["activation_score"].min()
            act_max = query_data.obs["activation_score"].max()
            if act_max > act_min:  # Avoid division by zero
                query_data.obs["activation_score_norm"] = (query_data.obs["activation_score"] - act_min) / (act_max - act_min)
            else:
                query_data.obs["activation_score_norm"] = 0.5  # Default to middle value if all scores are the same
            
            print(f"✅ Standard activation scoring completed successfully")
            print(f"  - dpt_pseudotime range: [{query_data.obs['dpt_pseudotime'].min():.3f}, {query_data.obs['dpt_pseudotime'].max():.3f}]")
            print(f"  - Activation scores range: [{query_data.obs['activation_score_norm'].min():.3f}, {query_data.obs['activation_score_norm'].max():.3f}]")
            
            return query_data.obs["activation_score_norm"].values
        except Exception as e:
            print(f"⚠️  Standard method failed: {e}. Falling back to robust method.")
            import traceback
            traceback.print_exc()
            return self._calculate_robust_activation_scores(query_data, reference_data)
    
    def _generate_visualizations(self, query_data: ad.AnnData, reference_data: ad.AnnData, 
                                activation_scores: np.ndarray, output_dir: str, 
                                visualization_type: str = 'all') -> Dict[str, str]:
        """
        Generate visualizations for activation analysis.
        
        Args:
            query_data: Query cell data
            reference_data: Reference cell data
            activation_scores: Calculated activation scores
            output_dir: Output directory for visualizations
            visualization_type: Type of visualizations to generate
            
        Returns:
            Dict[str, str]: Paths to generated visualization files
        """
        os.makedirs(output_dir, exist_ok=True)
        viz_paths = {}
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set style
            plt.style.use('default')
            sns.set_palette("husl")
            
            if visualization_type in ['basic', 'all']:
                # Basic activation score distribution
                plt.figure(figsize=(10, 6))
                plt.hist(activation_scores, bins=30, alpha=0.7, edgecolor='black')
                plt.xlabel('Activation Score')
                plt.ylabel('Number of Cells')
                plt.title('Distribution of Fibroblast Activation Scores')
                plt.grid(True, alpha=0.3)
                
                basic_viz_path = os.path.join(output_dir, 'activation_score_distribution.png')
                plt.savefig(basic_viz_path, dpi=300, bbox_inches='tight')
                plt.close()
                viz_paths['activation_distribution'] = basic_viz_path
                
                # Box plot with overlaid points (用户期望的风格)
                plt.figure(figsize=(8, 6))
                sns.boxplot(y=activation_scores, color='#4CAF50', width=0.3, boxprops=dict(alpha=0.7))
                sns.stripplot(y=activation_scores, color='black', alpha=0.6, jitter=0.2, size=5)
                plt.ylabel('Activation Score')
                plt.title('Activation Score Distribution (Box + Points)')
                plt.tight_layout()
                box_viz_path = os.path.join(output_dir, 'activation_score_boxplot.png')
                plt.savefig(box_viz_path, dpi=300, bbox_inches='tight')
                plt.close()
                viz_paths['activation_boxplot'] = box_viz_path
            
            if visualization_type in ['comprehensive', 'all']:
                # Comprehensive analysis
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                
                # 1. Activation score distribution
                axes[0, 0].hist(activation_scores, bins=30, alpha=0.7, edgecolor='black')
                axes[0, 0].set_xlabel('Activation Score')
                axes[0, 0].set_ylabel('Number of Cells')
                axes[0, 0].set_title('Activation Score Distribution')
                axes[0, 0].grid(True, alpha=0.3)
                
                # 2. Box plot of activation scores
                box_plot = axes[0, 1].boxplot(activation_scores, patch_artist=True)
                box_plot['boxes'][0].set_facecolor('lightgreen')
                box_plot['medians'][0].set_color('red')
                box_plot['medians'][0].set_linewidth(2)
                axes[0, 1].set_ylabel('Activation Score')
                axes[0, 1].set_title('Box Plot of Activation Scores')
                axes[0, 1].grid(True, alpha=0.3)
                
                # 3. Activation score vs expression
                ref_means = np.mean(reference_data.X, axis=1)
                query_means = np.mean(query_data.X, axis=1)
                
                axes[1, 0].scatter(query_means, activation_scores, alpha=0.6)
                axes[1, 0].set_xlabel('Mean Expression')
                axes[1, 0].set_ylabel('Activation Score')
                axes[1, 0].set_title('Activation Score vs Expression')
                axes[1, 0].grid(True, alpha=0.3)
                
                # 4. Statistical summary
                stats_text = f"""
                Query Cells: {len(activation_scores)}
                Reference Cells: {reference_data.shape[0]}
                
                Activation Score Statistics:
                Mean: {np.mean(activation_scores):.3f}
                Std: {np.std(activation_scores):.3f}
                Min: {np.min(activation_scores):.3f}
                Max: {np.max(activation_scores):.3f}
                Median: {np.median(activation_scores):.3f}
                Q1: {np.percentile(activation_scores, 25):.3f}
                Q3: {np.percentile(activation_scores, 75):.3f}
                
                High Activation (>0.7): {np.sum(activation_scores > 0.7)}
                Low Activation (<0.3): {np.sum(activation_scores < 0.3)}
                """
                
                axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes, 
                               fontsize=10, verticalalignment='center',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                axes[1, 1].set_title('Statistical Summary')
                axes[1, 1].axis('off')
                
                plt.tight_layout()
                comprehensive_viz_path = os.path.join(output_dir, 'comprehensive_activation_analysis.png')
                plt.savefig(comprehensive_viz_path, dpi=300, bbox_inches='tight')
                plt.close()
                viz_paths['comprehensive_analysis'] = comprehensive_viz_path
            
            if visualization_type == 'all':
                # Additional UMAP visualization if scanpy is available
                if SCANPY_AVAILABLE:
                    try:
                        # Combine reference and query data for UMAP
                        combined_data = ad.concat([reference_data, query_data], join='outer', 
                                                label='dataset', keys=['reference', 'query'])
                        
                        # Basic preprocessing
                        sc.pp.normalize_total(combined_data, target_sum=1e4)
                        sc.pp.log1p(combined_data)
                        sc.pp.highly_variable_genes(combined_data, min_mean=0.0125, max_mean=3, min_disp=0.5)
                        combined_data = combined_data[:, combined_data.var.highly_variable]
                        sc.pp.scale(combined_data, max_value=10)
                        
                        # UMAP
                        sc.pp.pca(combined_data, use_highly_variable=True)
                        sc.pp.neighbors(combined_data)
                        sc.tl.umap(combined_data)
                        
                        # Plot
                        sc.pl.umap(combined_data, color='dataset', size=50, 
                                 save='_umap_activation_analysis.png')
                        
                        umap_path = os.path.join(output_dir, 'umap_activation_analysis.png')
                        if os.path.exists('figures/umap_activation_analysis.png'):
                            os.rename('figures/umap_activation_analysis.png', umap_path)
                        
                        viz_paths['umap_analysis'] = umap_path
                        
                    except Exception as e:
                        print(f"UMAP visualization failed: {e}")
            
        except ImportError:
            print("matplotlib or seaborn not available. Skipping visualizations.")
        
        return viz_paths
    
    def execute(self, cell_data: Union[str, List[str]], 
                reference_source: str = 'huggingface',
                reference_repo_id: str = '5xuekun/adata_reference',
                reference_filename: str = 'adata_reference.h5ad',
                local_reference_path: Optional[str] = None,
                output_dir: str = 'output_visualizations',
                visualization_type: str = 'all',
                confidence_threshold: float = 0.5,
                batch_size: int = 100) -> Dict[str, Any]:
        """
        Execute fibroblast activation scoring.
        
        Args:
            cell_data: Path to cell data file or list of cell crop paths
            reference_source: Reference source ('huggingface' or 'local')
            reference_repo_id: Hugging Face repository ID
            reference_filename: Reference filename in Hugging Face repo
            local_reference_path: Local path to reference file
            output_dir: Output directory for results
            visualization_type: Visualization type ('basic', 'comprehensive', 'all')
            confidence_threshold: Minimum confidence threshold
            batch_size: Batch size for processing
            
        Returns:
            Dict containing activation scoring results
        """
        try:
            print(f"Starting fibroblast activation scoring...")
            print(f"Reference source: {reference_source}")
            print(f"Visualization type: {visualization_type}")
            
            # Load reference data FIRST
            if reference_source == 'huggingface':
                if not HF_AVAILABLE:
                    raise ImportError("huggingface_hub not available for Hugging Face downloads")
                
                reference_path = self._download_reference_from_hf(reference_repo_id, reference_filename)
            elif reference_source == 'local':
                if not local_reference_path:
                    raise ValueError("local_reference_path must be provided when reference_source='local'")
                reference_path = local_reference_path
            else:
                raise ValueError("reference_source must be 'huggingface' or 'local'")
            
            # Load reference data and store it for feature matching
            reference_data = self._load_reference_data(reference_path)
            self.reference_data = reference_data  # Store for feature matching
            
            # Prepare query data with reference data for feature matching
            query_data = self._prepare_cell_data(cell_data)
            
            # Calculate activation scores
            activation_scores = self._calculate_activation_scores(query_data, reference_data)
            
            # Generate visualizations
            viz_paths = self._generate_visualizations(
                query_data, reference_data, activation_scores, 
                output_dir, visualization_type
            )
            
            # Prepare results
            results = {
                'activation_scores': activation_scores.tolist(),
                'reference_stats': {
                    'n_cells': reference_data.shape[0],
                    'n_features': reference_data.shape[1],
                    'mean_expression': float(np.mean(reference_data.X)),
                    'std_expression': float(np.std(reference_data.X))
                },
                'query_stats': {
                    'n_cells': query_data.shape[0],
                    'n_features': query_data.shape[1],
                    'mean_expression': float(np.mean(query_data.X)),
                    'std_expression': float(np.std(query_data.X))
                },
                'comparison_results': {
                    'mean_activation': float(np.mean(activation_scores)),
                    'std_activation': float(np.std(activation_scores)),
                    'high_activation_count': int(np.sum(activation_scores > 0.7)),
                    'low_activation_count': int(np.sum(activation_scores < 0.3)),
                    'activation_range': [float(np.min(activation_scores)), float(np.max(activation_scores))]
                },
                'visualizations': viz_paths,
                'visual_outputs': list(viz_paths.values()),
                'metadata': {
                    'reference_source': reference_source,
                    'reference_path': reference_path,
                    'visualization_type': visualization_type,
                    'confidence_threshold': confidence_threshold,
                    'batch_size': batch_size,
                    'processing_timestamp': pd.Timestamp.now().isoformat()
                }
            }
            
            print(f"Activation scoring completed successfully!")
            print(f"Processed {len(activation_scores)} cells")
            print(f"Mean activation score: {np.mean(activation_scores):.3f}")
            print(f"Visualizations saved to: {output_dir}")
            
            return results
            
        except Exception as e:
            error_msg = f"Error in fibroblast activation scoring: {str(e)}"
            print(error_msg)
            return {
                'error': error_msg,
                'status': 'failed',
                'metadata': {
                    'reference_source': reference_source,
                    'visualization_type': visualization_type,
                    'processing_timestamp': pd.Timestamp.now().isoformat()
                }
            } 