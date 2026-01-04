#!/usr/bin/env python3
"""
Analysis Visualizer Tool - Creates statistical visualizations for multi-group comparison analysis.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import json
from collections import Counter, defaultdict

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
sys.path.insert(0, project_root)

from octotools.tools.base import BaseTool
from octotools.models.utils import VisualizationConfig

# Try to import anndata and scanpy for cell state analysis visualization
try:
    import anndata as ad
    import scanpy as sc
    ANNDATA_AVAILABLE = True
except ImportError:
    ANNDATA_AVAILABLE = False


class Analysis_Visualizer_Tool(BaseTool):
    """
    Creates statistical visualizations for pipeline analysis results, especially for multi-group comparisons.
    Supports various chart types (bar charts, pie charts, etc.) with automatic statistical testing.
    """
    
    def __init__(self):
        super().__init__(
            tool_name="Analysis_Visualizer_Tool",
            tool_description="Creates statistical visualizations for analysis results with multi-group comparisons. Supports bar charts, pie charts, and other visualization types with automatic statistical testing (t-test for 2 groups, ANOVA for 3+ groups).",
            tool_version="1.0.0",
            input_types={
                "analysis_data": "dict or str - Analysis results data (dict) or path to JSON file containing analysis results.",
                "chart_type": "str - Type of chart to create ('auto', 'bar', 'pie', 'box', 'violin', 'scatter', 'line'). Default: 'auto'.",
                "comparison_metric": "str - Metric to compare across groups (e.g., 'cell_count', 'cell_type_distribution', 'confidence_mean'). Default: 'cell_count'.",
                "group_column": "str - Column name in data that contains group labels. Default: 'group'.",
                "output_dir": "str - Directory to save visualization outputs. Default: 'output_visualizations'.",
                "figure_size": "tuple - Figure size (width, height) in inches. Default: (10, 6).",
                "dpi": "int - Resolution for saved figures. Default: 300."
            },
            output_type="dict - Visualization results with chart paths and statistical test results.",
            demo_commands=[
                "Create a bar chart comparing cell counts across groups",
                "Visualize cell type distribution using a pie chart",
                "Generate statistical comparison of confidence scores between groups",
                "Create automatic visualization for multi-group analysis results"
            ],
            user_metadata={
                "limitation": "Requires analysis data with group labels. Statistical tests assume normal distribution for t-test and ANOVA.",
                "best_practice": "Use 'auto' chart type to let the tool automatically select the most appropriate visualization based on data characteristics. For cell counts, bar charts are recommended. For proportions, pie charts work well.",
                "statistical_tests": "Automatically performs t-test for 2 groups and ANOVA for 3+ groups. Results are included in the output.",
                "color_scheme": "Automatically uses tab10 colors for ≤10 groups and tab20 colors for >10 groups."
            }
        )
        
        # Color schemes
        self.tab10_colors = plt.cm.tab10.colors
        self.tab20_colors = plt.cm.tab20.colors
        
    def _get_color_scheme(self, n_groups: int) -> List[tuple]:
        """
        Get color scheme based on number of groups.
        
        Args:
            n_groups: Number of groups to visualize
            
        Returns:
            List of color tuples (RGBA)
        """
        if n_groups <= 10:
            return [self.tab10_colors[i % 10] for i in range(n_groups)]
        else:
            return [self.tab20_colors[i % 20] for i in range(n_groups)]
    
    def _perform_statistical_test(self, data_by_group: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Perform appropriate statistical test based on number of groups.
        
        Args:
            data_by_group: Dictionary mapping group names to lists of values
            
        Returns:
            Dictionary with test results
        """
        groups = list(data_by_group.keys())
        n_groups = len(groups)
        
        if n_groups < 2:
            return {
                "test_type": "none",
                "message": "Insufficient groups for statistical comparison (need at least 2 groups)"
            }
        
        # Prepare data for testing
        group_data = [data_by_group[g] for g in groups]
        
        # Remove groups with insufficient data
        valid_groups = []
        valid_data = []
        for i, (g, d) in enumerate(zip(groups, group_data)):
            if len(d) >= 2:  # Need at least 2 samples for variance
                valid_groups.append(g)
                valid_data.append(d)
        
        if len(valid_groups) < 2:
            return {
                "test_type": "none",
                "message": "Insufficient data for statistical comparison (need at least 2 groups with 2+ samples each)"
            }
        
        if len(valid_groups) == 2:
            # Two-sample t-test
            try:
                stat, p_value = stats.ttest_ind(valid_data[0], valid_data[1])
                return {
                    "test_type": "t-test (independent samples)",
                    "statistic": float(stat),
                    "p_value": float(p_value),
                    "groups": valid_groups,
                    "significant": p_value < 0.05,
                    "interpretation": f"Groups {'differ significantly' if p_value < 0.05 else 'do not differ significantly'} (p={p_value:.4f})"
                }
            except Exception as e:
                return {
                    "test_type": "t-test (failed)",
                    "error": str(e)
                }
        else:
            # One-way ANOVA
            try:
                stat, p_value = stats.f_oneway(*valid_data)
                return {
                    "test_type": "ANOVA (one-way)",
                    "statistic": float(stat),
                    "p_value": float(p_value),
                    "groups": valid_groups,
                    "n_groups": len(valid_groups),
                    "significant": p_value < 0.05,
                    "interpretation": f"Groups {'differ significantly' if p_value < 0.05 else 'do not differ significantly'} (p={p_value:.4f})"
                }
            except Exception as e:
                return {
                    "test_type": "ANOVA (failed)",
                    "error": str(e)
                }
    
    def _recommend_chart_type(self, data: pd.DataFrame, comparison_metric: str, group_column: str) -> str:
        """
        Recommend appropriate chart type based on data characteristics.
        
        Args:
            data: DataFrame with analysis data
            comparison_metric: Metric to compare
            group_column: Column containing group labels
            
        Returns:
            Recommended chart type
        """
        if comparison_metric in ['cell_count', 'count', 'total', 'number']:
            return 'bar'
        elif comparison_metric in ['distribution', 'proportion', 'percentage', 'composition']:
            return 'pie'
        elif 'confidence' in comparison_metric.lower() or 'score' in comparison_metric.lower():
            # For continuous metrics, use box or violin plot
            if len(data[group_column].unique()) <= 5:
                return 'box'
            else:
                return 'violin'
        else:
            # Default to bar chart
            return 'bar'
    
    def _create_bar_chart(self, data_by_group: Dict[str, float], colors: List[tuple], 
                          title: str, ylabel: str, output_path: str, 
                          figure_size: tuple, dpi: int, stats_result: Optional[Dict] = None) -> str:
        """Create a bar chart comparing groups."""
        fig, ax = plt.subplots(figsize=figure_size, dpi=dpi)
        
        groups = list(data_by_group.keys())
        values = [data_by_group[g] for g in groups]
        
        bars = ax.bar(groups, values, color=colors[:len(groups)], alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}' if isinstance(val, float) else f'{val}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Group', fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add statistical test result
        if stats_result and stats_result.get('test_type') != 'none':
            test_info = f"{stats_result['test_type']}: p={stats_result['p_value']:.4f}"
            if stats_result.get('significant'):
                test_info += " *"
            ax.text(0.02, 0.98, test_info, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _create_pie_chart(self, data_by_group: Dict[str, float], colors: List[tuple],
                         title: str, output_path: str, figure_size: tuple, dpi: int) -> str:
        """Create a pie chart showing group composition."""
        fig, ax = plt.subplots(figsize=figure_size, dpi=dpi)
        
        groups = list(data_by_group.keys())
        values = [data_by_group[g] for g in groups]
        total = sum(values)
        
        # Calculate percentages
        percentages = [v/total*100 for v in values]
        
        # Create pie chart with labels
        wedges, texts, autotexts = ax.pie(values, labels=groups, colors=colors[:len(groups)],
                                          autopct='%1.1f%%', startangle=90,
                                          textprops={'fontsize': 10, 'fontweight': 'bold'})
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _create_box_plot(self, data_by_group: Dict[str, List[float]], colors: List[tuple],
                        title: str, ylabel: str, output_path: str, 
                        figure_size: tuple, dpi: int, stats_result: Optional[Dict] = None) -> str:
        """Create a box plot comparing distributions across groups."""
        fig, ax = plt.subplots(figsize=figure_size, dpi=dpi)
        
        groups = list(data_by_group.keys())
        data_list = [data_by_group[g] for g in groups]
        
        bp = ax.boxplot(data_list, labels=groups, patch_artist=True,
                       showmeans=True, meanline=True)
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], colors[:len(groups)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_xlabel('Group', fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add statistical test result
        if stats_result and stats_result.get('test_type') != 'none':
            test_info = f"{stats_result['test_type']}: p={stats_result['p_value']:.4f}"
            if stats_result.get('significant'):
                test_info += " *"
            ax.text(0.02, 0.98, test_info, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _create_violin_plot(self, data_by_group: Dict[str, List[float]], colors: List[tuple],
                           title: str, ylabel: str, output_path: str,
                           figure_size: tuple, dpi: int, stats_result: Optional[Dict] = None) -> str:
        """Create a violin plot comparing distributions across groups."""
        fig, ax = plt.subplots(figsize=figure_size, dpi=dpi)
        
        groups = list(data_by_group.keys())
        data_list = [data_by_group[g] for g in groups]
        
        parts = ax.violinplot(data_list, positions=range(len(groups)), showmeans=True, showmedians=True)
        
        # Color the violins
        for pc, color in zip(parts['bodies'], colors[:len(groups)]):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups)
        ax.set_xlabel('Group', fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add statistical test result
        if stats_result and stats_result.get('test_type') != 'none':
            test_info = f"{stats_result['test_type']}: p={stats_result['p_value']:.4f}"
            if stats_result.get('significant'):
                test_info += " *"
            ax.text(0.02, 0.98, test_info, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _prepare_data(self, analysis_data: Union[Dict, str], comparison_metric: str, 
                     group_column: str) -> tuple:
        """
        Prepare data for visualization.
        
        Returns:
            Tuple of (data_by_group dict, stats_result dict, groups list)
        """
        # Load data if it's a file path
        if isinstance(analysis_data, str):
            if os.path.exists(analysis_data):
                with open(analysis_data, 'r') as f:
                    analysis_data = json.load(f)
            else:
                raise ValueError(f"Analysis data file not found: {analysis_data}")
        
        # Convert to DataFrame for easier manipulation
        if isinstance(analysis_data, dict):
            # Check if it's a column-oriented dict (keys are column names, values are lists)
            is_column_oriented = (
                len(analysis_data) > 0 and
                all(isinstance(v, list) for v in analysis_data.values() if v is not None) and
                all(len(v) == len(list(analysis_data.values())[0]) for v in analysis_data.values() if isinstance(v, list) and v)
            )
            
            if is_column_oriented:
                # Column-oriented dict: convert to row-oriented for DataFrame
                # Ensure all lists have the same length
                max_len = max(len(v) if isinstance(v, list) else 1 for v in analysis_data.values())
                records = []
                for i in range(max_len):
                    record = {}
                    for key, value in analysis_data.items():
                        if isinstance(value, list):
                            record[key] = value[i] if i < len(value) else None
                        else:
                            record[key] = value
                    records.append(record)
                df = pd.DataFrame(records)
            elif 'per_image' in analysis_data:
                # Multi-image results
                records = []
                for img_result in analysis_data['per_image']:
                    group = img_result.get(group_column, img_result.get('group', 'default'))
                    if comparison_metric in img_result:
                        records.append({
                            group_column: group,
                            comparison_metric: img_result[comparison_metric]
                        })
                    elif 'statistics' in img_result and comparison_metric in img_result['statistics']:
                        records.append({
                            group_column: group,
                            comparison_metric: img_result['statistics'][comparison_metric]
                        })
                df = pd.DataFrame(records)
            elif 'results' in analysis_data:
                # Results list
                records = []
                for result in analysis_data['results']:
                    group = result.get(group_column, result.get('group', 'default'))
                    if comparison_metric in result:
                        records.append({
                            group_column: group,
                            comparison_metric: result[comparison_metric]
                        })
                df = pd.DataFrame(records)
            elif 'cell_metadata' in analysis_data and 'cell_crop_objects' not in analysis_data:
                # Cell metadata structure (only if not already handled by cell_crop_objects)
                records = []
                for metadata in analysis_data['cell_metadata']:
                    group = metadata.get(group_column, metadata.get('group', 'default'))
                    if comparison_metric in metadata:
                        records.append({
                            group_column: group,
                            comparison_metric: metadata[comparison_metric]
                        })
                    elif comparison_metric == 'cell_count':
                        records.append({
                            group_column: group
                        })
                df = pd.DataFrame(records)
            elif 'analysis_type' in analysis_data and analysis_data['analysis_type'] == 'cell_state_analysis':
                # Cell_State_Analyzer_Tool output - load from AnnData file
                if 'adata_path' in analysis_data and os.path.exists(analysis_data['adata_path']):
                    # Load AnnData and extract cluster/group information
                    if not ANNDATA_AVAILABLE:
                        raise ValueError("anndata/scanpy not available. Cannot process cell state analysis results.")
                    
                    adata = ad.read_h5ad(analysis_data['adata_path'])
                    cluster_key = analysis_data.get('cluster_key', 'leiden_0.5')
                    
                    # Extract data for visualization
                    records = []
                    for i in range(adata.n_obs):
                        record = {
                            group_column: adata.obs[group_column].iloc[i] if group_column in adata.obs else 'default',
                            'cluster': adata.obs[cluster_key].iloc[i] if cluster_key in adata.obs else None,
                            'umap_1': adata.obsm['X_umap'][i, 0] if 'X_umap' in adata.obsm else None,
                            'umap_2': adata.obsm['X_umap'][i, 1] if 'X_umap' in adata.obsm else None,
                        }
                        # Add comparison metric if available
                        if comparison_metric in adata.obs:
                            record[comparison_metric] = adata.obs[comparison_metric].iloc[i]
                        records.append(record)
                    
                    df = pd.DataFrame(records)
                    
                    # Store AnnData info for later use in visualization
                    analysis_data['_adata'] = adata
                    analysis_data['_cluster_key'] = cluster_key
                else:
                    raise ValueError(f"AnnData file not found: {analysis_data.get('adata_path', 'Not specified')}")
            elif 'cell_crop_objects' in analysis_data:
                # Single cell cropper output format
                # cell_crop_objects is a list of CellCrop objects (serialized as dicts)
                records = []
                for obj in analysis_data['cell_crop_objects']:
                    group = obj.get(group_column, obj.get('group', 'default'))
                    # Try to get comparison_metric from the object
                    if comparison_metric in obj:
                        records.append({
                            group_column: group,
                            comparison_metric: obj[comparison_metric]
                        })
                    # For cell_count metric, just record group (will count later)
                    elif comparison_metric == 'cell_count':
                        records.append({
                            group_column: group
                        })
                    # For other metrics like 'area', try to find in object
                    else:
                        # Check if metric exists in object (e.g., 'area' from CellCrop)
                        if comparison_metric in obj:
                            records.append({
                                group_column: group,
                                comparison_metric: obj[comparison_metric]
                            })
                df = pd.DataFrame(records) if records else pd.DataFrame()
                
                # If no records were created and we have cell_metadata, try that instead
                if df.empty and 'cell_metadata' in analysis_data:
                    records = []
                    for metadata in analysis_data['cell_metadata']:
                        group = metadata.get(group_column, metadata.get('group', 'default'))
                        if comparison_metric in metadata:
                            records.append({
                                group_column: group,
                                comparison_metric: metadata[comparison_metric]
                            })
                        elif comparison_metric == 'cell_count':
                            records.append({
                                group_column: group
                            })
                    df = pd.DataFrame(records) if records else pd.DataFrame()
            else:
                # Try to convert directly to DataFrame
                try:
                    df = pd.DataFrame([analysis_data])
                except:
                    # If that fails, try to extract summary statistics
                    if 'statistics' in analysis_data:
                        stats_dict = analysis_data['statistics']
                        if isinstance(stats_dict, dict):
                            # Create a single-row DataFrame
                            group = analysis_data.get(group_column, analysis_data.get('group', 'default'))
                            df = pd.DataFrame([{group_column: group, **stats_dict}])
                        else:
                            raise ValueError("Unable to parse analysis data structure")
                    else:
                        raise ValueError("Unable to parse analysis data structure")
        # Handle list case if df wasn't created above
        if 'df' not in locals() or ('df' in locals() and df is None):
            if isinstance(analysis_data, list):
                # List of records (row-oriented)
                if len(analysis_data) == 0:
                    raise ValueError("Empty data list provided")
                # Check if it's a list of dicts
                if all(isinstance(item, dict) for item in analysis_data):
                    # Check if this is a list of cell crop objects
                    first_item = analysis_data[0]
                    if 'group' in first_item or 'crop_path' in first_item:
                        # List of cell crop objects from single cell cropper
                        records = []
                        for obj in analysis_data:
                            group = obj.get(group_column, obj.get('group', 'default'))
                            if comparison_metric in obj:
                                records.append({
                                    group_column: group,
                                    comparison_metric: obj[comparison_metric]
                                })
                            elif comparison_metric == 'cell_count':
                                records.append({
                                    group_column: group
                                })
                        df = pd.DataFrame(records)
                    else:
                        df = pd.DataFrame(analysis_data)
                else:
                    raise ValueError(f"List must contain dictionaries, got {type(analysis_data[0])}")
            else:
                raise ValueError(f"Unsupported data type: {type(analysis_data)}")
        
        if df.empty:
            raise ValueError("No data found for visualization")
        
        if group_column not in df.columns:
            raise ValueError(f"Group column '{group_column}' not found in data. Available columns: {list(df.columns)}")
        
        # Handle cell_count metric specially (count rows per group)
        if comparison_metric == 'cell_count' and comparison_metric not in df.columns:
            # Count cells per group
            df = df.groupby(group_column).size().reset_index(name=comparison_metric)
        elif comparison_metric not in df.columns:
            raise ValueError(f"Comparison metric '{comparison_metric}' not found in data. Available columns: {list(df.columns)}")
        
        # Convert any list/array values in group_column to strings to avoid unhashable type errors
        if df[group_column].dtype == 'object':
            has_list_values = df[group_column].apply(lambda x: isinstance(x, (list, np.ndarray))).any()
            if has_list_values:
                df[group_column] = df[group_column].apply(lambda x: str(x) if isinstance(x, (list, np.ndarray)) else x)
        
        # Group data
        groups = df[group_column].unique().tolist()
        data_by_group = {}
        
        # Check if metric is continuous (list/array) or scalar
        first_value = df[comparison_metric].iloc[0]
        is_continuous = isinstance(first_value, (list, np.ndarray)) or (
            isinstance(first_value, (int, float)) and len(df[df[group_column] == groups[0]]) > 1
        )
        
        if is_continuous:
            # For continuous data, collect all values per group
            for group in groups:
                group_data = df[df[group_column] == group][comparison_metric]
                # Flatten if it's a list of lists
                values = []
                for val in group_data:
                    if isinstance(val, (list, np.ndarray)):
                        values.extend(list(val))
                    else:
                        values.append(val)
                data_by_group[group] = values
        else:
            # For scalar data, aggregate (sum or mean)
            for group in groups:
                group_data = df[df[group_column] == group][comparison_metric]
                # If all values are the same or it's a count, use sum
                if group_data.nunique() == 1 or 'count' in comparison_metric.lower():
                    data_by_group[group] = group_data.sum()
                else:
                    data_by_group[group] = group_data.mean()
        
        # Perform statistical test if we have continuous data
        stats_result = None
        if is_continuous:
            stats_result = self._perform_statistical_test(data_by_group)
        elif len(groups) > 1:
            # For scalar data, convert to lists for statistical testing
            scalar_data_by_group = {}
            for group in groups:
                group_data = df[df[group_column] == group][comparison_metric]
                scalar_data_by_group[group] = group_data.tolist()
            stats_result = self._perform_statistical_test(scalar_data_by_group)
        
        return data_by_group, stats_result, groups
    
    def _visualize_cell_state_analysis(self, analysis_data: Dict[str, Any], 
                                       output_dir: str, figure_size: tuple, dpi: int) -> Dict[str, Any]:
        """
        Generate publication-quality visualizations for cell state analysis results.
        
        Args:
            analysis_data: Cell_State_Analyzer_Tool output dictionary
            output_dir: Directory to save outputs
            figure_size: Figure size (width, height) in inches
            dpi: Resolution for saved figures
            
        Returns:
            Dictionary with visualization results
        """
        if not ANNDATA_AVAILABLE:
            return {
                "error": "anndata/scanpy not available. Cannot generate cell state visualizations.",
                "summary": "Visualization failed"
            }
        
        # Load AnnData
        adata_path = analysis_data.get('adata_path')
        if not adata_path or not os.path.exists(adata_path):
            return {
                "error": f"AnnData file not found: {adata_path}",
                "summary": "Visualization failed"
            }
        
        adata = ad.read_h5ad(adata_path)
        cluster_key = analysis_data.get('cluster_key', 'leiden_0.5')
        cluster_resolution = analysis_data.get('cluster_resolution', 0.5)
        
        visual_outputs = []
        
        # 1. Create publication-quality UMAP colored by cluster
        umap_cluster_path = self._create_publication_umap_by_cluster(
            adata, cluster_key, cluster_resolution, output_dir, figure_size, dpi
        )
        if umap_cluster_path:
            visual_outputs.append(umap_cluster_path)
        
        # 2. Create UMAP colored by group (if multiple groups exist)
        if 'group' in adata.obs and adata.obs['group'].nunique() > 1:
            umap_group_path = self._create_publication_umap_by_group(
                adata, output_dir, figure_size, dpi
            )
            if umap_group_path:
                visual_outputs.append(umap_group_path)
        
        # 3. Create cluster composition plot by group (if multiple groups)
        if 'group' in adata.obs and adata.obs['group'].nunique() > 1:
            composition_path = self._create_publication_cluster_composition(
                adata, cluster_key, output_dir, figure_size, dpi
            )
            if composition_path:
                visual_outputs.append(composition_path)
        
        # 4. Create cluster size bar chart
        cluster_size_path = self._create_cluster_size_chart(
            adata, cluster_key, output_dir, figure_size, dpi
        )
        if cluster_size_path:
            visual_outputs.append(cluster_size_path)
        
        return {
            "summary": f"Generated {len(visual_outputs)} publication-quality visualizations for cell state analysis",
            "visual_outputs": visual_outputs,
            "cell_count": adata.n_obs,
            "num_clusters": adata.obs[cluster_key].nunique() if cluster_key in adata.obs else 0,
            "cluster_key": cluster_key
        }
    
    def _create_publication_umap_by_cluster(self, adata, cluster_key: str, resolution: float,
                                           output_dir: str, figure_size: tuple, dpi: int) -> Optional[str]:
        """Create publication-quality UMAP visualization colored by cluster."""
        if 'X_umap' not in adata.obsm or cluster_key not in adata.obs:
            return None
        
        # Create figure with professional styling
        fig, ax = VisualizationConfig.create_professional_figure(figsize=(10, 8))
        
        # Get unique clusters and assign colors
        clusters = sorted(adata.obs[cluster_key].unique())
        n_clusters = len(clusters)
        colors = self._get_color_scheme(n_clusters)
        cluster_colors = {cluster: colors[i] for i, cluster in enumerate(clusters)}
        
        # Plot each cluster
        for cluster in clusters:
            cluster_mask = adata.obs[cluster_key] == cluster
            umap_coords = adata.obsm['X_umap'][cluster_mask]
            ax.scatter(umap_coords[:, 0], umap_coords[:, 1], 
                      c=[cluster_colors[cluster]], label=f'Cluster {cluster}',
                      s=50, alpha=0.6, edgecolors='white', linewidths=0.5)
        
        ax.set_xlabel('UMAP 1', fontsize=16, fontweight='bold')
        ax.set_ylabel('UMAP 2', fontsize=16, fontweight='bold')
        ax.set_title(f'UMAP Visualization - Leiden Clustering (resolution={resolution})', 
                    fontsize=18, fontweight='bold', pad=20)
        
        # Professional legend
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                 frameon=True, fancybox=True, shadow=True, fontsize=12)
        
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        output_path = os.path.join(output_dir, f"umap_cluster_res{resolution}.png")
        VisualizationConfig.save_professional_figure(fig, output_path)
        plt.close(fig)
        
        return output_path
    
    def _create_publication_umap_by_group(self, adata, output_dir: str, 
                                         figure_size: tuple, dpi: int) -> Optional[str]:
        """Create publication-quality UMAP visualization colored by group."""
        if 'X_umap' not in adata.obsm or 'group' not in adata.obs:
            return None
        
        fig, ax = VisualizationConfig.create_professional_figure(figsize=(10, 8))
        
        groups = sorted(adata.obs['group'].unique())
        n_groups = len(groups)
        colors = self._get_color_scheme(n_groups)
        group_colors = {group: colors[i] for i, group in enumerate(groups)}
        
        for group in groups:
            group_mask = adata.obs['group'] == group
            umap_coords = adata.obsm['X_umap'][group_mask]
            ax.scatter(umap_coords[:, 0], umap_coords[:, 1],
                      c=[group_colors[group]], label=str(group),
                      s=50, alpha=0.6, edgecolors='white', linewidths=0.5)
        
        ax.set_xlabel('UMAP 1', fontsize=16, fontweight='bold')
        ax.set_ylabel('UMAP 2', fontsize=16, fontweight='bold')
        ax.set_title('UMAP Visualization - Colored by Group',
                    fontsize=18, fontweight='bold', pad=20)
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                 frameon=True, fancybox=True, shadow=True, fontsize=12)
        
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        output_path = os.path.join(output_dir, "umap_by_group.png")
        VisualizationConfig.save_professional_figure(fig, output_path)
        plt.close(fig)
        
        return output_path
    
    def _create_publication_cluster_composition(self, adata, cluster_key: str,
                                               output_dir: str, figure_size: tuple, dpi: int) -> Optional[str]:
        """Create publication-quality cluster composition plot by group."""
        if 'group' not in adata.obs or cluster_key not in adata.obs:
            return None
        
        # Calculate composition
        composition = pd.crosstab(
            adata.obs['group'], 
            adata.obs[cluster_key],
            normalize='index'
        )
        
        fig, ax = VisualizationConfig.create_professional_figure(figsize=(12, 8))
        
        # Stacked bar chart
        composition.plot(kind='bar', stacked=True, ax=ax, 
                        colormap='tab20', width=0.8, edgecolor='white', linewidth=1.5)
        
        ax.set_xlabel('Group', fontsize=16, fontweight='bold')
        ax.set_ylabel('Proportion', fontsize=16, fontweight='bold')
        ax.set_title('Cluster Composition by Group', fontsize=18, fontweight='bold', pad=20)
        
        ax.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left',
                 frameon=True, fancybox=True, shadow=True, fontsize=11)
        
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        output_path = os.path.join(output_dir, "cluster_composition_by_group.png")
        VisualizationConfig.save_professional_figure(fig, output_path)
        plt.close(fig)
        
        return output_path
    
    def _create_cluster_size_chart(self, adata, cluster_key: str,
                                  output_dir: str, figure_size: tuple, dpi: int) -> Optional[str]:
        """Create cluster size distribution bar chart."""
        if cluster_key not in adata.obs:
            return None
        
        cluster_counts = adata.obs[cluster_key].value_counts().sort_index()
        
        fig, ax = VisualizationConfig.create_professional_figure(figsize=(10, 6))
        
        colors = self._get_color_scheme(len(cluster_counts))
        bars = ax.bar(range(len(cluster_counts)), cluster_counts.values, 
                     color=colors[:len(cluster_counts)], edgecolor='white', linewidth=1.5)
        
        ax.set_xlabel('Cluster', fontsize=16, fontweight='bold')
        ax.set_ylabel('Number of Cells', fontsize=16, fontweight='bold')
        ax.set_title('Cluster Size Distribution', fontsize=18, fontweight='bold', pad=20)
        
        ax.set_xticks(range(len(cluster_counts)))
        ax.set_xticklabels([f'Cluster {c}' for c in cluster_counts.index], rotation=45, ha='right')
        
        # Add value labels on bars
        for i, (bar, count) in enumerate(zip(bars, cluster_counts.values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(count)}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        output_path = os.path.join(output_dir, "cluster_size_distribution.png")
        VisualizationConfig.save_professional_figure(fig, output_path)
        plt.close(fig)
        
        return output_path
    
    def execute(self, analysis_data: Union[Dict, str], 
                chart_type: str = 'auto',
                comparison_metric: str = 'cell_count',
                group_column: str = 'group',
                output_dir: str = 'output_visualizations',
                figure_size: tuple = (10, 6),
                dpi: int = 300) -> Dict[str, Any]:
        """
        Execute visualization generation.
        
        Args:
            analysis_data: Analysis results data (dict) or path to JSON file
            chart_type: Type of chart ('auto', 'bar', 'pie', 'box', 'violin')
            comparison_metric: Metric to compare across groups
            group_column: Column name containing group labels
            output_dir: Directory to save outputs
            figure_size: Figure size (width, height) in inches
            dpi: Resolution for saved figures
            
        Returns:
            Dictionary with visualization results
        """
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Check if this is cell state analysis output (from Cell_State_Analyzer_Tool)
            is_cell_state_analysis = (
                isinstance(analysis_data, dict) and 
                analysis_data.get('analysis_type') == 'cell_state_analysis' and
                'adata_path' in analysis_data
            )
            
            # For cell state analysis, generate publication-quality UMAP and cluster composition plots
            if is_cell_state_analysis:
                return self._visualize_cell_state_analysis(
                    analysis_data, output_dir, figure_size, dpi
                )
            
            # Prepare data for regular visualizations
            data_by_group, stats_result, groups = self._prepare_data(
                analysis_data, comparison_metric, group_column
            )
            
            n_groups = len(groups)
            colors = self._get_color_scheme(n_groups)
            
            # Determine chart type
            if chart_type == 'auto':
                # Try to infer from data structure
                first_value = list(data_by_group.values())[0]
                if isinstance(first_value, list):
                    # Continuous data - use box or violin
                    chart_type = 'box' if n_groups <= 5 else 'violin'
                else:
                    # Scalar data - recommend based on metric name
                    if 'count' in comparison_metric.lower():
                        chart_type = 'bar'
                    elif 'distribution' in comparison_metric.lower() or 'proportion' in comparison_metric.lower():
                        chart_type = 'pie'
                    else:
                        chart_type = 'bar'
            
            # Generate title and labels
            title = f"{comparison_metric.replace('_', ' ').title()} Comparison Across Groups"
            ylabel = comparison_metric.replace('_', ' ').title()
            
            # Generate output filename
            safe_metric = comparison_metric.replace(' ', '_').replace('/', '_')
            output_filename = f"{chart_type}_{safe_metric}_comparison.png"
            output_path = os.path.join(output_dir, output_filename)
            
            # Create visualization
            visual_outputs = []
            
            if chart_type == 'bar':
                if isinstance(list(data_by_group.values())[0], list):
                    # Convert lists to means for bar chart
                    bar_data = {k: np.mean(v) if isinstance(v, list) else v 
                               for k, v in data_by_group.items()}
                else:
                    bar_data = data_by_group
                chart_path = self._create_bar_chart(
                    bar_data, colors, title, ylabel, output_path, figure_size, dpi, stats_result
                )
                visual_outputs.append(chart_path)
                
            elif chart_type == 'pie':
                if isinstance(list(data_by_group.values())[0], list):
                    # Convert lists to sums for pie chart
                    pie_data = {k: np.sum(v) if isinstance(v, list) else v 
                               for k, v in data_by_group.items()}
                else:
                    pie_data = data_by_group
                chart_path = self._create_pie_chart(
                    pie_data, colors, title, output_path, figure_size, dpi
                )
                visual_outputs.append(chart_path)
                
            elif chart_type == 'box':
                if not isinstance(list(data_by_group.values())[0], list):
                    # Convert scalars to lists
                    box_data = {k: [v] if not isinstance(v, list) else v 
                               for k, v in data_by_group.items()}
                else:
                    box_data = data_by_group
                chart_path = self._create_box_plot(
                    box_data, colors, title, ylabel, output_path, figure_size, dpi, stats_result
                )
                visual_outputs.append(chart_path)
                
            elif chart_type == 'violin':
                if not isinstance(list(data_by_group.values())[0], list):
                    # Convert scalars to lists
                    violin_data = {k: [v] if not isinstance(v, list) else v 
                                  for k, v in data_by_group.items()}
                else:
                    violin_data = data_by_group
                chart_path = self._create_violin_plot(
                    violin_data, colors, title, ylabel, output_path, figure_size, dpi, stats_result
                )
                visual_outputs.append(chart_path)
                
            else:
                raise ValueError(f"Unsupported chart type: {chart_type}")
            
            # Prepare result
            result = {
                "summary": f"Successfully created {chart_type} chart comparing {comparison_metric} across {n_groups} groups",
                "chart_type": chart_type,
                "comparison_metric": comparison_metric,
                "groups": groups,
                "n_groups": n_groups,
                "visual_outputs": visual_outputs,
                "output_path": output_path,
                "statistical_test": stats_result if stats_result else {"message": "No statistical test performed"},
                "data_summary": {
                    group: {
                        "mean": float(np.mean(v)) if isinstance(v, list) else float(v),
                        "std": float(np.std(v)) if isinstance(v, list) else None,
                        "n": len(v) if isinstance(v, list) else 1
                    } for group, v in data_by_group.items()
                }
            }
            
            return result
            
        except Exception as e:
            import traceback
            error_msg = f"Error creating visualization: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return {
                "error": error_msg,
                "summary": "Visualization generation failed"
            }

