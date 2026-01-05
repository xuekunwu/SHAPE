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
import random
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import cv2

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
                "color_scheme": "Automatically uses colorblindness-adjusted professional palettes (vega_10_scanpy, vega_20_scanpy, default_26, default_64) based on number of groups."
            }
        )
        
    def _get_color_scheme(self, n_groups: int) -> List[str]:
        """
        Get professional color scheme based on number of groups.
        Uses colorblindness-adjusted palettes from VisualizationConfig.
        
        Args:
            n_groups: Number of groups to visualize
            
        Returns:
            List of hex color strings
        """
        return VisualizationConfig.get_color_palette(n_groups)
    
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
    
    def _create_bar_chart(self, data_by_group: Dict[str, float], colors: List[str],
                         title: str, ylabel: str, output_path: str, 
                         figure_size: tuple, dpi: int, stats_result: Optional[Dict] = None) -> str:
        """Create a bar chart comparing groups."""
        fig, ax = plt.subplots(figsize=figure_size, dpi=dpi)

        groups = list(data_by_group.keys())
        values = [data_by_group[g] for g in groups]

        bars = ax.bar(range(len(groups)), values, color=colors[:len(groups)], 
                     width=0.6, alpha=0.7, edgecolor='white', linewidth=1.0)
        
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}' if isinstance(val, float) else f'{val}',
                   ha='center', va='bottom', fontsize=11, fontweight='normal')
        
        ax.set_xlabel('Group', fontsize=14, fontweight='normal')
        ax.set_ylabel(ylabel, fontsize=14, fontweight='normal')
        ax.set_title(title, fontsize=16, fontweight='normal', pad=15)
        ax.grid(axis='y', alpha=0.2, linestyle='-', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1)
        ax.spines['bottom'].set_linewidth(1)
        ax.tick_params(labelsize=11)
        
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
    
    def _create_pie_chart(self, data_by_group: Dict[str, float], colors: List[str],
                         title: str, output_path: str, figure_size: tuple, dpi: int) -> str:
        """Create a pie chart showing group composition."""
        fig, ax = VisualizationConfig.create_professional_figure(figsize=figure_size)
        
        groups = list(data_by_group.keys())
        values = [data_by_group[g] for g in groups]
        total = sum(values)
        
        # Calculate percentages
        percentages = [v/total*100 for v in values]
        
        # Create pie chart with labels
        wedges, texts, autotexts = ax.pie(values, labels=groups, colors=colors[:len(groups)],
                                          autopct='%1.1f%%', startangle=90,
                                          textprops={'fontsize': 11, 'fontweight': 'normal'})
        
        # Apply professional styling (title only for pie chart)
        VisualizationConfig.apply_professional_styling(ax, title=title)
        
        VisualizationConfig.save_professional_figure(fig, output_path, dpi=dpi)
        plt.close()
        
        return output_path
    
    def _create_box_plot(self, data_by_group: Dict[str, List[float]], colors: List[str],
                        title: str, ylabel: str, output_path: str, 
                        figure_size: tuple, dpi: int, stats_result: Optional[Dict] = None) -> str:
        """Create a box plot comparing distributions across groups."""
        fig, ax = VisualizationConfig.create_professional_figure(figsize=figure_size)
        
        groups = list(data_by_group.keys())
        data_list = [data_by_group[g] for g in groups]
        
        bp = ax.boxplot(data_list, labels=groups, patch_artist=True,
                       showmeans=True, meanline=True)
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], colors[:len(groups)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Apply professional styling
        VisualizationConfig.apply_professional_styling(ax, title=title, xlabel='Group', ylabel=ylabel)
        
        # Add statistical test result
        if stats_result and stats_result.get('test_type') != 'none':
            test_info = f"{stats_result['test_type']}: p={stats_result['p_value']:.4f}"
            if stats_result.get('significant'):
                test_info += " *"
            ax.text(0.02, 0.98, test_info, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.xticks(rotation=45, ha='right')
        VisualizationConfig.save_professional_figure(fig, output_path, dpi=dpi)
        plt.close()
        
        return output_path
    
    def _create_violin_plot(self, data_by_group: Dict[str, List[float]], colors: List[str],
                           title: str, ylabel: str, output_path: str,
                           figure_size: tuple, dpi: int, stats_result: Optional[Dict] = None) -> str:
        """Create a violin plot comparing distributions across groups."""
        fig, ax = VisualizationConfig.create_professional_figure(figsize=figure_size)
        
        groups = list(data_by_group.keys())
        data_list = [data_by_group[g] for g in groups]
        
        parts = ax.violinplot(data_list, positions=range(len(groups)), showmeans=True, showmedians=True)
        
        # Color the violins
        for pc, color in zip(parts['bodies'], colors[:len(groups)]):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups)
        
        # Apply professional styling
        VisualizationConfig.apply_professional_styling(ax, title=title, xlabel='Group', ylabel=ylabel)
        
        # Add statistical test result
        if stats_result and stats_result.get('test_type') != 'none':
            test_info = f"{stats_result['test_type']}: p={stats_result['p_value']:.4f}"
            if stats_result.get('significant'):
                test_info += " *"
            ax.text(0.02, 0.98, test_info, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.xticks(rotation=45, ha='right')
        VisualizationConfig.save_professional_figure(fig, output_path, dpi=dpi)
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
        if not adata_path:
            return {
                "error": "AnnData file path not provided in analysis_data",
                "summary": "Visualization failed"
            }
        
        # Try multiple possible paths if file doesn't exist at specified location
        if not os.path.exists(adata_path):
            # Try relative path from current working directory
            possible_paths = [
                adata_path,
                os.path.join(os.getcwd(), adata_path),
                os.path.abspath(adata_path)
            ]
            # Also try with common cache directories
            if 'solver_cache' in adata_path:
                possible_paths.extend([
                    os.path.join('solver_cache', os.path.basename(adata_path)),
                    os.path.join('.', adata_path)
                ])
            
            found = False
            for path in possible_paths:
                if os.path.exists(path):
                    adata_path = path
                    found = True
                    break
            
            if not found:
                return {
                    "error": f"AnnData file not found at: {analysis_data.get('adata_path')}. Tried: {', '.join(possible_paths[:3])}",
                    "summary": "Visualization failed - AnnData file not accessible"
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
        
        # 3. Create cluster composition plot by group (if group column exists)
        # Always generate if group column exists, even with single group (for consistency)
        if 'group' in adata.obs:
            composition_path = self._create_publication_cluster_composition(
                adata, cluster_key, output_dir, figure_size, dpi
            )
            if composition_path:
                visual_outputs.append(composition_path)
        
        # 4. Create pie chart of cluster proportions
        cluster_pie_path = self._create_cluster_proportion_pie_chart(
            adata, cluster_key, output_dir, figure_size, dpi
        )
        if cluster_pie_path:
            visual_outputs.append(cluster_pie_path)
        
        # 5. Create cluster exemplar montage (separate from UMAP)
        exemplar_path = self._create_cluster_exemplar_montage(
            adata, cluster_key, cluster_resolution, output_dir, figure_size, dpi
        )
        if exemplar_path:
            visual_outputs.append(exemplar_path)
        
        # Calculate cluster statistics
        cluster_stats = self._calculate_cluster_statistics(adata, cluster_key)
        
        return {
            "summary": f"Generated {len(visual_outputs)} publication-quality visualizations for cell state analysis",
            "visual_outputs": visual_outputs,
            "cell_count": adata.n_obs,
            "num_clusters": adata.obs[cluster_key].nunique() if cluster_key in adata.obs else 0,
            "cluster_key": cluster_key,
            "cluster_statistics": cluster_stats
        }
    
    def _create_publication_umap_by_cluster(self, adata, cluster_key: str, resolution: float,
                                           output_dir: str, figure_size: tuple, dpi: int) -> Optional[str]:
        """Create publication-quality UMAP visualization colored by cluster with sample cell crops."""
        if 'X_umap' not in adata.obsm or cluster_key not in adata.obs:
            return None
        
        # Create figure with professional styling - consistent size
        fig, ax = VisualizationConfig.create_professional_figure(figsize=figure_size)
        
        # Get unique clusters and assign colors
        clusters = sorted(adata.obs[cluster_key].unique())
        n_clusters = len(clusters)
        colors = self._get_color_scheme(n_clusters)
        cluster_colors = {cluster: colors[i] for i, cluster in enumerate(clusters)}
        
        # Plot each cluster with larger dots
        for cluster in clusters:
            cluster_mask = adata.obs[cluster_key] == cluster
            umap_coords = adata.obsm['X_umap'][cluster_mask]
            ax.scatter(umap_coords[:, 0], umap_coords[:, 1], 
                      c=[cluster_colors[cluster]], label=f'Cluster {cluster}',
                      s=100, alpha=0.7, edgecolors='white', linewidths=0.8)
        
        # Apply professional styling
        VisualizationConfig.apply_professional_styling(
            ax, 
            title=f'UMAP - Leiden Clustering (resolution={resolution})',
            xlabel='UMAP 1', 
            ylabel='UMAP 2'
        )
        
        # Professional legend
        VisualizationConfig.create_professional_legend(
            ax, title='Cluster', bbox_to_anchor=(1.02, 1), loc='upper left'
        )
        
        output_path = os.path.join(output_dir, f"umap_cluster_res{resolution}.png")
        VisualizationConfig.save_professional_figure(fig, output_path)
        plt.close(fig)
        
        return output_path
    
    def _load_and_resize_crop(self, crop_path: str, size: tuple = (60, 60)) -> Optional[np.ndarray]:
        """Load and resize cell crop image for display in UMAP plot."""
        try:
            # Try to load with PIL first (handles various formats)
            if os.path.exists(crop_path):
                # Try PIL
                try:
                    img = Image.open(crop_path)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img = img.resize(size, Image.Resampling.LANCZOS)
                    return np.array(img)
                except Exception:
                    # Fallback to cv2
                    try:
                        img = cv2.imread(crop_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            # Convert grayscale to RGB
                            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                            img_resized = cv2.resize(img_rgb, size, interpolation=cv2.INTER_LANCZOS4)
                            return img_resized
                    except Exception:
                        pass
            return None
        except Exception:
            return None
    
    def _create_publication_umap_by_group(self, adata, output_dir: str, 
                                         figure_size: tuple, dpi: int) -> Optional[str]:
        """Create publication-quality UMAP visualization colored by group."""
        if 'X_umap' not in adata.obsm or 'group' not in adata.obs:
            return None
        
        fig, ax = VisualizationConfig.create_professional_figure(figsize=figure_size)
        
        groups = sorted(adata.obs['group'].unique())
        n_groups = len(groups)
        colors = self._get_color_scheme(n_groups)
        group_colors = {group: colors[i] for i, group in enumerate(groups)}
        
        for group in groups:
            group_mask = adata.obs['group'] == group
            umap_coords = adata.obsm['X_umap'][group_mask]
            ax.scatter(umap_coords[:, 0], umap_coords[:, 1],
                      c=[group_colors[group]], label=str(group),
                      s=100, alpha=0.7, edgecolors='white', linewidths=0.8)
        
        # Apply professional styling
        VisualizationConfig.apply_professional_styling(
            ax, 
            title='UMAP - Colored by Group',
            xlabel='UMAP 1', 
            ylabel='UMAP 2'
        )
        
        # Professional legend
        VisualizationConfig.create_professional_legend(
            ax, title='Group', bbox_to_anchor=(1.02, 1), loc='upper left'
        )
        
        output_path = os.path.join(output_dir, "umap_by_group.png")
        VisualizationConfig.save_professional_figure(fig, output_path)
        plt.close(fig)
        
        return output_path
    
    def _create_publication_cluster_composition(self, adata, cluster_key: str,
                                               output_dir: str, figure_size: tuple, dpi: int) -> Optional[str]:
        """Create publication-quality cluster composition plot by group with grouped bar chart and statistical testing."""
        if 'group' not in adata.obs or cluster_key not in adata.obs:
            return None
        
        # Check if we have at least one group
        unique_groups = adata.obs['group'].nunique()
        if unique_groups == 0:
            return None
        
        # Calculate composition (proportions)
        composition = pd.crosstab(
            adata.obs['group'], 
            adata.obs[cluster_key],
            normalize='index'
        )
        
        # Calculate number of images per group (if image_name column exists)
        images_per_group = {}
        if 'image_name' in adata.obs:
            for group in composition.index:
                group_mask = adata.obs['group'] == group
                unique_images = adata.obs.loc[group_mask, 'image_name'].nunique()
                images_per_group[group] = unique_images
        else:
            # If no image_name, assume we can't determine image count
            for group in composition.index:
                images_per_group[group] = 0
        
        # Check if we should perform statistical testing (>=2 images per group)
        perform_statistical_test = all(count >= 2 for count in images_per_group.values()) and len(images_per_group) >= 2
        
        # Create figure with subplots if statistical test is needed
        if perform_statistical_test:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(figure_size[0] * 1.5, figure_size[1]), dpi=dpi)
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=figure_size, dpi=dpi)
            ax2 = None
        
        # Get professional colors for groups
        n_groups = len(composition.index)
        group_colors = self._get_color_scheme(n_groups)
        
        # Create grouped bar chart (each cluster is a group of bars, one per group)
        clusters = composition.columns.tolist()
        n_clusters = len(clusters)
        x = np.arange(n_clusters)
        width = 0.8 / n_groups  # Width of each bar
        
        for i, group in enumerate(composition.index):
            proportions = composition.loc[group].values
            offset = (i - n_groups / 2 + 0.5) * width
            ax1.bar(x + offset, proportions, width, label=str(group), 
                   color=group_colors[i], edgecolor='white', linewidth=1.0, alpha=0.8)
        
        # Apply professional styling
        VisualizationConfig.apply_professional_styling(
            ax1, title='Cluster Proportion by Group', 
            xlabel='Cluster', ylabel='Proportion'
        )
        
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'Cluster {c}' for c in clusters], rotation=45, ha='right')
        ax1.legend(title='Group', bbox_to_anchor=(1.02, 1), loc='upper left')
        ax1.set_ylim([0, 1])
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Perform statistical testing if conditions are met
        stats_text = ""
        if perform_statistical_test and ax2 is not None:
            # For each cluster, perform statistical test across groups
            # We'll use chi-square test or Fisher's exact test for proportions
            from scipy.stats import chi2_contingency, fisher_exact
            import numpy as np
            
            stats_results = []
            for cluster in clusters:
                # Get counts (not proportions) for this cluster across groups
                cluster_counts = pd.crosstab(adata.obs['group'], adata.obs[cluster_key] == cluster)
                
                # Prepare contingency table: [group1_not_in_cluster, group1_in_cluster], [group2_not_in_cluster, group2_in_cluster], ...
                contingency = []
                for group in composition.index:
                    if group in cluster_counts.index:
                        in_cluster = cluster_counts.loc[group, True] if True in cluster_counts.columns else 0
                        not_in_cluster = cluster_counts.loc[group, False] if False in cluster_counts.columns else 0
                        contingency.append([not_in_cluster, in_cluster])
                    else:
                        contingency.append([0, 0])
                
                contingency = np.array(contingency)
                
                # Perform chi-square test
                try:
                    if contingency.shape[0] == 2 and contingency.shape[1] == 2:
                        # 2x2 table: use Fisher's exact test
                        stat, p_value = fisher_exact(contingency)
                        test_name = "Fisher's exact"
                    else:
                        # Larger table: use chi-square test
                        stat, p_value, dof, expected = chi2_contingency(contingency)
                        test_name = "Chi-square"
                    
                    significant = p_value < 0.05
                    stats_results.append({
                        'cluster': cluster,
                        'test': test_name,
                        'p_value': p_value,
                        'significant': significant
                    })
                except Exception as e:
                    stats_results.append({
                        'cluster': cluster,
                        'test': 'Failed',
                        'p_value': None,
                        'error': str(e)
                    })
            
            # Display statistical results on second subplot
            ax2.axis('off')
            stats_lines = ["**Statistical Test Results**\n"]
            stats_lines.append(f"Test performed: Chi-square / Fisher's exact\n")
            stats_lines.append(f"Significance level: α = 0.05\n\n")
            
            for result in stats_results:
                if 'error' in result:
                    stats_lines.append(f"Cluster {result['cluster']}: {result['error']}\n")
                else:
                    sig_marker = "***" if result['significant'] else ""
                    stats_lines.append(f"Cluster {result['cluster']}: p = {result['p_value']:.4f} {sig_marker}\n")
            
            stats_text = "".join(stats_lines)
            ax2.text(0.1, 0.5, stats_text, transform=ax2.transAxes, 
                    fontsize=10, verticalalignment='center', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        elif perform_statistical_test:
            # If we should test but ax2 is None, add stats to summary
            stats_text = "\n\n**Note**: Statistical testing requires >=2 images per group."
        
        output_path = os.path.join(output_dir, "cluster_composition_by_group.png")
        VisualizationConfig.save_professional_figure(fig, output_path)
        plt.close(fig)
        
        return output_path
    
    def _create_cluster_exemplar_montage(self, adata, cluster_key: str, resolution: float,
                                        output_dir: str, figure_size: tuple, dpi: int) -> Optional[str]:
        """Create separate montage showing exemplar cell crops for each cluster."""
        if cluster_key not in adata.obs or 'crop_path' not in adata.obs:
            return None
        
        clusters = sorted(adata.obs[cluster_key].unique())
        crops_per_cluster = 5
        random.seed(42)  # Set seed for reproducibility
        
        # Calculate layout: rows = clusters, cols = crops_per_cluster
        n_rows = len(clusters)
        n_cols = crops_per_cluster
        crop_size = 100  # Size for each crop in montage
        
        # Use more compact figure size (similar to single-cell cropping summary)
        # Base size per cell crop: smaller for tighter layout
        base_width = 1.5  # Width per column (inches)
        base_height = 1.5  # Height per row (inches)
        montage_figsize = (base_width * n_cols + 1.0, base_height * n_rows + 0.8)  # Add space for labels and title
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=montage_figsize, dpi=dpi)
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Set very tight spacing between subplots (like single-cell cropping summary: wspace=0.05, hspace=0.05)
        plt.subplots_adjust(left=0.12, right=0.98, top=0.95, bottom=0.05, wspace=0.05, hspace=0.05)
        
        colors = self._get_color_scheme(len(clusters))
        cluster_colors = {cluster: colors[i] for i, cluster in enumerate(clusters)}
        
        for row_idx, cluster in enumerate(clusters):
            cluster_mask = adata.obs[cluster_key] == cluster
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) == 0:
                continue
            
            # Randomly sample cells
            n_samples = min(crops_per_cluster, len(cluster_indices))
            sampled_indices = random.sample(list(cluster_indices), n_samples)
            
            for col_idx in range(n_cols):
                ax = axes[row_idx, col_idx]
                ax.axis('off')
                
                if col_idx < len(sampled_indices):
                    idx = sampled_indices[col_idx]
                    crop_path = adata.obs.iloc[idx]['crop_path']
                    
                    # Handle path format
                    if isinstance(crop_path, (list, np.ndarray)) and len(crop_path) > 0:
                        crop_path = crop_path[0] if isinstance(crop_path[0], str) else str(crop_path[0])
                    elif not isinstance(crop_path, str):
                        crop_path = str(crop_path) if crop_path is not None else None
                    
                    if crop_path and os.path.exists(crop_path):
                        try:
                            crop_img = self._load_and_resize_crop(crop_path, size=(crop_size, crop_size))
                            if crop_img is not None:
                                ax.imshow(crop_img, cmap='gray' if len(crop_img.shape) == 2 else None)
                                # Add border with cluster color
                                for spine in ax.spines.values():
                                    spine.set_visible(True)
                                    spine.set_color(cluster_colors[cluster])
                                    spine.set_linewidth(2)
                        except Exception:
                            pass
                
                # Add cluster label on first column (adjusted position for tighter layout)
                if col_idx == 0:
                    ax.text(-0.15, 0.5, f'Cluster {cluster}', transform=ax.transAxes,
                           fontsize=12, fontweight='normal', va='center', ha='right')
        
        plt.suptitle(f'Cluster Exemplars (resolution={resolution})', 
                    fontsize=14, fontweight='normal', y=0.98)
        
        output_path = os.path.join(output_dir, f"cluster_exemplars_res{resolution}.png")
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig)
        
        return output_path
    
    def _create_cluster_proportion_pie_chart(self, adata, cluster_key: str,
                                            output_dir: str, figure_size: tuple, dpi: int) -> Optional[str]:
        """Create pie chart showing cluster proportions."""
        if cluster_key not in adata.obs:
            return None
        
        cluster_counts = adata.obs[cluster_key].value_counts().sort_index()
        total_cells = cluster_counts.sum()
        proportions = cluster_counts / total_cells * 100
        
        fig, ax = VisualizationConfig.create_professional_figure(figsize=figure_size)
        
        colors = self._get_color_scheme(len(cluster_counts))
        labels = [f'Cluster {c}\n({p:.1f}%)' for c, p in zip(cluster_counts.index, proportions)]
        
        wedges, texts, autotexts = ax.pie(cluster_counts.values, labels=labels,
                                          colors=colors[:len(cluster_counts)],
                                          autopct='', startangle=90,
                                          textprops={'fontsize': 11, 'fontweight': 'normal'})
        
        # Apply professional styling
        VisualizationConfig.apply_professional_styling(ax, title='Cluster Proportions')
        
        output_path = os.path.join(output_dir, "cluster_proportion_pie_chart.png")
        VisualizationConfig.save_professional_figure(fig, output_path)
        plt.close(fig)
        
        return output_path
    
    def _calculate_cluster_statistics(self, adata, cluster_key: str) -> Dict[str, Any]:
        """Calculate statistics for each cluster."""
        if cluster_key not in adata.obs:
            return {}
        
        cluster_counts = adata.obs[cluster_key].value_counts().sort_index()
        total_cells = cluster_counts.sum()
        
        stats = {}
        for cluster_id in cluster_counts.index:
            cluster_mask = adata.obs[cluster_key] == cluster_id
            cluster_size = cluster_counts[cluster_id]
            
            stats[str(cluster_id)] = {
                'cell_count': int(cluster_size),
                'proportion': float(cluster_size / total_cells * 100)
            }
            
            # Add area statistics if available
            if 'area' in adata.obs:
                cluster_areas = adata.obs.loc[cluster_mask, 'area']
                stats[str(cluster_id)]['mean_area'] = float(cluster_areas.mean())
                stats[str(cluster_id)]['median_area'] = float(cluster_areas.median())
        
        return stats
    
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
            
            # Load data from file if path is provided
            if isinstance(analysis_data, str):
                if os.path.exists(analysis_data):
                    # Check file extension to ensure it's a JSON file
                    file_ext = os.path.splitext(analysis_data)[1].lower()
                    if file_ext not in ['.json', '.jsonl']:
                        # Try to detect if it's a binary file (image, etc.)
                        try:
                            with open(analysis_data, 'rb') as f:
                                first_bytes = f.read(4)
                            if first_bytes.startswith(b'\x89PNG') or first_bytes.startswith(b'\xff\xd8') or first_bytes.startswith(b'GIF'):
                                return {
                                    "error": f"Expected JSON file but got image file: {analysis_data}. Analysis_Visualizer_Tool requires a dict or JSON file path, not image paths.",
                                    "summary": "Visualization failed: invalid file type"
                                }
                        except Exception:
                            pass
                        return {
                            "error": f"Expected JSON file but got '{file_ext}' file: {analysis_data}. Analysis_Visualizer_Tool requires analysis_data to be a dict or a JSON file path.",
                            "summary": "Visualization failed: invalid file type"
                        }
                    
                    # Load JSON file
                    try:
                        with open(analysis_data, 'r', encoding='utf-8') as f:
                            analysis_data = json.load(f)
                    except json.JSONDecodeError as e:
                        return {
                            "error": f"Invalid JSON format in file {analysis_data}: {str(e)}",
                            "summary": "Visualization failed: JSON decode error"
                        }
                    except UnicodeDecodeError as e:
                        return {
                            "error": f"File {analysis_data} is not a valid JSON file (encoding error: {str(e)}). It may be a binary file (image, etc.). Analysis_Visualizer_Tool requires analysis_data to be a dict or a JSON file path.",
                            "summary": "Visualization failed: invalid file encoding"
                        }
                else:
                    return {
                        "error": f"File not found: {analysis_data}",
                        "summary": "Visualization failed"
                    }
            
            # Check if this is cell state analysis output (from Cell_State_Analyzer_Tool)
            # Detect by: (1) explicit analysis_type flag, or (2) presence of adata_path + cluster_key
            # Also check if group_column is a cluster key (starts with 'leiden_' or matches cluster_key)
            is_cell_state_analysis = (
                isinstance(analysis_data, dict) and 
                'adata_path' in analysis_data and
                (analysis_data.get('analysis_type') == 'cell_state_analysis' or 
                 'cluster_key' in analysis_data or  # If cluster_key is present, assume cell state analysis
                 group_column.startswith('leiden_') or  # If group_column is a cluster key
                 group_column == analysis_data.get('cluster_key') or  # If group_column matches cluster_key
                 os.path.exists(analysis_data.get('adata_path', '')))
            )
            
            # For cell state analysis, generate publication-quality UMAP and cluster composition plots
            if is_cell_state_analysis:
                return self._visualize_cell_state_analysis(
                    analysis_data, output_dir, figure_size, dpi
                )
            
            # Prepare data for regular visualizations (only if not cell state analysis)
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

