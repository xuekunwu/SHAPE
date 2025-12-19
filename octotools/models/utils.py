# import json

# def truncate_result(result, max_length: int = 100000, truncation_indicator: str = "...") -> str:
#     """
#     Truncate the result to specified length while preserving JSON structure when possible.
    
#     Args:
#         result: The result to truncate (can be str, list, dict, or other types)
#         max_length: Maximum length of the output string (default: 1000)
#         truncation_indicator: String to indicate truncation (default: "...")
        
#     Returns:
#         str: Truncated string representation of the result
#     """
#     if isinstance(result, (dict, list)):
#         try:
#             result_str = json.dumps(result, ensure_ascii=False)
#         except:
#             result_str = str(result)
#     else:
#         result_str = str(result)
    
#     indicator_length = len(truncation_indicator)
    
#     if len(result_str) > max_length:
#         # For JSON-like strings, try to find the last complete structure
#         if result_str.startswith('{') or result_str.startswith('['):
#             # Find last complete element
#             pos = max_length - indicator_length
#             while pos > 0 and not (
#                 result_str[pos] in ',]}' and 
#                 result_str[pos:].count('"') % 2 == 0
#             ):
#                 pos -= 1
#             if pos > 0:
#                 return result_str[:pos + 1] + truncation_indicator
        
#         # Default truncation if not JSON or no suitable truncation point found
#         return result_str[:max_length - indicator_length] + truncation_indicator
    
#     return result_str

import json
import numpy as np
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from pathlib import Path
import random
import sys

def set_reproducibility(seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Set global seeds for reproducibility (Issue 5) and return environment info.
    """
    seed_val = seed if seed is not None else 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    env = {
        "seed": seed_val,
        "python_version": sys.version,
        "numpy_version": np.__version__,
    }
    try:
        import torch
        torch.manual_seed(seed_val)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_val)
        env["torch_version"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
    except Exception:
        env["torch_version"] = None
        env["cuda_available"] = False
    return env

def make_json_safe(obj: Any) -> Any:
    """
    Recursively convert common non-serializable types (Path, numpy, tuples) into JSON-safe types.
    Does NOT drop data. (Execution provenance hardening)
    """
    from pathlib import Path
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, tuple):
        return [make_json_safe(o) for o in obj]
    if isinstance(obj, list):
        return [make_json_safe(o) for o in obj]
    if isinstance(obj, dict):
        return {make_json_safe(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj

def make_json_serializable(obj):
    """Convert numpy arrays and other non-serializable objects to JSON-serializable format."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    else:
        return obj

class VisualizationConfig:
    """
    Centralized configuration for all visualizations to ensure consistency.
    All tools should use this configuration for generating images.
    """
    
    # Professional visualization settings
    PROFESSIONAL_STYLE = {
        'figure.figsize': (12, 8),
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.format': 'png',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'font.size': 16,  # Increased base font size
        'axes.titlesize': 22,  # Increased title size
        'axes.labelsize': 20,  # Increased label size
        'xtick.labelsize': 18,  # Increased tick label size
        'ytick.labelsize': 18,  # Increased tick label size
        'legend.fontsize': 17,  # Increased legend font size
        'legend.title_fontsize': 18,  # Increased legend title size
        'lines.linewidth': 2.5,  # Increased line width
        'axes.linewidth': 2.0,  # Increased axis line width
        'grid.linewidth': 1.0,  # Increased grid line width
        'grid.alpha': 0.4,  # Increased grid alpha
        'axes.grid': True,
        'axes.axisbelow': True,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.spines.top': True,
        'axes.spines.right': True,
        'axes.spines.bottom': True,
        'axes.spines.left': True,
        'text.usetex': False,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
    }
    
    # Output directory configuration
    OUTPUT_DIR = "output_visualizations"
    
    @classmethod
    def setup_professional_style(cls):
        """Apply professional style settings to matplotlib."""
        plt.style.use('default')  # Reset to default first
        mpl.rcParams.update(cls.PROFESSIONAL_STYLE)
    
    @classmethod
    def get_output_dir(cls, query_cache_dir: str = None) -> str:
        """
        Get the standardized output directory for visualizations.
        
        Args:
            query_cache_dir: Optional cache directory, if None uses default OUTPUT_DIR
            
        Returns:
            Path to output directory
        """
        if query_cache_dir:
            output_dir = os.path.join(query_cache_dir, cls.OUTPUT_DIR)
        else:
            output_dir = cls.OUTPUT_DIR
        
        # Ensure directory exists
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    
    @classmethod
    def clear_output_dir(cls, query_cache_dir: str = None, force_clear: bool = False):
        """
        Clear the output directory (only when explicitly requested).
        
        Args:
            query_cache_dir: Optional cache directory
            force_clear: If True, clear the directory. If False, preserve all files (default: False)
        """
        output_dir = cls.get_output_dir(query_cache_dir)
        if force_clear and os.path.exists(output_dir):
            import shutil
            shutil.rmtree(output_dir)
            print(f"🧹 Cleared output directory: {output_dir}")
        elif not force_clear:
            print(f"📁 Preserving output directory: {output_dir}")
        
        # Always ensure directory exists
        os.makedirs(output_dir, exist_ok=True)
    
    @classmethod
    def save_professional_figure(cls, fig, output_path: str, title: str = "", 
                                dpi: int = 300, bbox_inches: str = 'tight'):
        """
        Save figure with professional settings and error handling.
        
        Args:
            fig: matplotlib figure object
            output_path: Path to save the figure
            title: Optional title for the figure
            dpi: DPI setting (default 300)
            bbox_inches: Bbox setting (default 'tight')
        """
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Apply professional style
            cls.setup_professional_style()
            
            # Save with professional settings
            fig.savefig(
                output_path,
                dpi=dpi,
                bbox_inches=bbox_inches,
                pad_inches=0.1,
                format='png',
                facecolor='white',
                edgecolor='none',
                transparent=False
            )
            
            print(f"✅ Professional figure saved: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ Error saving figure {output_path}: {str(e)}")
            return None
    
    @classmethod
    def create_professional_figure(cls, figsize: tuple = (12, 8), **kwargs):
        """
        Create a new figure with professional settings.
        
        Args:
            figsize: Figure size (default 12x8 inches)
            **kwargs: Additional arguments for plt.subplots
            
        Returns:
            tuple: (fig, ax) matplotlib figure and axis objects
        """
        cls.setup_professional_style()
        return plt.subplots(figsize=figsize, **kwargs)
    
    @classmethod
    def get_professional_colors(cls):
        """
        Get professional color palette for cell states.
        
        Returns:
            dict: Color mapping for different cell states
        """
        return {
            'dead': '#808080', 
            'np-MyoFb': '#A65A9F', 
            'p-MyoFb': '#D6B8D8', 
            'proto-MyoFb': '#F8BD6F', 
            'q-Fb': '#66B22F'
        }
    
    @classmethod
    def apply_professional_styling(cls, ax, title: str = "", xlabel: str = "", ylabel: str = ""):
        """
        Apply professional styling to a matplotlib axis.
        
        Args:
            ax: matplotlib axis object
            title: Title for the plot
            xlabel: X-axis label
            ylabel: Y-axis label
        """
        if title:
            ax.set_title(title, fontsize=22, fontweight='bold', pad=20)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=20, fontweight='bold')
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=20, fontweight='bold')
        
        # Apply professional grid and spine styling
        ax.grid(True, alpha=0.4, linewidth=1.0)
        ax.tick_params(axis='both', which='major', labelsize=18, width=2, length=6)
        
        # Make spines more prominent
        for spine in ax.spines.values():
            spine.set_linewidth(2.0)
            spine.set_color('black')
    
    @classmethod
    def create_professional_legend(cls, ax, title: str = "Cell States", **kwargs):
        """
        Create a professional legend with consistent formatting.
        
        Args:
            ax: matplotlib axis object
            title: Legend title
            **kwargs: Additional legend arguments
            
        Returns:
            matplotlib legend object
        """
        legend = ax.legend(
            title=title,
            title_fontsize=18,
            fontsize=17,
            frameon=True,
            fancybox=True,
            shadow=True,
            **kwargs
        )
        
        # Style the legend frame
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.9)
        legend.get_frame().set_linewidth(1.5)
        
        return legend

def make_json_serializable_truncated(obj, max_length: int = 100000):
    if isinstance(obj, (int, float, bool, type(None))):
        if isinstance(obj, (int, float)) and len(str(obj)) > max_length:
            return str(obj)[:max_length - 3] + "..."
        return obj
    elif isinstance(obj, str):
        return obj if len(obj) <= max_length else obj[:max_length - 3] + "..."
    elif isinstance(obj, dict):
        return {make_json_serializable_truncated(key, max_length): make_json_serializable_truncated(value, max_length) 
                for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable_truncated(element, max_length) for element in obj]
    elif hasattr(obj, '__dict__'):
        return make_json_serializable_truncated(obj.__dict__, max_length)
    else:
        result = str(obj)
        return result if len(result) <= max_length else result[:max_length - 3] + "..."

def normalize_tool_name(tool_name: str, available_tools=None) -> str:
    """Normalize the tool name to match the available tools."""
    if available_tools is None:
        return tool_name
    for tool in available_tools:
        if tool.lower() in tool_name.lower():
            return tool
    return "No matched tool given: " + tool_name
    
    
