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
from typing import Any, Dict, List
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from pathlib import Path

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

def sanitize_tool_output_for_llm(result: Any) -> Dict[str, Any]:
    """
    Sanitize tool output to separate LLM-visible summary from artifacts (file paths).
    
    CRITICAL: Local image paths must NEVER be sent to the LLM as image inputs.
    This function separates:
    - summary: Text/numeric data safe for LLM (no file paths)
    - artifacts: File paths for executor/cache only (not sent to LLM)
    
    Args:
        result: Raw tool output (dict, list, or other)
        
    Returns:
        dict with 'summary' (LLM-safe) and 'artifacts' (paths only)
    """
    if not isinstance(result, dict):
        # For non-dict results, return as summary with no artifacts
        return {
            "summary": result,
            "artifacts": {}
        }
    
    # Image path keys that should be moved to artifacts
    image_path_keys = [
        'processed_image_path', 'visual_outputs', 'output_path', 'mask_path',
        'image_path', 'nuclei_mask', 'comparison_plot', 'analyzed_h5ad_path',
        'h5ad_path', 'output_file', 'result_path', 'visualization_path'
    ]
    
    summary = {}
    artifacts = {}
    
    for key, value in result.items():
        # Check if key suggests it's an image path
        is_image_path = any(path_key in key.lower() for path_key in ['path', 'output', 'visual', 'image', 'mask', 'plot', 'file'])
        
        # Check if value is a file path (string ending with image extensions)
        is_file_path = False
        if isinstance(value, str):
            # Check if it looks like a file path
            is_file_path = (
                value.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.h5ad', '.json', '.csv')) or
                os.path.sep in value or
                '/' in value or
                '\\' in value
            )
        
        # Check if value is a list of file paths
        is_path_list = False
        if isinstance(value, list):
            is_path_list = all(
                isinstance(item, str) and (
                    item.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.h5ad', '.json', '.csv')) or
                    os.path.sep in item or '/' in item or '\\' in item
                )
                for item in value
            )
        
        # Move image paths to artifacts
        if (key in image_path_keys or (is_image_path and is_file_path)) and isinstance(value, str):
            artifacts[key] = value
        elif (key in image_path_keys or (is_image_path and is_path_list)) and isinstance(value, list):
            artifacts[key] = value
        else:
            # Keep in summary, but recursively sanitize nested dicts
            if isinstance(value, dict):
                sanitized = sanitize_tool_output_for_llm(value)
                summary[key] = sanitized['summary']
                if sanitized['artifacts']:
                    artifacts[f"{key}_artifacts"] = sanitized['artifacts']
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                # List of dicts - sanitize each
                sanitized_list = []
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        sanitized = sanitize_tool_output_for_llm(item)
                        sanitized_list.append(sanitized['summary'])
                        if sanitized['artifacts']:
                            artifacts[f"{key}_{i}_artifacts"] = sanitized['artifacts']
                    else:
                        sanitized_list.append(item)
                summary[key] = sanitized_list
            else:
                # Safe to include in summary (text, numbers, etc.)
                summary[key] = value
    
    return {
        "summary": summary,
        "artifacts": artifacts
    }

def get_llm_safe_result(result: Any) -> Any:
    """
    Get LLM-safe version of tool result (summary only, no file paths).
    
    Args:
        result: Raw tool output
        
    Returns:
        LLM-safe result (summary only)
    """
    sanitized = sanitize_tool_output_for_llm(result)
    return sanitized['summary']

def get_tool_artifacts(result: Any) -> Dict[str, Any]:
    """
    Extract artifacts (file paths) from tool result for executor/cache use.
    
    Args:
        result: Raw tool output
        
    Returns:
        dict of artifacts (file paths)
    """
    sanitized = sanitize_tool_output_for_llm(result)
    return sanitized['artifacts']
    
    