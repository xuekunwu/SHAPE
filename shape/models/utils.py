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
from matplotlib import colors, cm
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
    Based on professional plotting standards from plotting/palettes.py and plotting/utils.py
    """
    
    # Professional color palettes (based on plotting/palettes.py)
    # Colorblindness-adjusted vega_10_scanpy palette
    VEGA_10_SCANPY = [
        "#1f77b4",  # blue
        "#279e68",  # green (adjusted for colorblindness)
        "#ff7f0e",  # orange
        "#aa40fc",  # purple (adjusted)
        "#d62728",  # red
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#b5bd61",  # khaki (adjusted)
        "#17becf",  # cyan
        "#bcbd22"   # yellow-green
    ]
    
    # Extended vega_20_scanpy palette
    VEGA_20_SCANPY = [
        *["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#279e68", "#98df8a", 
          "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94",
          "#e377c2", "#f7b6d3", "#aa40fc", "#c7c7c7", "#bcbd22", "#dbdb8d",
          "#b5bd61", "#dede5f"],
        "#ad494a",
        "#8c6d31"
    ]
    
    # Default palettes for different numbers of categories
    DEFAULT_26 = [
        "#023fa5", "#7d87b9", "#bec1d4", "#d6bcc0", "#bb7784", "#8e063b", "#4a6fe3",
        "#8595e1", "#b5bbe3", "#e6afb9", "#e07b91", "#d33f6a", "#11c638", "#8dd593",
        "#c6dec7", "#ead3c6", "#f0b98d", "#ef9708", "#0fcfc0", "#9cded6", "#d5eae7",
        "#f3e1eb", "#f6c4e1", "#f79cd4", "#7f7f7f", "#c7c7c7"
    ]
    
    DEFAULT_64 = [
        "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
        "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF",
        "#997D87", "#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF",
        "#4A3B53", "#FF2F80", "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92",
        "#FF90C9", "#B903AA", "#D16100", "#DDEFFF", "#000035", "#7B4F4B", "#A1C299",
        "#300018", "#0AA6D8", "#013349", "#00846F", "#372101", "#FFB500", "#C2FFED",
        "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09", "#00489C", "#6F0062",
        "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66", "#885578",
        "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C",
        "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F"
    ]
    
    # Professional visualization settings (optimized for publication quality)
    PROFESSIONAL_STYLE = {
        'figure.figsize': (10, 6),  # Consistent figure size
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.format': 'png',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'font.size': 11,  # Base font size (professional standard)
        'axes.titlesize': 16,  # Title size
        'axes.labelsize': 14,  # Label size
        'xtick.labelsize': 11,  # Tick label size
        'ytick.labelsize': 11,  # Tick label size
        'legend.fontsize': 11,  # Legend font size
        'legend.title_fontsize': 11,  # Legend title size
        'lines.linewidth': 1.5,  # Line width
        'axes.linewidth': 1.0,  # Axis line width (cleaner look)
        'grid.linewidth': 0.5,  # Grid line width
        'grid.alpha': 0.2,  # Light grid (professional standard)
        'axes.grid': True,
        'axes.axisbelow': True,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.spines.top': False,  # Remove top spine (cleaner)
        'axes.spines.right': False,  # Remove right spine (cleaner)
        'axes.spines.bottom': True,
        'axes.spines.left': True,
        'text.usetex': False,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif'],
        'font.weight': 'normal',  # Normal weight (not bold) for cleaner look
        'axes.labelweight': 'normal',
        'axes.titleweight': 'normal'
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
            Path to output directory (always absolute)
        """
        if query_cache_dir:
            # query_cache_dir should be absolute, join with OUTPUT_DIR
            output_dir = os.path.join(query_cache_dir, cls.OUTPUT_DIR)
        else:
            # If query_cache_dir is None, use default OUTPUT_DIR
            # But ensure it's resolved relative to current working directory
            output_dir = cls.OUTPUT_DIR
        
        # Ensure absolute path: if output_dir is relative, resolve it
        if not os.path.isabs(output_dir):
            # If query_cache_dir was provided but output_dir is still relative,
            # it means query_cache_dir itself was relative - resolve it
            if query_cache_dir and not os.path.isabs(query_cache_dir):
                # Resolve query_cache_dir first, then join with OUTPUT_DIR
                query_cache_dir_abs = os.path.abspath(query_cache_dir)
                output_dir = os.path.join(query_cache_dir_abs, cls.OUTPUT_DIR)
            else:
                # Resolve output_dir relative to current working directory
                output_dir = os.path.abspath(output_dir)
        
        # Normalize path separators
        output_dir = os.path.normpath(output_dir)
        
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
            print(f"馃Ч Cleared output directory: {output_dir}")
        elif not force_clear:
            print(f"馃搧 Preserving output directory: {output_dir}")
        
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
            
            print(f"鉁?Professional figure saved: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"鉂?Error saving figure {output_path}: {str(e)}")
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
    def get_color_palette(cls, n_colors: int) -> List[str]:
        """
        Get professional color palette for given number of colors.
        Uses colorblindness-adjusted palettes from plotting standards.
        
        Args:
            n_colors: Number of colors needed
            
        Returns:
            List of hex color strings
        """
        if n_colors <= 10:
            return cls.VEGA_10_SCANPY[:n_colors]
        elif n_colors <= 20:
            return cls.VEGA_20_SCANPY[:n_colors]
        elif n_colors <= 26:
            return cls.DEFAULT_26[:n_colors]
        elif n_colors <= 64:
            return cls.DEFAULT_64[:n_colors]
        else:
            # For more than 64 colors, cycle through default_64
            palette = cls.DEFAULT_64 * ((n_colors // 64) + 1)
            return palette[:n_colors]
    
    @classmethod
    def apply_professional_styling(cls, ax, title: str = "", xlabel: str = "", ylabel: str = ""):
        """
        Apply professional styling to a matplotlib axis.
        Based on plotting/utils.py standards for publication-quality figures.
        
        Args:
            ax: matplotlib axis object
            title: Title for the plot
            xlabel: X-axis label
            ylabel: Y-axis label
        """
        if title:
            ax.set_title(title, fontsize=16, fontweight='normal', pad=15)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=14, fontweight='normal')
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=14, fontweight='normal')
        
        # Apply professional grid and spine styling (cleaner, publication-quality)
        ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, axis='both')
        ax.tick_params(axis='both', which='major', labelsize=11, width=1.0, length=4)
        
        # Clean spine styling (remove top/right, thin lines)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.0)
        ax.spines['bottom'].set_linewidth(1.0)
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')
    
    @classmethod
    def create_professional_legend(cls, ax, title: str = "Cell States", **kwargs):
        """
        Create a professional legend with consistent formatting.
        Clean, simple style without shadows for publication quality.
        
        Args:
            ax: matplotlib axis object
            title: Legend title
            **kwargs: Additional legend arguments
            
        Returns:
            matplotlib legend object
        """
        # Set default kwargs for professional style
        default_kwargs = {
            'title': title,
            'title_fontsize': 11,
            'fontsize': 11,
            'frameon': True,
            'fancybox': False,
            'shadow': False,
            'edgecolor': 'gray',
            'facecolor': 'white'
        }
        default_kwargs.update(kwargs)
        
        legend = ax.legend(**default_kwargs)
        
        # Style the legend frame (clean, simple)
        if legend:
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_alpha(1.0)
            legend.get_frame().set_linewidth(0.8)
            legend.get_frame().set_edgecolor('gray')
        
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
    
    # Strip any "No matched tool given: " prefix if present (handle recursive calls)
    clean_name = tool_name
    if "No matched tool given: " in tool_name:
        # Handle multiple nested prefixes
        while "No matched tool given: " in clean_name:
            clean_name = clean_name.split("No matched tool given: ")[-1].strip()
    
    # First try exact match (case-insensitive)
    for tool in available_tools:
        if tool.lower() == clean_name.lower():
            print(f"normalize_tool_name: Exact match found: '{tool_name}' -> '{tool}'")
            return tool
    
    # Then try partial match (tool name contained in the given string)
    for tool in available_tools:
        if tool.lower() in clean_name.lower() or clean_name.lower() in tool.lower():
            print(f"normalize_tool_name: Partial match found: '{tool_name}' -> '{tool}'")
            return tool
    
    # If still no match, return error with cleaned name
    print(f"normalize_tool_name: No match found for '{tool_name}' (cleaned: '{clean_name}'). Available tools: {available_tools[:5]}...")
    return "No matched tool given: " + clean_name

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


def sanitize_paths_in_dict(obj: Any, depth: int = 0, max_depth: int = 10) -> Any:
    """
    Recursively remove file paths from dict/list structures.
    This ensures LLM-safe data by removing any remaining file paths that might
    have been missed in initial sanitization.
    
    Args:
        obj: Object to sanitize (dict, list, str, or other)
        depth: Current recursion depth
        max_depth: Maximum recursion depth to prevent infinite loops
        
    Returns:
        Sanitized object with file paths replaced with placeholders
    """
    if depth > max_depth:
        return obj  # Prevent infinite recursion
    
    # Check if string looks like a file path
    def is_file_path(value: str) -> bool:
        """Check if a string appears to be a file path."""
        if not isinstance(value, str) or len(value) < 3:
            return False
        # Check for file extensions or path separators
        path_indicators = [
            value.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.h5ad', '.json', '.csv', '.txt', '.pdf')),
            os.path.sep in value,
            '/' in value and len(value.split('/')) > 2,
            '\\' in value and len(value.split('\\')) > 2,
            value.startswith('/tmp/'),
            value.startswith('solver_cache/'),
            'output_visualizations' in value
        ]
        return any(path_indicators)
    
    if isinstance(obj, dict):
        sanitized = {}
        for key, value in obj.items():
            if isinstance(value, str) and is_file_path(value):
                # Replace file path with placeholder
                sanitized[key] = f"[file_path: {os.path.basename(value)}]"
            elif isinstance(value, dict):
                sanitized[key] = sanitize_paths_in_dict(value, depth + 1, max_depth)
            elif isinstance(value, list):
                sanitized[key] = sanitize_paths_in_dict(value, depth + 1, max_depth)
            else:
                sanitized[key] = value
        return sanitized
    elif isinstance(obj, list):
        sanitized = []
        for item in obj:
            if isinstance(item, str) and is_file_path(item):
                sanitized.append(f"[file_path: {os.path.basename(item)}]")
            elif isinstance(item, (dict, list)):
                sanitized.append(sanitize_paths_in_dict(item, depth + 1, max_depth))
            else:
                sanitized.append(item)
        return sanitized
    elif isinstance(obj, str) and is_file_path(obj):
        return f"[file_path: {os.path.basename(obj)}]"
    else:
        return obj
    
    

