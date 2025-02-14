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

def make_json_serializable(obj):
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, dict):
        return {make_json_serializable(key): make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(element) for element in obj]
    elif hasattr(obj, '__dict__'):
        return make_json_serializable(obj.__dict__)
    else:
        return str(obj)
    

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
    