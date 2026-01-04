"""
Unified response parser utility.
Eliminates code duplication in parsing LLM responses across planner and executor.
"""

from typing import Dict, Any, Tuple, Optional
import re

# Import logger - use lazy import to avoid circular dependency
try:
    from octotools.utils.logger import logger
except ImportError:
    # Fallback if logger not available yet
    import logging
    logger = logging.getLogger('octotools')


class ResponseParser:
    """Unified parser for LLM structured responses."""
    
    @staticmethod
    def normalize_code(code: str) -> str:
        """Remove markdown code blocks from code string."""
        # Remove ```python at the beginning
        code = re.sub(r'^```python\s*', '', code)
        # Remove ``` at the end (handle both with and without newlines)
        code = re.sub(r'\s*```\s*$', '', code)
        return code.strip()
    
    @staticmethod
    def parse_next_step(response: Any, available_tools: list) -> Tuple[str, str, str]:
        """
        Parse NextStep response from LLM.
        
        Returns:
            Tuple of (context, sub_goal, tool_name)
        """
        # Check if response is a NextStep object
        if hasattr(response, 'context') and hasattr(response, 'sub_goal') and hasattr(response, 'tool_name'):
            context = response.context.strip()
            sub_goal = response.sub_goal.strip()
            tool_name = ResponseParser._normalize_tool_name(response.tool_name.strip(), available_tools)
            return context, sub_goal, tool_name
        
        # Check if response is a string (fallback for non-structured models)
        elif isinstance(response, str):
            return ResponseParser._parse_next_step_string(response, available_tools)
        else:
            logger.warning(f"Unknown response type: {type(response)}")
            return "", "", "Error parsing tool name"
    
    @staticmethod
    def _parse_next_step_string(response: str, available_tools: list) -> Tuple[str, str, str]:
        """Parse NextStep from string response."""
        lines = response.split('\n')
        context = ""
        sub_goal = ""
        tool_name = ""
        
        for line in lines:
            line = line.strip()
            # Handle <context>...</context> tags
            if line.lower().startswith('<context>') and not line.lower().startswith('<context>:'):
                context = line.split('<context>')[1].split('</context>')[0].strip()
            elif line.lower().startswith('context:'):
                parts = line.split('context:', 1)
                if len(parts) > 1:
                    context = parts[1].lstrip(' :')
            
            # Handle <sub_goal>...</sub_goal> tags
            elif line.lower().startswith('<sub_goal>') and not line.lower().startswith('<sub_goal>:'):
                sub_goal = line.split('<sub_goal>')[1].split('</sub_goal>')[0].strip()
            elif line.lower().startswith('sub_goal:'):
                parts = line.split('sub_goal:', 1)
                if len(parts) > 1:
                    sub_goal = parts[1].lstrip(' :')
            
            # Handle <tool_name>...</tool_name> tags
            elif line.lower().startswith('<tool_name>') and not line.lower().startswith('<tool_name>:'):
                tool_name = line.split('<tool_name>')[1].split('</tool_name>')[0].strip()
            elif line.lower().startswith('tool_name:'):
                parts = line.split('tool_name:', 1)
                if len(parts) > 1:
                    tool_name = parts[1].lstrip(' :')
        
        tool_name = ResponseParser._normalize_tool_name(tool_name, available_tools)
        return context, sub_goal, tool_name
    
    @staticmethod
    def parse_tool_command(response: Any) -> Tuple[str, str, str]:
        """
        Parse ToolCommand response from LLM.
        
        Returns:
            Tuple of (analysis, explanation, command)
        """
        # Check if response is a ToolCommand object
        if hasattr(response, 'analysis') and hasattr(response, 'explanation') and hasattr(response, 'command'):
            analysis = response.analysis.strip()
            explanation = response.explanation.strip()
            command = ResponseParser.normalize_code(response.command.strip())
            return analysis, explanation, command
        
        # Check if response is a string (fallback)
        elif isinstance(response, str):
            return ResponseParser._parse_tool_command_string(response)
        else:
            logger.warning(f"Unknown ToolCommand response type: {type(response)}")
            return "Error parsing analysis", "Error parsing explanation", "execution = tool.execute(error='Error parsing command')"
    
    @staticmethod
    def _parse_tool_command_string(response: str) -> Tuple[str, str, str]:
        """Parse ToolCommand from string response."""
        lines = response.split('\n')
        analysis = ""
        explanation = ""
        command = ""
        
        for line in lines:
            line = line.strip()
            # Handle <analysis>...</analysis> tags
            if line.lower().startswith('<analysis>') and not line.lower().startswith('<analysis>:'):
                analysis = line.split('<analysis>')[1].split('</analysis>')[0].strip()
            elif line.lower().startswith('analysis:'):
                parts = line.split('analysis:', 1)
                if len(parts) > 1:
                    analysis = parts[1].lstrip(' :')
            
            # Handle <explanation>...</explanation> tags
            elif line.lower().startswith('<explanation>') and not line.lower().startswith('<explanation>:'):
                explanation = line.split('<explanation>')[1].split('</explanation>')[0].strip()
            elif line.lower().startswith('explanation:'):
                parts = line.split('explanation:', 1)
                if len(parts) > 1:
                    explanation = parts[1].lstrip(' :')
            
            # Handle <command>...</command> tags
            elif line.lower().startswith('<command>') and not line.lower().startswith('<command>:'):
                command = line.split('<command>')[1].split('</command>')[0].strip()
            elif line.lower().startswith('command:'):
                parts = line.split('command:', 1)
                if len(parts) > 1:
                    command = parts[1].lstrip(' :')
        
        # Normalize command code
        if command:
            command = ResponseParser.normalize_code(command)
        
        # Use defaults if missing
        if not analysis:
            analysis = "No analysis provided"
        if not explanation:
            explanation = "No explanation provided"
        if not command:
            # Use unified error format: dictionary instead of tool.execute(error=...)
            command = "execution = {'error': 'No command provided', 'status': 'failed'}"
        
        return analysis, explanation, command
    
    @staticmethod
    def parse_memory_verification(response: Any) -> Tuple[str, bool]:
        """
        Parse MemoryVerification response from LLM.
        
        Returns:
            Tuple of (analysis, stop_signal)
        """
        # Check if response is a MemoryVerification object
        if hasattr(response, 'analysis') and hasattr(response, 'stop_signal'):
            return response.analysis, response.stop_signal
        
        # Check if response is a string (fallback)
        elif isinstance(response, str):
            return ResponseParser._parse_memory_verification_string(response)
        else:
            logger.warning(f"Unknown MemoryVerification response type: {type(response)}")
            return str(response), False
    
    @staticmethod
    def _parse_memory_verification_string(response: str) -> Tuple[str, bool]:
        """Parse MemoryVerification from string response."""
        lines = response.split('\n')
        analysis = ""
        stop_signal = False
        
        for line in lines:
            line = line.strip()
            if line.lower().startswith('analysis:'):
                analysis = line[9:].strip()
            elif line.lower().startswith('stop_signal:'):
                signal_text = line[12:].strip().lower()
                stop_signal = signal_text in ['true', 'yes', '1']
        
        # If we couldn't parse properly, use the whole response as analysis
        if not analysis:
            analysis = response
        
        return analysis, stop_signal
    
    @staticmethod
    def _normalize_tool_name(tool_name: str, available_tools: list) -> str:
        """Normalize tool name to match available tools."""
        # Strip any error prefix if present
        clean_name = tool_name
        if "No matched tool given: " in tool_name:
            while "No matched tool given: " in clean_name:
                clean_name = clean_name.split("No matched tool given: ")[-1].strip()
        
        # First try exact match (case-insensitive)
        for tool in available_tools:
            if tool.lower() == clean_name.lower():
                return tool
        
        # Then try partial match
        for tool in available_tools:
            if tool.lower() in clean_name.lower() or clean_name.lower() in tool.lower():
                return tool
        
        # If still no match, return error
        if 'logger' in globals():
            logger.warning(f"No match found for '{tool_name}' (cleaned: '{clean_name}')")
        return "No matched tool given: " + clean_name
