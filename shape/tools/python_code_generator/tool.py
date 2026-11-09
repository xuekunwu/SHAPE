# shape/tools/python_code_generator/tool.py

import os
import re
import sys
import json
import subprocess
from io import StringIO
import contextlib

from shape.tools.base import BaseTool
from shape.engine.factory import create_engine

# Code is executed in a fresh interpreter via subprocess so a real wall-clock
# timeout can terminate runaway loops/long computations (a thread-based timer
# cannot interrupt the main thread's exec()). The child reads code from stdin and
# returns a JSON {printed_output, variables} or {error}.
EXEC_TIMEOUT_SECONDS = 10

_SUBPROCESS_RUNNER = r"""
import sys, json
from io import StringIO
code = sys.stdin.read()
local_vars = {}
buf = StringIO()
_old = sys.stdout
sys.stdout = buf
try:
    exec(code, {}, local_vars)
    sys.stdout = _old
    used = {}
    for k, v in local_vars.items():
        if k.startswith("__"):
            continue
        try:
            json.dumps(v)
            used[k] = v
        except Exception:
            used[k] = repr(v)
    print(json.dumps({"printed_output": buf.getvalue().strip(), "variables": used}))
except Exception as e:
    sys.stdout = _old
    print(json.dumps({"error": str(e)}))
"""


class Python_Code_Generator_Tool(BaseTool):
    require_llm_engine = True
    require_api_key = True

    def __init__(self, model_string="gpt-4o-mini", api_key=None):
        super().__init__(
            tool_name="Python_Code_Generator_Tool",
            tool_description="A tool that generates and executes simple Python code snippets for basic arithmetical calculations and math-related problems. The generated code runs in a separate subprocess with a wall-clock timeout so runaway loops are killed. NOTE: this is NOT a security sandbox -- the subprocess is a normal interpreter that can import modules and touch the filesystem/network; use it only for trusted, self-contained computation.",
            tool_version="1.0.0",
            input_types={
                "query": "str - A clear, specific description of the arithmetic calculation or math problem to be solved, including any necessary numerical inputs."},
            output_type="dict - A dictionary containing the generated code, calculation result, and any error messages.",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(query="Calculate the factorial of 5")',
                    "description": "Generate a Python code snippet to calculate the factorial of 5."
                },
                {
                    "command": 'execution = tool.execute(query="Find the sum of prime numbers up to 50")',
                    "description": "Generate a Python code snippet to find the sum of prime numbers up to 50."
                },
                {
                    "command": 'query="Given the list [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], calculate the sum of squares of odd numbers"\nexecution = tool.execute(query=query)',
                    "description": "Generate a Python function for a specific mathematical operation on a given list of numbers."
                },
            ],
            user_metadata = {
                "limitations": [
                    "Intended for basic Python arithmetic and built-in mathematical functions.",
                    "Best used for simple, self-contained mathematical calculations.",
                    "A wall-clock timeout kills long-running/infinite code, but execution is NOT permission-isolated: the subprocess can import modules and access the filesystem/network. Do not run untrusted code with this tool.",
                    "Input must be provided directly in the query string.",
                    "Output is limited to numerical results or simple lists/tuples of numbers."
                ],
                "best_practices": [
                    "Provide clear and specific queries that describe the desired mathematical calculation.",
                    "Include all necessary numerical inputs directly in the query string.",
                    "Keep tasks focused on basic arithmetic, algebraic calculations, or simple mathematical algorithms.",
                    "Ensure all required numerical data is included in the query.",
                    "Verify that the query only involves mathematical operations and does not require any data processing or complex algorithms.",
                    "Review generated code to ensure it only uses basic Python arithmetic operations and built-in math functions."
                ]
            }
        )
        print(f"\nInitializing Python_Code_Generator_Tool with model_string: {model_string}")
        self.llm_engine = create_engine(model_string=model_string, is_multimodal=False, api_key=api_key) if model_string else None

    @staticmethod
    def preprocess_code(code):
        """
        Preprocesses the generated code snippet by extracting it from the response.

        Parameters:
            code (str): The response containing the code snippet.

        Returns:
            str: The extracted code snippet.
        """
        # Ensure code is a string
        if not isinstance(code, str):
            if code is None:
                raise ValueError("Code is None. LLM engine may not have returned a valid response.")
            code = str(code)
        
        # Try to extract code from markdown code blocks
        match = re.search(r"```python(.*?)```", code, re.DOTALL)
        if match:
            code = match.group(1).strip()
        else:
            # If no code block found, try to find code without markdown
            # Check if the response looks like code (contains common Python keywords)
            if any(keyword in code for keyword in ['import ', 'def ', 'print(', 'if ', 'for ', '=']):
                # Assume the entire response is code
                code = code.strip()
            else:
                # If it doesn't look like code, return as-is (might be an error message)
                code = code.strip()
        
        return code

    @contextlib.contextmanager
    def capture_output(self):
        """
        Context manager to capture the standard output.

        Yields:
            StringIO: The captured output.
        """
        new_out = StringIO()
        old_out = sys.stdout
        sys.stdout = new_out
        try:
            yield sys.stdout
        finally:
            sys.stdout = old_out

    def execute_code_snippet(self, code):
        """
        Executes the given Python code snippet.

        Parameters:
            code (str): The Python code snippet to be executed.

        Returns:
            dict: A dictionary containing the printed output and local variables.
        """
        # Ensure code is a string
        if not isinstance(code, str):
            if code is None:
                return {"error": "Code is None. LLM engine may not have returned a valid response."}
            code = str(code)
        
        # Check for dangerous functions and remove them
        dangerous_functions = ['exit', 'quit', 'sys.exit']
        for func in dangerous_functions:
            if func in code:
                print(f"Warning: Removing unsafe '{func}' call from code")
                # Use regex to remove function calls with any arguments
                try:
                    code = re.sub(rf'{func}\s*\([^)]*\)', 'break', code)
                except (TypeError, AttributeError) as e:
                    print(f"Warning: Failed to remove unsafe function '{func}': {e}")
                    continue

        try:
            execution_code = self.preprocess_code(code)

            # Run in a fresh interpreter with a real, enforceable timeout. The
            # child process is killed if it exceeds EXEC_TIMEOUT_SECONDS, so an
            # infinite loop or long computation can no longer hang the agent.
            try:
                proc = subprocess.run(
                    [sys.executable, "-c", _SUBPROCESS_RUNNER],
                    input=execution_code,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    timeout=EXEC_TIMEOUT_SECONDS,
                )
            except subprocess.TimeoutExpired:
                return {"error": f"Execution timed out after {EXEC_TIMEOUT_SECONDS} seconds"}

            out = (proc.stdout or "").strip()
            if not out:
                return {"error": (proc.stderr or "No output produced").strip()}
            try:
                return json.loads(out.splitlines()[-1])
            except Exception:
                # child printed something non-JSON; surface raw output
                return {"printed_output": out, "variables": {}}

        except Exception as e:
            print(f"Error executing code: {e}")
            return {"error": str(e)}

    def execute(self, query):
        """
        Generates and executes Python code based on the provided query.

        Parameters:
            query (str): A query describing the desired operation.

        Returns:
            dict: A dictionary containing the executed output, local variables, or any error message.
        """

        if not self.llm_engine:
            raise ValueError("LLM engine not initialized. Please provide a valid model_string when initializing the tool.")

        task_description = """
        Given a query, generate a Python code snippet that performs the specified operation on the provided data. Please think step by step. Ensure to break down the process into clear, logical steps. Make sure to print the final result in the generated code snippet with a descriptive message explaining what the output represents. The final output should be presented in the following format:

        ```python
        <code snippet>
        ```
        """
        task_description = task_description.strip()
        full_prompt = f"Task:\n{task_description}\n\nQuery:\n{query}"

        try:
            response = self.llm_engine(full_prompt)
            # Ensure response is a string
            if not isinstance(response, str):
                if response is None:
                    return {"error": "LLM engine returned None. Please check API key and model configuration."}
                response = str(response)
            
            result_or_error = self.execute_code_snippet(response)
            return result_or_error
        except Exception as e:
            error_msg = f"Error in LLM engine or code execution: {str(e)}"
            print(error_msg)
            return {"error": error_msg}

    def get_metadata(self):
        """
        Returns the metadata for the Python_Code_Generator_Tool.

        Returns:
            dict: A dictionary containing the tool's metadata.
        """
        metadata = super().get_metadata()
        metadata["require_llm_engine"] = self.require_llm_engine # NOTE: can be removed if not needed
        return metadata


if __name__ == "__main__":
    # Test command:
    """
    Run the following commands in the terminal to test the script:
    
    cd shape/tools/python_code_generator
    python tool.py
    """

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Example usage of the Python_Code_Generator_Tool
    tool = Python_Code_Generator_Tool()
    tool = Python_Code_Generator_Tool(model_string="gpt-4o-mini")

    # Get tool metadata
    metadata = tool.get_metadata()
    print(metadata)

    # Sample query for generating and executing Python code
    queries = [
        "Given the number list: [1, 2, 3, 4, 5], calculate the sum of all the numbers in the list.",
    ]
    for query in queries:
        print(f"\n###Query: {query}")
        # Execute the tool with the sample query
        try:
            execution = tool.execute(query=query)
            print("\n###Execution Result:", execution)
        except ValueError as e:
            print(f"Execution failed: {e}")

    print("Done!")



