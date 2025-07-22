import ast
import os
from langchain_core.runnables import RunnableLambda
from typing import Dict, Any


class ValidatePythonCodeAgent(RunnableLambda):
    """
    A StateGraph-compatible agent to validate Python code syntax in all .py files
    and TDD test files within a Git repository.
    """

    def __init__(self):
        """
        Initialize the validation agent with a runnable lambda.
        """
        super().__init__(self.validate_code)

    def validate_code(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates all Python files in the repo for syntax errors.

        Args:
            input (Dict[str, Any]): Workflow state, expects 'repo_path'.

        Returns:
            Dict[str, Any]: Updated state with validation status and details.
        """
        state = input  # 👈 Align with LangGraph tool input
        repo_path = state.get("repo_path")
        if not repo_path or not os.path.isdir(repo_path):
            error_msg = f"❌ Invalid or missing repo_path: {repo_path}"
            print(error_msg)
            state.update({
                "validation_status": "failed",
                "validation_message": error_msg,
                "validation_errors": []
            })
            return state

        print(f"🔍 Validating Python files in repo: {repo_path}")
        errors = []

        # Walk through all files in repo
        for root, _, files in os.walk(repo_path):
            for file in files:
                if file.endswith(".py"):  # Target only Python files
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            code = f.read()
                            ast.parse(code)  # Parse for syntax errors
                    except SyntaxError as e:
                        error_detail = {
                            "file": file_path,
                            "error": f"SyntaxError: {e.msg} (line {e.lineno}, col {e.offset})"
                        }
                        print(f"❌ {error_detail}")
                        errors.append(error_detail)
                    except Exception as e:
                        error_detail = {
                            "file": file_path,
                            "error": f"UnexpectedError: {str(e)}"
                        }
                        print(f"❌ {error_detail}")
                        errors.append(error_detail)

        if errors:
            summary_msg = f"🚨 Found syntax errors in {len(errors)} file(s)."
            state.update({
                "validation_status": "failed",
                "validation_message": summary_msg,
                "validation_errors": errors
            })
        else:
            success_msg = "✅ All Python files are syntactically correct."
            print(success_msg)
            state.update({
                "validation_status": "success",
                "validation_message": success_msg,
                "validation_errors": []
            })

        return state
