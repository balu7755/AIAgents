from langchain.tools import BaseTool
from typing import Dict, Any
import subprocess
import os
import re
import traceback


class GenerateCoverageAgent(BaseTool):
    """
    LangChain Agent Tool for StateGraph workflows.
    Runs pytest coverage and updates workflow state with the result.
    """

    name: str = "generate_code_coverage"
    description: str = (
        "Runs pytest coverage on the repository and updates the workflow state. "
        "It does NOT call any other agent; only reports coverage and status."
    )

    def _run(self, input: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Run the agent to check code coverage.

        Args:
            input (Dict[str, Any]): The current workflow state.

        Returns:
            Dict[str, Any]: Updated workflow state.
        """
        state = input  # 👈 Aligning with LangGraph's expected tool input
        try:
            repo_path = state.get("repo_path")
            threshold = state.get("coverage_threshold", 90)

            if not repo_path:
                error_msg = "❌ Missing 'repo_path' in workflow state."
                print(error_msg)
                state["coverage_status"] = "failed"
                state["coverage_message"] = error_msg
                return state

            print(f"📊 Running pytest coverage in repository: {repo_path}")

            # ✅ Ensure pytest and pytest-cov are installed
            self._install_pytest()

            # ▶️ Run pytest with coverage
            output = self._run_pytest_coverage(repo_path)

            # 📈 Parse coverage percentage
            coverage_percent = self._parse_coverage(output)
            if coverage_percent is None:
                msg = "⚠️ Could not determine code coverage from pytest output."
                print(msg)
                state["coverage_status"] = "failed"
                state["coverage_message"] = msg
                return state

            print(f"📈 Code coverage: {coverage_percent}%")
            state["coverage_percent"] = coverage_percent

            if coverage_percent >= threshold:
                success_msg = f"✅ Code coverage {coverage_percent}% meets threshold {threshold}%."
                print(success_msg)
                state["coverage_status"] = "success"
                state["coverage_message"] = success_msg
            else:
                warning_msg = (
                    f"⚠️ Coverage {coverage_percent}% is below threshold {threshold}%. "
                    "No additional actions taken. Workflow should decide next step."
                )
                print(warning_msg)
                state["coverage_status"] = "below_threshold"
                state["coverage_message"] = warning_msg

            return state

        except Exception as e:
            print(traceback.format_exc())
            state["coverage_status"] = "failed"
            state["coverage_message"] = f"❌ Exception occurred: {e}"
            return state

    def invoke(self, input: Dict[str, Any],config: Any = None, **kwargs) -> Dict[str, Any]:
        """
        Adapter for LangGraph to pass state correctly to _run().
        """
        return self._run(input, **kwargs)

    def _install_pytest(self):
        """
        Ensure pytest and pytest-cov are installed.
        """
        try:
            subprocess.run(
                ["pip", "install", "pytest", "pytest-cov"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError("❌ Failed to install pytest/pytest-cov.") from e

    def _run_pytest_coverage(self, repo_path: str) -> str:
        """
        Run pytest with coverage in the specified repo.
        Returns combined stdout and stderr output.
        """
        result = subprocess.run(
            ["pytest", "--cov=.", "--cov-report=term-missing"],
            cwd=repo_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        output = result.stdout + "\n" + result.stderr
        print(output)
        return output

    def _parse_coverage(self, output: str) -> int:
        """
    Extract the coverage percentage from pytest output robustly.
    """
    # Match TOTAL line like: TOTAL     38      0   100%
        match = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+)%", output)
        if match:
            return int(match.group(1))

    # Fallback: Match lines like: Coverage: 100%
        fallback = re.search(r"Coverage:\s+(\d+)%", output, re.IGNORECASE)
        if fallback:
            return int(fallback.group(1))

    # If neither found, return None
        return None


    def _arun(self, *args, **kwargs):
        raise NotImplementedError("Async execution is not supported.")
