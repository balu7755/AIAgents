from langchain.tools import BaseTool
from typing import Dict, Any
from pydantic import PrivateAttr
import os
import traceback


class GenerateReadmeAgent(BaseTool):
    """
    LangChain Agent Tool for StateGraph workflows.
    Generates a README.md file using an LLM and updates the workflow state.
    """

    name: str = "generate_readme"
    description: str = (
        "Generates a README.md file using an LLM based on the project description "
        "and updates the workflow state with status and paths."
    )

    # 👇 Mark LLM as private attribute
    _llm: Any = PrivateAttr(default=None)

    def __init__(self, llm=None, **kwargs):
        """
        Initialize the agent.

        Args:
            llm: Any LLM instance that implements `.invoke(prompt)` for generation.
            kwargs: Additional BaseTool parameters.
        """
        super().__init__(**kwargs)
        self._llm = llm

    def _run(self, input: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Run the agent to generate a README.md file.

        Args:
            input (Dict[str, Any]): Current workflow state.

        Returns:
            Dict[str, Any]: Updated workflow state with status and results.
        """
        state = input  # 👈 Align with LangGraph expectations
        try:
            # Use injected LLM (already shared via build_graph)
            llm = self._llm

            if llm is None:
                error_msg = "❌ No LLM instance provided to GenerateReadmeAgent."
                print(error_msg)
                state.update({
                    "readme_status": "failed",
                    "readme_message": error_msg
                })
                return state

            # 📄 Prepare prompt for README generation
            project_name = state.get("project_name", "My Project")
            project_desc = state.get("code_prompt", "No description provided.")
            readme_path = os.path.join(state.get("repo_path", "./local_repo"), "README.md")

            llm_prompt = (
                f"You are an expert technical writer.\n"
                f"Create a professional README.md for the Python project '{project_name}'.\n"
                f"Project Description: {project_desc}\n"
                "Include the following sections:\n"
                "- 📖 Overview\n"
                "- ⚙️ Installation Instructions\n"
                "- 🚀 Usage Examples\n"
                "- ✅ Testing Information\n"
                "- 📜 License\n"
                "Respond ONLY with valid markdown for README.md.\n"
            )

            print(f"📄 Generating README.md using LLM...")

            # 🔥 Send to LLM
            response = llm.invoke(llm_prompt)
            readme_content = getattr(response, "content", str(response)).strip()

            if not readme_content:
                error_msg = "⚠️ LLM returned an empty README response."
                print(error_msg)
                state.update({
                    "readme_status": "failed",
                    "readme_message": error_msg
                })
                return state

            # 📝 Write README.md file
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(readme_content)

            success_msg = f"✅ README.md created at: {readme_path}"
            print(success_msg)
            state.update({
                "readme_status": "success",
                "readme_message": success_msg,
                "readme_path": readme_path,
                "readme_content": readme_content
            })

            return state

        except Exception as e:
            exception_msg = f"❌ Exception occurred: {e}"
            print(traceback.format_exc())
            state.update({
                "readme_status": "failed",
                "readme_message": exception_msg
            })
            return state

    def invoke(self, input: Dict[str, Any],config: Any = None, **kwargs) -> Dict[str, Any]:
        """
        Adapter for LangGraph to pass state correctly to _run().
        """
        return self._run(input, **kwargs)

    def _arun(self, *args, **kwargs):
        raise NotImplementedError("Async execution is not supported.")
