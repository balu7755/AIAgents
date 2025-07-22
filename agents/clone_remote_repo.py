from langchain.tools import BaseTool
from git import Repo, GitCommandError
import os
import traceback
from typing import Dict, Any


class CloneNewRepoAgent(BaseTool):
    """
    LangChain Agent Tool for StateGraph workflows.
    Clones a GitHub repository to a local directory using
    username and token authentication.
    """

    name: str = "clone_new_repo"
    description: str = (
        "Clones a GitHub repository to a specified local directory. "
        "Skips cloning if the repo already exists locally. Updates workflow state with results."
    )

    def _run(self, input: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Run the agent to clone the repository.

        Args:
            input (Dict[str, Any]): The current workflow state.

        Returns:
            Dict[str, Any]: Updated workflow state with status and details.
        """
        state = input  # 📌 Aligning with workflow state
        try:
            username = state.get("username")
            token = state.get("token")
            repo_url = state.get("repo_url")
            repo_path = state.get("repo_path", "./cloned_repo").strip()

            # ✅ Validate required parameters
            missing = []
            if not username:
                missing.append("username")
            if not token:
                missing.append("token")
            if not repo_url:
                missing.append("repo_url")
            if not repo_path:
                missing.append("repo_path")

            if missing:
                error_msg = (
                    f"❌ Missing required parameters in state: {', '.join(missing)}.\n"
                    f"Please ensure these are set before running CloneNewRepoAgent."
                )
                print(error_msg)
                state.update({
                    "clone_status": "failed",
                    "clone_message": error_msg
                })
                return state

            # 📂 Check if repo already exists locally
            if os.path.isdir(os.path.join(repo_path, ".git")):
                msg = f"⚠️ Repository already exists locally at {repo_path}. Skipping clone."
                print(msg)
                state.update({
                    "clone_status": "already_exists",
                    "clone_message": msg
                })
                return state

            # 🔐 Inject credentials into remote URL
            if repo_url.startswith("https://"):
                auth_url = repo_url.replace(
                    "https://",
                    f"https://{username}:{token}@"
                )
            else:
                error_msg = "❌ Invalid remote URL. Must use HTTPS."
                print(error_msg)
                state.update({
                    "clone_status": "failed",
                    "clone_message": error_msg
                })
                return state

            print(f"🔄 Cloning repository from {repo_url} to {repo_path}...")
            Repo.clone_from(auth_url, repo_path)

            success_msg = f"✅ Repository cloned successfully to {repo_path}."
            print(success_msg)
            state.update({
                "clone_status": "success",
                "clone_message": success_msg
            })
            return state

        except GitCommandError as e:
            error_msg = f"❌ Git error while cloning: {str(e)}"
            print(error_msg)
            state.update({
                "clone_status": "failed",
                "clone_message": error_msg
            })
            return state

        except Exception as e:
            print(traceback.format_exc())
            exception_msg = f"❌ Unexpected error while cloning repository: {str(e)}"
            state.update({
                "clone_status": "failed",
                "clone_message": exception_msg
            })
            return state

    def invoke(self, input: Dict[str, Any], config: Any = None, **kwargs) -> Dict[str, Any]:
        """
        Adapter for LangGraph to pass state correctly to _run().
        """
        return self._run(input, **kwargs)

    def _arun(self, *args, **kwargs):
        raise NotImplementedError("Async is not supported for this agent.")
