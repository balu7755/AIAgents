from langchain.tools import BaseTool
from typing import Dict, Any
import requests
import traceback

class CreateRemoteRepoTool(BaseTool):
    """
    Creates a new GitHub repository using the API.
    """
    name: str = "create_remote_repo"
    description: str = "Creates a new GitHub repository for the workflow."

    def _run(self, input: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Args:
            input (Dict[str, Any]): Current workflow state.
        Returns:
            Dict[str, Any]: Updated workflow state with repo creation result.
        """
        state = input  # 📌 StateGraph passes the entire workflow state
        try:
            username = state.get("username")
            token = state.get("token")
            new_repo_name = state.get("new_repo_name")

            if not username or not token or not new_repo_name:
                error_msg = "❌ Missing username, token, or new_repo_name in state."
                print(error_msg)
                state.update({
                    "repo_creation_status": "failed",
                    "repo_creation_message": error_msg
                })
                return state

            print(f"📦 Creating new GitHub repo: {new_repo_name}")
            api_url = "https://api.github.com/user/repos"
            headers = {
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github.v3+json"
            }
            payload = {
                "name": new_repo_name,
                "private": False
            }

            response = requests.post(api_url, headers=headers, json=payload)

            if response.status_code == 201:
                repo_data = response.json()
                repo_url = repo_data.get("html_url")
                success_msg = f"✅ Successfully created repo '{new_repo_name}': {repo_url}"
                print(success_msg)
                state.update({
                    "repo_creation_status": "success",
                    "repo_creation_message": success_msg,
                    "repo_url": repo_url  # update state with new repo URL
                })
            else:
                error_msg = (
                    f"❌ Failed to create repo: "
                    f"{response.status_code} {response.text}"
                )
                print(error_msg)
                state.update({
                    "repo_creation_status": "failed",
                    "repo_creation_message": error_msg
                })

            return state

        except Exception as e:
            exception_msg = f"❌ Exception: {str(e)}"
            print(traceback.format_exc())
            state.update({
                "repo_creation_status": "failed",
                "repo_creation_message": exception_msg
            })
            return state

    def invoke(self, input: Dict[str, Any],config: Any = None, **kwargs) -> Dict[str, Any]:
        """
        Adapter for LangGraph to pass state correctly to _run().
        """
        return self._run(input, **kwargs)

    def _arun(self, *args, **kwargs):
        raise NotImplementedError("Async not supported.")
