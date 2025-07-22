from langchain.tools import BaseTool
from typing import Dict, Any
import requests
import traceback


class GetRepoUrlTool(BaseTool):
    """
    LangChain tool for StateGraph workflows.
    Fetches the GitHub repository URL for a specified user and repo name,
    and updates workflow state with the result.
    """

    name: str = "get_repo_url"
    description: str = (
        "Fetches the GitHub repository URL for a specified username and repo name. "
        "Updates workflow state with the result."
    )

    def _run(self, state: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Run the tool to fetch the GitHub repository URL.

        Args:
            state (Dict[str, Any]): Current workflow state.

        Returns:
            Dict[str, Any]: Updated workflow state with repo URL and status.
        """
        try:
            username = state.get("username")
            token = state.get("token")
            repo_name = state.get("repo_name")

            # Validate required parameters
            if not username or not token or not repo_name:
                error_msg = "❌ Missing required parameters: username, token, or repo_name."
                print(error_msg)
                state.update({
                    "repo_url_status": "failed",
                    "repo_url_message": error_msg
                })
                return state

            api_url = f"https://api.github.com/repos/{username}/{repo_name}"
            headers = {
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github+json"
            }

            print(f"🔗 Fetching GitHub URL for repository '{repo_name}'...")
            response = requests.get(api_url, headers=headers)

            if response.status_code == 200:
                repo_data = response.json()
                repo_url = repo_data.get("html_url")
                success_msg = f"✅ GitHub URL retrieved: {repo_url}"
                print(success_msg)
                state.update({
                    "repo_url_status": "success",
                    "repo_url_message": success_msg,
                    "repo_url": repo_url
                })
            else:
                error_msg = (
                    f"❌ Failed to fetch repository details: "
                    f"{response.status_code} {response.reason} - {response.text}"
                )
                print(error_msg)
                state.update({
                    "repo_url_status": "failed",
                    "repo_url_message": error_msg
                })

            return state

        except Exception as e:
            exception_msg = f"❌ Unexpected error during API call: {str(e)}"
            print(traceback.format_exc())
            state.update({
                "repo_url_status": "failed",
                "repo_url_message": exception_msg
            })
            return state

    def _arun(self, *args, **kwargs):
        raise NotImplementedError("Async execution is not supported.")
