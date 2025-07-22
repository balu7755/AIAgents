from langchain.tools import BaseTool
from typing import Dict, Any
import subprocess
import traceback

class CheckRemoteRepoExistsAgent(BaseTool):
    name: str = "check_remote_repo_exists"
    description: str = "Checks if a GitHub repo exists and if branch is accessible."

    def _run(self, input: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        state = input
        try:
            username = state.get("username")
            token = state.get("token")
            repo_url = state.get("repo_url")
            branch = state.get("branch")

            if not (username and token and repo_url and branch):
                error_msg = "❌ Missing required parameters (username, token, repo_url, branch)."
                print(error_msg)
                state["repo_check_status"] = "failed"
                state["repo_check_message"] = error_msg
                return state

            print(f"🔗 Checking remote repository: {repo_url} (branch: {branch})")
            if repo_url.startswith("https://"):
                auth_url = repo_url.replace("https://", f"https://{username}:{token}@")
            else:
                error_msg = "❌ Invalid repository URL. Must start with https://"
                print(error_msg)
                state["repo_check_status"] = "failed"
                state["repo_check_message"] = error_msg
                return state

            result = subprocess.run(
                ["git", "ls-remote", "--heads", auth_url, branch],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10
            )

            if result.returncode == 0 and result.stdout:
                success_msg = f"✅ Repo '{repo_url}' exists and branch '{branch}' is accessible."
                print(success_msg)
                state["repo_check_status"] = "success"
                state["repo_check_message"] = success_msg
            elif result.returncode == 0 and not result.stdout:
                warning_msg = f"⚠️ Repo exists but branch '{branch}' not found."
                print(warning_msg)
                state["repo_check_status"] = "branch_not_found"
                state["repo_check_message"] = warning_msg
            else:
                error_msg = f"❌ Repo '{repo_url}' not accessible.\nGit Error: {result.stderr.strip()}"
                print(error_msg)
                state["repo_check_status"] = "failed"
                state["repo_check_message"] = error_msg

            return state

        except Exception as e:
            error_msg = f"❌ Exception occurred: {e}"
            print(traceback.format_exc())
            state["repo_check_status"] = "failed"
            state["repo_check_message"] = error_msg
            return state

    def invoke(self, input: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:
        return self._run(input, **kwargs)

    def _arun(self, *args, **kwargs):
        raise NotImplementedError("Async not supported.")
