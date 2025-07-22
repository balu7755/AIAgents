from langchain.tools import BaseTool
from typing import Dict, Any
import subprocess
import os
import traceback


class GitCommitPushAgent(BaseTool):
    """
    LangChain Agent Tool for StateGraph workflows.
    Commits all changes in the local repository and pushes them to the specified branch.
    Initializes the repository if empty.
    """

    name: str = "git_commit_push"
    description: str = (
        "Commits all changes in the local repository and pushes them to the remote branch. "
        "If repository is empty, initializes it with an initial commit."
    )

    def _run(self, input: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Run the agent to commit and push changes, updating workflow state.

        Args:
            input (Dict[str, Any]): The current workflow state.

        Returns:
            Dict[str, Any]: Updated workflow state.
        """
        state = input  # 👈 Aligning with LangGraph's expected input
        try:
            repo_path = state.get("repo_path")
            username = state.get("username")
            token = state.get("token")
            repo_url = state.get("repo_url")
            new_branch = state.get("new_branch", "main")
            commit_message = state.get("commit_message", "🚀 Auto commit by GitCommitPushAgent")

            if not (repo_path and username and token and repo_url):
                error_msg = "❌ Missing required parameters in state (repo_path, username, token, repo_url)."
                print(error_msg)
                state["git_push_status"] = "failed"
                state["git_push_message"] = error_msg
                return state

            print(f"📁 Working in repository: {repo_path}")

            # Configure Git credentials
            self._configure_git_credentials(username, token, repo_url, repo_path)

            # Configure Git user identity
            self._configure_git_identity(state, repo_path)

            # Determine if repo is empty
            if self._is_repo_empty(repo_path):
                print("📂 Repository is empty. Initializing...")
                # Create and checkout new branch
                self._run_git_command(
                    ["git", "checkout", "-b", new_branch],
                    repo_path,
                    f"Creating and switching to branch {new_branch}"
                )
                # Create placeholder file for initial commit
                placeholder_file = os.path.join(repo_path, ".gitkeep")
                with open(placeholder_file, "w") as f:
                    f.write("# Placeholder file for initial commit\n")
                self._run_git_command(["git", "add", "."], repo_path, "Adding placeholder file")
                self._run_git_command(
                    ["git", "commit", "-m", "✨ Initial commit (auto-generated)"],
                    repo_path,
                    "Creating initial commit"
                )
            else:
                # Add and commit changes
                self._run_git_command(["git", "add", "."], repo_path, "Adding changes")
                commit_output = self._run_git_command(
                    ["git", "commit", "-m", commit_message],
                    repo_path,
                    "Creating commit",
                    allow_fail=True
                )
                if "nothing to commit" in commit_output.lower():
                    msg = "✅ No changes to commit. Working directory clean."
                    print(msg)
                    state["git_push_status"] = "clean"
                    state["git_push_message"] = msg
                    return state

            # Push changes
            print(f"🚀 Pushing to remote branch: {new_branch}")
            self._run_git_command(
                ["git", "push", "-u", "origin", new_branch],
                repo_path,
                f"Pushing to remote branch {new_branch}"
            )

            success_msg = f"✅ Successfully committed and pushed changes to branch: {new_branch}"
            print(success_msg)
            state["git_push_status"] = "success"
            state["git_push_message"] = success_msg
            return state

        except Exception as e:
            print(traceback.format_exc())
            error_msg = f"❌ Exception during commit & push: {e}"
            state["git_push_status"] = "failed"
            state["git_push_message"] = error_msg
            return state

    def invoke(self, input: Dict[str, Any],config: Any = None, **kwargs) -> Dict[str, Any]:
        """
        Adapter for LangGraph to pass state correctly to _run().
        """
        return self._run(input, **kwargs)

    def _configure_git_credentials(self, username, token, repo_url, repo_path):
        """
        Configure Git remote URL with username and token.
        """
        if not repo_url.startswith("https://"):
            raise ValueError("⚠️ Only HTTPS GitHub URLs are supported for token authentication.")
        auth_repo_url = repo_url.replace("https://", f"https://{username}:{token}@")

        print("🔗 Updating Git remote URL with credentials...")
        self._run_git_command(
            ["git", "remote", "set-url", "origin", auth_repo_url],
            repo_path,
            "Updating remote URL"
        )

    def _configure_git_identity(self, state: Dict[str, Any], repo_path: str):
        """
        Ensure git user.name and user.email are set for the repository.
        """
        user_name = state.get("git_user_name", "Agentic Bot")
        user_email = state.get("git_user_email", "agentic@example.com")

        print(f"👤 Setting Git user.name='{user_name}' and user.email='{user_email}'")
        self._run_git_command(["git", "config", "user.name", user_name], repo_path, "Configuring git user.name")
        self._run_git_command(["git", "config", "user.email", user_email], repo_path, "Configuring git user.email")

    def _is_repo_empty(self, repo_path: str) -> bool:
        """
        Check if the Git repository is empty (no commits yet).
        """
        try:
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_path,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            return False  # Repo has commits
        except subprocess.CalledProcessError:
            return True  # Repo is empty

    def _run_git_command(self, command, cwd, action="Git command", allow_fail=False):
        """
        Helper to run a Git command.
        """
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            print(f"✅ {action} succeeded:\n{result.stdout.strip()}")
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f"❌ {action} failed:\n{e.stderr.strip()}")
            if allow_fail:
                return e.stderr.strip()
            raise

    def _arun(self, *args, **kwargs):
        raise NotImplementedError("Async execution is not supported.")
