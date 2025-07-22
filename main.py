from langgraph.graph import StateGraph
from pydantic import BaseModel
from typing import Dict, Any
import yaml
import sys

# 🧠 Import agents
from agents.check_remote_repo_exits import CheckRemoteRepoExistsAgent
from agents.create_remote_repo import CreateRemoteRepoTool
from agents.clone_remote_repo import CloneNewRepoAgent
from agents.generate_code import GeneratePythonCodeAgent
from agents.add_tdds import GenerateTestsAgent
from agents.add_readme import GenerateReadmeAgent
from agents.commit_push import GitCommitPushAgent
from agents.coverage_checker import GenerateCoverageAgent
from agents.RetryAgent import RetryAgent
from llm_utils import get_llm


# ✅ Define WorkflowState
class WorkflowState(BaseModel):
    username: str
    token: str
    user_email: str
    repo_url: str
    branch: str
    new_branch: str
    new_repo_name: str
    repo_path: str
    project_name: str
    module_name: str
    code_prompt: str
    branch_prefix: str
    tdd_coverage: int
    diagram_format: str
    code_style: str
    repo_check_status: str = "pending"
    repo_check_message: str = ""
    repo_creation_status: str = "pending"
    clone_status: str = "pending"
    code_generation_status: str = "pending"
    test_generation_status: str = "pending"
    readme_status: str = "pending"
    git_push_status: str = "pending"
    coverage_status: str = "pending"
    workflow_status: str = "in_progress"


def build_graph(llm) -> StateGraph:
    """
    Builds the workflow graph with all agents and retry logic.
    """
    graph = StateGraph(state_schema=dict)

    # ✅ Core agents
    check_remote_repo_agent = CheckRemoteRepoExistsAgent()
    create_remote_repo_agent = CreateRemoteRepoTool()
    clone_new_repo_agent = CloneNewRepoAgent()
    generate_code_agent = GeneratePythonCodeAgent(llm=llm)
    generate_tests_agent = GenerateTestsAgent(llm=llm)
    generate_coverage_agent = GenerateCoverageAgent()
    generate_readme_agent = GenerateReadmeAgent(llm=llm)
    git_commit_push_agent = GitCommitPushAgent()

    # 🔁 Wrap retryable agents
    retry_generate_code_agent = RetryAgent(
        target_agent=generate_code_agent,
        max_retries=3,
        retry_delay=2.0,
        failure_key="code_generation_status",
        success_value="success",
        name="retry_generate_code"
    )

    retry_generate_tests_agent = RetryAgent(
        target_agent=generate_tests_agent,
        max_retries=3,
        retry_delay=2.0,
        failure_key="test_generation_status",
        success_value="success",
        name="retry_generate_tests"
    )

    # 🪝 Add all agents as nodes
    graph.add_node("check_remote_repo", check_remote_repo_agent)
    graph.add_node("create_remote_repo", create_remote_repo_agent)
    graph.add_node("clone_new_repo", clone_new_repo_agent)
    graph.add_node("generate_code", retry_generate_code_agent)
    graph.add_node("generate_tests", retry_generate_tests_agent)
    graph.add_node("check_coverage", generate_coverage_agent)
    graph.add_node("generate_readme", generate_readme_agent)
    graph.add_node("git_commit_push", git_commit_push_agent)

    # 🛤 Add routing logic
    def repo_check_router(state: Dict[str, Any]) -> str:
        if state.get("repo_check_status") == "success":
            return "clone_new_repo"
        elif state.get("repo_check_status") in ["branch_not_found", "failed"]:
            return "create_remote_repo"
        else:
            raise ValueError(f"Unknown repo_check_status: {state.get('repo_check_status')}")

    def coverage_check_router(state: Dict[str, Any]) -> str:
        if state.get("coverage_status") in ["below_threshold", "failed"]:
            return "generate_tests"
        elif state.get("coverage_status") == "success":
            return "generate_readme"
        else:
            raise ValueError(f"Unknown coverage_status: {state.get('coverage_status')}")

    graph.add_conditional_edges("check_remote_repo", repo_check_router)
    graph.add_edge("create_remote_repo", "clone_new_repo")
    graph.add_edge("clone_new_repo", "generate_code")
    graph.add_edge("generate_code", "generate_tests")
    graph.add_edge("generate_tests", "check_coverage")#check_coverage
    #graph.add_conditional_edges("check_coverage", coverage_check_router)
    graph.add_edge("check_coverage", "generate_readme")
    graph.add_edge("generate_readme", "git_commit_push")

    # 🚀 Set entry and exit points
    graph.set_entry_point("check_remote_repo")
    graph.set_finish_point("git_commit_push")

    return graph.compile()


if __name__ == "__main__":
    try:
        # 📥 Ask user for prompt
        user_prompt = input("💡 Enter the code requirement (prompt): ").strip()
        if not user_prompt:
            print("❌ Prompt cannot be empty. Exiting.")
            sys.exit(1)

        # 📖 Load config
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)

        # 🧠 Initialize shared LLM
        llm = get_llm(config)

        # 🏁 Prepare plain dict for state
        initial_state = {
            "username": config["github"]["username"],
            "token": config["github"]["token"],
            "user_email": config["github"]["user_email"],
            "repo_url": config["github"]["repo_url"],
            "branch": config["github"]["branch"],
            "new_branch": config["github"]["new_branch"],
            "new_repo_name": config["github"]["new_repo_name"],
            "repo_path": config["project"]["repo_path"],
            "project_name": config["project"]["project_name"],
            "module_name": config["project"]["module_name"],
            "code_prompt": user_prompt,
            "branch_prefix": config["settings"]["branch_prefix"],
            "tdd_coverage": config["settings"]["tdd_coverage"],
            "diagram_format": config["settings"]["diagram_format"],
            "code_style": config["settings"]["code_style"]
        }

        # 🛠 Build workflow graph
        workflow_graph = build_graph(llm=llm)

        # 🚀 Invoke workflow
        print("\n🚀 Starting workflow...\n")
        for step in workflow_graph.stream(initial_state):
            print(f"🔄 Step: {step.get('node')}")
            print(f"📦 State: {step.get('state')}")

        print("\n🎉 Workflow finished successfully!")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
