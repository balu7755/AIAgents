from langchain.tools import BaseTool
from typing import Dict, Any, List
from pydantic import PrivateAttr
import os
import traceback
import re
import subprocess
import sys
import pkgutil


class GeneratePythonCodeAgent(BaseTool):
    """
    LangChain Agent Tool for StateGraph workflows.
    Generates a Python package project using an LLM, validates syntax,
    sets up a virtual environment, detects third-party dependencies,
    and configures IntelliJ project structure (with test runner).
    """

    name: str = "generate_python_code"
    description: str = (
        "Generates Python code using an LLM prompt, validates syntax, "
        "creates a Python package structure, detects dependencies, "
        "sets up a virtual environment, and configures IntelliJ project structure."
    )

    _llm: Any = PrivateAttr(default=None)

    def __init__(self, llm=None, **kwargs):
        """
        Initialize the agent.

        Args:
            llm: Any LLM instance that implements `.invoke(prompt)`.
            kwargs: Additional BaseTool parameters.
        """
        super().__init__(**kwargs)
        self._llm = llm
        self._stdlib_modules = set(sys.builtin_module_names).union(
            {module.name for module in pkgutil.iter_modules()}
        )

    def invoke(self, input: Dict[str, Any], config: Any = None, **kwargs) -> Dict[str, Any]:
        """
        Adapter for LangGraph or external callers to execute the agent
        with the given workflow state.
        """
        print("⚙️ Invoking GeneratePythonCodeAgent...")
        return self._run(input, **kwargs)

    def _run(self, input: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Run the agent to generate and save Python code as a package.

        Args:
            input (Dict[str, Any]): Current workflow state.

        Returns:
            Dict[str, Any]: Updated workflow state with status and results.
        """
        state = input
        try:
            if self._llm is None:
                return self._fail_state(state, "❌ No LLM instance provided to GeneratePythonCodeAgent.")

            prompt = state.get("code_prompt")
            project_name = state.get("project_name", "my_project")
            module_name = state.get("module_name", "main")
            repo_path = state.get("repo_path", "./local_repo")
            package_path = os.path.join(repo_path, "src", project_name)
            output_file = os.path.join(package_path, f"{module_name}.py")

            if not prompt:
                return self._fail_state(state, "❌ No code generation prompt provided in state.")

            # 📁 Setup project structure
            self._create_project_structure(repo_path, project_name)

            # 🐍 Setup virtual environment
            venv_python = self._setup_virtualenv(repo_path)

            # 📝 Setup IntelliJ IDEA project structure (with test runner)
            self._setup_intellij_project_structure(repo_path, project_name, venv_python)

            # 📜 Prepare strict prompt
            llm_prompt = self._build_prompt(prompt, project_name)
            print(f"🚀 Sending prompt to LLM...")

            response = self._llm.invoke(llm_prompt)
            llm_text = getattr(response, "content", str(response)).strip()

            if not llm_text:
                return self._fail_state(state, "⚠️ LLM returned an empty response.")

            # 🧹 Clean and validate code
            generated_code = self._clean_llm_code_response(llm_text)
            if not generated_code or len(generated_code.splitlines()) < 2:
                return self._fail_state(state, "❌ LLM response does not contain valid Python code.", generated_code)

            # ✅ Add file path comment at the top
            generated_code = f"# {project_name}/{module_name}.py\n" + generated_code

            # ✅ Validate syntax and imports
            if self._validate_python_code(generated_code):
                self._write_file(output_file, generated_code)

                # 📦 Detect dependencies and write to requirements.txt
                dependencies = self._detect_third_party_imports(generated_code)
                self._write_requirements_file(repo_path, dependencies)

                success_msg = f"✅ Python package and IntelliJ project created at: {repo_path}"
                print(success_msg)
                return self._update_state(state, "success", success_msg, generated_code, output_file)
            else:
                return self._fail_state(state, "❌ Generated code contains syntax errors.", generated_code)

        except Exception as e:
            exception_msg = f"❌ Exception occurred: {e}"
            print(traceback.format_exc())
            return self._fail_state(state, exception_msg)

    def _fail_state(self, state: Dict[str, Any], error_msg: str, code: str = None) -> Dict[str, Any]:
        """
        Update state in case of failure.
        """
        print(error_msg)
        return self._update_state(state, "failed", error_msg, code)

    def _setup_virtualenv(self, repo_path: str) -> str:
        """
        Sets up a Python virtual environment in the project directory.

        Returns:
            str: Path to the Python executable inside the virtual environment.
        """
        venv_path = os.path.join(repo_path, "venv")
        print(f"📦 Creating virtual environment at: {venv_path}")
        try:
            subprocess.check_call([sys.executable, "-m", "venv", venv_path])
            print("✅ Virtual environment created successfully.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create virtual environment: {e}")
        python_executable = os.path.join(venv_path, "Scripts", "python.exe") if os.name == "nt" \
            else os.path.join(venv_path, "bin", "python")
        return python_executable

    def _setup_intellij_project_structure(self, repo_path: str, project_name: str, sdk_path: str):
        """
        Sets up IntelliJ project structure so it can be opened directly.

        Args:
            repo_path (str): Path to the root of the project.
            project_name (str): Name of the project/module.
            sdk_path (str): Path to Python interpreter (venv).
        """
        print("📝 Setting up IntelliJ IDEA project files...")
        idea_dir = os.path.join(repo_path, ".idea")
        run_configs_dir = os.path.join(idea_dir, "runConfigurations")
        os.makedirs(run_configs_dir, exist_ok=True)

        # Create .idea/misc.xml
        misc_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<project version="4">
  <component name="ProjectRootManager" version="2" project-jdk-name="{sdk_path}"
             project-jdk-type="Python SDK" languageLevel="Python 3.11" />
</project>
"""
        self._write_file(os.path.join(idea_dir, "misc.xml"), misc_xml)

        # Create .idea/modules.xml
        modules_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<project version="4">
  <component name="ProjectModuleManager">
    <modules>
      <module fileurl="file://$PROJECT_DIR$/{project_name}.iml"
              filepath="$PROJECT_DIR$/{project_name}.iml" />
    </modules>
  </component>
</project>
"""
        self._write_file(os.path.join(idea_dir, "modules.xml"), modules_xml)

        # Create project_name.iml file
        iml_file = f"""<?xml version="1.0" encoding="UTF-8"?>
<module type="PYTHON_MODULE" version="4">
  <component name="NewModuleRootManager">
    <content url="file://$MODULE_DIR$">
      <sourceFolder url="file://$MODULE_DIR$/src" isTestSource="false" />
      <sourceFolder url="file://$MODULE_DIR$/tests" isTestSource="true" />
    </content>
    <orderEntry type="jdk" jdkName="{sdk_path}" jdkType="Python SDK" />
    <orderEntry type="sourceFolder" forTests="false" />
  </component>
</module>
"""
        self._write_file(os.path.join(repo_path, f"{project_name}.iml"), iml_file)

        # Create runConfigurations/AllTests.xml for pytest
        all_tests_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<component name="ProjectRunConfigurationManager">
  <configuration default="false" name="All Tests" type="PythonTestConfigurationType" factoryName="py.test">
    <module name="{project_name}" />
    <option name="INTERPRETER_OPTIONS" value="" />
    <option name="WORKING_DIRECTORY" value="$PROJECT_DIR$" />
    <option name="TARGET" value="tests" />
    <option name="USE_MODULE_SDK" value="true" />
    <option name="TEST_TYPE" value="TEST_FOLDER" />
  </configuration>
</component>
"""
        self._write_file(os.path.join(run_configs_dir, "AllTests.xml"), all_tests_xml)

        print("✅ IntelliJ project and test runner configured.")

    def _create_project_structure(self, repo_path: str, package_name: str):
        """
        Create a Python project directory structure.
        """
        print("📁 Setting up Python project structure...")
        src_package_path = os.path.join(repo_path, "src", package_name)
        tests_path = os.path.join(repo_path, "tests")

        os.makedirs(src_package_path, exist_ok=True)
        os.makedirs(tests_path, exist_ok=True)

        # Create __init__.py files
        for path in [src_package_path, os.path.join(repo_path, "src"), tests_path]:
            self._write_file(os.path.join(path, "__init__.py"), "")

        # Add README.md
        readme_path = os.path.join(repo_path, "README.md")
        if not os.path.exists(readme_path):
            self._write_file(readme_path, f"# {package_name}\n\nAuto-generated Python package.\n")

    def _build_prompt(self, user_prompt: str, package_name: str) -> str:
        """
        Build a strict LLM prompt enforcing Python-only output.
        """
        return (
            f"You are a highly experienced Python developer.\n"
            f"Your task is to generate a **self-contained Python module** inside a package named '{package_name}'.\n"
            f"💡 Requirement:\n{user_prompt}\n\n"
            "🎯 OBJECTIVES:\n"
            "- ✅ Write clean, modular code using classes, methods, and functions (OOP where suitable).\n"
            "- ✅ Add file path comments at the top of each module.\n"
            "- ✅ Follow PEP8 style, use type hints, and include meaningful docstrings.\n"
            "- ✅ Ensure the code runs without modification.\n"
            "⚠️ STRICT RULES:\n"
            "- 🚫 No markdown, code fences, or explanations.\n"
            "- ✅ Return only valid Python code.\n"
        )

    def _clean_llm_code_response(self, llm_text: str) -> str:
        """
        Cleans LLM response by removing markdown markers and extra text.
        """
        llm_text = re.sub(r"^```(?:python)?", "", llm_text.strip(), flags=re.IGNORECASE | re.MULTILINE)
        llm_text = re.sub(r"```$", "", llm_text.strip(), flags=re.MULTILINE)
        return llm_text.strip()

    def _validate_python_code(self, code: str) -> bool:
        """
        Check if the given Python code is syntactically valid.
        """
        try:
            compile(code, "<string>", "exec")
            return True
        except (SyntaxError, IndentationError, UnicodeDecodeError) as e:
            print(f"❌ Syntax error in generated code: {e}")
            return False

    def _detect_third_party_imports(self, code: str) -> List[str]:
        """
        Detect third-party imports in the code (ignores standard library).
        """
        print("🔍 Detecting third-party imports...")
        imports = re.findall(r"^(?:from|import)\s+([a-zA-Z_][\w.]*)", code, re.MULTILINE)
        third_party = {module.split(".")[0] for module in imports if module.split(".")[0] not in self._stdlib_modules}
        for lib in third_party:
            print(f"➕ Detected third-party library: {lib}")
        return list(third_party)

    def _write_file(self, path: str, content: str):
        """
        Safely write content to a file.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    def _write_requirements_file(self, repo_path: str, dependencies: List[str]):
        """
        Write detected dependencies to requirements.txt.
        """
        requirements_path = os.path.join(repo_path, "requirements.txt")
        print(f"📜 Writing requirements.txt with dependencies...")
        self._write_file(requirements_path, "\n".join(dependencies))
        print(f"✅ requirements.txt created at: {requirements_path}")

    def _update_state(self, state: Dict[str, Any], status: str, message: str, code: str = None, path: str = None) -> Dict[str, Any]:
        """
        Update the workflow state with the given status and message.
        """
        state.update({
            "code_generation_status": status,
            "code_generation_message": message
        })
        if code:
            state["generated_code"] = code
        if path:
            state["generated_code_path"] = path
        return state

    def _arun(self, *args, **kwargs):
        raise NotImplementedError("Async execution is not supported.")
