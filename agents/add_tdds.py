from langchain.tools import BaseTool
from typing import Dict, Any, Tuple, List
from pydantic import PrivateAttr
import os
import traceback
import re


class GenerateTestsAgent(BaseTool):
    """
    LangChain Agent Tool for StateGraph workflows.
    Generates unit tests for a Python project using an LLM and updates workflow state.
    """

    name: str = "generate_tests"
    description: str = (
        "Generates unit tests for a Python project using an LLM and "
        "updates the workflow state with status and test file paths."
    )

    _llm: Any = PrivateAttr(default=None)

    def __init__(self, llm: Any = None, **kwargs: Any):
        super().__init__(**kwargs)
        self._llm = llm

    def _run(self, input: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        state: Dict[str, Any] = input
        try:
            llm = self._llm

            if llm is None:
                error_msg = "❌ No LLM instance provided to GenerateTestsAgent."
                print(error_msg)
                return self._update_state(state, "failed", error_msg)

            # 📄 Extract project info
            generated_code: str = state.get("generated_code", "")
            repo_path: str = state.get("repo_path", "./local_repo")
            project_name: str = state.get("project_name", "my_package")
            module_name: str = state.get("module_name", "main").replace(".py", "")  # Remove .py if present
            tests_path: str = os.path.join(repo_path, "tests")
            improve_existing: bool = state.get("improve_existing_tests", False)

            if not generated_code.strip():
                error_msg = "❌ No generated code found in state for test creation."
                print(error_msg)
                return self._update_state(state, "failed", error_msg)

            # 🧠 Extract public API
            public_functions, public_classes = self._extract_public_functions_and_classes(generated_code)

            if not public_functions and not public_classes:
                error_msg = (
                        "❌ No public functions or classes found in generated code. "
                        "Here’s the code I analyzed:\n"
                        + "─" * 60 + "\n"
                        + f"{generated_code}\n" + "─" * 60
                )
                print(error_msg)
                return self._update_state(state, "failed", error_msg)

            # 🗂 Compute import path for the module
            module_import_path, sys_path_insert = self._build_module_import_path(repo_path, project_name, module_name)

            # 🚨 Import only top-level functions and classes (not methods!)
            import_members = public_functions + list(public_classes.keys())

            print(f"🧪 Generating tests for public API using LLM...")

            # 📝 Build strict prompt
            llm_prompt: str = self._build_prompt(
                generated_code,
                module_import_path,
                sys_path_insert,
                import_members,
                public_functions,
                public_classes,
                improve_existing
            )

            # 🔥 Send to LLM
            response = llm.invoke(llm_prompt)
            tests_content: str = getattr(response, "content", str(response)).strip()

            # 📝 Log raw LLM output for debugging
            print("\n📄 Raw LLM Response (before cleanup):")
            print("─" * 60)
            print(tests_content)
            print("─" * 60)

            # 🧹 Clean and validate LLM response
            clean_tests: str = self._clean_llm_code_response(tests_content)

            if not clean_tests or len(clean_tests.splitlines()) < 2:
                fail_msg = "❌ LLM returned invalid or empty test code."
                print(fail_msg)
                return self._update_state(state, "failed", fail_msg)

            if self._validate_python_code(clean_tests):
                # 📝 Always write to test_<module_name>.py (overwrite if exists)
                os.makedirs(tests_path, exist_ok=True)
                test_file_path = os.path.join(tests_path, f"test_{module_name}.py")

                with open(test_file_path, "w", encoding="utf-8") as f:
                    f.write(clean_tests)

                success_msg = f"✅ Test file created at: {test_file_path}"
                print(success_msg)
                return self._update_state(state, "success", success_msg, clean_tests, test_file_path)
            else:
                fail_msg = "❌ Syntax error in generated test code. LLM must produce valid Python code."
                print(fail_msg)
                return self._update_state(state, "failed", fail_msg)

        except Exception as e:
            exception_msg = f"❌ Exception occurred: {e}"
            print(traceback.format_exc())
            return self._update_state(state, "failed", exception_msg)

    def _build_module_import_path(self, repo_path: str, project_name: str, module_name: str) -> Tuple[str, str]:
        """
        Build the Python import path for the module and the sys.path.insert string.
        """
        sys_path_insert: str = (
            "sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))"
        )
        module_import_path: str = f"src.{project_name}.{module_name}"
        return module_import_path, sys_path_insert

    def _build_prompt(
            self,
            generated_code: str,
            module_import_path: str,
            sys_path_insert: str,
            import_members: List[str],
            functions: List[str],
            classes: Dict[str, Dict],
            improve_existing: bool
    ) -> str:
        """
        Build a strict LLM prompt for test generation with proper imports.
        """
        prompt: str = (
            f"You are a **senior Python testing expert**.\n"
            f"Write **pytest-based unit tests** for the following Python module.\n"
            f"\n📦 Module Code:\n{generated_code}\n\n"
            "🎯 OBJECTIVES:\n"
            "- ✅ Write tests for **all public functions and class methods**.\n"
            "- ✅ Instantiate classes properly for testing their methods (e.g., `sut = ClassName()`).\n"
            "- ✅ Include positive and negative test cases.\n"
            "- ✅ Use type-safe assertions:\n"
            "    - Use `math.isclose()` for comparing floating-point values.\n"
            "    - Use `math.isnan(x)` for NaN comparisons.\n"
            "    - Use `math.isinf(x)` for Infinity comparisons.\n"
            "    - Use `==` for all other types (int, str, bool, etc).\n"
            "- ✅ Ensure clean, PEP8-compliant code with required imports only.\n"
            "- ✅ At the top of the file, add these imports:\n"
            "import sys, os, math\n"
            f"{sys_path_insert}\n"
            "import pytest\n"
            f"from {module_import_path} import {', '.join(import_members)}\n\n"
            "⚠️ STRICT RULES:\n"
            "- 🚫 DO NOT include markdown, explanations, or comments.\n"
            "- ✅ Return **only valid Python code**.\n"
            "- 🚫 If you cannot generate syntactically valid Python code, return an **empty response**.\n"
        )

        if improve_existing:
         prompt += (
        "\n🆙 Existing tests detected. Improve them by:\n"
        "- ✅ Fixing **only the failed test cases** by analyzing the failure traceback.\n"
        "- ✅ Modify the implementation code in the module if required to ensure the failed tests pass.\n"
        "- 🚫 Do NOT rewrite or modify any test cases that are already passing.\n"
        "- 🚫 Do NOT remove any existing test cases.\n"
        "- ⚠️ If a failed test involves the `main()` function or another critical entry point that cannot be tested:\n"
        "    - 🟡 Add a `pytest.mark.skip(reason='main() failed')` decorator to skip it gracefully.\n"
        "- ✅ Return **only the updated Python test file** with fixes for failed tests.\n"
        "- 🚨 STRICT RULES:\n"
        "    - Only modify failed test cases.\n"
        "    - Leave passing tests untouched.\n"
        "    - Ensure the module implementation changes (if any) do not break existing passing tests.\n"
        "    - Return valid Python code without markdown, explanations, or comments.\n"
    )


        return prompt

    def _extract_public_functions_and_classes(self, code: str) -> Tuple[List[str], Dict[str, Dict]]:
        """
        Extract public functions and classes + their public methods.
        """
        all_functions: List[str] = re.findall(r"^def (\w+)\(", code, flags=re.MULTILINE)
        public_functions: List[str] = [f for f in all_functions if not f.startswith("_")]

        class_pattern = re.compile(r"class (\w+)\s*\(?.*?\)?:")
        method_pattern = re.compile(r"^\s+def (\w+)\(", flags=re.MULTILINE)
        public_classes: Dict[str, Dict] = {}

        for class_match in class_pattern.finditer(code):
            class_name = class_match.group(1)
            if class_name.startswith("_"):
                continue  # Skip private classes

            class_body: str = self._get_class_body(code, class_name)
            methods: List[str] = method_pattern.findall(class_body)
            public_methods: List[str] = [m for m in methods if not m.startswith("_")]
            public_classes[class_name] = {"methods": public_methods}

        return public_functions, public_classes

    def _get_class_body(self, code: str, class_name: str) -> str:
        pattern: str = rf"class {class_name}\(.*?\):([\s\S]+?)(?=^class |\Z)"
        match = re.search(pattern, code, flags=re.MULTILINE)
        return match.group(1) if match else ""

    def _clean_llm_code_response(self, llm_text: str) -> str:
        """
        Clean up LLM response and extract only valid Python code.
        """
        code_match = re.search(r"```(?:python)?\s*(.*?)```", llm_text, re.DOTALL | re.IGNORECASE)
        if code_match:
            return code_match.group(1).strip()
        return llm_text.strip()

    def _validate_python_code(self, code: str) -> bool:
        """
        Validate Python code syntax.
        """
        try:
            compile(code, "<string>", "exec")
            return True
        except Exception as e:
            print(f"\n❌ Syntax error detected in cleaned LLM code:\n{e}")
            return False

    def _update_state(
            self,
            state: Dict[str, Any],
            status: str,
            message: str,
            code: str = None,
            path: str = None
    ) -> Dict[str, Any]:
        state.update({
            "status": status,
            "test_generation_message": message
        })
        if code:
            state["tests_content"] = code
        if path:
            state["test_file_path"] = path
        return state

    def invoke(self, input: Dict[str, Any], config: Any = None, **kwargs: Any) -> Dict[str, Any]:
        return self._run(input, **kwargs)

    def _arun(self, *args: Any, **kwargs: Any):
        raise NotImplementedError("Async execution is not supported.")
