from langchain.tools import BaseTool
from typing import Dict, Any
from pydantic import PrivateAttr
import time
import traceback


class RetryAgent(BaseTool):
    """
    Retry wrapper for agents. Retries the target agent if it fails.
    """

    name: str = "retry_agent"
    description: str = "Retries the target agent up to N times on failure."

    # 👇 Mark target_agent as a private attribute
    _target_agent: BaseTool = PrivateAttr()
    _max_retries: int = PrivateAttr(default=1)
    _retry_delay: float = PrivateAttr(default=2.0)
    _failure_key: str = PrivateAttr(default="status")
    _success_value: Any = PrivateAttr(default="success")

    def __init__(self, target_agent: BaseTool, max_retries=3, retry_delay=2.0,
                 failure_key="status", success_value="success", **kwargs):
        """
        Initialize the retry agent.

        Args:
            target_agent (BaseTool): The agent to retry on failure.
            max_retries (int): Number of retry attempts.
            retry_delay (float): Delay between retries in seconds.
            failure_key (str): State key to check for success/failure.
            success_value (Any): Expected success value for the failure_key.
            kwargs: Additional BaseTool parameters.
        """
        super().__init__(**kwargs)
        self._target_agent = target_agent
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._failure_key = failure_key
        self._success_value = success_value

    def _run(self, input: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Retry logic for the wrapped agent.

        Args:
            input (Dict[str, Any]): Workflow state.

        Returns:
            Dict[str, Any]: Updated workflow state.
        """
        state = input  # 👈 Align with LangGraph tool input
        attempt = 0
        while attempt < self._max_retries:
            print(f"🔁 RetryAgent attempt {attempt + 1}/{self._max_retries}...")
            try:
                state = self._target_agent.invoke(state)
                if state.get(self._failure_key) == self._success_value:
                    print(f"✅ Success on attempt {attempt + 1}.")
                    return state
                else:
                    print(f"⚠️ Attempt {attempt + 1} failed. Retrying...")
            except Exception as e:
                print(f"❌ Exception in RetryAgent attempt {attempt + 1}: {e}")
                print(traceback.format_exc())

            attempt += 1
            time.sleep(self._retry_delay)

        # Mark as failed after retries exhausted
        print(f"❌ All {self._max_retries} attempts failed.")
        state.update({
            self._failure_key: "failed",
            "retry_status": f"Failed after {self._max_retries} retries"
        })
        return state

    def invoke(self, input: Dict[str, Any], config: Any = None,**kwargs) -> Dict[str, Any]:
        """
        Adapter for LangGraph to call _run with workflow state.
        """
        return self._run(input, **kwargs)

    def _arun(self, *args, **kwargs):
        raise NotImplementedError("Async execution is not supported.")
