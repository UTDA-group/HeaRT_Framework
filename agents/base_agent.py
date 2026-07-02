from openai import OpenAI
# from anthropic import Anthropic  # uncomment for open-source
# from together import Together    # uncomment for open-source
# from google import genai         # uncomment for open-source
# from google.genai import types as genai_types  # uncomment for open-source
import os
import json
from typing import Callable, Dict, List, Any


class Agent:
    def __init__(
        self,
        name: str,
        tools: List[dict],
        available_functions: Dict[str, Callable],
        system_prompt: str,
        model: str = "gpt-4o",
    ):
        self.name = name
        # self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # uncomment for open-source
        # self.anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))  # uncomment for open-source
        # self.together_client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))  # uncomment for open-source
        # self.deepseek_v3_client = OpenAI(  # uncomment for open-source
        #     api_key=os.environ.get("DEEPSEEK_API_KEY"),
        #     base_url="https://api.deepseek.com",
        # )
        # self.gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))  # uncomment for open-source
        # NOTE: NVIDIA proxy client — delete before open-sourcing
        self.nvidia_client = OpenAI(
            base_url=os.environ.get("NVIDIA_OPENAI_BASE_URL") or "https://inference-api.nvidia.com/v1",
            api_key=os.environ.get("NVIDIA_API_KEY"),
        )

        self.system_prompt = system_prompt
        self.messages = [{"role": "system", "content": system_prompt}]
        self.tools = tools
        self.available_functions = available_functions
        self.model = model
        self.last_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def _execute_tool_call_recommended_by_assistant_and_get_outputs(
        self, tool_call
    ) -> Dict[str, Any]:
        func_name = tool_call.function.name
        func_args = json.loads(tool_call.function.arguments)

        print(f"[{self.name}] Calling function '{func_name}' with arguments: {func_args}")

        func_to_call = self.available_functions.get(func_name)

        if func_to_call is None:
            print(f"Function '{func_name}' is not available.")
            return {"role": "tool", "tool_call_id": tool_call.id, "content": "null"}

        try:
            output = func_to_call(**func_args)
            if isinstance(output, (dict, list)):
                output = json.dumps(output)
            elif not isinstance(output, str):
                output = str(output)
        except Exception as e:
            print(f"[{self.name}] Error calling '{func_name}': {e}")
            output = None

        return {"role": "tool", "tool_call_id": tool_call.id, "content": output}

    def _is_gemini_model(self) -> bool:
        m = self.model.lower()
        return m.startswith("gemini") or "gemini" in m

    def _is_deepseek_model(self) -> bool:
        return self.model.lower().startswith("deepseek")

    def _is_openai_model(self) -> bool:
        m = self.model.lower()
        return m.startswith("gpt-") or m.startswith("o1") or m.startswith("o3") or m.startswith("azure/openai/")

    def _is_llama_model(self) -> bool:
        m = self.model.lower()
        return "llama-3.3-70b" in m or m.startswith("meta-llama/")

    def _is_claude_model(self) -> bool:
        return self.model.lower().startswith("claude") or "anthropic" in self.model.lower()

    def _run_gemini(self, user_message: str) -> str:
        resp = self.gemini_client.models.generate_content(
            model=self.model,  # e.g. "gemini-2.5-pro"
            contents=user_message,
            config=genai_types.GenerateContentConfig(system_instruction=self.system_prompt),
        )
        text = resp.text
        in_tok = getattr(resp.usage_metadata, "prompt_token_count", 0) or 0
        out_tok = getattr(resp.usage_metadata, "candidates_token_count", 0) or 0
        self.last_usage = {
            "prompt_tokens": in_tok,
            "completion_tokens": out_tok,
            "total_tokens": in_tok + out_tok,
        }
        return text

    def _run_deepseek(self, user_message: str, temperature: float = 0.03) -> str:
        self.deepseek_messages = [{"role": "system", "content": self.system_prompt}]
        self.deepseek_messages.append({"role": "user", "content": user_message})
        resp = self.deepseek_v3_client.chat.completions.create(
            model=self.model,
            messages=self.deepseek_messages,
            stream=False,
        )
        usage = getattr(resp, "usage", None)
        if usage:
            self.last_usage = {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                "total_tokens": getattr(usage, "total_tokens", 0) or 0,
            }
        return resp.choices[0].message.content

    def _run_llama(self, user_message: str, temperature: float = 0.03) -> str:
        self.llama_messages = [{"role": "system", "content": self.system_prompt}]
        self.llama_messages.append({"role": "user", "content": user_message})
        resp = self.together_client.chat.completions.create(
            model=self.model,
            messages=self.llama_messages,
            max_tokens=20000,
            temperature=temperature,
        )
        usage = getattr(resp, "usage", None)
        if usage:
            self.last_usage = {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                "total_tokens": getattr(usage, "total_tokens", 0) or 0,
            }
        return resp.choices[0].message.content

    def _run_claude(self, user_message: str, temperature: float = 0.8) -> str:
        # -------------------------------------------------------------------------
        # OPEN-SOURCE NOTE: The block below (until END NVIDIA BLOCK) uses NVIDIA's
        # internal OpenAI-compatible proxy to call Claude and should be DELETED
        # before open-sourcing. Replace with the commented-out Anthropic SDK block
        # beneath it, which is the correct open-source implementation.
        # -------------------------------------------------------------------------
        # BEGIN NVIDIA BLOCK
        # _NVIDIA_CLAUDE_MODEL = "azure/anthropic/claude-sonnet-4-6"
        _NVIDIA_CLAUDE_MODEL = "azure/anthropic/claude-opus-4-8"
        # _NVIDIA_CLAUDE_MODEL = "azure/openai/gpt-5.5"
        resp = self.nvidia_client.chat.completions.create(
            model=_NVIDIA_CLAUDE_MODEL,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=20000,
            temperature=temperature,
        )
        text = resp.choices[0].message.content.strip()
        usage = getattr(resp, "usage", None)
        self.last_usage = {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "total_tokens": getattr(usage, "total_tokens", 0) or 0,
        }
        return text
        # END NVIDIA BLOCK

        # -------------------------------------------------------------------------
        # OPEN-SOURCE IMPLEMENTATION (restore this when deleting the NVIDIA block):
        # -------------------------------------------------------------------------
        # self.claude_messages = [
        #     {"role": "user", "content": [{"type": "text", "text": user_message}]}
        # ]
        # resp = self.anthropic_client.messages.create(
        #     model=self.model,
        #     system=self.system_prompt,
        #     messages=self.claude_messages,
        #     max_tokens=20000,
        #     temperature=temperature,
        # )
        # text = "".join(
        #     block.text
        #     for block in resp.content
        #     if getattr(block, "type", None) == "text"
        # ).strip()
        # in_tok = getattr(resp.usage, "input_tokens", 0) or 0
        # out_tok = getattr(resp.usage, "output_tokens", 0) or 0
        # self.last_usage = {
        #     "prompt_tokens": in_tok,
        #     "completion_tokens": out_tok,
        #     "total_tokens": in_tok + out_tok,
        # }
        # self.claude_messages.append(
        #     {"role": "assistant", "content": [{"type": "text", "text": text}]}
        # )
        # return text

    def _run_openai_via_nvidia(self, user_message: str, temperature: float = 0.8) -> str:
        # Route GPT models through NVIDIA proxy (OpenAI-compatible)
        nvidia_model = self.model if self.model.startswith("azure/") else f"azure/openai/{self.model}"
        resp = self.nvidia_client.chat.completions.create(
            model=nvidia_model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=16000,
            temperature=temperature,
        )
        text = resp.choices[0].message.content.strip()
        usage = getattr(resp, "usage", None)
        self.last_usage = {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "total_tokens": getattr(usage, "total_tokens", 0) or 0,
        }
        return text

    def run(
        self, user_message: str, temperature: float = 0.8, reasoning_effort="medium"
    ) -> str:
        """
        Run the agent on a user message. Handles multi-turn tool calls automatically.
        """
        print()
        self.messages.append({"role": "user", "content": user_message})

        if self._is_claude_model():
            if self.tools:
                raise NotImplementedError(
                    "Claude tool-calling path not wired in this agent yet."
                )
            return self._run_claude(user_message=user_message, temperature=temperature)

        if self._is_openai_model():
            return self._run_openai_via_nvidia(user_message=user_message, temperature=temperature)

        if self._is_deepseek_model():
            return self._run_deepseek(user_message=user_message, temperature=temperature)

        if self._is_llama_model():
            return self._run_llama(user_message=user_message, temperature=temperature)

        if self._is_gemini_model():
            return self._run_gemini(user_message=user_message)

        while True:
            run = self.openai_client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self.tools,
                tool_choice="auto",
            )

            usage = getattr(run, "usage", None)
            if usage is not None:
                self.last_usage = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                    "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                    "total_tokens": getattr(usage, "total_tokens", 0) or 0,
                }

            msg = run.choices[0].message
            finish_reason = run.choices[0].finish_reason

            if finish_reason == "tool_calls":
                tool_calls = msg.tool_calls
                tool_outputs = list(map(
                    self._execute_tool_call_recommended_by_assistant_and_get_outputs,
                    tool_calls,
                ))
                self.messages.append(msg)
                self.messages.extend(tool_outputs)
                continue
            else:
                self.messages.append(msg)
                return msg.content
