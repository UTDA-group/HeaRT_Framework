import re
import json
import sys
import os
import argparse
from typing import List

# Add parent directory (Multi_Agent) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from base_agent import Agent
from contexts.local_context import LocalContext
from contexts.global_context import GlobalContext

from agents.prompts import HIERARCHICAL_AGENT_SYSTEM_PROMPT
working_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Start with the generic prompt above. The XG Boosting within the Iterative Refinement will make it more and more subcircuit specific, and make it simulation-grounded.


class HierarchicalAgent(Agent):

    def __init__(
        self,
        name,
        system_prompt,
        tools,
        available_functions,
        top_level_port_names,
        toplevelcircuit: str,
        model="gpt-5",  # gpt-4o
    ):
        super().__init__(name, tools, available_functions, system_prompt, model)
        self.local_context = LocalContext(name)
        self.top_level_port_names = top_level_port_names
        self.toplevelcircuit = toplevelcircuit
        self.device_list = {}

    def _safe_parse_json(self, text: str):
        # Match triple backticks optionally followed by 'json' and capture the content inside
        match = re.search(r"```(?:[a-zA-Z]+)?\s*([\s\S]*?)\s*```", text)
        if match:
            json_text = match.group(1)
            return json.loads(json_text)

        # Otherwise try extracting first {...} JSON block
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            return json.loads(match.group(0))
        return json.loads(text)

    def hierarchical_agglomeration_tree(
        self, subcircuits: List[dict], system_prompt: str = None
    ) -> dict:
        """
        Build hierarchical tree of circuit based on global context.
        Input: list of subcircuits (filtered from global context)
        Output: hierarchical JSON tree
        """

        if system_prompt is None:  # Nothing provided
            system_prompt = self.system_prompt

        for message in self.messages:
            if message["role"] == "system":
                message["content"] = system_prompt
                break

        user_message = f"""
            Top-level Circuit Netlist:
            {self.toplevelcircuit}

            Identified Subcircuits (Leaf Nodes):
            {json.dumps(subcircuits, indent=2)}
            
        """

        output_format_prompt = """

            Please output the result strictly as a JSON tree.
            Each node in the tree must follow this schema:

            {
                "unique_name": "<string, name of the node>",
                "children": [ ... ],   // list of child nodes using the same schema recursively
            }

        """

        response = self.run(user_message + output_format_prompt)

        try:
            summary = self._safe_parse_json(response)
        except Exception:
            summary = {"raw_output": response}

        return summary


def main():
    parser = argparse.ArgumentParser(description="Hierarchical agglomeration tree builder")
    parser.add_argument("--circuit", required=True, help="Circuit ID from benchmark_circuits.json (e.g. ckt1)")
    args = parser.parse_args()

    benchmark_path = os.path.join(parent_dir, "benchmark_circuits.json")
    with open(benchmark_path) as f:
        benchmark = json.load(f)

    if args.circuit not in benchmark:
        print(f"Unknown circuit '{args.circuit}'. Available: {sorted(benchmark.keys())}")
        sys.exit(1)

    netlist_filepath = os.path.join(parent_dir, benchmark[args.circuit]["netlist"])
    parent_netlist_path = os.path.dirname(os.path.dirname(netlist_filepath))

    # Load full netlist text
    with open(netlist_filepath, "r") as f:
        full_netlist_content = f.read()

    top_level_netlist_filename = os.path.basename(netlist_filepath)
    top_level_circuit_name = os.path.splitext(top_level_netlist_filename)[0]

    generated_subcircuits_path = os.path.join(parent_netlist_path, "Generated_Subcircuits")

    circuit_global_context = GlobalContext()
    circuit_global_context.load(
        os.path.join(generated_subcircuits_path, "circuit_global_context.json")
    )

    global_ctx_data = circuit_global_context._store
    global_subcircuits_list = global_ctx_data.get("subcircuits", [])

    hierarchical_agent = HierarchicalAgent(
        name=top_level_circuit_name,
        system_prompt=HIERARCHICAL_AGENT_SYSTEM_PROMPT,
        tools=[],
        available_functions={},
        top_level_port_names=global_ctx_data.get("top_level_port_names", []),
        toplevelcircuit=full_netlist_content,
        model="claude-opus-4-8",
        # model="deepseek-reasoner",
        # model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    )

    generated_response = hierarchical_agent.hierarchical_agglomeration_tree(
        global_subcircuits_list
    )

    print(json.dumps(generated_response, indent=2))
    output_path = os.path.join(generated_subcircuits_path, "hierarchical_agglomeration_tree.json")
    with open(output_path, "w") as f:
        json.dump(generated_response, f, indent=2)
    print(f"Hierarchical tree saved to {output_path}")
    tokens = hierarchical_agent.last_usage.get("total_tokens", 0)
    print(f"Total tokens: {tokens}")


if __name__ == "__main__":
    main()
