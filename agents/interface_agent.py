import re
import json
import sys
import os
import argparse
import networkx as nx
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from base_agent import Agent



parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def add_edges_from_json(node, G, parent=None):
    node_name = node.get("unique_name", node.get("name", "Unnamed"))
    if parent:
        G.add_edge(parent, node_name)
    for child in node.get("children", []):
        add_edges_from_json(child, G, node_name)


def hierarchy_pos(G, root, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5):
    # From Joel's answer at https://stackoverflow.com/a/29597209/2966723
    def _hierarchy_pos(
        G, root, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None
    ):
        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.successors(root))
        if len(children) != 0:
            dx = width / len(children)
            nextx = xcenter - width / 2 - dx / 2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(
                    G,
                    child,
                    width=dx,
                    vert_gap=vert_gap,
                    vert_loc=vert_loc - vert_gap,
                    xcenter=nextx,
                    pos=pos,
                    parent=root,
                )
        return pos

    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)


def plot_graph(G, root, width=1.5, vert_gap=0.2, figsize=(12, 8)):
    pos = hierarchy_pos(G, root, width=width, vert_gap=vert_gap)
    plt.figure(figsize=figsize)
    nx.draw(
        G,
        pos=pos,
        with_labels=True,
        arrows=True,
        node_size=1200,
        node_color="#A0CBE2",
        font_size=9,
    )
    plt.title("Hierarchical Circuit Tree")
    plt.show()


def extract_subtree(root, target_name):
    if root.get("unique_name") == target_name:
        return root
    for child in root.get("children", []):
        result = extract_subtree(child, target_name)
        if result:
            return result
    return None


def reasoning_guided_tree_scope(
    query, tree, llm, uniform_thresh=0.15, high_thresh=0.7, low_thresh=0.4
):
    """
    Reasoning-guided hierarchical scoping to identify relevant subtree for a query.

    Args:
        query (str): user query
        tree (dict): hierarchical circuit JSON
        llm (object): model interface with `.run(prompt)` method
    Returns:
        dict: {"path": [...], "final_node": node_dict, "trace": [...]}
    """

    current_node = tree
    path = [current_node["unique_name"]]
    trace = []

    while current_node.get("children"):
        children = current_node["children"]

        subtree_chunks = [
            {"id": c["unique_name"], "subtree": json.dumps(c, indent=2)}
            for c in children
        ]

        prompt = f"""
        You are an **expert analog and mixed-signal circuit design reasoning agent**.

        Your task is to determine which part of a hierarchical analog circuit design tree is most relevant to a user's query.

        You will be provided with:
        - The **user query**, describing what the user wants to understand or analyze.
        - A list of **subtrees**, each representing a **hierarchical functional block abstraction** within the full circuit (including its role descriptions, class/category, netlist, and sub-blocks).

        For each subtree:
        - Read the full JSON content carefully — it includes the block's internal hierarchy, role hints, class/category, role descriptions, and contextual information.
        - Judge **how relevant that subtree is** to answering the query, using your expert understanding of analog architectures, signal paths, and functional roles.
        - Score relevance on a **continuous scale from 0.0 to 1.0**, where:
            - **1.0 = Directly and specifically addresses the query.**
            - **0.5 = Somewhat related or indirectly contributes.**
            - **0.0 = Completely irrelevant.**

        Return your final judgement **strictly as a single valid JSON object**, mapping each subtree ID to its relevance score.

        Example format:
        {{
        "childA": 0.85,
        "childB": 0.40,
        "childC": 0.05
        }}

        Now analyze and score the following:

        **User Query:** "{query}"

        **Candidate Subtrees (each a complete JSON fragment):**
        {json.dumps(subtree_chunks, indent=2)}

        """

        response = llm.run(prompt)
        scores = _safe_parse_json(response)

        # Record reasoning step
        trace.append(
            {
                "parent": current_node["unique_name"],
                "scores": scores,
                "chosen": max(scores, key=scores.get),
            }
        )

        # Stop criteria
        vals = list(scores.values())
        spread = max(vals) - min(vals)
        if (
            spread < uniform_thresh
            or all(s > high_thresh for s in vals)
            or all(s < low_thresh for s in vals)
        ):
            # Scores uniform, high, or uniformly low → current node is scope
            break

        # Otherwise descend into best-scoring child
        best_child_id = max(scores, key=scores.get)
        next_node = next(c for c in children if c["unique_name"] == best_child_id)
        current_node = next_node
        path.append(best_child_id)

    result = {"path": path, "final_node": current_node, "trace": trace}
    return result


def _safe_parse_json(text: str):
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


class InterfaceAgent(Agent):
    def __init__(
        self,
        name,
        system_prompt,
        tools,
        available_functions,
        enriched_tree,
        toplevelcircuit: str,
        model="azure/openai/gpt-5",
    ):
        super().__init__(name, tools, available_functions, system_prompt, model)
        self.toplevelcircuit = toplevelcircuit
        self.tree = enriched_tree

    def handle_query(self, query):
        print(f"\n Handling query: {query}\n")

        # Step 1: Run reasoning-guided traversal
        scope_result = reasoning_guided_tree_scope(query, self.tree, self)

        print("Traversal complete.\n")
        print("Path of Interest:", " -> ".join(scope_result["path"]))
        print("Final Focus Node:", scope_result["final_node"]["unique_name"])

        return scope_result

    def respond_with_retrieved_subtree(self, query, scope_result):
        """
        Takes the scoping result and answers the query using the retrieved subtree context.
        """
        final_node = scope_result["final_node"]
        focused_subtree = extract_subtree(self.tree, final_node["unique_name"])

        prompt = f"""
            You are an **expert analog circuit design reasoning assistant**.

            You will be provided with:
            - A **user query** describing the design-related question or analysis objective. 
            - The **relevant subtree** of a hierarchical analog circuit design (including role descriptions, netlist fragments, and functional context). This is your **primary focus area**. 
            - The **entire global circuit hierarchy JSON** for broader context.

            Use the global context to understand **how this subtree fits within the overall circuit architecture**.  
            However, base your reasoning and explanations **primarily on the detailed content of the identified most relevant subtree**.  

            Provide clear, technically accurate reasoning.  

            **User Query:** "{query}"

            **Relevant Subtree Context:**
            {json.dumps(focused_subtree, indent=2)}

            **Global Circuit Context (Full Hierarchical JSON):**
            {json.dumps(self.tree, indent=2)}

        """

        answer = self.run(prompt)

        return {
            "query": query,
            "answer": answer,
            "path": scope_result["path"],
            "trace": scope_result["trace"],
            "focused_node": final_node,
        }


def main():
    parser = argparse.ArgumentParser(description="Interface agent for hierarchical circuit query")
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

    with open(netlist_filepath, "r") as f:
        full_netlist_content = f.read()

    generated_subcircuits_path = os.path.join(
        parent_netlist_path, "Generated_Subcircuits"
    )

    lightweight_analysis_path = os.path.join(
        generated_subcircuits_path, "Lightweight_Analysis_Bottom_Up.json"
    )

    with open(lightweight_analysis_path, "r") as f:
        enriched_tree = json.load(f)

    G = nx.DiGraph()
    add_edges_from_json(enriched_tree, G)

    root = enriched_tree.get("unique_name", enriched_tree.get("name", "Unnamed"))
    plot_graph(G, root)

    interface_agent = InterfaceAgent(
        name="AnalogInterfaceAgent",
        system_prompt="You are a hierarchical analog circuit understanding and reasoning interface agent.",
        tools=[],
        available_functions={},
        enriched_tree=enriched_tree,
        toplevelcircuit=full_netlist_content,
    )

    # Example query
    query = "Which block contributes the most to the SNR?"

    # Traversal
    scope_result = interface_agent.handle_query(query)

    print("\n Reasoning Trace:")
    for step in scope_result["trace"]:
        print(f"Parent: {step['parent']}, Scores: {step['scores']}, Chosen: {step['chosen']}")

    # Retrieval and response generation
    result = interface_agent.respond_with_retrieved_subtree(query, scope_result)

    print("\n================= FINAL ANSWER =================")
    print(result["answer"])
    print("\n================================================")


if __name__ == "__main__":
    main()
