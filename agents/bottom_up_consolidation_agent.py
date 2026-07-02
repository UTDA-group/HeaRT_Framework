import re
import json
import sys
import os
import argparse
from typing import List
import networkx as nx
import matplotlib.pyplot as plt

# Add parent directory (Multi_Agent) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from base_agent import Agent
from contexts.local_context import LocalContext
from contexts.global_context import GlobalContext

from agents.prompts import (
    BOTTOM_UP_LIGHTWEIGHT_ANALYSIS_INTEGRATOR_LOOP_AND_PROXY_DETERMINATION_SYSTEM_PROMPT,
)


working_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def clean_leaf_nodes_and_collapse(node):
    # Recursively process children first
    if "children" in node:
        # Clean and collapse children
        new_children = []
        for child in node["children"]:
            cleaned_child = clean_leaf_nodes_and_collapse(child)
            if cleaned_child is not None:
                new_children.append(cleaned_child)
        node["children"] = new_children

        # Collapse parent with only one child
        if len(node["children"]) == 1:
            # Merge this node with its only child
            only_child = node["children"][0]

            # Here, we keep the child and drop this node's info except for the parent name
            # if "unique_name" in node:
            #    only_child["broad_scope_hint"] = node[
            #        "unique_name"
            #    ]  # Renamed for clarity

            if "class_category" in node:
                only_child["class_category"] = node[
                    "class_category"
                ]  # Retain parent's class_category: Since now good enough scope to understand the category
            return only_child

        # If this is a leaf node, remove class_category, since leaf nodes might be too small of a context to decide the class category
        if len(node["children"]) == 0:
            node.pop("class_category", None)
    return node


# Convert JSON tree to a directed graph (network x)
# Visualize


def add_edges_from_json(node, G, parent=None):
    node_name = node.get("unique_name", node.get("name", "Unnamed"))
    if parent:
        G.add_edge(parent, node_name)
    for child in node.get("children", []):
        add_edges_from_json(child, G, node_name)


def print_tree(node, indent=0):
    print("  " * indent + node.get("unique_name", "Unnamed"))
    for child in node.get("children", []):
        print_tree(child, indent + 1)


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


def plot_graph(G, root, width=1.5, vert_gap=0.2, figsize=(12, 8), save_path=None):
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
    # plt.show()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def add_global_context_fields_to_leaves(node, global_subcircuits):
    if "children" not in node or len(node["children"]) == 0:
        node_name = node.get("unique_name")
        for subckt_dict in global_subcircuits:
            subckt_identifier = subckt_dict.get("subcircuit_id", subckt_dict.get("id"))
            if node_name == subckt_identifier:  # Leaf Nodes
                for field in [
                    "netlist",
                    "signal_ports",
                    "supply_ports",
                    "role_hint",
                    "io_interface_type",
                    "is_analog",
                    "is_digital",
                ]:
                    if field in subckt_dict:
                        node[field] = subckt_dict[field]

                # if "netlist" in node:
                #    node["devices"] = extract_device_names(node["netlist"])
                # else:
                #    node["devices"] = []
                break
        return node
    # Recursive case: parent node
    elif "children" in node:
        # all_devices = set()
        for i, child in enumerate(node["children"]):
            node["children"][i] = add_global_context_fields_to_leaves(
                child, global_subcircuits
            )
            # Accumulate devices from children
            # child_devices = node["children"][i].get("devices", [])
            # all_devices.update(child_devices)
        # node["devices"] = sorted(all_devices)
        return node


def extract_device_names_from_netlist_str(netlist_str):

    # Split into lines
    lines = netlist_str.split("\n")

    # Extract lines between .subckt and .ends
    device_lines = lines[1:-1]

    # Initialize lists to store results
    device_names = []
    device_types = []

    # Possible device types
    valid_types = {"pmos", "nmos", "cap", "res"}

    for line in device_lines:
        parts = line.split()
        device_name = parts[0]
        # Find device type within the line; it should be one of the valid types
        device_type = None
        for p in parts:
            if p.lower() in valid_types:
                device_type = p.lower()
                break
        device_names.append(device_name)
        device_types.append(device_type)

    return device_names


def compute_net_degree_for_net(net_name, netlist_lines):
    """
    Compute the degree (number of connections) of a specific net in a SPICE netlist.

    Args:
        net_name (str): Target net whose degree to compute.
        netlist_str (str): SPICE netlist text.

    Returns:
        int: Degree (number of connections) of the given net.
    """
    valid_types = {"pmos", "nmos", "res", "cap"}
    degree = 0

    # Clean netlist lines (ignore comments, directives)
    lines = [
        l.strip()
        for l in netlist_lines
        if l.strip() and not l.strip().startswith((".", "*", "+", "//"))
    ]

    for line in lines:
        parts = line.split()
        if len(parts) < 3:
            continue

        # Check if this line defines a recognized device type
        device_type = None
        for p in parts:
            if p.lower() in valid_types:
                device_type = p.lower()
                break
        if not device_type:
            continue  # Skip if not in recognized device set

        # Extract net names depending on device type
        if device_type in ("pmos", "nmos"):
            nets = parts[1:5]  # D, G, S, B
        elif device_type in ("res", "cap"):
            nets = parts[1:3]  # Two-terminal devices
        else:
            nets = []

        # Count occurrences for the target net
        degree += sum(1 for n in nets if n == net_name)

    return degree


def add_device_list_to_nodes(node):
    # If node has a netlist string, extract device names and add them
    if "netlist" in node:
        node["device_list"] = extract_device_names_from_netlist_str(node["netlist"])
    # Recursively apply to children if any
    if "children" in node:
        for child in node["children"]:
            add_device_list_to_nodes(child)
    return node


def flatten_netlists_bottom_up_with_ports(
    node, global_netlist_lines, top_level_ports, visited=None
):
    """
    Recursively concatenate child netlists bottom-up.
    Each parent node gets a flattened .subckt once all its children have netlists.

    Args:
        node (dict): current node in the hierarchical JSON.
        visited (set): set of node IDs already processed.

    Returns:
        str: flattened netlist for this node.
    """
    if visited is None:
        visited = set()

    # Base case: node already has its own netlist → mark visited
    if "netlist" in node and node["netlist"]:
        visited.add(node["unique_name"])

        # Ensure leaf has explicit port fields
        node.setdefault("supply_ports", [])
        node.setdefault("signal_ports", [])
        return node["netlist"]

    # Recursive case: process children first
    child_netlists = []
    all_child_supply_ports = set()
    all_child_signal_ports = set()

    for child in node.get("children", []):
        child_netlist = flatten_netlists_bottom_up_with_ports(
            child, global_netlist_lines, top_level_ports, visited
        )
        if not child_netlist:
            continue

        child_netlists.append(child_netlist)

        # Collect child ports
        child_supply = set(child.get("supply_ports", []))
        child_signal = set(child.get("signal_ports", []))
        all_child_supply_ports |= child_supply
        all_child_signal_ports |= child_signal

    # Parent-level ports
    parent_supply_ports = sorted(all_child_supply_ports)

    # --- Concatenate all child netlists ---
    # Only build this node's netlist if all children have netlists
    if len(child_netlists) == len(node.get("children", [])) and child_netlists:
        # Concatenate children netlists, removing redundant .subckt/.ends lines
        combined_body = []
        parent_signal_ports = []
        for subckt in child_netlists:
            for line in subckt.splitlines():
                if not line.strip().lower().startswith((".subckt", ".ends")):
                    combined_body.append(line)

        for net in sorted(all_child_signal_ports):
            local_deg = compute_net_degree_for_net(net, combined_body)
            global_deg = compute_net_degree_for_net(net, global_netlist_lines)

            # Always include if it's a known top-level port
            if net in top_level_ports:
                parent_signal_ports.append(net)
                continue

            if (
                local_deg < global_deg
            ):  # Meaning External Port to this Parent Node But Internal to the Overall Design
                parent_signal_ports.append(net)

        parent_signal_ports = sorted(parent_signal_ports)

        # Wrap subckt name in quotes to support spaces
        flattened_netlist = (
            f'.subckt "{node["unique_name"]}" '
            + " ".join(parent_supply_ports + parent_signal_ports)
            + "\n"
            + "\n".join(combined_body)
            + f'\n.ends "{node["unique_name"]}"'
        )

        node["netlist"] = flattened_netlist
        node["supply_ports"] = parent_supply_ports
        node["signal_ports"] = parent_signal_ports
        visited.add(node["unique_name"])
        return flattened_netlist

    # If no children or incomplete → return None
    return None


class BottomUpLightweightAnalysisAgent(Agent):
    def __init__(
        self,
        name,
        system_prompt,
        tools,
        available_functions,
        toplevelcircuit: str,
        model="gpt-5",  # gpt-4o
    ):
        super().__init__(name, tools, available_functions, system_prompt, model)
        self.local_context = LocalContext(name)
        self.toplevelcircuit = toplevelcircuit

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

    def bottom_up_lightweight_analysis(
        self, tree: dict, system_prompt: str = None
    ) -> dict:

        if system_prompt is None:  # Nothing provided
            system_prompt = self.system_prompt

        for message in self.messages:
            if message["role"] == "system":
                message["content"] = system_prompt
                break

        user_message = f"""
            Top-level Circuit Netlist:
            {self.toplevelcircuit}

            Hierarchical JSON Tree:
            {json.dumps(tree, indent=2)}

        """

        output_format_prompt = """           
            Please return the **entire hierarchical tree** in valid JSON format.  
            Each node (root, intermediate, leaf) must strictly follow this schema:

            {
                "unique_name": "<string, unique identifier of the node>",
                "role_description": "<one-line functional role in context of the immediate parent>",
                "children": [ ... ]  // list of child nodes, each following the same schema recursively
            }
        
            Notes:
                - Always preserve `"unique_name"` exactly as in the input JSON tree.  
                - Ensure the **entire tree** is returned.
        """

        response = self.run(user_message + output_format_prompt)

        try:
            summary = self._safe_parse_json(response)
        except Exception:
            summary = {"raw_output": response}

        return summary




# This function walks through two trees at the same time, lines up nodes whose 'unique_name' fields are the same, and copies 'role_description' from the second tree into the first. It does this all the way down through their children.
def merge_trees_by_unique_name(struct_tree, role_tree):
    """
    Recursively merges role descriptions into the structural tree by matching 'unique_name'.
    Keeps all fields from struct_tree and adds 'role_description' from role_tree.
    """
    if not struct_tree or not role_tree:
        return struct_tree or role_tree

    # Merge if unique_name matches
    if struct_tree.get("unique_name") == role_tree.get("unique_name"):
        if "role_description" in role_tree:
            struct_tree["role_description"] = role_tree["role_description"]

        # Recursively merge children
        struct_children = struct_tree.get("children", [])
        role_children = role_tree.get("children", [])
        if struct_children and role_children:
            merged_children = []
            for s_child in struct_children:
                # Find corresponding child in role tree
                match = next(
                    (
                        r_child
                        for r_child in role_children
                        if r_child.get("unique_name") == s_child.get("unique_name")
                    ),
                    None,
                )
                merged_children.append(merge_trees_by_unique_name(s_child, match))
            struct_tree["children"] = merged_children

    return struct_tree


def main():
    parser = argparse.ArgumentParser(description="Bottom-up lightweight analysis consolidation")
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

    top_level_netlist_filename = os.path.basename(netlist_filepath)
    top_level_circuit_name = os.path.splitext(top_level_netlist_filename)[0]

    generated_subcircuits_path = os.path.join(parent_netlist_path, "Generated_Subcircuits")

    # Read in the Global Context JSON
    circuit_global_context = GlobalContext()
    circuit_global_context.load(
        os.path.join(generated_subcircuits_path, "circuit_global_context.json")
    )

    global_ctx_data = circuit_global_context._store
    global_subcircuits_list = global_ctx_data.get("subcircuits", [])

    # Load full netlist text
    with open(netlist_filepath, "r") as f:
        full_netlist_content = f.read()

    full_netlist_lines = full_netlist_content.splitlines()

    with open(os.path.join(generated_subcircuits_path, "hierarchical_agglomeration_tree.json")) as f:
        tree = json.load(f)

    # Clean and collapse
    cleaned_tree = clean_leaf_nodes_and_collapse(tree)

    # Enrich leaves with global context fields
    augmented_tree = add_global_context_fields_to_leaves(cleaned_tree, global_subcircuits_list)
    top_level_ports = global_ctx_data.get("top_level_port_names", [])
    augmented_tree["port_names"] = top_level_ports
    flatten_netlists_bottom_up_with_ports(augmented_tree, full_netlist_lines, top_level_ports)

    agent = BottomUpLightweightAnalysisAgent(
        name=top_level_circuit_name,
        system_prompt=BOTTOM_UP_LIGHTWEIGHT_ANALYSIS_INTEGRATOR_LOOP_AND_PROXY_DETERMINATION_SYSTEM_PROMPT,
        tools=[],
        available_functions={},
        toplevelcircuit=full_netlist_content,
        model="claude-opus-4-8",
        # model="deepseek-reasoner",
        # model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    )

    role_description_tree = agent.bottom_up_lightweight_analysis(augmented_tree)

    enriched_tree = merge_trees_by_unique_name(augmented_tree, role_description_tree)
    enriched_tree = add_device_list_to_nodes(enriched_tree)

    # Plot the enriched tree
    G_enriched = nx.DiGraph()
    add_edges_from_json(enriched_tree, G_enriched)
    root_enriched = enriched_tree.get("unique_name", enriched_tree.get("name", "Unnamed"))
    plot_graph(G_enriched, root_enriched, save_path=os.path.join(generated_subcircuits_path, "tree_enriched.png"))

    output_path = os.path.join(generated_subcircuits_path, "Lightweight_Analysis_Bottom_Up.json")
    with open(output_path, "w") as f:
        json.dump(enriched_tree, f, indent=2)

    print(f"Bottom-up analysis saved to {output_path}")
    print(f"Total tokens: {agent.last_usage.get('total_tokens', 0)}")


if __name__ == "__main__":
    main()


# In the final tree save addtionally:
#### > Netlist per node, not just leaf node
#### > The device list per node
#### > The feedback and feedforward loops under ther covered content under each node
#### > Tie the Performance Proxies based on Look-up to each node. Keep this part for later during the top-down agent design.
