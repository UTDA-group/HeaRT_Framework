# HeaRT: Hierarchical Circuit Reasoning Tree

**Paper:** [HeaRT: A Hierarchical Circuit Reasoning Tree-Based Agentic Framework for AMS Design Optimization]

**Authors:** Souradip Poddar, Chia-Tung (Mark) Ho, Ziming Wei, Weidong Cao, Haoxing Ren, David Z. Pan

---

## Overview

HeaRT is a multi-agent framework for hierarchical understanding and reasoning over analog/mixed-signal (AMS) SPICE netlists. It decomposes a circuit into a structured reasoning tree that can be queried to answer design questions.

## Pipeline

The framework runs in four sequential steps:

1. **Subcircuit Decomposition** (`agents/subcircuit_decomposition_agent.py`)
   Parses the netlist into a bipartite device-net graph, identifies DC-connected component groups, and uses an LLM to decompose the circuit into functional subcircuits.

2. **Hierarchical Tree Construction** (`agents/hierarchical_tree_agent.py`)
   Takes the leaf-level subcircuits and builds a hierarchical agglomeration tree (JSON) representing the circuit's functional hierarchy.

3. **Bottom-Up Knowledge Consolidation** (`agents/bottom_up_consolidation_agent.py`)
   Traverses the tree bottom-up, annotating each node with role descriptions and flattened netlists to produce the final enriched reasoning tree.

4. **Query Interface** (`agents/interface_agent.py`)
   Accepts natural-language queries and uses LLM-guided tree traversal to identify the most relevant subtree and generate a grounded answer.

## Supported Models

OpenAI (GPT), Anthropic (Claude), Google (Gemini 2.5 Pro), DeepSeek, and Llama (via Together AI).

## Usage

Set API keys as environment variables:
```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...
export DEEPSEEK_API_KEY=...
export TOGETHER_API_KEY=...
```

Run each step in order, pointing to your netlist directory. Example netlists are provided under `netlists/`.
