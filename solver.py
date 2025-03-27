import json
import argparse
import sys

import exact_sat
import exact_cover_set
import heuristic_cover_set

from graph import Graph

def load_config(config_path):
    """
    Load configuration from a JSON file.

    Args:
        config_path (str): Path to the configuration JSON file.

    Returns:
        dict: Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"Configuration file not found: {config_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Invalid JSON in configuration file: {config_path}")
        sys.exit(1)

def solve_graph(config):
    """
    Solve the graph dominating set problem based on configuration.

    Args:
        config (dict): Configuration dictionary
    """
    # Load graph
    input_file = config['input_file']
    graph = Graph(input_file)

    # Determine solution method
    if config['method'] == 'exact_cover_set':
        graph = exact_cover_set.solve(graph, config)
    elif config['method'] == 'exact_sat':
        graph = exact_sat.solve(graph, config)
    elif config['method'] == 'heuristic_cover_set':
        graph = heuristic_cover_set.solve(graph, config)
    else:
        print(f"Unknown solution method: {config['method']}")
        sys.exit(1)

    # Optional verbose output
    if config.get('verbose', False):
        print(f"Method: {config['method']} for {input_file}")
        print(f"\tSolution size: {len(graph.dominating_set)} vertices")

    if config.get('save_sol', False):
        sol_file = config.get('sol_file', 'solution_output.sol')
        graph.to_sol(sol_file)
        print(f"Solution saved to {sol_file}")

    # Render graph visualization
    if config.get('render', False):
        gv = graph.to_graphviz()
        render_file = config.get('render_file', 'solution_output.pdf')
        gv.render(render_file.replace(".pdf", ""))
        print(f"Graph rendered to {render_file}")

def main():
    parser = argparse.ArgumentParser(description="Graph Dominating Set Solver")
    parser.add_argument("-c", "--config", required=True, help="Path to JSON configuration file")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Solve graph
    solve_graph(config)

if __name__ == "__main__":
    main()