# Dominating Set Solver

This repository provides a solver for the **minimum dominating set problem** using both **exact** and **heuristic** methods. The solver takes a graph as input and finds a minimal set of vertices such that every other vertex in the graph is either in the set or adjacent to a vertex in the set.

## How to Use

### Prerequisites
Ensure you have Python installed and install the required dependencies:
```bash
pip install networkx matplotlib pysat
```

### Running the Solver
To run the solver, use the following command:
```bash
python solver.py -c config.json
```
This will load the configuration from `config.json` and execute the solver accordingly.

## Configuration File (`config.json`)
The configuration file specifies how the solver should process the graph. Below is an example configuration:
```json
{
    "input_file": "testGraphs/bremen_subgraph_20.gr",
    "save_sol": true,
    "sol_file": "solution_output.sol",
    "method": "exact_cover_set",
    "reduction_rules": [1, 2, 3, 4, 5, 6, 7],
    "verbose": true,
    "render": true,
    "render_file": "solution_output.pdf"
}
```
### Configuration Options
- `input_file` (str): Path to the input graph file.
- `save_sol` (bool): Whether to save the solution to a file.
- `sol_file` (str): Output file for the solution.
- `method` (str): The method to use for solving. Options:
  - `exact_sat`: Uses SAT reduction for an exact solution.
  - `exact_cover_set`: Uses set cover reduction rules for an exact solution.
  - `heuristic_cover_set`: Uses a greedy heuristic.
- `reduction_rules` (list[int]): List of reduction rules (1-7) to apply (used with `exact_cover_set`).
- `verbose` (bool): Whether to print additional details.
- `render` (bool): Whether to generate a visualization of the solution.
- `render_file` (str): Output PDF file for visualization.

## Solver Methods

### 1. SAT Reduction (`exact_sat`)
This method formulates the minimum dominating set problem as a **Boolean satisfiability problem (SAT)**. The solver:
- Creates a SAT encoding where each variable represents whether a vertex is in the dominating set.
- Uses a SAT solver to find the minimum valid solution.
- Performs binary search to minimize the size of the set.

### 2. Set Cover Reduction (`exact_cover_set`)
This method reduces the problem to the **minimum set cover problem**, using:
- A conversion of the graph into a set cover problem.
- A recursive application of **reduction rules** to simplify the problem.
- An exact algorithm to find the optimal cover.

### 3. Heuristic Set Cover (`heuristic_cover_set`)
This method uses a **greedy heuristic** to approximate a solution:
- Iteratively selects the node that covers the most uncovered vertices.
- Provides a fast but non-optimal solution.

## References
- [An Exact Algorithm for the Minimum Dominating Set Problem](https://www.ijcai.org/proceedings/2023/622)
- [Exact algorithms for dominating set](https://www.sciencedirect.com/science/article/pii/S0166218X11002393)

This project provides flexible solvers for tackling the dominating set problem, allowing trade-offs between speed and optimality.