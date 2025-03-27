from graph import Graph
from pysat.solvers import Solver
import json
import argparse
import sys

def add_clauses_from_graph(graph, solver):
    """
    Converts the graph dominating set problem to SAT by adding clauses to the solver.

    Args:
        graph (Graph): The input graph to be converted to SAT clauses.
        solver (Solver): The SAT solver to which clauses will be added.
    """
    g = graph.neighbors

    for v in g:
        clause = [int(v)]
        for u in g[v]:
            clause.append(int(u))
        solver.add_clause(clause)


def minimize_solver(graph):
    """
    Finds the minimum dominating set size using binary search on a SAT solver.

    Args:
        graph (Graph): The input graph to find the minimum dominating set for.

    Returns:
        tuple: A tuple containing:
            - int: The size of the minimum dominating set
            - list: The list of nodes in the minimum dominating set
    """
    # Get the list of possible literals (nodes)
    litterals = []
    for v in graph.nodes:
        litterals.append(int(v))

    # Binary search for minimum dominating set size
    upper_bound = graph.nb_vertices
    lower_bound = 1

    while upper_bound - lower_bound > 1:
        # Create a new SAT solver with graph constraints
        s = Solver(name='mc')
        add_clauses_from_graph(graph, s)

        # Calculate midpoint and add at-most constraint
        k = (upper_bound + lower_bound)//2
        s.add_atmost(lits=litterals, k=k, no_return=False)
        print(k)
        
        if s.solve():
            print("here")
            upper_bound = k
        else:
            lower_bound = k

    # Get the final solution
    sol = s.get_model()

    # Filter positive literals (nodes in the dominating set)
    positive_sol = [x for x in sol if x > 0]

    return k, positive_sol


def solve(graph, config):
    """
    Solves the dominating set problem for the given graph using SAT solver.

    Args:
        graph (Graph): The input graph to solve.
        config (dict): Configuration dictionary (not used in this implementation).

    Returns:
        Graph: The graph with the dominating set added and colored.
    """
    s = Solver(name='mc')
    add_clauses_from_graph(graph, s)
    dom_set_size, sol = minimize_solver(graph)
    graph.add_dominating_set_from_list(sol)
    return graph