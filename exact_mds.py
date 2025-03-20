from Graph import Graph
import numpy as np
from enum import Enum


class GraphMDS:
    def __init__(graph):
        self.graph = graph
        self.d0 = set()
        self.d0_dominated_nodes = set()
        self.df = []

    def prep_is_dominating(self):
        for node in self.graph.nodes:
            if node.label not in self.d0_dominated_nodes:
                return false
        return true

    def prep_get_next_node(self):
        max_nb_node = 0
        best_node = None
        for node in self.graph.nodes:
            if node.label in d0:
                continue
            cur_nb_node = 0
            for neigh in self.graph.neighbors[node.label]:
                if not neigh in self.d0_dominated_nodes:
                    cur_nb_node += 1
            if cur_nb_node > max_nb_node:
                max_nb_node = cur_nb_node
                best_node = node.label
        return best_node

    def preprocessing(self):
        while(not self.prep_is_dominating()):
            # Find element with most non flagged neighbors
            new_node = self.prep_get_next_node()
            self.d0.append(new_node)
            self.d0_dominated_nodes.insert(self.graphs.neighbors[new_node])
        

    def get_dominating_set(self):
        self.preprocessing()
        # TODO: change this
        return [node for node in d0] 



if __name__ == "__main__":
    nb_nodes = [20, 50, 100]

    test_directory = "tests"
    sols_directory = "solutions"

    verbose = True
    render = True

    for nb_node in nb_nodes:
        if verbose:
            print(f"Begin n = {nb_node} nodes...")

        file_path = f"{test_directory}/bremen_subgraph_{nb_node}.gr"
        sol_path = f"{test_directory}/bremen_subgraph_{nb_node}.sol"

        # Optimal graph
        optimal_graph = Graph(file_path, sol_path)
        optimal_sol = list(optimal_graph.dominating_set)
        if verbose:
            print(f"Optimal solution: {len(optimal_sol)} vertices")

        gv_optimal = optimal_graph.to_graphviz()
        if render:
            gv_optimal.render(f"{sols_directory}/optimal/solution_{nb_node}.gv")

        output_graph = Graph(file_path)
        mds_graph = GraphMDS(output_graph)
        output_sol = mds_graph.get_dominating_set()

        output_graph.add_dominating_set_from_list(output_sol)
        if verbose:
            print(f"Solution found: {len(output_sol)} vertices")

        gv_outputs = output_graph.to_graphviz()
        if render:
            gv_outputs.render(f"{sols_directory}/cover_set/solution_{nb_node}.gv")

        if verbose:
            print(f"")
