from Graph import Graph
import numpy as np
from enum import Enum

class SetCoverHeuristicCostType(Enum):
    NbElementsAdded = 1,
    Frequencies = 2,


class Set:
    """
    Represents a set in the cover set problem
    """
    def __init__(self, graph, node):
        """
        Initializes a set

        Args: 
            graph (Graph): Graph to get the information from
            node (str): Node label in the graph that will serves as the set representative
        """
        self.label = node
        self.content = [n for n in graph.neighbors[node]]
        self.content.append(node)

    def get_cost(self, cover_set, cost_type, node_frequencies):
        """
        Computes the score of the cost of adding the current set to the given cover set.
        The score is the number of elements that would be effectively added in the cover set

        Args:
            cover_set (set(str)): A set of node label
            cost_type (SetCoverHeuristicCostType): The type of cost heuristic
            node_frequencies (dict(str, float)): The frequency of each node

        Returns:
            int: The cost
        """
        cost = 0.
        if(cost_type == SetCoverHeuristicCostType.Frequencies):
            cost = np.sum([(1. / node_frequencies[node]) for node in self.content])

        elif(cost_type == SetCoverHeuristicCostType.NbElementsAdded):
            cost = len(self.content)
            for label in cover_set:
                if label in self.content:
                    cost -= 1.
        return cost



def create_sets(graph):
    """
    Creates a collection of sets given a graph
    
    Args:
        graph (Graph): An undirected graph

    Returns:
        list(Set): A list of Sets
    """
    output = []
    for node in graph.nodes:
        cur_set = Set(graph, node)
        output.append(cur_set)
    return output


def greedy_set_cover(sets, cost_type = SetCoverHeuristicCostType.NbElementsAdded, node_frequencies = None):
    """
    Computes a greedy solution for the cover set problem

    Args:
        sets (list(Sets)): A list of Sets
        cost_type (SetCoverHeuristicCostType): The type of cost heuristic
        node_frequencies (dict(str, float)): The frequency of each node

    Returns:
        list(str): A list of covering sets represented by their labels
    """
    target = set([n.label for n in sets])
    cover_set = set()
    output = []
    available_sets = sets

    # While the cover set is not complete
    while(len(cover_set) < len(target)):
        # Find the available set with the highest score
        max_cost = -np.inf
        best_set_index = 0
        for index in range(len(available_sets)):
            cur_set = available_sets[index]
            cur_cost = cur_set.get_cost(cover_set, cost_type, node_frequencies)
            if(cur_cost > max_cost):
                max_cost = cur_cost
                best_set_index = index
        
        # Update the output list
        best_set = available_sets[best_set_index]
        output.append(best_set.label)

        # Update the cover set
        for label in best_set.content:
            cover_set.add(label)

        # Remove the set to not pick it again
        available_sets.pop(best_set_index)
    
    return output


def create_node_frequencies(sets):
    """
    Computes the node frequencies in the sets

    Args:
        sets (list(Sets)): A list of Sets        

    Returns:
        node_frequencies (dict(str, float)): The frequency of each node
    """
    node_frequencies = {}
    nb_total_nodes = 0
    for cur_set in sets:
        for node in cur_set.content:
            if(node not in node_frequencies):
                node_frequencies[node] = 1.
            else:
                node_frequencies[node] += 1.
            nb_total_nodes += 1
    for node in node_frequencies:
        node_frequencies[node] /= nb_total_nodes
    return node_frequencies



if __name__ == "__main__":
    nb_nodes = [20, 50, 100, 150, 200, 250, 300]

    test_directory = "tests"
    sols_directory = "solutions"

    verbose = True
    render = True

    cost_heuristic = SetCoverHeuristicCostType.NbElementsAdded
    # cost_heuristic = SetCoverHeuristicCostType.Frequencies

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
        output_sets = create_sets(output_graph)

        if(cost_heuristic == SetCoverHeuristicCostType.NbElementsAdded):
            output_sol = greedy_set_cover(output_sets, cost_type=cost_heuristic)
        elif(cost_heuristic == SetCoverHeuristicCostType.Frequencies):
            node_frequencies = create_node_frequencies(output_sets)
            output_sol = greedy_set_cover(output_sets, cost_type=cost_heuristic, node_frequencies=node_frequencies)

        output_graph.add_dominating_set_from_list(output_sol)
        if verbose:
            print(f"Solution found: {len(output_sol)} vertices")

        gv_outputs = output_graph.to_graphviz()
        if render:
            gv_outputs.render(f"{sols_directory}/cover_set/solution_{nb_node}.gv")

        if verbose:
            print(f"")
