from graph import Graph
import numpy as np
import json
import argparse
import sys

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

    def get_cost(self, cover_set):
        """
        Computes the score of the cost of adding the current set to the given cover set.
        The score is the number of elements that would be effectively added in the cover set

        Args:
            cover_set (set(str)): A set of node label

        Returns:
            int: The cost
        """
        cost = len(self.content)
        for label in cover_set:
            if label in self.content:
                cost -= 1
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


def greedy_set_cover(sets):
    """
    Computes a greedy solution for the cover set problem

    Args:
        sets (list(Sets)): A list of Sets

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
            cur_cost = cur_set.get_cost(cover_set)
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


def solve(graph, config):
    """
    Solves the dominating set problem for the given graph using an approximated greedy method.

    Args:
        graph (Graph): The input graph to solve.
        config (dict): Configuration dictionary (not used in this implementation).

    Returns:
        Graph: The graph with the dominating set added and colored.
    """
    output_sets = create_sets(graph)
    output_sol = greedy_set_cover(output_sets)
    graph.add_dominating_set_from_list(output_sol)