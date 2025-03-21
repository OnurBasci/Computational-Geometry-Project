from Graph import Graph
import os
import time
import matplotlib.pyplot as plt
import numpy as np


def minimum_set_cover(S, U):
    """
    Recursively computes the minimum set cover.
    :param S: A set of subsets (set of frozensets).
    :param U: The universal set that needs to be covered (set).
    :return: The minimum set cover as a set of frozensets or None if no cover exists.
    """
    # Base case: If no more subsets are left
    if len(S) == 0:
        return set() if len(U) == 0 else None
    

    # Select the set with maximum cardinality (largest subset)
    S_max = max(S, key=len)
    S_rest = S - {S_max}  # Remove selected subset from S
    S_removed = {s - S_max for s in S_rest if len(s - S_max) > 0} #Each element that are in S_max is removed from S
    
    #we need a mappint from s_removed to S_rest to get the original mapping
    s_map = dict()
    for s in S_rest:
        removed = s - S_max
        s_map[removed] = s

    # Case 1: Take S_max in the set cover
    new_U = U - S_max  # Remove covered elements from U
    cover_with_S_max = minimum_set_cover(S_removed, new_U)

    if cover_with_S_max is not None:
        #get the original mapping from the minimum set
        cover_with_S_max = {frozenset(s_map[s]) for s in cover_with_S_max}
        cover_with_S_max.add(S_max)
        #print(cover_with_S_max)
    
    # Case 2: Do not take S_max in the set cover
    cover_without_S_max = minimum_set_cover(S_rest, U)
    
    # Return None if both cover sets are None
    if cover_without_S_max is None and cover_with_S_max is None:
        return None

    # Take the smallest between cover_with_S_max and cover_without_S_max
    if cover_with_S_max is None:
        smallest_set_cover = cover_without_S_max
    elif cover_without_S_max is None:
        smallest_set_cover = cover_with_S_max
    else:
        smallest_set_cover = cover_with_S_max if len(cover_with_S_max) < len(cover_without_S_max) else cover_without_S_max
    
    return smallest_set_cover


def minimum_set_cover_reduction_rules(S, U):
    """
    Recursively computes the minimum set cover.
    In this version of the MSC, if there is an element of frequency 1 with the size
    of 1 and frequency of 1
    :param S: A set of subsets (set of frozensets).
    :param U: The universal set that needs to be covered (set).
    :return: The minimum set cover as a set of frozensets or None if no cover exists.
    """
    # Base case: If no more subsets are left
    if len(S) == 0:
        return set() if len(U) == 0 else None
    
    # Reduction Rule 1
    counter = dict()
    for s in S:
        for e in s:
            counter[e] = counter.get(e, (0, s))
            counter[e] = (counter[e][0] + 1, s)
    for e in counter:
        if counter[e][0] == 1:
            R = counter[e][1]
            new_U = U - R
            S_rest = S - {R}
            S_removed = {s - R for s in S_rest if len(s - R) > 0}
            # print(f"\n\ne: {e}, R: {R}\nS: {S}\ncounter: {counter}\nS_rest: {S_rest}\nS_removed: {S_removed}\nnew_U: {new_U}\n\n")

            s_map = dict()
            for s in S_rest:
                removed = s - R
                if len(removed) > 0:
                    s_map[removed] = s

            cover = minimum_set_cover_reduction_rules(S_removed, new_U)

            if cover is not None:
                #get the original mapping from the minimum set
                cover = {frozenset(s_map[s]) for s in cover}
                cover.add(R)
                return cover
            else:
                return {R}

    # Reduction Rule 3
    for Q in S:
        for R in S:
            if len(Q) >= len(R) and Q != R:
                is_R_a_Q_subset = True
                for r in R:
                    if r not in Q:
                        is_R_a_Q_subset = False
                        break
                if is_R_a_Q_subset:
                    # print(f"\n\nQ: {Q}, R: {R}\n\nS: {S}\n\nS - R: {S-{R}}\n\n")
                    return minimum_set_cover_reduction_rules(S - {R}, U)

    # Reduction Rule 5
    elements_to_sets = dict()
    for s in S:
        for e in s:
            if e not in elements_to_sets:
                elements_to_sets[e] = set()
            elements_to_sets[e].add(s)
    sorted_elements_to_sets = sorted(elements_to_sets.keys(), key=lambda e: len(elements_to_sets[e]))
    removed_elements = set()
    for i, e1 in enumerate(sorted_elements_to_sets):
        if e1 in removed_elements:
            continue
        for e2 in sorted_elements_to_sets[i+1:]:
            if e2 in removed_elements:
                continue
            if elements_to_sets[e1] <= elements_to_sets[e2]: # elements_to_sets[e1] include in elements_to_sets[e2]
                removed_elements.add(e2)
    if len(removed_elements) > 0:
        new_U = U - removed_elements
        s_map = dict()
        for s in S:
            removed = s - removed_elements
            if len(removed) > 0:
                s_map[removed] = s
        S_removed = {s - removed_elements for s in S if len(s - removed_elements) > 0}
        cover = minimum_set_cover_reduction_rules(S_removed, new_U)
        if cover is not None:
            #get the original mapping from the minimum set
            cover = {frozenset(s_map[s]) for s in cover}
            return cover
        else:
            return None

    # Select the set with maximum cardinality (largest subset)
    S_max = max(S, key=len)
    S_rest = S - {S_max}  # Remove selected subset from S
    S_removed = {s - S_max for s in S_rest if len(s - S_max) > 0} #Each element that are in S_max is removed from S
    
    #we need a mappint from s_removed to S_rest to get the original mapping
    s_map = dict()
    for s in S_rest:
        removed = s - S_max
        if len(removed) > 0:
            s_map[removed] = s

    # Case 1: Take S_max in the set cover
    new_U = U - S_max  # Remove covered elements from U
    cover_with_S_max = minimum_set_cover_reduction_rules(S_removed, new_U)

    if cover_with_S_max is not None:
        #get the original mapping from the minimum set
        cover_with_S_max = {frozenset(s_map[s]) for s in cover_with_S_max}
        cover_with_S_max.add(S_max)
    else:
        cover_with_S_max = {S_max}
    
    # Case 2: Do not take S_max in the set cover
    cover_without_S_max = minimum_set_cover_reduction_rules(S_rest, U)
    
    # Return None if both cover sets are None
    if cover_without_S_max is None and cover_with_S_max is None:
        return None

    # Take the smallest between cover_with_S_max and cover_without_S_max
    if cover_with_S_max is None:
        smallest_set_cover = cover_without_S_max
    elif cover_without_S_max is None:
        smallest_set_cover = cover_with_S_max
    else:
        smallest_set_cover = cover_with_S_max if len(cover_with_S_max) < len(cover_without_S_max) else cover_without_S_max
    
    return smallest_set_cover


def graph_to_set_cover(graph):
    """
    Transforma a graph to a set cover structure
    graph: an instance of Graph data structure
    S: List of sets
    U: univers
    node_mapping: dictionary that maps to set from node
    """
    g = graph.neighbors

    U = set()
    S = set()

    node_mapping = {}  # Dictionary to map sets to nodes
    for v in g:
        U.add(int(v))
        s = {int(v)}
        for u in g[v]:
            s.add(int(u))
        S.add(frozenset(s))
        node_mapping[frozenset(s)] = int(v)  # Store mapping from set to node

    return U, S, node_mapping


def test_complexity(solver, graph_folder_path):
    """
    solver: the function that solves the set cover
    """

    folder_path = graph_folder_path
    execution_times = []
    for file_name in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path): 
            print(file_name)
            graph = Graph(file_path, sol_path=None)
            U, S, node_map = graph_to_set_cover(graph)

            time_before = time.time()
            solution = solver(S, U)
            time_after = time.time()
            print(f"length of solution {len(solution)}")
            dt = time_after - time_before
            execution_times.append(dt)
            print(f"time passed to solve {dt}")
            print("Minimum Set Cover:", solution)
        
    return execution_times


def plot_execution_time(execution_times, save_name = "execution time", lower_bound = 5, upper_bound = 40):
    """
    execution times: a list containing execution times
    """
    x_values = np.linspace(lower_bound, upper_bound, len(execution_times))

    plt.plot(x_values, execution_times)
    plt.title("Execution time over edge probability")
    plt.ylabel("execution time")
    plt.savefig(save_name, format='png')

def test():
    # Example Usage

    """S = {
        frozenset({1, 2, 3}),
        frozenset({2, 4}),
        frozenset({3, 5}),
        frozenset({4, 5, 6}),
        frozenset({6})
    }
    U = {1, 2, 3, 4, 5, 6}


    solution = trivial_set_cover(S, U)
    print("Minimum Set Cover:", solution)"""

    file_path = "tests/bremen_subgraph_20.gr"
    # file_path = "generated_graphs_increasing_vertices/graph_n55_p_0.5.gr"

    graph = Graph(file_path, sol_path=None)

    U, S, node_map = graph_to_set_cover(graph)
    
    #print(f"U: {U}")
    print(f"S: {S}")

    # solution = minimum_set_cover(S, U)
    solution = minimum_set_cover_reduction_rules(S, U)
    print(f"length of solution {len(solution)}")
    print("Minimum Set Cover:", solution)

    dominating_set = [node_map[frozenset(s)] for s in solution]
    print(f"dominating set is {dominating_set}")

    graph.add_dominating_set_from_list(dominating_set)

    gv = graph.to_graphviz()
    gv.render('set_cover_sol')

if __name__ == "__main__":
    #a = frozenset({5})
    # test()
    # """
    #graph_folder_path = "generated_graphs_increasing_edge"
    
    graph_folder_path = "generated_graphs_increasing_vertices"
    # execution_times = test_complexity(minimum_set_cover, graph_folder_path)
    execution_times = test_complexity(minimum_set_cover_reduction_rules, graph_folder_path)
    plot_execution_time(execution_times, save_name="execution_time_vertices.png", lower_bound=5, upper_bound=60)
    print(execution_times)

    # """
    """file_path = "./generated_graphs_increasing_vertices/graph_n55_p_0.5.gr"
    g = Graph(file_path, sol_path=None)
    gv = g.to_graphviz()
    gv.render('test')"""
