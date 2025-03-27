import argparse
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from Graph import Graph

def parse_arguments():
    parser = argparse.ArgumentParser(description="Minimum Set Cover with Reduction Rules")
    
    parser.add_argument("-i", "--input", required=True, help="Path to input .gr graph file")
    parser.add_argument("-o", "--output", default="set_cover_solution.pdf", help="Path to output PDF file")
    parser.add_argument("-r", "--rules", nargs="*", type=int, choices=range(1, 8), default=list(range(1, 8)),
                        help="List of reduction rules to apply (1-7). Default is all rules.")
    
    return parser.parse_args()


def minimum_set_cover_reduction_rules(S, U, rules=list(range(1, 8))):
    """
    Recursively computes the minimum set cover using reduction rules
    """
    # Convert rules list to dictionary for quick lookup
    rules_dict = {i: (i in rules) for i in range(1, 8)}

    # Base case: If no more subsets are left
    if len(S) == 0:
        return set() if len(U) == 0 else None

    # Select the set with maximum cardinality (largest subset)
    S_max = max(S, key=len)
    
    # Reduction Rule 2
    if rules_dict[2]:
        if(len(S_max) <= 1):
            return {frozenset({e}) for e in U}

    # Reduction Rule 1
    if rules_dict[1]:
        counter = dict()
        sets_containing = dict()
        for s in S:
            for e in s:
                if e not in counter:
                    counter[e] = (0,s)
                    sets_containing[e] = set()
                counter[e] = (counter[e][0]+1, s)
                sets_containing[e].add(s)

        for e in counter:
            if counter[e][0] == 1:
                R = counter[e][1]
                new_U = U - R
                S_rest = S - {R}
                S_removed = {s - R for s in S_rest if len(s - R) > 0}

                s_map = dict()
                for s in S_rest:
                    removed = s - R
                    if len(removed) > 0:
                        s_map[removed] = s

                cover = minimum_set_cover_reduction_rules(S_removed, new_U, rules)

                if cover is not None:
                    #get the original mapping from the minimum set
                    cover = {frozenset(s_map[s]) for s in cover}
                    cover.add(R)
                    return cover
                else:
                    return {R}

    # Reduction Rule 3
    if rules_dict[3]:
        for Q in S:
            for R in S:
                if len(Q) >= len(R) and Q != R:
                    is_R_a_Q_subset = True
                    for r in R:
                        if r not in Q:
                            is_R_a_Q_subset = False
                            break
                    if is_R_a_Q_subset:
                        return minimum_set_cover_reduction_rules(S - {R}, U, rules)

    # Reduction Rule 5
    if rules_dict[5]:
        removed_elements = set()
        sorted_keys = sorted(sets_containing.keys(), key=lambda e: len(sets_containing[e]))
        for i, e1 in enumerate(sorted_keys):
            if e1 in removed_elements:
                continue
            for e2 in sorted_keys[i+1:]:
                if e2 in removed_elements:
                    continue
                if sets_containing[e1].issubset(sets_containing[e2]):
                    removed_elements.add(e2)
                    continue
                if sets_containing[e2].issubset(sets_containing[e1]):
                    removed_elements.add(e1)
                    continue
        if len(removed_elements) > 0:
            new_U = U - removed_elements
            s_map = dict()
            for s in S:
                removed = s - removed_elements
                if len(removed) > 0:
                    s_map[removed] = s
            S_removed = {s - removed_elements for s in S if len(s - removed_elements) > 0}
            cover = minimum_set_cover_reduction_rules(S_removed, new_U, rules)
            if cover is not None:
                cover = {frozenset(s_map[s]) for s in cover}
                return cover
            else:
                return None


    # Redcution Rule 6
    if rules_dict[6]:
        for R in S:
            r2_elements = {e for e in R if counter[e][0] == 2}
            if len(r2_elements) > 0:
                other_sets = set()
                for e in R:
                    if counter[e][0] == 2:
                        for s in sets_containing[e]:
                            if s != R:
                                other_sets.add(s)
                other_elements = set().union(*other_sets) - set(R)
                if len(other_elements) < len(r2_elements):
                    new_U = U - R
                    S_rest = S - {R}
                    S_removed = {s - R for s in S_rest if len(s - R) > 0}

                    s_map = dict()
                    for s in S_rest:
                        removed = s - R
                        if len(removed) > 0:
                            s_map[removed] = s
                    
                    cover = minimum_set_cover_reduction_rules(S_removed, new_U, rules)
                    if cover is not None:
                        cover = {frozenset(s_map[s]) for s in cover}
                        cover.add(R)
                        return cover
                    else:
                        return {R}


    # Reduction Rule 7
    if rules_dict[7]:
        for R in S:
            if len(R) != 2:
                continue
            e_list = list(R)
            e1, e2 = e_list[0], e_list[1]
            # Check if both e1 and e2 have frequency 2
            if counter[e1] == 2 and counter[e2] == 2:
                # Get the unique other set containing e1 and the one containing e2
                R1 = [s for s in sets_containing[e1] if s != R][0]
                R2 = [s for s in sets_containing[e2] if s != R][0]

                Q = frozenset((R1 | R2) - R)
                new_U = U - R
                S_rest = (S - {R, R1, R2}).union({Q})
                S_removed = set()
                s_map = {}
                for s in S_rest:
                    s_removed = s - R
                    s_removed_frozen = frozenset(s_removed)
                    if s_removed_frozen:
                        S_removed.add(s_removed_frozen)
                        s_map[s_removed_frozen] = s

                C = minimum_set_cover_reduction_rules(S_removed, new_U, rules)
                if C is not None:
                    cover_mapped = set()
                    for s in C:
                        original_s = s_map.get(s)
                        if original_s is not None:
                            cover_mapped.add(original_s)
                    if Q in cover_mapped:
                        final_cover = (cover_mapped - {Q}).union({R1, R2})
                    else:
                        final_cover = cover_mapped.union({R})
                    return final_cover
                else:
                    return {R}
    
    # Reduction Rule 4
    if rules_dict[4]:
        if len(S_max) <= 2:
            #use nx library to compute maximum matchnig in polynomial time
            G = nx.Graph()

            #add vertices
            for e in U:
                G.add_node(e)
            #add edges
            for s in S:
                if len(s) == 2:
                    e1, e2 = s
                    G.add_edge(e1, e2)

            matching = nx.max_weight_matching(G, maxcardinality=True)

            cover = set()
            matched_vertices = set()
            #add edges of the maximum edges
            for e1, e2 in matching:
                cover.add(frozenset({e1, e2}))
                matched_vertices.update([e1, e2])

            #add also the sets that contains the uncovered vertex
            unmatched = U - matched_vertices
            for e in unmatched:
                for s in S:
                    if e in s:
                        cover.add(s)
                        break

            return cover


    # Select the set with maximum cardinality (largest subset)
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
    cover_with_S_max = minimum_set_cover_reduction_rules(S_removed, new_U, rules)

    if cover_with_S_max is not None:
        #get the original mapping from the minimum set
        cover_with_S_max = {frozenset(s_map[s]) for s in cover_with_S_max}
        cover_with_S_max.add(S_max)
    else:
        cover_with_S_max = {S_max}
    
    # Case 2: Do not take S_max in the set cover
    cover_without_S_max = minimum_set_cover_reduction_rules(S_rest, U, rules)
    
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




def main():
    args = parse_arguments()
    
    # Load Graph
    graph = Graph(args.input, sol_path=None)
    U, S, node_map = graph_to_set_cover(graph)
    
    # Solve Set Cover Problem
    solution = minimum_set_cover_reduction_rules(S, U, args.rules)
    
    print(f"Minimum Set Cover Size: {len(solution)}")
    print("Cover Sets:", solution)
    
    # Convert solution to a dominating set
    dominating_set = [node_map[frozenset(s)] for s in solution]
    graph.add_dominating_set_from_list(dominating_set)
    
    # Generate PDF Output
    gv = graph.to_graphviz()
    gv.render(args.output.replace(".pdf", ""))
    print(f"Output saved to {args.output}")

if __name__ == "__main__":
    main()





# def test_complexity(solver, graph_folder_path):
#     """
#     solver: the function that solves the set cover
#     """

#     folder_path = graph_folder_path
#     execution_times = []
#     for file_name in sorted(os.listdir(folder_path)):
#         file_path = os.path.join(folder_path, file_name)
#         if os.path.isfile(file_path): 
#             print(file_name)
#             graph = Graph(file_path, sol_path=None)
#             U, S, node_map = graph_to_set_cover(graph)

#             time_before = time.time()
#             solution = solver(S, U)
#             time_after = time.time()
#             print(f"length of solution {len(solution)}")
#             dt = time_after - time_before
#             execution_times.append(dt)
#             print(f"time passed to solve {dt}")
#             print("Minimum Set Cover:", solution)
        
#     return execution_times


# def plot_execution_time(execution_times, save_name = "execution time", lower_bound = 5, upper_bound = 40):
#     """
#     execution times: a list containing execution times
#     """
#     x_values = np.linspace(lower_bound, upper_bound, len(execution_times))

#     plt.plot(x_values, execution_times)
#     plt.title("Execution time over edge probability")
#     plt.ylabel("execution time")
#     plt.savefig(save_name, format='png')

# def test():
#     # Example Usage

#     """S = {
#         frozenset({1, 2, 3}),
#         frozenset({2, 4}),
#         frozenset({3, 5}),
#         frozenset({4, 5, 6}),
#         frozenset({6})
#     }
#     U = {1, 2, 3, 4, 5, 6}


#     solution = trivial_set_cover(S, U)
#     print("Minimum Set Cover:", solution)"""

#     file_path = "tests/bremen_subgraph_20.gr"
#     # file_path = "generated_graphs_increasing_vertices/graph_n55_p_0.5.gr"

#     graph = Graph(file_path, sol_path=None)

#     U, S, node_map = graph_to_set_cover(graph)
    
#     #print(f"U: {U}")
#     print(f"S: {S}")

#     # solution = minimum_set_cover(S, U)
#     rules = [1, 2, 3, 4, 5, 6, 7, 8]
#     solution = minimum_set_cover_reduction_rules(S, U, rules)
#     print(f"length of solution {len(solution)}")
#     print("Minimum Set Cover:", solution)

#     dominating_set = [node_map[frozenset(s)] for s in solution]
#     print(f"dominating set is {dominating_set}")

#     graph.add_dominating_set_from_list(dominating_set)

#     gv = graph.to_graphviz()
#     gv.render('set_cover_sol')


# if __name__ == "__main__":
#     #a = frozenset({5})
#     # test()
#     # """
#     #graph_folder_path = "generated_graphs_increasing_edge"
    
#     graph_folder_path = "generated_graphs_increasing_vertices"
#     # execution_times = test_complexity(minimum_set_cover, graph_folder_path)
#     execution_times = test_complexity(minimum_set_cover_reduction_rules, graph_folder_path)
#     plot_execution_time(execution_times, save_name="execution_time_vertices.png", lower_bound=5, upper_bound=60)
#     print(execution_times)

#     # """
#     """file_path = "./generated_graphs_increasing_vertices/graph_n55_p_0.5.gr"
#     g = Graph(file_path, sol_path=None)
#     gv = g.to_graphviz()
#     gv.render('test')"""
