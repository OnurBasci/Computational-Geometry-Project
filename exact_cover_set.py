import argparse
import json
import sys
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from graph import Graph

def minimum_set_cover_reduction_rules(S, U, rules=list(range(1, 8))):
    """
    Recursively computes the minimum set cover using various reduction rules.

    Args:
        S (set): A collection of sets to cover the universe
        U (set): The universe of elements to be covered
        rules (list, optional): List of reduction rules to apply. Defaults to all rules.

    Returns:
        set: A minimal set cover, or None if no valid cover exists
    """
    # Convert rules list to dictionary for quick lookup
    rules_dict = {i: (i in rules) for i in range(1, 8)}

    # Base case: If no more subsets are left
    if len(S) == 0:
        return set() if len(U) == 0 else None

    # Select the set with maximum cardinality (largest subset)
    S_max = max(S, key=len)
    
    # Reduction Rule 2
    if(len(S_max) <= 1):
        return {frozenset({e}) for e in U}
    
    counter = dict()
    sets_containing = dict()
    for s in S:
        for e in s:
            if e not in counter:
                counter[e] = (0,s)
                sets_containing[e] = set()
            counter[e] = (counter[e][0]+1, s)
            sets_containing[e].add(s)

    # Reduction Rule 1
    if rules_dict[1]:
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
    Transforms a graph data structure into a set cover problem representation.

    Args:
        graph (Graph): Input graph to be transformed

    Returns:
        tuple: A 3-tuple containing:
            - U (set): Universe of elements
            - S (set): Collection of sets
            - node_mapping (dict): Mapping between sets and original nodes
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




def solve(graph, config):
    """
    Solves the dominating set problem for the given graph using a cover set reduction.

    Args:
        graph (Graph): The input graph to solve.
        config (dict): Configuration dictionary.

    Returns:
        Graph: The graph with the dominating set added and colored.
    """
    # Exact method using set cover reduction rules
    U, S, node_map = graph_to_set_cover(graph)
    
    # Apply selected reduction rules
    rules = config.get('reduction_rules', list(range(1, 8)))
    solution = minimum_set_cover_reduction_rules(S, U, rules)
    
    # Convert solution to a dominating set
    dominating_set = [node_map[frozenset(s)] for s in solution]
    graph.add_dominating_set_from_list(dominating_set)
    return graph