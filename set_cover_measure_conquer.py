from Graph import Graph

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



if __name__ == "__main__":
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

    graph = Graph(file_path, sol_path=None)

    U, S, node_map = graph_to_set_cover(graph)
    
    #print(f"U: {U}")
    print(f"S: {S}")

    solution = minimum_set_cover(S, U)
    print(f"length of solution {len(solution)}")
    print("Minimum Set Cover:", solution)

    dominating_set = [node_map[frozenset(s)] for s in solution]
    print(f"dominating set is {dominating_set}")

    graph.add_dominating_set_from_list(dominating_set)

    gv = graph.to_graphviz()
    gv.render('set_cover_sol')
