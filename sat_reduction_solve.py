from Graph import Graph
from pysat.solvers import Solver

def add_clauses_from_graph(graph, solver):
    """
    This function takes a graph and a solver and it add the litterals to the solver
    by applying a reduction of the minimum dominating set to sat
    """
    g = graph.neighbors

    for v in g:
        clause = [int(v)]
        for u in g[v]:
            clause.append(int(u))
        solver.add_clause(clause)


def minimize_solver(graph):
    """
    this function try to find minimum number of ones in the given solver
    """
    #get the list of litterals
    litterals = []
    for v in graph.nodes:
        litterals.append(int(v))

    #binary search for minimum k
    upper_bound = graph.nb_vertices
    lower_bound = 1

    while upper_bound - lower_bound > 1:
        #solver generation with new constraints
        s = Solver(name='mc')
        add_clauses_from_graph(graph, s)

        k = (upper_bound + lower_bound)//2
        s.add_atmost(lits=litterals, k = k, no_return=False)
        print(k)
        if(s.solve()):
            print("here")
            upper_bound = k
        else:
            lower_bound = k

    sol = s.get_model()

    positive_sol = [x for x in sol if x > 0]

    return k, positive_sol


if __name__ == "__main__":
    file_path = "tests/bremen_subgraph_20.gr"
    sol_path = "tests/bremen_subgraph_20.sol"

    graph = Graph(file_path, sol_path)

    s = Solver(name='mc')

    add_clauses_from_graph(graph, s)

    dom_set_size, sol =  minimize_solver(graph)

    graph.add_dominating_set_from_list(sol)

    print(dom_set_size)
    print(sol)

    gv = graph.to_graphviz()
    gv.render('my_solution')
