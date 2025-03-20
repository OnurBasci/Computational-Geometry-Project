import random
from Graph import Graph

class Graph_Generator:

    def __init__(self, n, m, p):
        self.n = n #number of vertices
        self.m = m #number of edges
        self.p = p #probability for erdos renyi graph
        self.vertices = {}
        self.edges = []

    def generate(self):
        """
        generates a list of edges that corresponds to the graph
        """
        edges = set()

        neighbor_map = dict()
        for node in range(self.n):
            neighbor_map[node] = []

        # Ensure each node gets m neighbors
        for node in range(self.n):
            
            for j in range(len(neighbor_map[node]), self.m):
                neighbor = random.randint(0, self.n - 1)
                #while neighbor == node or neighbor in neighbor_map[node] or node in neighbor_map[neighbor]:
                while neighbor == node or tuple(sorted((node, neighbor))) in edges:
                    neighbor = random.randint(0, self.n - 1)
        
                neighbor_map[node].append(neighbor)
                neighbor_map[neighbor].append(node)
                #print(neighbor == node)
                #print(f"node {node}, neighbor {neighbor}")
                print(sorted((node,neighbor)))
                edge = tuple(sorted((node, neighbor)))  # Ensure undirected edges
                edges.add(edge)
            print(f"neighbors n {len(neighbor_map[node])}")
            print(neighbor_map[node])

        self.edges = list(edges)
        return self.edges
    
    def generate_erdos_renyi_graph(self):
        """
        An Erdos-Renyi graph is characterized by two parameters: the number of nodes n and the probability p
        that any two nodes are connected
        """
        for i in range(self.n):
            for j in range(i + 1, self.n):  # Avoid self-loops and duplicate edges
                if random.random() < self.p:
                    self.edges.append((i, j))

        return self.edges

    def write_gr_file(self, filename="graph.gr"):

        m = self.m 
        n = self.n

        with open(filename, "w") as f:
            # Write the header line
            f.write(f"p ds {n} {m}\n")
            
            # Write each edge, converting 0-based indices to 1-based
            for u, v in self.edges:
                f.write(f"{u+1} {v+1}\n")  # Convert 0-based to 1-based indices

        print(f"Graph saved to {filename}")


def generate_increasing_edge_graph_and_save():
    n = 20
    probability_lover_bound = 0.2
    probability_upper_bound = 0.8
    step = 0.02
    p = probability_lover_bound
    while p < probability_upper_bound:
        print(p)
        graph = Graph_Generator(n, 5, p)
        edge_list = graph.generate_erdos_renyi_graph()

        #save the file
        file_path = "generated_graphs/graph_n" + str(n) + "_p_" + str(p) + ".gr"
        graph.write_gr_file(file_path)

        p += step


def generate_increasing_edge_graph_and_save(vertice_num=20):
    """
    vertice_num: fixed number of vertices
    """
    n = n
    probability_lover_bound = 0.2
    probability_upper_bound = 0.8
    step = 0.02
    p = probability_lover_bound
    while p < probability_upper_bound:
        print(p)
        graph = Graph_Generator(n, 5, p)
        edge_list = graph.generate_erdos_renyi_graph()

        #save the file
        file_path = "generated_graphs/graph_n" + str(n) + "_p_" + str(p) + ".gr"
        graph.write_gr_file(file_path)

        p += step


def genereate_increasing_vertex_graph(edge_prob=0.5, lower_bound = 5, upper_bound = 30):
    """
    edge_prob: fixed probability of adding edges
    """
    lower_bound_vertex = lower_bound
    upper_bound_vertex = upper_bound
    step = 5
    n = lower_bound_vertex
    p = edge_prob
    i = 0
    while n <= upper_bound_vertex:
        i+=1
        graph = Graph_Generator(n, 5, p)
        graph.generate_erdos_renyi_graph()

        #save the file
        file_path = f"generated_graphs_increasing_vertices/graph_n_{n:03}_p_{p}.gr"
        graph.write_gr_file(file_path)
        n += step


def test():
    n = 20
    m = 5
    p = 0.2
    graph = Graph_Generator(n, m, p)
    edge_list = graph.generate_erdos_renyi_graph()
    print(edge_list)

    #save graph
    file_path = "deneme.gr"
    graph.write_gr_file(file_path)

    g = Graph(file_path, sol_path=None)
    gv = g.to_graphviz()
    gv.render('test')

if __name__ == "__main__":
    genereate_increasing_vertex_graph(edge_prob=0.5, lower_bound=5, upper_bound=60)
    #generate_increasing_edge_graph_and_save()
