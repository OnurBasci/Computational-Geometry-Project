import random
from Graph import Graph

class Graph_Generator:
    """
    A class for generating random graphs with various methods.
    """

    def __init__(self, n, m, p):
        """
        Initializes a Graph_Generator with specified parameters.

        Args:
            n (int): Number of vertices in the graph.
            m (int): Number of edges per vertex.
            p (float): Probability of edge creation in Erdős-Rényi model.
        """
        self.n = n  # number of vertices
        self.m = m  # number of edges
        self.p = p  # probability for erdos renyi graph
        self.vertices = {}
        self.edges = []

    def generate(self):
        """
        Generates a list of edges for a random graph with a fixed number of edges per vertex.

        Returns:
            list: A list of unique edges in the graph.
        """
        edges = set()
        neighbor_map = dict()
        
        # Initialize neighbor map
        for node in range(self.n):
            neighbor_map[node] = []

        # Ensure each node gets m neighbors
        for node in range(self.n):
            for j in range(len(neighbor_map[node]), self.m):
                neighbor = random.randint(0, self.n - 1)
                
                # Find a valid neighbor that doesn't create a duplicate edge
                while neighbor == node or tuple(sorted((node, neighbor))) in edges:
                    neighbor = random.randint(0, self.n - 1)
        
                neighbor_map[node].append(neighbor)
                neighbor_map[neighbor].append(node)
                
                print(sorted((node,neighbor)))
                edge = tuple(sorted((node, neighbor)))  # Ensure undirected edges
                edges.add(edge)
            
            print(f"neighbors n {len(neighbor_map[node])}")
            print(neighbor_map[node])

        self.edges = list(edges)
        return self.edges
    
    def generate_erdos_renyi_graph(self):
        """
        Generates an Erdős-Rényi random graph.

        An Erdos-Renyi graph is characterized by two parameters:
        - n: number of nodes
        - p: probability that any two nodes are connected

        Returns:
            list: A list of edges in the Erdős-Rényi graph.
        """
        for i in range(self.n):
            for j in range(i + 1, self.n):  # Avoid self-loops and duplicate edges
                if random.random() < self.p:
                    self.edges.append((i, j))

        return self.edges

    def write_gr_file(self, filename="graph.gr"):
        """
        Writes the generated graph to a .gr file in PACE challenge format.

        Args:
            filename (str, optional): Path to the output file. Defaults to "graph.gr".
        """
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
    """
    Generates a series of Erdős-Rényi graphs with increasing edge probability.

    Creates graphs with:
    - Fixed number of vertices (20)
    - Probability ranging from 0.2 to 0.8
    - Incremental step of 0.02
    """
    n = 20
    probability_lower_bound = 0.2
    probability_upper_bound = 0.8
    step = 0.02
    p = probability_lower_bound
    
    while p < probability_upper_bound:
        print(p)
        graph = Graph_Generator(n, 5, p)
        edge_list = graph.generate_erdos_renyi_graph()

        # Save the file
        file_path = f"generated_graphs/graph_n{n}_p_{p}.gr"
        graph.write_gr_file(file_path)

        p += step


def generate_increasing_edge_graph_and_save(vertice_num=20):
    """
    Generates a series of Erdős-Rényi graphs with increasing edge probability.

    Args:
        vertice_num (int, optional): Fixed number of vertices. Defaults to 20.
    """
    n = vertice_num
    probability_lower_bound = 0.2
    probability_upper_bound = 0.8
    step = 0.02
    p = probability_lower_bound
    
    while p < probability_upper_bound:
        print(p)
        graph = Graph_Generator(n, 5, p)
        edge_list = graph.generate_erdos_renyi_graph()

        # Save the file
        file_path = f"generated_graphs/graph_n{n}_p_{p}.gr"
        graph.write_gr_file(file_path)

        p += step


def generate_increasing_vertex_graph(edge_prob=0.5, lower_bound=5, upper_bound=30):
    """
    Generates a series of Erdős-Rényi graphs with increasing number of vertices.

    Args:
        edge_prob (float, optional): Fixed probability of adding edges. Defaults to 0.5.
        lower_bound (int, optional): Minimum number of vertices. Defaults to 5.
        upper_bound (int, optional): Maximum number of vertices. Defaults to 30.
    """
    lower_bound_vertex = lower_bound
    upper_bound_vertex = upper_bound
    step = 5
    n = lower_bound_vertex
    p = edge_prob
    
    while n <= upper_bound_vertex:
        graph = Graph_Generator(n, 5, p)
        graph.generate_erdos_renyi_graph()

        # Save the file
        file_path = f"generated_graphs_increasing_vertices/graph_n_{n:03}_p_{p}.gr"
        graph.write_gr_file(file_path)
        
        n += step


if __name__ == "__main__":
    generate_increasing_vertex_graph(edge_prob=0.5, lower_bound=5, upper_bound=60)
    #generate_increasing_edge_graph_and_save()
