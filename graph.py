import graphviz

class Node:
    """
    Represents a node in a graph.
    """
    def __init__(self, label, color=None):
        """
        Initializes a node with a label and an optional color.

        Args:
            label (str): The label or identifier of the node.
            color (str, optional): The color of the node. Defaults to None.
        """
        self.label = label
        self.color = color

    def __str__(self):
        """
        Returns a string representation of the node.

        Returns:
            str: A string showing the node's label and color.
        """
        return f"(label: {self.label}, color: {self.color})"

    __repr__ = __str__


class Graph:
    """
    Represents a graph with nodes and edges. Can be initialized from a file and optionally a solution file.
    """
    def __init__(self, file_path=None, sol_path=None):
        """
        Initializes a graph. If file paths are provided, it reads the graph structure and solution from the files.

        Args:
            file_path (str, optional): Path to the .gr file containing the graph structure. Defaults to None.
            sol_path (str, optional): Path to the .sol file containing the dominating set solution. Defaults to None.
        """
        self.nodes = {}  # Dictionary to store nodes by their labels
        self.neighbors = {}  # Dictionary to store adjacency lists for each node
        
        self.nb_vertices = 0  # Number of vertices in the graph
        self.nb_edges = 0  # Number of edges in the graph

        self.nb_vertices_dominating_set = 0  # Number of vertices in the dominating set
        self.dominating_set = set()  # Set of nodes in the dominating set

        if file_path is not None:
            with open(file_path, 'r') as file:
                for line in file:
                    line = line.strip()
                    if not line or line.startswith('c'):
                        continue  # Ignore comment lines
                    
                    if line.startswith('p'):
                        parts = line.split()
                        if len(parts) != 4 or parts[1] != 'ds':
                            raise ValueError("Invalid problem descriptor in .gr file")
                        continue
                    
                    parts = line.split()
                    if len(parts) == 2:
                        try:
                            u, v = int(parts[0]), int(parts[1])
                            u = Node(str(u))
                            v = Node(str(v))
                            
                            self.add_node(u)
                            self.add_node(v)
                            self.add_edge(u,v)
                        except ValueError:
                            raise ValueError(f"Invalid edge definition: {line}")

        if (sol_path is not None) and (file_path is not None):
            with open(sol_path, 'r') as file:
                first_line_found = False
                for line in file:
                    line = line.strip()
                    if not line or line.startswith('c'):
                        continue  # Ignore comment lines

                    if not first_line_found:
                        self.nb_vertices_dominating_set = int(line[0])
                        first_line_found = True
                        continue

                    node_label = str(int(line))
                    node = self.nodes[node_label]
                    node.color = "red"
                    self.dominating_set.add(node)

    def add_node(self, node):
        """
        Adds a node to the graph.

        Args:
            node (Node): The node to add.
        """
        if not node.label in self.nodes:
            self.nodes[node.label] = node
            self.neighbors[node.label] = []
            self.nb_vertices += 1

    def add_edge(self, node_1, node_2):
        """
        Adds an edge between two nodes in the graph.

        Args:
            node_1 (Node): The first node.
            node_2 (Node): The second node.

        Raises:
            AssertionError: If the nodes are not already in the graph or if the edge already exists.
        """
        assert(node_1.label in self.nodes.keys())
        assert(node_2.label in self.nodes.keys())
        assert(node_2.label not in self.neighbors[node_1.label])
        assert(node_1.label not in self.neighbors[node_2.label])
        self.neighbors[node_1.label].append(node_2.label)
        self.neighbors[node_2.label].append(node_1.label)

    def add_dominating_set_from_list(self, dominating_list):
        """
        Adds a dominating set to the graph from a list of node labels.

        Args:
            dominating_list (list): A list of node labels representing the dominating set.
        """
        for u in dominating_list:
            node = self.nodes[str(u)]
            node.color = "red"
            self.dominating_set.add(node)

    def to_graphviz(self):
        """
        Converts the graph to a Graphviz object for visualization.

        Returns:
            graphviz.Graph: A Graphviz representation of the graph.
        """
        g = graphviz.Graph()
        edges = []
        for node_label, node in self.nodes.items():
            g.node(node.label, fillcolor=node.color, style="filled")
            for neighbor_label in self.neighbors[node_label]:
                neighbor = self.nodes[neighbor_label]
                if (node.label, neighbor.label) not in edges and (neighbor.label, node.label) not in edges:
                    edges.append((node.label, neighbor.label))
        for i,j in edges:
            g.edge(i,j)
        return g

    def to_sol(self, path):
        """
        Creates a solution file formated for the PACE challenge

        Args:
            path (str): Path to the output file
        """
        with open(path, 'w') as file:
            file.write(f"{len(self.dominating_set)}\n")
            for node in self.dominating_set:
                file.write(f"{node.label}\n")
