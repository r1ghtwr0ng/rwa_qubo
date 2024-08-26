import random
import numpy as np
import networkx as nx

# Create a graph from a list of nodes and edges
def create_graph(nodes, edges):
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return G

# Generate a Watts-Strogatz graph (well-connected)
def create_watts_strogatz(nodes, nearest_neighbors=4, rewiring_prob=0.25):
    # Generate a new graph
    return nx.connected_watts_strogatz_graph(nodes, nearest_neighbors, rewiring_prob)

# Just relablels nodes to remove gaps in names
def relabel_nodes_sequentially(G):
    # Relabel nodes to be sequential starting from 0
    mapping = {node: new_label for new_label, node in enumerate(G.nodes)}
    return nx.relabel_nodes(G, mapping)

def create_network_graph(nodes, min_degree, max_degree):
    # Start with a random graph
    G = nx.gnm_random_graph(nodes, random.randint(nodes - 1, nodes * (nodes - 1) // 2))

    # Adjust the graph to meet degree constraints
    for node in list(G.nodes):
        degree = G.degree[node]
        while degree < min_degree or degree > max_degree:
            if degree > max_degree:
                neighbors = list(G.neighbors(node))
                G.remove_edge(node, random.choice(neighbors))
            elif degree < min_degree:
                potential_nodes = set(G.nodes) - set(G.neighbors(node)) - {node}
                if potential_nodes:
                    G.add_edge(node, random.choice(list(potential_nodes)))
            degree = G.degree[node]

    # Remove isolated nodes if any
    G.remove_nodes_from(list(nx.isolates(G)))

    # Relabel nodes to be sequential
    return relabel_nodes_sequentially(G)

# Convert G's adjacency dict_itemiterator class to a matrix so I can use JIT down the line
def build_adjacency(G):
    # Initialize a zero matrix
    num_nodes = len(G.nodes())
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype='uint8')

    # Populate the matrix
    for node, neighbors in G.adjacency():
        for neighbor in neighbors:
            adj_matrix[node][neighbor] = 1
    
    return adj_matrix

def load_graph(filename):
    """
    Load the graph from a GML file as a MultiGraph.
    """
    try:
        MG = nx.MultiGraph()
        with open(filename, 'r') as f:
            G = nx.parse_gml(f, label='id')
            MG.add_nodes_from(G.nodes(data=True))
            MG.add_edges_from(G.edges(data=True))
        return MG
    except Exception as e:
        print(f"Error loading graph: {e}")
        return None

def remove_isolated_nodes(G):
    """
    Remove nodes with no edges from the graph.
    """
    isolated_nodes = [node for node, degree in dict(G.degree()).items() if degree == 0]
    G.remove_nodes_from(isolated_nodes)
    return G

# Generate a random route in a network
def generate_random_routes(G, n, avg_length):
    routes = []
    nodes = list(G.nodes)

    for _ in range(n):
        route_length = max(2, int(random.gauss(avg_length, 1)))
        while True:
            start_node = random.choice(nodes)
            end_node = start_node
            while end_node == start_node:
                end_node = random.choice(nodes)
            
            try:
                path = nx.shortest_path(G, source=start_node, target=end_node)
                if len(path) <= route_length:
                    routes.append(path)
                    break
            except nx.NetworkXNoPath:
                continue

    return routes

# Generate graph to color from a list of routes and underlying network topology
def generate_route_graph(routes, network_graph):
    route_graph = nx.Graph()
    
    # Add nodes representing each route
    for i, route in enumerate(routes):
        route_graph.add_node(i, route=route)
    
    # Add edges between routes that share edges in the network graph
    for i, route1 in enumerate(routes):
        for j, route2 in enumerate(routes):
            if i < j:
                if any((route1[k], route1[k+1]) in network_graph.edges or 
                       (route1[k+1], route1[k]) in network_graph.edges for k in range(len(route1) - 1) for l in range(len(route2) - 1)):
                    route_graph.add_edge(i, j)
    
    return route_graph