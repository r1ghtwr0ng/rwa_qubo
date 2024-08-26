import re
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
from dimod import BinaryQuadraticModel
from dwave.samplers import SimulatedAnnealingSampler
import numpy as np

class ConnectedNetwork:
    "Contains precomputed information on all routes in a network"
    def __init__(self, filename):
        node_regex = r'^\[(\d{1,5})\]$'
        nodes_regex = r'^NUMBER OF NODES:\s{0,1}(\d{1,5})'
        links_regex = r'^NUMBER OF LINKS:\s{0,1}(\d{1,5})'
        with open(filename) as f:
            # Read nodes and links count from file
            nodes = re.findall(nodes_regex, f.readline().upper())
            links = re.findall(links_regex, f.readline().upper())
            if len(nodes) == 0 or len(links) == 0:
                err_msg = f"[-] Invalid file format for: {filename}"
                raise InvalidFileError(err_msg)
            self.__nodes = int(nodes[0])
            self.__links = int(links[0])

            # Create a graph with the correct number of nodes
            self.__G = nx.Graph()
            self.__G.add_nodes_from(range(self.__nodes))

            # Load all entries
            entries = f.readlines()[3:]

        #print(f"[i] {filename}: nodes - {self.__nodes} links - {self.__links}")
        self.__routes_dbase = {}
        self.__link_mappings = {}
        self.__routes_selection = {}
        # Loop through each possible route
        for route in entries:
            values = route.split()

            # Skip empty lines
            if len(values) == 0:
                continue

            try:
                src = int(re.findall(node_regex, values[0])[0])
                dest = int(re.findall(node_regex, values[1])[0])
            except (IndexError, ValueError):
                err_msg = '[-] Invalid src/dest node entry'
                raise InvalidFileError(err_msg)
            
            hops = values[2].split('/')
            paths = values[3].split('/')

            # Skip self-reference routes
            if hops[0] == '0':
                continue
            elif hops[0] == '1':
                link_idx = int(paths[0])
                self.__G.add_edge(src, dest, index=link_idx)
                self.__link_mappings[link_idx] = (src, dest)

            self.__routes_dbase[(src, dest)] = []
            self.__routes_selection[(src, dest)] = 0
            for path in paths:
                if path == '-':
                    break
                links_arr = [int(link_id) for link_id in path.split('-')]
                self.__routes_dbase[(src, dest)].append(links_arr)
    
    # ---- GETTERS ----
    def get_graph(self):
        return self.__G.copy()

    def get_routes(self, pair=None):
        try:
            return self.__routes_dbase[pair]
        except KeyError:
            return None
    
    def get_pairs(self):
        return list(self.__routes_dbase.keys())

    def build_routing_graph(self, path_pairs=None):
        G = nx.Graph()
        if path_pairs is None:
            path_pairs = self.__routes_dbase.keys()
        
        idx = 0
        routes_map = {}
        for pair in path_pairs:
            for route in self.__routes_dbase[pair]:
                routes_map[idx] = (pair, route, len(route))
                G.add_node(idx)

                # Add edge for routes which share a link
                for i in range(idx):
                    if self.common_link(route, routes_map[i][1]):
                        G.add_edge(i, idx)

                idx += 1
        
        return G, routes_map
    
    def link_lookup(self, link_id):
        try:
            return self.__link_mappings[link_id]
        except KeyError:
            return None

    def plot_route(self, route, src, dest, pos, filename):
        plt.figure(figsize=(10, 8))

        # Set colors and layout
        node_color = '#DAA520'
        edge_color = '#000000'
        label_color = '#000000'
        background_color = '#F0F0F0'

        # Draw nodes and edges
        nx.draw_networkx_nodes(self.__G, pos, node_size=700, node_color=node_color, edgecolors='black')
        nx.draw_networkx_edges(self.__G, pos, width=2.0, alpha=0.7, edge_color=edge_color)

        # Highlight the edges in the route
        route_edges = [self.__link_mappings[link_id] for link_id in route]
        nx.draw_networkx_edges(self.__G, pos, edgelist=route_edges, edge_color='red', width=4)

        # Highlight the nodes in the route
        route_nodes = set()
        for edge in route_edges:
            route_nodes.update(edge)
        nx.draw_networkx_nodes(self.__G, pos, nodelist=list(route_nodes), node_size=700, node_color=node_color, edgecolors='red')

        # Highlight the source and destination nodes
        nx.draw_networkx_nodes(self.__G, pos, nodelist=[src], node_color='green', node_size=700, edgecolors='red')
        nx.draw_networkx_nodes(self.__G, pos, nodelist=[dest], node_color='blue', node_size=700, edgecolors='red')

        # Labels with adjusted font and size
        nx.draw_networkx_labels(self.__G, pos, font_size=10, font_family="sans-serif", font_color=label_color)

        plt.axis("off")
        plt.gca().set_facecolor(background_color)  # Set background color
        plt.title(f"Route from {src} to {dest}")
        plt.savefig(filename)
        print(f"[i] Plotting: {filename} (new run)")
        plt.close()

    def print_mappings(self):
        for k, v in self.__link_mappings.items():
            print(f"{k} -> {v}")

    def generate_route_images(self, prefix=""):
        pos = nx.spring_layout(self.__G)
        nodes = sorted(self.__G.nodes)
        for src in nodes:
            for dest in nodes:
                if src == dest:
                    continue
                routes = self.__routes_dbase[(src, dest)]
                for i, route in enumerate(routes):
                    filename = f"../other/images/{prefix}route_{src:02d}_{dest:02d}_path_{i+1}.png"
                    self.plot_route(route, src, dest, pos, filename)
    
    def network_to_gcp(self, routes=None):
        # Best routes selection
        if routes is None:
            routes = [(x, y) for x in range(self.__nodes) for y in range(self.__nodes) if x != y]
        
        link_to_route_map = {}
        gcp = nx.Graph()
        gcp.add_nodes_from(range(len(routes)))

        for i, pair in enumerate(routes):
            (x, y) = pair
            if x == y:
                continue
            
            route_idx = self.__routes_selection[pair]
            edges = self.__routes_dbase[pair][route_idx]
            for edge in edges:
                gcp, link_to_route_map = self.update_graph_and_map(gcp, link_to_route_map, edge, i)
        
        return gcp

    @staticmethod
    def get_max_hops(reverse_mapping):
        return max(lst for _, _, lst in reverse_mapping.values())

    @staticmethod
    def common_link(route_1, route_2):
        return any(x in route_1 for x in route_2)

    @staticmethod
    def update_graph_and_map(graph, links_map, edge, route):
        if edge not in links_map:
            links_map[edge] = [route]
        else:
            if route not in links_map[edge]:
                for existing_route in links_map[edge]:
                    graph.add_edge(existing_route, route)
                links_map[edge].append(route)
        
        return graph, links_map

    @staticmethod
    def check_solution(sample, pair_routes, reverse_mapping):
        pair_status = {pair: False for pair in pair_routes}

        for route_idx, value in sample.items():
            if value == 1:
                pair = reverse_mapping[route_idx][0]
                if not pair_status[pair]:
                    pair_status[pair] = True
                else:
                    #print(f"Not zero! {sample}")
                    return False

        if all(pair_status.values()):
            #print(f"Valid solution")
            #for pair, routes in pair_routes.items():
                #print(f"PATH: [ {pair[0]} ==> {pair[1]} ]")
                #for rid in routes:
                    #if sample[rid] == 1:
                        #print(f"** {rid}")
            return True
        elif any(pair_status.values()):
            #print(f"Not zero! {sample}")
            return False
        else:
            return False

    def solve_qubo(self, c1, c2, c3, c4, num_runs=10, sampler=SimulatedAnnealingSampler()):
        G, reverse_mapping = self.build_routing_graph()

        pair_routes = {}
        Q = {}
        for route_1, (pair_1, _, hops_1) in reverse_mapping.items():
            # Construct pair routes hashmap for easy access
            if pair_1 not in pair_routes:
                pair_routes[pair_1] = [route_1]
            else:
                pair_routes[pair_1].append(route_1)
                
            for route_2, (pair_2, _, hops_2) in reverse_mapping.items():
                if route_1 == route_2:
                    value = c1
                elif pair_1 == pair_2:
                    value = c2 + hops_1 + hops_2
                elif G.has_edge(route_1, route_2):
                    value = c3 + hops_1 + hops_2
                else:
                    value = c4 + hops_1 + hops_2
                Q[(route_1, route_2)] = value

        bqm = BinaryQuadraticModel.from_qubo(Q)
        sampleset = sampler.sample(bqm, num_reads=num_runs, label="RWA QUBO Solving") #, randomize_order=True)
        return sampleset, pair_routes, reverse_mapping, G

    def analyze_solutions(self, sampleset, pair_routes, reverse_mapping):
        total_routes_selected = 0
        pair_selection_count = {pair: 0 for pair in pair_routes}
        
        for sample in sampleset.samples():
            for route_idx, value in sample.items():
                if value == 1:
                    pair = reverse_mapping[route_idx][0]
                    pair_selection_count[pair] += 1
                    total_routes_selected += 1

        num_pairs = len(pair_routes)
        avg_routes_per_pair = total_routes_selected / num_pairs

        return avg_routes_per_pair, pair_selection_count

    def optimize_parameters(self, param_space, num_reads, target_num_solutions, num_sweeps=1000, beta_range=(0.1, 4.2), beta_schedule_type='geometric', plot=False):
        best_params = None
        best_solutions = None
        best_score = float('-inf')

        results = []
        avg_routes_per_pair_results = []
        pair_distribution_results = []
        
        for c1, c2 in tqdm(param_space, desc="Optimizing Parameters"):
            sampleset, pair_routes, reverse_mapping = self.solve_qubo(c1, c2, num_reads, num_sweeps, beta_range, beta_schedule_type)
            avg_routes_per_pair, pair_selection_count = self.analyze_solutions(sampleset, pair_routes, reverse_mapping)

            avg_routes_per_pair_results.append((c1, c2, avg_routes_per_pair))
            pair_distribution_results.append((c1, c2, pair_selection_count))

            feasible_solutions = [sample for sample in sampleset.samples() if self.check_solution(sample, pair_routes, reverse_mapping)]

            if feasible_solutions:
                num_solutions = len(feasible_solutions)
                solutions_list = [list(sample.values()) for sample in feasible_solutions]
                diversity = len(set(map(tuple, solutions_list)))
                if diversity == 0:
                    continue
                
                score = num_solutions + diversity
                
                if num_solutions == target_num_solutions:
                    best_params = (c1, c2)
                    best_solutions = feasible_solutions
                    best_score = score
                    print(f"Found target number of solutions: Best Parameters so far: {best_params}, Best Score: {best_score}")
                    break
                elif score > best_score:
                    best_params = (c1, c2)
                    best_solutions = feasible_solutions
                    best_score = score
                    print(f"New best parameters found: {best_params}, Best Score: {best_score}")

                results.append((c1, c2, score))

        if plot:
            self.plot_heatmap(avg_routes_per_pair_results, "Average Number of Routes per Pair", "avg_routes_per_pair_heatmap.png")
            self.plot_heatmap(pair_distribution_results, "Pair Selection Distribution", "pair_distribution_heatmap.png")
        
        return best_params, best_solutions, best_score

    def plot_heatmap(self, results, title, filename):
        import matplotlib.pyplot as plt
        import numpy as np

        c1_values = sorted(set([r[0] for r in results]))
        c2_values = sorted(set([r[1] for r in results]))

        heatmap = np.zeros((len(c1_values), len(c2_values)))

        for c1, c2, value in results:
            i = c1_values.index(c1)
            j = c2_values.index(c2)
            if isinstance(value, dict):  # For pair distribution results, calculate the average
                value = np.mean(list(value.values()))
            heatmap[i, j] = value

        plt.figure(figsize=(10, 8))
        plt.imshow(heatmap, cmap='inferno', interpolation='nearest', aspect='auto', origin='lower')
        plt.colorbar(label='Value')
        plt.xticks(ticks=range(len(c2_values)), labels=c2_values, rotation=90)
        plt.yticks(ticks=range(len(c1_values)), labels=c1_values)
        plt.xlabel('c2')
        plt.ylabel('c1')
        plt.title(title)
        plt.savefig(filename)
        plt.show()
