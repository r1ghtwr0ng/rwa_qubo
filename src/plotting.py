import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from .graph import load_graph, remove_isolated_nodes

# Generate a list of W colors used to draw the nodes on the graph
def generate_color_list(W):
    colors = ['red', 'green', 'skyblue', 'orange', 'yellow', 'purple', 'cyan', 'magenta', 'lime', 'pink', 'teal', 'lavender', 'brown', 'beige', 'maroon', 'mint', 'olive', 'coral', 'navy', 'grey']
    
    # Index checking
    if W > len(colors):
        raise ValueError("Not enough predefined colors for the requested number W.")

    return colors[:W]

# Plot a graph with matplotlib
def display_graph(G, seed=None):
    plt.figure(figsize=(10, 8))

    # Set colors and layout
    node_color = '#FFFFFF'
    edge_color = '#000000'
    label_color = '#000000'
    background_color = '#F0F0F0'

    if seed is None:
        seed = random.randint(1, 2**32 - 1)
        print(f"[+] DEBUG: Setting seed for plotting position {seed}")
    
    pos = nx.spring_layout(G, k=0.8, iterations=100, threshold=1e-4, scale=2, seed=seed)

    # Draw nodes and edges
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color=node_color, edgecolors='black')
    nx.draw_networkx_edges(G, pos, width=2.0, alpha=0.7, edge_color=edge_color)

    # Labels with adjusted font and size
    nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif", font_color=label_color)

    plt.axis("off")
    plt.gca().set_facecolor(background_color)  # Set background color
    plt.show()

# Draw the assigned colors onto the nodes and show the solution
def display_solution(G, solution, colors):
    node_colors = []
    for color in solution:
        # Check the number of ones found
        if color == -1:
            node_colors.append('black')
        else:
            node_colors.append(colors[color])
    
    pos = nx.spring_layout(G)

    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=700, edgecolors='black')
    plt.show()

def get_boundaries(pos):
    """
    Calculate the boundaries for the Basemap based on node positions.
    """
    lons, lats = zip(*pos.values())
    margin = 5  # degrees of margin around the furthest nodes
    llcrnrlon = min(lons) - margin
    urcrnrlon = max(lons) + margin
    llcrnrlat = min(lats) - margin
    urcrnrlat = max(lats) + margin
    return llcrnrlon, urcrnrlon, llcrnrlat, urcrnrlat

def visualize_network(filename):
    """
    Load, check, and visualize the network.
    """
    # Load the graph
    G = load_graph(filename)
    if G is None:
        return

    # Check for duplicate labels
    G = remove_isolated_nodes(G)

    # Extract positions from the node attributes with default values for missing data
    pos = {node: (data.get('Longitude', -75), data.get('Latitude', 40)) for node, data in G.nodes(data=True)}

    # Plot using the geographical positions
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos=pos, with_labels=False, node_size=50, node_color='darkred', edge_color='green')
    plt.show()

    # Calculate boundaries
    llcrnrlon, urcrnrlon, llcrnrlat, urcrnrlat = get_boundaries(pos)

    # For more accurate geographical plotting using Basemap
    plt.figure(figsize=(12, 8))

    # Setup Basemap with dynamic boundaries
    m = Basemap(projection='merc', llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, lat_ts=20, resolution='i')

    # Draw coastlines and countries
    m.drawcoastlines()
    m.drawcountries()

    # Convert positions to map projection with default values for missing data
    mapped_pos = {node: m(data.get('Longitude', -75), data.get('Latitude', 40)) for node, data in G.nodes(data=True)}

    # Draw the graph
    nx.draw(G, pos=mapped_pos, with_labels=False, node_size=50, node_color='darkred', edge_color='green')
    plt.show()

def plot_colored_solution(G, solution, seed=None):
    wavelengths = max(solution.values()) + 1
    # Plot solution
    print(f"[+] Solution found, wavelengths: {wavelengths}")
    
    colors = generate_color_list(wavelengths)

    node_colors = []
    for color in solution.values():
        # Check the number of ones found
        if color == -1:
            node_colors.append('black')
        else:
            node_colors.append(colors[color])

    if seed is None:
        seed = random.randint(1, 2**32 - 1)
        print(f"[+] DEBUG: Setting seed for plotting position {seed}")

    pos = nx.spring_layout(G, k=0.8, iterations=100, threshold=1e-4, scale=2, seed=seed)
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=700, edgecolors='black')
    plt.show()

# Plot histogram for RWA solution quality based on route selection (best to worst, depending on hop count)
def plot_histogram(counts):
    # Average the counts
    percentages = [x / sum(counts) * 100 for x in counts]

    # Data for histogram
    labels = ['Optimal Route', 'Second Best Route', 'Third Best Route']
    x = np.arange(len(labels))

    # Adjusted positions to bring bars closer together
    spacing = 0.4  # Adjust this value as needed to control spacing
    x_adjusted = x * (1 - spacing)

    # Plot histogram with enhanced y-axis and grid lines
    plt.figure(figsize=(6, 8))
    bars = plt.bar(x_adjusted, percentages, color='skyblue', width=0.5)

    # Add percentages above the bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.5, f'{height:.1f}%', ha='center', va='bottom', fontsize=15)

    plt.xticks(x_adjusted, labels, fontsize=11)
    plt.ylabel('Selection Frequency (%)', fontsize=15)
    plt.title('Selected Routes by Hop Count', fontsize=16)
    plt.ylim(0, max(percentages) + 10)
    plt.yticks(range(0, 101, 10))
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.show()

def plot_combined_histogram(sim_counts, q_counts, hybrid_counts, labels):
    # Calculate percentages for each set of counts
    sim_percentages = [x / sum(sim_counts) * 100 for x in sim_counts]
    q_percentages = [x / sum(q_counts) * 100 for x in q_counts]
    hybrid_percentages = [x / sum(hybrid_counts) * 100 for x in hybrid_counts]

    # Data for histogram
    categories = ['Least Hops (Best) Route', 'Second Best Route', 'Third Best Route']
    x = np.arange(len(categories))

    # Width and positions for grouped bars
    bar_width = 0.25
    x_sim = x - bar_width
    x_q = x
    x_hybrid = x + bar_width

    # Plot histogram with enhanced y-axis and grid lines
    plt.figure(figsize=(10, 6))
    bars_sim = plt.bar(x_sim, sim_percentages, width=bar_width, color='skyblue', label='Simulated Annealing')
    bars_q = plt.bar(x_q, q_percentages, width=bar_width, color='orange', label='Quantum Annealing')
    bars_hybrid = plt.bar(x_hybrid, hybrid_percentages, width=bar_width, color='green', label='Hybrid Solver')

    # Set labels and title
    plt.xticks(x, categories, fontsize=11)
    plt.ylabel('Selection Frequency (%)', fontsize=15)
    plt.title('Selected Routes by Hop Count', fontsize=16)
    plt.ylim(0, max(sim_percentages + q_percentages + hybrid_percentages) + 10)
    plt.yticks(range(0, 101, 10))
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add legend
    plt.legend()

    plt.show()

# Define a function to plot the results in a 3x2 grid
def plot_coloring_bmark_results(results, title, density_interval, topology_files):
    densities = np.linspace(0.1, 1, density_interval)
    markers = ['s', '^', 'o', 'D', 'x', '*']  # Square, triangle, circle, diamond, x, star
    colors = ['green', 'darkred', 'orange', 'blue', 'brown', 'black']
    
    fig, axs = plt.subplots(2, 3, figsize=(18, 8))  # Adjusted figsize for more horizontal squashing
    axs = axs.flatten()  # Flatten the 2D array of axes to make it easier to iterate
    
    for idx, name in enumerate(topology_files):
        ax = axs[idx]
        
        for (solver, color_list), marker, color in zip(results.items(), markers, colors):
            valid_densities = [densities[i] for i in range(density_interval) if color_list[name][i] is not None]
            valid_results = [color_list[name][i] for i in range(density_interval) if color_list[name][i] is not None]
            ax.plot(valid_densities, valid_results, marker=marker, color=color, label=solver)
        
        ax.set_title(f"{name} Network Results")
        ax.set_xlabel("Route Density")
        ax.set_ylabel("Colors")
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='best')
    
    # Hide any empty subplots
    for i in range(len(topology_files), len(axs)):
        fig.delaxes(axs[i])
    
    plt.tight_layout()
    #plt.suptitle(title, fontsize=12)
    plt.subplots_adjust(top=0.92)  # Adjust the top to make room for the super title
    plt.show()
