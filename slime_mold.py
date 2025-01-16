import numpy as np
import networkx as nx
from queue import PriorityQueue
from scipy.sparse import lil_matrix, csr_matrix

# Initialize the network graph
G = nx.Graph()
G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])  # Example graph

# Parameters
N = 10  # Number of agents
evaporation_rate = 0.1  # Slime evaporation rate
slime_concentration = lil_matrix((1, len(G.nodes)))  # Sparse matrix for slime concentration

# Initialize agents
plasmodium = np.zeros((N, len(G.nodes)))  # Agent states for all nodes
start_nodes = np.random.choice(list(G.nodes), size=N, replace=True)  # Random start nodes
end_node = np.random.choice(list(G.nodes))  # Single target node

# Set initial concentration at start nodes
for i, start_node in enumerate(start_nodes):
    plasmodium[i, start_node] = 1

# Agent propagation
for agent in range(N):
    queue = PriorityQueue()
    queue.put((-plasmodium[agent, start_nodes[agent]], start_nodes[agent]))  # Start node priority

    while not queue.empty():
        current_node = queue.get()[1]
        if current_node == end_node:  # Stop if end node is reached
            break

        neighbors = list(G.neighbors(current_node))
        if not neighbors:  # Skip nodes with no neighbors
            continue

        # Distribute concentration to neighbors
        for neighbor in neighbors:
            next_concentration = plasmodium[agent, current_node] / len(neighbors) if len(neighbors) > 0 else 0
            next_concentration = min(next_concentration, 1000000)

            if next_concentration > slime_concentration[0, neighbor]:
                slime_concentration[0, neighbor] = next_concentration
                queue.put((-next_concentration, neighbor))

            # Modified line with both accumulation limit and immediate evaporation
            plasmodium[agent, neighbor] = min(plasmodium[agent, neighbor] + next_concentration, 1.0) * (1 - evaporation_rate)

# Aggregate slime concentrations across agents
slime_concentration = np.sum(plasmodium, axis=0)
slime_concentration *= (1 - evaporation_rate)  # Global evaporation
slime_concentration = csr_matrix(slime_concentration)  # Convert to sparse matrix

# Find shortest path based on slime concentration
path = [start_nodes[0]]  # Example start node
while path[-1] != end_node:
    neighbors = list(G.neighbors(path[-1]))
    if not neighbors:  # Stop if no neighbors
        break
    # Choose the neighbor with highest slime concentration
    next_node = neighbors[np.argmax([slime_concentration[0, neighbor] for neighbor in neighbors])]
    path.append(next_node)

# Output results
print("Start nodes for agents:", start_nodes)
print("End node:", end_node)
print("Shortest path based on slime concentration:", [int(node) for node in path])
