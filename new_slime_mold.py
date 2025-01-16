import numpy as np
import networkx as nx
from queue import PriorityQueue
from scipy.sparse import lil_matrix, csr_matrix
from typing import List, Tuple


class SlimeMoldOptimizer:
    def __init__(self,
                 graph: nx.Graph,
                 n_agents: int = 10,
                 evaporation_rate: float = 0.1,
                 min_concentration: float = 1e-6):
        """
        Initialize the slime mold optimizer.

        Args:
            graph: NetworkX graph representing the network
            n_agents: Number of plasmoid agents
            evaporation_rate: Rate of slime evaporation
            min_concentration: Minimum concentration threshold
        """
        self.G = graph
        self.N = n_agents
        self.evaporation_rate = evaporation_rate
        self.min_concentration = min_concentration
        self.n_nodes = len(self.G.nodes)

        # Initialize matrices
        self.slime_concentration = lil_matrix((self.N, self.n_nodes))
        self.plasmoid = np.zeros((self.N, self.n_nodes))

    def initialize_agents(self, start_nodes: List[int] = None, end_node: int = None) -> Tuple[List[int], int]:
        """
        Initialize agent positions and target.

        Args:
            start_nodes: Optional list of start nodes. If None, randomly chosen.
            end_node: Optional end node. If None, randomly chosen.

        Returns:
            Tuple of (start_nodes, end_node)
        """
        if start_nodes is None:
            start_nodes = np.random.choice(list(self.G.nodes), size=self.N, replace=True)
        if end_node is None:
            end_node = np.random.choice(list(self.G.nodes))

        # Initialize plasmoid states
        for i, start_node in enumerate(start_nodes):
            self.plasmoid[i, start_node] = 1.0

        return start_nodes, end_node

    def propagate_agent(self, agent_idx: int, start_node: int, end_node: int):
        """
        Propagate a single agent through the network.

        Args:
            agent_idx: Index of the agent to propagate
            start_node: Starting node for this agent
            end_node: Target end node
        """
        queue = PriorityQueue()
        queue.put((-self.plasmoid[agent_idx, start_node], start_node))
        visited = set()

        while not queue.empty():
            current_node = queue.get()[1]

            if current_node == end_node or current_node in visited:
                continue

            visited.add(current_node)
            neighbors = list(self.G.neighbors(current_node))

            if not neighbors:
                continue

            concentration_share = self.plasmoid[agent_idx, current_node] / len(neighbors)

            for neighbor in neighbors:
                if concentration_share > self.min_concentration:
                    self.slime_concentration[agent_idx, neighbor] = concentration_share
                    self.plasmoid[agent_idx, neighbor] += concentration_share
                    queue.put((-concentration_share, neighbor))

        # Apply evaporation for this agent
        self.slime_concentration[agent_idx] *= (1 - self.evaporation_rate)

    def find_path(self, start_node: int, end_node: int) -> List[int]:
        """
        Find the optimal path based on slime concentration.

        Args:
            start_node: Starting node for the path
            end_node: Target end node

        Returns:
            List of nodes representing the optimal path
        """
        # Aggregate slime concentration from all agents
        total_concentration = csr_matrix(np.sum(self.plasmoid, axis=0))
        total_concentration *= (1 - self.evaporation_rate)

        path = [start_node]
        visited = {start_node}

        while path[-1] != end_node:
            current = path[-1]
            neighbors = list(self.G.neighbors(current))

            if not neighbors:
                break

            # Filter out already visited neighbors
            unvisited = [n for n in neighbors if n not in visited]
            if not unvisited:
                break

            # Choose next node based on highest concentration
            concentrations = [total_concentration[0, n] for n in unvisited]
            next_node = unvisited[np.argmax(concentrations)]

            path.append(next_node)
            visited.add(next_node)

        return path

    def optimize(self, start_nodes: List[int] = None, end_node: int = None) -> Tuple[List[int], List[int], int]:
        """
        Run the full optimization process.

        Returns:
            Tuple of (optimal_path, start_nodes, end_node)
        """
        start_nodes, end_node = self.initialize_agents(start_nodes, end_node)

        # Propagate all agents
        for i, start_node in enumerate(start_nodes):
            self.propagate_agent(i, start_node, end_node)

        # Find optimal path
        path = self.find_path(start_nodes[0], end_node)

        return path, start_nodes, end_node


def main():
    # Create example graph
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])

    # Initialize and run optimizer
    optimizer = SlimeMoldOptimizer(G)
    path, start_nodes, end_node = optimizer.optimize()

    # Print results
    print("Start nodes for agents:", start_nodes)
    print("End node:", end_node)
    print("Shortest path based on slime concentration:", [int(node) for node in path])


if __name__ == "__main__":
    main()
