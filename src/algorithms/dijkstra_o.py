# algorithms/dijkstra.py
import networkx as nx
import time

class DijkstraRouter:
    def __init__(self, G):
        self.G = G
        
    def find_route(self, source, target):
        """
        Find shortest path using Dijkstra's algorithm
        
        Args:
            source: Source node ID
            target: Target node ID
            
        Returns:
            path: List of nodes in the path
            distance: Total distance of path
            travel_time: Estimated travel time in seconds
        """
        start_time = time.time()
        
        try:
            # Use NetworkX's implementation of Dijkstra's algorithm
            path = nx.dijkstra_path(self.G, source, target, weight='weight')
            
            # Calculate total distance and travel time
            distance = 0
            travel_time = 0
            
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                # There might be multiple edges between u and v, get the one with minimum weight
                edge_data = min(self.G.get_edge_data(u, v).values(), 
                               key=lambda x: x.get('weight', float('inf')))
                
                # Add distance
                distance += edge_data.get('length', 0)
                
                # Add travel time (edge weight) - IMPORTANT: ensure it's a float
                weight = float(edge_data.get('weight', 0))
                travel_time += weight
                
                # Add node delay if present (for intersections)
                node_delay = float(self.G.nodes[v].get('delay', 0))
                travel_time += node_delay
                
                # Debug print to verify weights are being properly summed
                if weight > 0.001:  # Only print significant weights to reduce noise
                    print(f"Dijkstra Edge ({u}, {v}): Weight={weight:.2f}s, Node delay={node_delay:.2f}s")
            
            computation_time = time.time() - start_time
            
            print(f"Dijkstra total path weight: {travel_time:.2f}s")
            
            return {
                'path': path,
                'distance': distance,
                'travel_time': travel_time,
                'computation_time': computation_time
            }
        except nx.NetworkXNoPath:
            print(f"No path found between nodes {source} and {target}")
            return None
        except Exception as e:
            print(f"Error in Dijkstra routing: {e}")
            return None