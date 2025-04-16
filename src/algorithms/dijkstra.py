# algorithms/dijkstra.py
import networkx as nx
import time
import heapq
from collections import defaultdict

class DijkstraRouter:
    def __init__(self, G):
        self.G = G
        
    def find_route(self, source, target):
        """
        Find shortest path using pure Dijkstra's algorithm without any heuristics
        
        Args:
            source: Source node ID
            target: Target node ID
            
        Returns:
            dict containing path, distance, travel_time, and computation_time
        """
        start_time = time.time()
        
        try:
            # Implement optimized Dijkstra algorithm directly rather than using NetworkX
            # This gives us more visibility and control over the process
            
            # Initialize data structures
            distances = {source: 0}
            previous = {source: None}
            priority_queue = [(0, source)]  # (distance, node)
            nodes_expanded = 0
            
            while priority_queue:
                # Get node with smallest distance
                current_distance, current_node = heapq.heappop(priority_queue)
                nodes_expanded += 1
                
                # If we've reached the target, we're done
                if current_node == target:
                    break
                
                # If we've already found a better path to this node, skip it
                if current_distance > distances.get(current_node, float('inf')):
                    continue
                
                # Explore all neighbors
                for neighbor in self.G.neighbors(current_node):
                    # Dijkstra considers only edge weights without any heuristic
                    # Get the edge with minimum weight if multiple edges exist
                    edge_data = min(self.G.get_edge_data(current_node, neighbor).values(), 
                                  key=lambda x: x.get('weight', float('inf')))
                    
                    # Calculate new distance
                    weight = float(edge_data.get('weight', float('inf')))
                    new_distance = current_distance + weight
                    
                    # Update if we found a better path
                    if new_distance < distances.get(neighbor, float('inf')):
                        distances[neighbor] = new_distance
                        previous[neighbor] = current_node
                        heapq.heappush(priority_queue, (new_distance, neighbor))
            
            # Reconstruct path
            if target not in previous and target != source:
                print(f"No path found between nodes {source} and {target}")
                return None
            
            path = []
            current = target
            while current is not None:
                path.append(current)
                current = previous.get(current)
            path.reverse()
            
            # Calculate metrics
            distance = 0
            travel_time = 0
            
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                edge_data = min(self.G.get_edge_data(u, v).values(), 
                              key=lambda x: x.get('weight', float('inf')))
                
                # Add distance
                distance += edge_data.get('length', 0)
                
                # Add travel time (edge weight)
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
            print(f"Dijkstra expanded {nodes_expanded} nodes")
            
            return {
                'path': path,
                'distance': distance,
                'travel_time': travel_time,
                'computation_time': computation_time,
                'nodes_expanded': nodes_expanded
            }
            
        except Exception as e:
            print(f"Error in Dijkstra routing: {e}")
            import traceback
            traceback.print_exc()
            return None