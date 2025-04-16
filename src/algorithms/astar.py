# algorithms/astar.py
import networkx as nx
import time
import numpy as np
import heapq
from functools import lru_cache
from datetime import datetime

class AStarRouter:
    def __init__(self, G):
        self.G = G
        # Precompute max speed for the heuristic
        self._max_speed = self._get_max_speed()
        # Precompute node coordinates for faster access
        self._node_coords = {}
        for node, data in G.nodes(data=True):
            if 'x' in data and 'y' in data:
                self._node_coords[node] = (data['x'], data['y'])
        # Store target node globally for use in heuristic
        self.target_node = None
        
    def _get_max_speed(self):
        """Determine the maximum realistic speed in the network"""
        max_speed = 0
        for u, v, data in self.G.edges(data=True):
            if 'traffic_speed' in data and data['traffic_speed'] > 0:
                max_speed = max(max_speed, data['traffic_speed'])
            elif 'highway' in data:
                # Estimate max speed based on road type if no traffic speed
                highway_type = data['highway']
                if highway_type in ['motorway', 'trunk']:
                    max_speed = max(max_speed, 100)  # 100 km/h
                elif highway_type in ['primary']:
                    max_speed = max(max_speed, 80)  # 80 km/h
                elif highway_type in ['secondary']:
                    max_speed = max(max_speed, 60)  # 60 km/h
        
        # Convert to m/s and use a conservative value
        if max_speed > 0:
            return max_speed * 1000 / 3600  # km/h to m/s
        else:
            # Default if no speed data found
            return 33.33  # 120 km/h = 33.33 m/s

    @lru_cache(maxsize=100000)
    def haversine_distance(self, u, v):
        """Calculate haversine distance between two nodes with caching"""
        # Get coordinates from precomputed dict for speed
        if u not in self._node_coords or v not in self._node_coords:
            return 0
        
        u_lon, u_lat = self._node_coords[u]
        v_lon, v_lat = self._node_coords[v]
        
        # Convert to radians
        u_lon, u_lat = np.radians(u_lon), np.radians(u_lat)
        v_lon, v_lat = np.radians(v_lon), np.radians(v_lat)
        
        # Haversine formula
        dlon = v_lon - u_lon
        dlat = v_lat - u_lat
        a = np.sin(dlat/2)**2 + np.cos(u_lat) * np.cos(v_lat) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371000  # Earth radius in meters
        return c * r
        
    def astar_heuristic(self, u, v):
        """
        Enhanced A* heuristic that makes the algorithm behave differently from Dijkstra
        
        Key improvements:
        1. Uses road type to prefer main roads
        2. Considers traffic conditions more aggressively
        3. Makes A* more "greedy" toward the target
        
        Returns:
        Estimated travel time (in seconds) from u to v
        """
        # Calculate straight-line distance
        distance = self.haversine_distance(u, v)
        
        # Base calculation - similar to original A* but more aggressive
        # We'll use a faster speed estimate to make A* explore different paths
        # This creates a more aggressive heuristic that will prioritize getting closer to the target
        heuristic_speed_factor = 1.3  # 30% faster than reality - will make A* greedier
        base_speed = self._max_speed * heuristic_speed_factor
        
        # Calculate time estimate based on distance and speed
        time_estimate = distance / base_speed
        
        # Adjust heuristic based on road types and traffic conditions
        # This is where we make A* behavior significantly different from Dijkstra
        
        # 1. Look at current node's edges to detect road type preferences
        if len(list(self.G.edges(u))) > 0:
            # Check if current node is on a highway/main road
            is_on_major_road = False
            traffic_speed_sum = 0
            edge_count = 0
            
            for _, _, data in self.G.edges(u, data=True):
                highway_type = data.get('highway', '')
                if highway_type in ['motorway', 'trunk', 'primary']:
                    is_on_major_road = True
                    
                # Calculate average traffic speed around current node
                if 'traffic_speed' in data:
                    traffic_speed_sum += data['traffic_speed']
                    edge_count += 1
            
            # If we're on a major road and not in traffic, A* should prefer staying on it
            if is_on_major_road:
                avg_traffic_speed = traffic_speed_sum / max(1, edge_count)
                if avg_traffic_speed > 40:  # If traffic is flowing well on major roads
                    time_estimate *= 0.8  # Reduce estimated time to bias toward major roads
                    
        # 2. Check for traffic congestion between current node and target
        if self.target_node is not None:
            # Calculate bearing toward target
            u_coords = self._node_coords.get(u)
            target_coords = self._node_coords.get(self.target_node)
            
            if u_coords and target_coords:
                # Create vector toward target
                dx = target_coords[0] - u_coords[0]
                dy = target_coords[1] - u_coords[1]
                
                # Check if there are edges in that general direction
                for neighbor in self.G.neighbors(u):
                    n_coords = self._node_coords.get(neighbor)
                    if not n_coords:
                        continue
                    
                    # Vector toward neighbor
                    n_dx = n_coords[0] - u_coords[0]
                    n_dy = n_coords[1] - u_coords[1]
                    
                    # Calculate dot product to check alignment with target direction
                    dot_product = dx * n_dx + dy * n_dy
                    
                    if dot_product > 0:  # This neighbor is in the general direction of the target
                        # Check if this direction has traffic congestion
                        edge_data = min(self.G.get_edge_data(u, neighbor).values(), 
                                      key=lambda x: x.get('weight', float('inf')))
                        
                        if 'traffic_speed' in edge_data:
                            traffic_speed = edge_data['traffic_speed']
                            if traffic_speed < 20:  # Heavy congestion in target direction
                                # Increase time estimate to make A* explore other directions
                                time_estimate *= 1.2
        
        # 3. Adjust heuristic based on time of day
        current_hour = datetime.now().hour
        if (7 <= current_hour <= 9) or (16 <= current_hour <= 18):  # Rush hours
            # During rush hour, encourage exploration of alternate routes
            time_estimate *= 0.9  # Make A* slightly more optimistic during rush hours
        
        return time_estimate
    
    def find_route(self, source, target):
        """Find shortest path using enhanced A* algorithm"""
        start_time = time.time()
        self.target_node = target
        nodes_expanded = 0
        
        try:
            # Implement A* algorithm directly for better control
            # Initialize data structures
            open_set = [(0, 0, source)]  # (f_score, g_score, node)
            closed_set = set()
            g_score = {source: 0}  # Cost from start to node
            f_score = {source: self.astar_heuristic(source, target)}  # Estimated total cost
            came_from = {source: None}
            
            while open_set:
                # Get node with lowest f_score
                _, current_g, current = heapq.heappop(open_set)
                nodes_expanded += 1
                
                # If we've reached the target, we're done
                if current == target:
                    break
                
                # Skip if we've already processed this node
                if current in closed_set:
                    continue
                    
                closed_set.add(current)
                
                # Check if current g_score is still the best
                if current_g > g_score.get(current, float('inf')):
                    continue
                
                # Explore all neighbors
                for neighbor in self.G.neighbors(current):
                    # Get the edge with minimum weight if multiple edges exist
                    edge_data = min(self.G.get_edge_data(current, neighbor).values(), 
                                  key=lambda x: x.get('weight', float('inf')))
                    
                    # Calculate new g_score
                    weight = float(edge_data.get('weight', float('inf')))
                    if weight == float('inf'):
                        continue
                        
                    new_g = g_score[current] + weight
                    
                    # Update if we found a better path
                    if new_g < g_score.get(neighbor, float('inf')):
                        came_from[neighbor] = current
                        g_score[neighbor] = new_g
                        new_f = new_g + self.astar_heuristic(neighbor, target)
                        f_score[neighbor] = new_f
                        
                        # Add to open set with new scores
                        heapq.heappush(open_set, (new_f, new_g, neighbor))
            
            # Reconstruct path
            if target not in came_from and target != source:
                print(f"No path found between nodes {source} and {target}")
                return None
                
            path = []
            current = target
            while current is not None:
                path.append(current)
                current = came_from.get(current)
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
                
                if weight > 0.001:  # Only print significant weights
                    print(f"A* Edge ({u}, {v}): Weight={weight:.2f}s, Node delay={node_delay:.2f}s")
            
            computation_time = time.time() - start_time
            
            print(f"A* total path weight: {travel_time:.2f}s")
            print(f"A* expanded {nodes_expanded} nodes")
            
            return {
                'path': path,
                'distance': distance,
                'travel_time': travel_time,
                'computation_time': computation_time,
                'nodes_expanded': nodes_expanded
            }
        except Exception as e:
            print(f"Error in A* routing: {e}")
            import traceback
            traceback.print_exc()
            return None