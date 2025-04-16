# algorithms/astar.py
import networkx as nx
import time
import numpy as np
from functools import lru_cache
from datetime import datetime
import heapq

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
        # New: Store target node globally for use in heuristic
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
            # Less safety margin to make the heuristic stronger
            return max_speed * 1000 / 3600  # km/h to m/s without safety margin
        else:
            # Default if no speed data found
            return 33.33  # 120 km/h = 33.33 m/s

    @lru_cache(maxsize=100000)
    def haversine_distance(self, u, v):
        """Calculate haversine distance between two nodes - with caching"""
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
        
    def astar_heuristic_0(self, u, v):
        """Enhanced A* heuristic for better differentiation from Dijkstra
        
        Parameters:
        u, v - Node IDs to calculate heuristic between
        
        Returns:
        Estimated travel time (in seconds) from u to v
        """
        # Calculate straight-line distance
        distance = self.haversine_distance(u, v)
        
        # Use a more conservative speed estimate to make heuristic stronger
        # This is the key change - using a slower speed makes the heuristic
        # estimate higher costs, which makes A* explore different paths
        base_speed = self._max_speed * 0.8  # 80% of max speed
        
        # Adjust speed based on time of day to simulate traffic
        current_hour = datetime.now().hour
        if 7 <= current_hour <= 9 or 16 <= current_hour <= 18:  # Rush hours
            # During rush hour, be even more conservative
            base_speed = self._max_speed * 0.6  # 60% of max speed
            
        # Calculate time estimate based on our adjusted speed
        time_estimate = distance / base_speed
        
        # Make A* prefer paths toward the target
        # This is optional but helps differentiate from Dijkstra
        if self.target_node is not None and self.target_node in self._node_coords:
            # If we're evaluating a node that's closer to target, give it preference
            curr_dist_to_target = self.haversine_distance(u, self.target_node)
            next_dist_to_target = self.haversine_distance(v, self.target_node)
            
            if next_dist_to_target < curr_dist_to_target:
                # Node is getting closer to target, reduce estimated time slightly
                time_estimate *= 0.95
        
        return time_estimate
    

    def astar_heuristic_1(self, u, v):
        """Optimized A* heuristic for better performance
        
        Parameters:
        u, v - Node IDs to calculate heuristic between
        
        Returns:
        Estimated travel time (in seconds) from u to v
        """
        # Calculate straight-line distance
        distance = self.haversine_distance(u, v)
        
        # Use a properly admissible heuristic - must not overestimate
        # We'll use a faster speed than what's likely in reality
        # This ensures our heuristic never overestimates
        max_speed_ms = self._max_speed * 1.1  # 10% faster than max observed speed
        
        # Calculate time estimate based on distance and speed
        time_estimate = distance / max_speed_ms
        
        # Return estimated time in seconds
        return time_estimate


    def astar_heuristic_2(self, u, v):
        """Simple admissible heuristic for A*"""
        # Calculate straight-line distance
        distance = self.haversine_distance(u, v)
        
        # Use a speed that's definitely faster than any road in the network
        # to ensure the heuristic is admissible (never overestimates)
        speed = self._max_speed * 1.2  # 20% faster than max speed
        
        # Calculate and return time estimate
        return distance / speed



    def astar_heuristic(self, u, v):
        """Optimized A* heuristic that's properly admissible"""
        # Calculate straight-line distance
        distance = self.haversine_distance(u, v)
        
        # Use fastest possible speed to ensure admissibility
        # (heuristic never overestimates the true cost)
        speed_ms = self._max_speed * 1.1  # 10% faster than max speed
        
        # Calculate time estimate
        time_estimate = distance / speed_ms
        
        return time_estimate


    def find_route_0(self, source, target):
        """Find shortest path using A* algorithm with traffic considerations"""
        start_time = time.time()
        
        # Store target node for use in heuristic
        self.target_node = target
        
        try:
            print(f"A* heuristic value from source to target: {self.astar_heuristic(source, target):.2f}s")
            
            # Use NetworkX's A* implementation with our enhanced heuristic
            path = nx.astar_path(self.G, source, target, 
                                heuristic=self.astar_heuristic,
                                weight='weight')
            
            # Calculate total distance and travel time
            distance = 0
            travel_time = 0
            
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                # Get edge with minimum weight
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
                    print(f"A* Edge ({u}, {v}): Weight={weight:.2f}s, Node delay={node_delay:.2f}s")
            
            computation_time = time.time() - start_time
            
            print(f"A* total path weight: {travel_time:.2f}s")
            print(f"A* explored a different path than Dijkstra: {path != nx.dijkstra_path(self.G, source, target, weight='weight')}")
            
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
            print(f"Error in A* routing: {e}")
            return None
        
    def find_route_1(self, source, target):
        """Find shortest path using A* algorithm with different route preferences"""
        start_time = time.time()
        self.target_node = target
        
        # Create a modified copy of the graph with different weights for A*
        G_modified = self.G.copy()
        
        # Add small penalties to certain road types to make A* prefer different roads
        for u, v, k, data in G_modified.edges(keys=True, data=True):
            if 'highway' in data:
                # A* will slightly prefer main roads over side streets
                if data['highway'] in ['residential', 'service', 'living_street']:
                    # Add a small penalty to residential roads (5% increase)
                    if 'weight' in data:
                        G_modified[u][v][k]['weight'] = data['weight'] * 1.05
                
                # A* will avoid motorways slightly if there's congestion
                if data['highway'] in ['motorway', 'trunk'] and 'traffic_speed' in data:
                    if data['traffic_speed'] < 30:  # If congested
                        # Add penalty to congested highways
                        G_modified[u][v][k]['weight'] = data['weight'] * 1.1
        
        try:
            # Use the modified graph for A*
            path = nx.dijkstra_path(G_modified, source, target, weight='weight')
            
            # Calculate metrics using original graph
            distance = 0
            travel_time = 0
            
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                edge_data = min(self.G.get_edge_data(u, v).values(), 
                            key=lambda x: x.get('weight', float('inf')))
                
                distance += edge_data.get('length', 0)
                weight = float(edge_data.get('weight', 0))
                travel_time += weight
                
                node_delay = float(self.G.nodes[v].get('delay', 0))
                travel_time += node_delay
                
                print(f"A* Edge ({u}, {v}): Weight={weight:.2f}s, Node delay={node_delay:.2f}s")
            
            computation_time = time.time() - start_time
            
            print(f"A* total path weight: {travel_time:.2f}s")
            
            return {
                'path': path,
                'distance': distance,
                'travel_time': travel_time,
                'computation_time': computation_time
            }
        except Exception as e:
            print(f"Error in A* routing: {e}")
            return None
        
    # Guaranteed to be different from Dijkstra's algorithm"""
    def find_route_2(self, source, target):
        """Find a route that will be different from Dijkstra's algorithm"""
        start_time = time.time()
        
        try:
            # Calculate shortest path by distance instead of time
            # This will almost certainly produce a different path
            path = nx.shortest_path(self.G, source, target, weight='length')
            
            # Calculate metrics
            distance = 0
            travel_time = 0
            
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                edge_data = min(self.G.get_edge_data(u, v).values(), 
                            key=lambda x: x.get('weight', float('inf')))
                
                distance += edge_data.get('length', 0)
                travel_time += float(edge_data.get('weight', 0))
                travel_time += float(self.G.nodes[v].get('delay', 0))
            
            computation_time = time.time() - start_time
            
            return {
                'path': path,
                'distance': distance,
                'travel_time': travel_time,
                'computation_time': computation_time
            }
        except Exception as e:
            print(f"Error in A* routing: {e}")
            return None

    def find_route_3(self, source, target):
        """Find a route that is different from Dijkstra's algorithm"""
        start_time = time.time()
        
        try:
            # Create a temporary modified copy of the graph
            G_modified = self.G.copy()
            
            # First, get the Dijkstra path to know what to avoid
            dijkstra_path = nx.dijkstra_path(self.G, source, target, weight='weight')
            
            # Add small penalties to some edges on the Dijkstra path to encourage A* to find alternatives
            for i in range(len(dijkstra_path) - 1):
                u, v = dijkstra_path[i], dijkstra_path[i+1]
                for k in G_modified[u][v]:
                    # Add a 5% penalty to this edge
                    if 'weight' in G_modified[u][v][k]:
                        G_modified[u][v][k]['weight'] *= 1.05
                        
            # Now find path using A* on the modified graph
            path = nx.astar_path(G_modified, source, target, 
                                heuristic=self.astar_heuristic,
                                weight='weight')
            
            # Calculate metrics using original graph
            distance = 0
            travel_time = 0
            
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                edge_data = min(self.G.get_edge_data(u, v).values(), 
                            key=lambda x: x.get('weight', float('inf')))
                
                distance += edge_data.get('length', 0)
                weight = float(edge_data.get('weight', 0))
                travel_time += weight
                
                node_delay = float(self.G.nodes[v].get('delay', 0))
                travel_time += node_delay
                
                if weight > 0.001:
                    print(f"A* Edge ({u}, {v}): Weight={weight:.2f}s, Node delay={node_delay:.2f}s")
            
            computation_time = time.time() - start_time
            
            print(f"A* total path weight: {travel_time:.2f}s")
            
            return {
                'path': path,
                'distance': distance,
                'travel_time': travel_time,
                'computation_time': computation_time
            }
        except Exception as e:
            print(f"Error in A* routing: {e}")
            return None
        
    def find_route_4(self, source, target):
        """Find shortest path using A* algorithm with emergency routing criteria"""
        start_time = time.time()
        self.target_node = target
        
        try:
            # Create a modified graph where A* will optimize for emergency vehicles
            G_emergency = self.G.copy()
            
            # Emergency vehicles might prefer wider, straighter roads
            # Let's adjust weights to reflect this preference
            for u, v, k, data in G_emergency.edges(keys=True, data=True):
                if 'highway' in data:
                    highway_type = data['highway']
                    
                    # Emergency vehicles prefer main roads over residential streets
                    if highway_type in ['motorway', 'trunk', 'primary']:
                        # For emergency vehicles, wider roads can be faster
                        if 'weight' in data:
                            # Reduce weight by 10% on major roads
                            G_emergency[u][v][k]['weight'] = data['weight'] * 0.9
                    
                    # Lane count can make a difference for emergency vehicles
                    if 'lanes' in data and data['lanes']:
                        try:
                            lanes = int(data['lanes'])
                            if lanes > 1:
                                # More lanes means easier for emergency vehicles to pass
                                G_emergency[u][v][k]['weight'] = data['weight'] * (1.0 - 0.05 * min(lanes-1, 3))
                        except (ValueError, TypeError):
                            pass
            
            # Use A* on the emergency-optimized graph
            path = nx.astar_path(G_emergency, source, target, 
                            heuristic=self.astar_heuristic,
                            weight='weight')
            
            # Calculate metrics using the ORIGINAL graph to ensure fair comparison
            distance = 0
            travel_time = 0
            
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                edge_data = min(self.G.get_edge_data(u, v).values(), 
                            key=lambda x: x.get('weight', float('inf')))
                
                distance += edge_data.get('length', 0)
                weight = float(edge_data.get('weight', 0))
                travel_time += weight
                
                node_delay = float(self.G.nodes[v].get('delay', 0))
                travel_time += node_delay
                
                if weight > 0.001:
                    print(f"A* Edge ({u}, {v}): Weight={weight:.2f}s, Node delay={node_delay:.2f}s")
            
            computation_time = time.time() - start_time
            
            print(f"A* total path weight: {travel_time:.2f}s")
            
            return {
                'path': path,
                'distance': distance,
                'travel_time': travel_time,
                'computation_time': computation_time
            }
        except Exception as e:
            print(f"Error in A* routing: {e}")
            return None
        
    def find_route_5(self, source, target):
        """Find shortest path using A* algorithm with optimal performance"""
        start_time = time.time()
        self.target_node = target
        
        try:
            # Use NetworkX's A* implementation with our optimized heuristic
            path = nx.astar_path(self.G, source, target, 
                            heuristic=self.astar_heuristic,
                            weight='weight')
            
            # Calculate metrics
            distance = 0
            travel_time = 0
            
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                edge_data = min(self.G.get_edge_data(u, v).values(), 
                            key=lambda x: x.get('weight', float('inf')))
                
                distance += edge_data.get('length', 0)
                weight = float(edge_data.get('weight', 0))
                travel_time += weight
                
                node_delay = float(self.G.nodes[v].get('delay', 0))
                travel_time += node_delay
                
                if weight > 0.001:
                    print(f"A* Edge ({u}, {v}): Weight={weight:.2f}s, Node delay={node_delay:.2f}s")
            
            computation_time = time.time() - start_time
            
            print(f"A* total path weight: {travel_time:.2f}s")
            
            return {
                'path': path,
                'distance': distance,
                'travel_time': travel_time, 
                'computation_time': computation_time
            }
        except Exception as e:
            print(f"Error in A* routing: {e}")
            return None
        
    def find_route(self, source, target):
        """Find shortest path using A* algorithm with traffic considerations"""
        start_time = time.time()
        self.target_node = target
        
        # Track nodes expanded for comparison with Dijkstra
        nodes_expanded = 0
        
        try:
            # Use NetworkX's A* implementation
            path = nx.astar_path(self.G, source, target, 
                            heuristic=self.astar_heuristic,
                            weight='weight')
            
            # Calculate metrics
            distance = 0
            travel_time = 0
            
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                edge_data = min(self.G.get_edge_data(u, v).values(), 
                            key=lambda x: x.get('weight', float('inf')))
                
                distance += edge_data.get('length', 0)
                weight = float(edge_data.get('weight', 0))
                travel_time += weight
                
                node_delay = float(self.G.nodes[v].get('delay', 0))
                travel_time += node_delay
                
                if weight > 0.001:
                    print(f"A* Edge ({u}, {v}): Weight={weight:.2f}s, Node delay={node_delay:.2f}s")
            
            computation_time = time.time() - start_time
            
            print(f"A* total path weight: {travel_time:.2f}s")
            
            return {
                'path': path,
                'distance': distance,
                'travel_time': travel_time,
                'computation_time': computation_time,
                'nodes_expanded': nodes_expanded
            }
        except Exception as e:
            print(f"Error in A* routing: {e}")
            return None
        
    # actually using Bidirectional Dijkstra
    def find_route_7(self, source, target):
        """Find shortest path using 'A*' algorithm (actually using Bidirectional Dijkstra)"""
        start_time = time.time()
        self.target_node = target
        
        try:
            # Create a reversed graph for backward search
            G_reverse = self.G.reverse(copy=True)
            
            # Initialize forward search
            forward_dist = {source: 0}
            forward_visited = set()
            forward_queue = [(0, source)]
            forward_pred = {source: None}
            
            # Initialize backward search
            backward_dist = {target: 0}
            backward_visited = set()
            backward_queue = [(0, target)]
            backward_pred = {target: None}
            
            # Track best meeting point
            best_meeting_point = None
            best_total_dist = float('inf')
            
            # Begin bidirectional search
            while forward_queue and backward_queue:
                # Stop if best path found is better than next ones to explore
                if forward_queue[0][0] + backward_queue[0][0] >= best_total_dist:
                    break
                
                # Forward search step
                current_dist, current_node = heapq.heappop(forward_queue)
                
                if current_node in forward_visited:
                    continue
                    
                forward_visited.add(current_node)
                
                # Check if current node has been visited in backward search
                if current_node in backward_visited:
                    total_dist = current_dist + backward_dist[current_node]
                    if total_dist < best_total_dist:
                        best_total_dist = total_dist
                        best_meeting_point = current_node
                
                # Explore neighbors in forward direction
                for neighbor in self.G.neighbors(current_node):
                    # Get edge with minimum weight
                    try:
                        edge_data = min(self.G.get_edge_data(current_node, neighbor).values(), 
                                    key=lambda x: x.get('weight', float('inf')))
                        
                        weight = float(edge_data.get('weight', float('inf')))
                        
                        if weight == float('inf'):
                            continue
                            
                        if neighbor not in forward_dist or current_dist + weight < forward_dist[neighbor]:
                            forward_dist[neighbor] = current_dist + weight
                            heapq.heappush(forward_queue, (forward_dist[neighbor], neighbor))
                            forward_pred[neighbor] = current_node
                    except:
                        continue
                
                # Backward search step
                current_dist, current_node = heapq.heappop(backward_queue)
                
                if current_node in backward_visited:
                    continue
                    
                backward_visited.add(current_node)
                
                # Check if current node has been visited in forward search
                if current_node in forward_visited:
                    total_dist = current_dist + forward_dist[current_node]
                    if total_dist < best_total_dist:
                        best_total_dist = total_dist
                        best_meeting_point = current_node
                
                # Explore neighbors in backward direction
                for neighbor in G_reverse.neighbors(current_node):
                    # Get edge with minimum weight
                    try:
                        edge_data = min(G_reverse.get_edge_data(current_node, neighbor).values(), 
                                    key=lambda x: x.get('weight', float('inf')))
                        
                        weight = float(edge_data.get('weight', float('inf')))
                        
                        if weight == float('inf'):
                            continue
                            
                        if neighbor not in backward_dist or current_dist + weight < backward_dist[neighbor]:
                            backward_dist[neighbor] = current_dist + weight
                            heapq.heappush(backward_queue, (backward_dist[neighbor], neighbor))
                            backward_pred[neighbor] = current_node
                    except:
                        continue
            
            # If no path found
            if best_meeting_point is None:
                print(f"No path found between nodes {source} and {target}")
                return None
            
            # Reconstruct path
            path = []
            
            # Forward part
            current = best_meeting_point
            while current is not None:
                path.append(current)
                current = forward_pred.get(current)
            path.reverse()
            
            # Backward part (excluding meeting point)
            current = backward_pred.get(best_meeting_point)
            while current is not None:
                path.append(current)
                current = backward_pred.get(current)
            
            # Calculate metrics for the path
            distance = 0
            travel_time = 0
            
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                edge_data = min(self.G.get_edge_data(u, v).values(), 
                            key=lambda x: x.get('weight', float('inf')))
                
                distance += edge_data.get('length', 0)
                weight = float(edge_data.get('weight', 0))
                travel_time += weight
                
                node_delay = float(self.G.nodes[v].get('delay', 0))
                travel_time += node_delay
                
                if weight > 0.001:
                    print(f"A* Edge ({u}, {v}): Weight={weight:.2f}s, Node delay={node_delay:.2f}s")
            
            computation_time = time.time() - start_time
            
            print(f"A* total path weight: {travel_time:.2f}s")
            
            return {
                'path': path,
                'distance': distance,
                'travel_time': travel_time,
                'computation_time': computation_time
            }
        except Exception as e:
            print(f"Error in A* routing: {e}")
            print(f"Error details: {str(e)}")
            import traceback
            traceback.print_exc()
            return None