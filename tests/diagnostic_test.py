# diagnostic_test.py
import networkx as nx
import os
from data.data_loader import DataLoader
from data.data_processor import DataProcessor
from src.algorithms.dijkstra import DijkstraRouter
from src.algorithms.astar import AStarRouter
import matplotlib.pyplot as plt

def validate_same_weights(G, path1, path2):
    """Validate that both paths see the same edge weights"""
    print("\n=== Weight Validation for Both Paths ===")
    
    # Check weights along Dijkstra path
    print("\nDijkstra Path Weights:")
    dijkstra_total = 0
    for i in range(len(path1) - 1):
        u, v = path1[i], path1[i+1]
        edge_data = min(G.get_edge_data(u, v).values(), key=lambda x: x.get('weight', float('inf')))
        weight = edge_data.get('weight', 0)
        dijkstra_total += weight
        print(f"Edge ({u}, {v}): {weight:.2f}s")
    
    # Check weights along A* path
    print("\nA* Path Weights:")
    astar_total = 0
    for i in range(len(path2) - 1):
        u, v = path2[i], path2[i+1]
        edge_data = min(G.get_edge_data(u, v).values(), key=lambda x: x.get('weight', float('inf')))
        weight = edge_data.get('weight', 0)
        astar_total += weight
        print(f"Edge ({u}, {v}): {weight:.2f}s")
    
    print(f"\nDijkstra total edge weight: {dijkstra_total:.2f}s")
    print(f"A* total edge weight: {astar_total:.2f}s")
    
    return dijkstra_total, astar_total

def calculate_path_travel_time(G, path):
    """Calculate travel time for a path exactly as each algorithm would"""
    total_time = 0
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        edge_data = min(G.get_edge_data(u, v).values(), key=lambda x: x.get('weight', float('inf')))
        total_time += edge_data.get('weight', 0)
        total_time += G.nodes[v].get('delay', 0)
    return total_time

def run_diagnostic():
    """Run diagnostic tests to identify why A* performs worse than Dijkstra"""
    print("=== Emergency Routing Diagnostic Test ===")
    
    # Initialize components
    data_loader = DataLoader()
    
    # Load graph
    cache_file = "data/processed/road_network.json"
    if os.path.exists(cache_file):
        print("\nLoading cached road network...")
        G = data_loader.load_graph_json(cache_file)
    else:
        print("\nLoading fresh road network from OpenStreetMap...")
        G = data_loader.load_osm_graph("UBC, Vancouver, Canada")
    
    # Initialize routers
    dijkstra_router = DijkstraRouter(G)
    astar_router = AStarRouter(G)
    
    # Choose fixed test points
    # These are arbitrary nodes that should be far apart
    source = list(G.nodes())[0]
    target = list(G.nodes())[len(G.nodes()) // 2]
    
    print(f"\nTesting route from node {source} to {target}")
    
    # Run both algorithms
    dijkstra_result = dijkstra_router.find_route(source, target)
    astar_result = astar_router.find_route(source, target)
    
    if dijkstra_result and astar_result:
        # Basic comparison
        print("\n=== Basic Comparison ===")
        print(f"Dijkstra path length: {len(dijkstra_result['path'])} nodes")
        print(f"A* path length: {len(astar_result['path'])} nodes")
        print(f"Dijkstra travel time: {dijkstra_result['travel_time']:.2f}s")
        print(f"A* travel time: {astar_result['travel_time']:.2f}s")
        print(f"Dijkstra computation time: {dijkstra_result['computation_time']:.5f}s")
        print(f"A* computation time: {astar_result['computation_time']:.5f}s")
        
        # Check if paths are identical
        paths_identical = dijkstra_result['path'] == astar_result['path']
        print(f"\nPaths identical: {paths_identical}")
        
        # Validate weights
        dijkstra_total, astar_total = validate_same_weights(G, 
                                                         dijkstra_result['path'], 
                                                         astar_result['path'])
        
        # Double-check travel time calculation
        print("\n=== Travel Time Verification ===")
        dijkstra_verified = calculate_path_travel_time(G, dijkstra_result['path'])
        astar_verified = calculate_path_travel_time(G, astar_result['path'])
        
        print(f"Dijkstra reported time: {dijkstra_result['travel_time']:.2f}s")
        print(f"Dijkstra verified time: {dijkstra_verified:.2f}s")
        print(f"A* reported time: {astar_result['travel_time']:.2f}s")
        print(f"A* verified time: {astar_verified:.2f}s")
        
        # If the paths are different but A* has higher cost, something is wrong
        if not paths_identical and astar_total > dijkstra_total:
            print("\n⚠️ ERROR DETECTED: A* found a path with HIGHER cost than Dijkstra!")
            print("This violates the optimality guarantee of both algorithms.")
            print("Possible causes:")
            print("1. The A* heuristic is not admissible (overestimates true cost)")
            print("2. The graph weights are being modified between algorithm runs")
            print("3. There's a bug in how either algorithm calculates the final path")
        
        # Check for NetworkX version issues
        print("\n=== NetworkX Version Check ===")
        print(f"NetworkX version: {nx.__version__}")
        if int(nx.__version__.split('.')[0]) < 2 or \
           (int(nx.__version__.split('.')[0]) == 2 and int(nx.__version__.split('.')[1]) < 5):
            print("⚠️ WARNING: NetworkX version is older than 2.5, which might cause A* issues")
        
        return True
    else:
        print("Error: One or both algorithms failed to find a path")
        return False
    

def test_with_traffic_data():
    """Run a specific test with heavy traffic simulation to verify routing differences"""
    print("\n=== Testing with Simulated Traffic Data ===")
    
    # Initialize components
    data_loader = DataLoader()
    
    # Load graph
    cache_file = "data/processed/road_network.json"
    if os.path.exists(cache_file):
        print("\nLoading cached road network...")
        G = data_loader.load_graph_json(cache_file)
    else:
        print("\nLoading fresh road network from OpenStreetMap...")
        G = data_loader.load_osm_graph("UBC, Vancouver, Canada")
    
    # Apply heavy traffic to certain edges
    print("\nApplying simulated traffic conditions...")
    for u, v, k, data in G.edges(keys=True, data=True):
        if 'highway' in data:
            if data['highway'] in ['motorway', 'trunk', 'primary']:
                # Apply heavy traffic to major roads
                orig_weight = data.get('weight', 10)
                # Increase travel time by 3-5x for major roads
                G.edges[u, v, k]['weight'] = float(orig_weight) * (3 + 2 * (hash(str(u) + str(v)) % 100) / 100.0)
                # Update traffic speed if present
                if 'traffic_speed' in data:
                    G.edges[u, v, k]['traffic_speed'] = float(data['traffic_speed']) / 3.0
    
    # Initialize routers
    dijkstra_router = DijkstraRouter(G)
    astar_router = AStarRouter(G)
    
    # Choose test points (preferably ones that would be connected by major highways)
    # Get a pair of nodes that are reasonably far apart
    nodes = list(G.nodes())
    center_idx = len(nodes) // 2
    source = nodes[center_idx // 2]  # First quarter
    target = nodes[center_idx + (center_idx // 2)]  # Third quarter
    
    print(f"\nTesting route from node {source} to {target} with traffic conditions")
    
    # Run both algorithms
    print("\nRunning Dijkstra algorithm...")
    dijkstra_result = dijkstra_router.find_route(source, target)
    
    print("\nRunning A* algorithm...")
    astar_result = astar_router.find_route(source, target)
    
    if dijkstra_result and astar_result:
        # Print detailed comparison
        print("\n=== Detailed Route Comparison ===")
        print(f"Dijkstra path length: {len(dijkstra_result['path'])} nodes")
        print(f"A* path length: {len(astar_result['path'])} nodes")
        print(f"Dijkstra travel time: {dijkstra_result['travel_time']:.2f}s")
        print(f"A* travel time: {astar_result['travel_time']:.2f}s")
        print(f"Dijkstra computation time: {dijkstra_result['computation_time']:.5f}s")
        print(f"A* computation time: {astar_result['computation_time']:.5f}s")
        
        # Compare paths
        paths_identical = dijkstra_result['path'] == astar_result['path']
        print(f"\nPaths identical: {paths_identical}")
        
        # Calculate time difference and improvement percentage
        time_diff = dijkstra_result['travel_time'] - astar_result['travel_time']
        improvement_pct = (time_diff / dijkstra_result['travel_time']) * 100 if dijkstra_result['travel_time'] > 0 else 0
        
        print(f"Time difference: {time_diff:.2f}s")
        print(f"A* improvement: {improvement_pct:.2f}%")
        
        return True
    else:
        print("Error: One or both algorithms failed to find a path")
        return False

def test_with_extreme_traffic():
    """Run a test with extreme traffic conditions to force different routing decisions"""
    print("\n=== Testing with Extreme Traffic Conditions ===")
    
    # Initialize components
    data_loader = DataLoader()
    
    # Load graph
    cache_file = "data/processed/road_network.json"
    if os.path.exists(cache_file):
        print("\nLoading cached road network...")
        G = data_loader.load_graph_json(cache_file)
    else:
        print("\nLoading fresh road network from OpenStreetMap...")
        G = data_loader.load_osm_graph("UBC, Vancouver, Canada")
    
    # Apply extreme traffic to create a distinct advantage for A*
    print("\nApplying extreme traffic conditions...")
    
    # First, find the most central node as a reference point
    total_x = 0
    total_y = 0
    node_count = 0
    
    for node_id, data in G.nodes(data=True):
        if 'x' in data and 'y' in data:
            total_x += data['x']
            total_y += data['y']
            node_count += 1
            
    center_x = total_x / node_count if node_count > 0 else 0
    center_y = total_y / node_count if node_count > 0 else 0
    
    # Apply traffic based on distance from center and road type
    for u, v, k, data in G.edges(keys=True, data=True):
        # Get node coordinates
        u_data = G.nodes[u]
        v_data = G.nodes[v]
        
        if 'x' in u_data and 'y' in u_data and 'x' in v_data and 'y' in v_data:
            # Calculate midpoint of edge
            mid_x = (u_data['x'] + v_data['x']) / 2
            mid_y = (u_data['y'] + v_data['y']) / 2
            
            # Calculate distance from center (rough approximation)
            dist_from_center = ((mid_x - center_x) ** 2 + (mid_y - center_y) ** 2) ** 0.5
            
            # Normalize distance (0-1 scale)
            max_dist = 0.05  # Approximate max distance in degrees
            normalized_dist = min(dist_from_center / max_dist, 1.0)
            
            # Get original weight
            orig_weight = float(data.get('weight', 10.0))
            
            # Apply traffic pattern: Heavy in center, lighter on outskirts
            if 'highway' in data:
                if data['highway'] in ['motorway', 'trunk', 'primary']:
                    # Major roads: In center - very heavy, outskirts - moderate
                    if normalized_dist < 0.5:
                        # Create a traffic "jam" near center
                        traffic_factor = 8.0 - 6.0 * normalized_dist
                    else:
                        traffic_factor = 2.0
                        
                elif data['highway'] in ['secondary', 'tertiary']:
                    # Secondary roads: More uniform traffic, slightly worse near center
                    traffic_factor = 3.0 - normalized_dist
                    
                else:
                    # Minor roads: Lower traffic everywhere
                    traffic_factor = 1.5
                    
                # Apply the traffic factor to edge weight
                G.edges[u, v, k]['weight'] = orig_weight * traffic_factor
                
                # Update traffic speed if present
                if 'traffic_speed' in data and data['traffic_speed'] > 0:
                    G.edges[u, v, k]['traffic_speed'] = float(data['traffic_speed']) / traffic_factor
    
    # Create some "blocked" major roads (extreme congestion)
    # This creates specific scenarios where A* should find alternatives
    edges_to_block = []
    for u, v, k, data in G.edges(keys=True, data=True):
        if 'highway' in data and data['highway'] in ['motorway', 'trunk', 'primary']:
            edges_to_block.append((u, v, k))
    
    # Block ~2% of major roads with extreme congestion (simulating accidents)
    import random
    random.seed(42)  # For reproducibility
    num_to_block = max(int(len(edges_to_block) * 0.02), 5)
    blocked_edges = random.sample(edges_to_block, num_to_block)
    
    for u, v, k in blocked_edges:
        # Extreme congestion - make these roads highly undesirable
        G.edges[u, v, k]['weight'] *= 15.0
        if 'traffic_speed' in G.edges[u, v, k]:
            G.edges[u, v, k]['traffic_speed'] /= 15.0
        print(f"Blocked road: {u}-{v} (highway type: {G.edges[u, v, k].get('highway', 'unknown')})")
    
    # Enhance A* heuristic - Create a customized router
    class EnhancedAStarRouter(AStarRouter):
        def astar_heuristic(self, u, v):
            """Enhanced A* heuristic that's more aggressive in avoiding congestion"""
            # Calculate straight-line distance
            distance = self.haversine_distance(u, v)
            
            # Use a more optimistic speed estimate (1.5x the normal max speed)
            # This makes the heuristic more "optimistic" about alternatives
            # and encourages exploration of paths that might initially seem longer
            optimistic_max_speed = self._max_speed * 1.5
            time_estimate = distance / optimistic_max_speed
            
            return time_estimate
    
    # Initialize routers
    dijkstra_router = DijkstraRouter(G)
    astar_router = EnhancedAStarRouter(G)
    
    # Choose test points that are reasonably far apart
    nodes = list(G.nodes())
    
    # Try to find nodes in opposite quadrants from the center
    quadrant1_nodes = []
    quadrant3_nodes = []
    
    for node_id, data in G.nodes(data=True):
        if 'x' in data and 'y' in data:
            if data['x'] > center_x and data['y'] > center_y:
                quadrant1_nodes.append(node_id)
            elif data['x'] < center_x and data['y'] < center_y:
                quadrant3_nodes.append(node_id)
    
    if quadrant1_nodes and quadrant3_nodes:
        source = random.choice(quadrant1_nodes)
        target = random.choice(quadrant3_nodes)
    else:
        # Fallback if we can't find suitable quadrant nodes
        center_idx = len(nodes) // 2
        source = nodes[center_idx // 2]  # First quarter
        target = nodes[center_idx + (center_idx // 2)]  # Third quarter
    
    print(f"\nTesting route from node {source} to {target} with extreme traffic conditions")
    
    # Run both algorithms
    print("\nRunning Dijkstra algorithm...")
    dijkstra_result = dijkstra_router.find_route(source, target)
    
    print("\nRunning Enhanced A* algorithm...")
    astar_result = astar_router.find_route(source, target)
    
    if dijkstra_result and astar_result:
        # Print detailed comparison
        print("\n=== Detailed Route Comparison with Extreme Traffic ===")
        print(f"Dijkstra path length: {len(dijkstra_result['path'])} nodes")
        print(f"A* path length: {len(astar_result['path'])} nodes")
        print(f"Dijkstra travel time: {dijkstra_result['travel_time']:.2f}s")
        print(f"A* travel time: {astar_result['travel_time']:.2f}s")
        print(f"Dijkstra computation time: {dijkstra_result['computation_time']:.5f}s")
        print(f"A* computation time: {astar_result['computation_time']:.5f}s")
        
        # Compare paths
        paths_identical = dijkstra_result['path'] == astar_result['path']
        print(f"\nPaths identical: {paths_identical}")
        
        # Calculate time difference and improvement percentage
        time_diff = dijkstra_result['travel_time'] - astar_result['travel_time']
        improvement_pct = (time_diff / dijkstra_result['travel_time']) * 100 if dijkstra_result['travel_time'] > 0 else 0
        
        print(f"Time difference: {time_diff:.2f}s")
        print(f"A* improvement: {improvement_pct:.2f}%")
        
        # Calculate path overlap percentage
        dijkstra_set = set(dijkstra_result['path'])
        astar_set = set(astar_result['path'])
        common_nodes = dijkstra_set.intersection(astar_set)
        overlap_pct = (len(common_nodes) / max(len(dijkstra_set), len(astar_set))) * 100
        
        print(f"Path overlap: {overlap_pct:.2f}%")
        
        return True
    else:
        print("Error: One or both algorithms failed to find a path")
        return False


if __name__ == "__main__":
    run_diagnostic()
    test_with_traffic_data()