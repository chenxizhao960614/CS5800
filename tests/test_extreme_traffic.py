# test_extreme_traffic.py
# Standalone script to test the algorithms with extreme traffic conditions

import os
import networkx as nx
import random
import matplotlib.pyplot as plt
import pandas as pd
from data.data_loader import DataLoader
from src.algorithms.dijkstra import DijkstraRouter 
from src.algorithms.astar import AStarRouter

def run_extreme_traffic_test():
    print("=== Extreme Traffic Differentiation Test ===")
    
    # Initialize data loader
    data_loader = DataLoader()
    
    # Load graph
    cache_file = "data/processed/road_network.json"
    if os.path.exists(cache_file):
        print("\nLoading cached road network...")
        G = data_loader.load_graph_json(cache_file)
    else:
        print("\nLoading fresh road network from OpenStreetMap...")
        G = data_loader.load_osm_graph("UBC, Vancouver, Canada")
    
    print(f"Loaded graph with {len(G.nodes)} nodes and {len(G.edges)} edges")
    
    # Create extreme traffic conditions
    print("\nCreating extreme traffic conditions...")
    
    # 1. Apply different traffic factors based on road type
    for u, v, k, data in G.edges(keys=True, data=True):
        highway_type = data.get('highway', 'residential')
        
        # Get base weight
        orig_weight = float(data.get('weight', 10.0))
        
        # Different multipliers for different road types
        if highway_type in ['motorway', 'trunk']:
            # Major highways - extremely congested
            multiplier = random.uniform(6.0, 8.0)
        elif highway_type in ['primary']:
            # Primary roads - heavily congested
            multiplier = random.uniform(4.0, 5.0)
        elif highway_type in ['secondary']:
            # Secondary roads - moderately congested
            multiplier = random.uniform(2.0, 3.0)
        elif highway_type in ['tertiary']:
            # Tertiary roads - light congestion
            multiplier = random.uniform(1.5, 2.0)
        else:
            # Residential and other minor roads - minimal congestion
            multiplier = random.uniform(1.0, 1.3)
        
        # Apply multiplier
        G.edges[u, v, k]['weight'] = orig_weight * multiplier
        
        # Update traffic speed if it exists
        if 'traffic_speed' in data:
            G.edges[u, v, k]['traffic_speed'] = data['traffic_speed'] / multiplier
    
    # 2. Create road blockages/incidents on major roads
    major_edges = []
    for u, v, k, data in G.edges(keys=True, data=True):
        if data.get('highway') in ['motorway', 'trunk', 'primary']:
            major_edges.append((u, v, k))
    
    # Block 20 major roads (or fewer if there are less than 20)
    num_to_block = min(20, len(major_edges))
    blocked_edges = random.sample(major_edges, num_to_block)
    
    print(f"Blocking {num_to_block} major roads...")
    for u, v, k in blocked_edges:
        # Extreme congestion - almost blocking the road
        G.edges[u, v, k]['weight'] *= 20.0
        if 'traffic_speed' in G.edges[u, v, k]:
            G.edges[u, v, k]['traffic_speed'] /= 20.0
    
    # Initialize routers
    dijkstra_router = DijkstraRouter(G)
    
    # Create enhanced A* router with more aggressive heuristic
    class EnhancedAStarRouter(AStarRouter):
        def astar_heuristic(self, u, v):
            """More aggressive A* heuristic to find traffic-avoiding routes"""
            # Calculate straight-line distance
            distance = self.haversine_distance(u, v)
            
            # More optimistic speed estimate makes heuristic push harder towards destination
            # This helps find alternative routes that avoid congestion
            optimistic_speed = self._max_speed * 1.5  # 50% more optimistic
            
            # Estimated time based on optimistic speed
            time_estimate = distance / optimistic_speed
            
            return time_estimate
    
    astar_router = EnhancedAStarRouter(G)
    
    # Run tests for 5 random source-target pairs
    print("\nRunning comparison tests...")
    
    results = []
    random.seed(42)  # For reproducibility
    
    # Try to find nodes that are reasonably far apart
    nodes = list(G.nodes())
    
    for i in range(5):
        # Pick source and target nodes that are likely to be far apart
        first_half = nodes[:len(nodes)//2]
        second_half = nodes[len(nodes)//2:]
        
        source = random.choice(first_half)
        target = random.choice(second_half)
        
        print(f"\nTest {i+1}: Route from {source} to {target}")
        
        # Run Dijkstra
        print("Running Dijkstra algorithm...")
        dijkstra_result = dijkstra_router.find_route(source, target)
        
        # Run A*
        print("Running A* algorithm...")
        astar_result = astar_router.find_route(source, target)
        
        if dijkstra_result and astar_result:
            # Calculate path overlap
            dijkstra_nodes = set(dijkstra_result['path'])
            astar_nodes = set(astar_result['path'])
            common_nodes = dijkstra_nodes.intersection(astar_nodes)
            overlap_pct = len(common_nodes) / max(len(dijkstra_nodes), len(astar_nodes)) * 100
            
            # Calculate time difference
            time_diff = dijkstra_result['travel_time'] - astar_result['travel_time']
            if dijkstra_result['travel_time'] > 0:
                improvement_pct = (time_diff / dijkstra_result['travel_time']) * 100
            else:
                improvement_pct = 0
            
            print(f"Dijkstra path length: {len(dijkstra_result['path'])} nodes")
            print(f"A* path length: {len(astar_result['path'])} nodes")
            print(f"Dijkstra travel time: {dijkstra_result['travel_time']:.2f}s")
            print(f"A* travel time: {astar_result['travel_time']:.2f}s")
            print(f"Paths identical: {dijkstra_result['path'] == astar_result['path']}")
            print(f"Path overlap: {overlap_pct:.2f}%")
            print(f"Time difference: {time_diff:.2f}s")
            print(f"A* improvement: {improvement_pct:.2f}%")
            
            results.append({
                'test': i+1,
                'source': source,
                'target': target,
                'dijkstra_travel_time': dijkstra_result['travel_time'],
                'astar_travel_time': astar_result['travel_time'],
                'dijkstra_path_length': len(dijkstra_result['path']),
                'astar_path_length': len(astar_result['path']),
                'paths_identical': dijkstra_result['path'] == astar_result['path'],
                'path_overlap_pct': overlap_pct,
                'time_diff': time_diff,
                'improvement_pct': improvement_pct
            })
    
    # Summarize results
    if results:
        df = pd.DataFrame(results)
        
        print("\n=== Summary of Results ===")
        print(f"Tests with different paths: {(~df['paths_identical']).sum()} out of {len(df)}")
        print(f"Average path overlap: {df['path_overlap_pct'].mean():.2f}%")
        print(f"Average time difference: {df['time_diff'].mean():.2f}s")
        print(f"Average A* improvement: {df['improvement_pct'].mean():.2f}%")
        
        # Save results to CSV for further analysis
        result_file = "data/processed/extreme_traffic_results.csv"
        df.to_csv(result_file, index=False)
        print(f"\nResults saved to {result_file}")
    
    return df





if __name__ == "__main__":
    results = run_extreme_traffic_test()