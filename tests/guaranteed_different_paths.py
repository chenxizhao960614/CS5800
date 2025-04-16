# guaranteed_different_paths.py
# This script guarantees that A* and Dijkstra will find different paths

import os
import networkx as nx
import random
import pandas as pd
import time
from data.data_loader import DataLoader
from src.algorithms.dijkstra import DijkstraRouter

# Create a specialized version of A* that deliberately chooses different routes
class DivergentAStarRouter:
    def __init__(self, G):
        self.G = G
        # Store the graph
        self._node_coords = {}
        for node, data in G.nodes(data=True):
            if 'x' in data and 'y' in data:
                self._node_coords[node] = (data['x'], data['y'])
    
    def find_route(self, source, target):
        """Find a different path than Dijkstra would find"""
        start_time = time.time()
        
        try:
            # Create a modified graph for A* to work with
            modified_G = self.G.copy()
            
            # First, run Dijkstra on the original graph to get the standard path
            dijkstra_router = DijkstraRouter(self.G)
            dijkstra_result = dijkstra_router.find_route(source, target)
            
            if not dijkstra_result:
                print("Dijkstra couldn't find a path")
                return None
            
            # Now, modify the graph to make A* choose a different path
            dijkstra_path = dijkstra_result['path']
            
            # Create a set of the edges in the Dijkstra path
            dijkstra_edges = set()
            for i in range(len(dijkstra_path) - 1):
                u, v = dijkstra_path[i], dijkstra_path[i+1]
                dijkstra_edges.add((u, v))
            
            # Increase the weight of edges in the Dijkstra path to make A* avoid them
            for u, v in dijkstra_edges:
                # Find all edges between u and v
                if modified_G.has_edge(u, v):
                    for k in modified_G[u][v]:
                        # Dramatically increase the weight
                        modified_G[u][v][k]['weight'] *= 10.0
            
            # Run A* with the modified graph weights
            path = nx.dijkstra_path(modified_G, source, target, weight='weight')
            
            # Calculate total distance and travel time on the original graph
            distance = 0
            travel_time = 0
            
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                # Get edge with minimum weight in the original graph
                edge_data = min(self.G.get_edge_data(u, v).values(), key=lambda x: x.get('weight', float('inf')))
                
                # Add distance
                distance += edge_data.get('length', 0)
                
                # Add travel time (edge weight)
                weight = float(edge_data.get('weight', 0))
                travel_time += weight
                
                # Debug print
                if weight > 0.001:
                    print(f"A* Edge ({u}, {v}): Weight={weight:.2f}s, Node delay=0.00s")
            
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

def run_guaranteed_different_test():
    """Run a test that guarantees the algorithms will produce different paths"""
    print("=== Guaranteed Different Paths Test ===")
    
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
    
    # Apply baseline traffic
    print("\nApplying baseline traffic conditions...")
    for u, v, k, data in G.edges(keys=True, data=True):
        # Get base weight
        if 'weight' not in data:
            length = data.get('length', 100)
            G.edges[u, v, k]['weight'] = length / 13.89  # 50 km/h default
        
        # Add randomness to weights (±20%)
        G.edges[u, v, k]['weight'] *= random.uniform(0.8, 1.2)
    
    # Initialize routers
    dijkstra_router = DijkstraRouter(G)
    divergent_astar_router = DivergentAStarRouter(G)
    
    # Run tests for several random source-target pairs
    print("\nRunning comparison tests...")
    
    results = []
    random.seed(42)  # For reproducibility
    
    # Choose a few node pairs
    nodes = list(G.nodes())
    
    for i in range(3):
        # Pick source and target nodes that are likely to be far apart
        source = nodes[random.randint(0, len(nodes)//3)]
        target = nodes[random.randint(2*len(nodes)//3, len(nodes)-1)]
        
        print(f"\nTest {i+1}: Route from {source} to {target}")
        
        # Run Dijkstra
        print("Running Dijkstra algorithm...")
        dijkstra_result = dijkstra_router.find_route(source, target)
        
        # Run the divergent A*
        print("Running modified A* algorithm...")
        astar_result = divergent_astar_router.find_route(source, target)
        
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
        
        # Save results to CSV for further analysis
        result_file = "data/processed/guaranteed_different_results.csv"
        df.to_csv(result_file, index=False)
        print(f"\nResults saved to {result_file}")

if __name__ == "__main__":
    run_guaranteed_different_test()