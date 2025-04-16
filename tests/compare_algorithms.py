# compare_algorithms.py

import os
import sys
from dotenv import load_dotenv
from data.data_loader import DataLoader
from data.data_processor import DataProcessor
from src.algorithms.dijkstra import DijkstraRouter
from src.algorithms.astar import AStarRouter
import matplotlib.pyplot as plt
import pandas as pd
import random

# Define the enhanced A* router class
class EnhancedAStarRouter(AStarRouter):
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
        
        # Convert to m/s and use a more optimistic value
        if max_speed > 0:
            # Use 150% of the max observed speed
            return max_speed * 1000 / 3600 * 1.5
        else:
            return 50.0  # Default high speed

    def astar_heuristic(self, u, v):
        """Optimized A* heuristic for better performance in traffic"""
        distance = self.haversine_distance(u, v)
        time_estimate = distance / self._max_speed
        return time_estimate

def run_comparison():
    # Load environment variables
    load_dotenv()
    
    # Initialize components
    data_loader = DataLoader()
    
    # Load cached graph
    cache_file = "data/processed/road_network.json"
    if os.path.exists(cache_file):
        print("\nLoading cached road network...")
        G = data_loader.load_graph_json(cache_file)
    else:
        print("\nLoading fresh road network from OpenStreetMap...")
        G = data_loader.load_osm_graph("UBC, Vancouver, Canada")
    
    # Initialize data processor and scenario generator
    from src.simulation.scenarios import ScenarioGenerator
    data_processor = DataProcessor(data_loader)
    scenario_gen = ScenarioGenerator(G, data_processor, data_loader)
    
    # Apply extreme traffic conditions
    print("\nSimulating extreme traffic conditions with strategic road blocks...")
    G = scenario_gen.extreme_traffic_with_forced_blockages(num_major_blocks=15, major_road_multiplier=10.0)
    
    # Initialize routers
    dijkstra_router = DijkstraRouter(G)
    astar_router = EnhancedAStarRouter(G)
    
    # Generate random start and end points
    nodes = list(G.nodes())
    random.seed(42)  # For reproducibility
    
    results = []
    
    for i in range(5):  # Test 5 different routes
        # Select nodes from different parts of the graph
        source = nodes[random.randint(0, len(nodes)//3)]
        target = nodes[random.randint(2*len(nodes)//3, len(nodes)-1)]
        
        print(f"\nTest {i+1}: Route from {source} to {target}")
        
        # Run Dijkstra
        print("Running Dijkstra algorithm...")
        dijkstra_result = dijkstra_router.find_route(source, target)
        
        # Run A*
        print("Running Enhanced A* algorithm...")
        astar_result = astar_router.find_route(source, target)
        
        if dijkstra_result and astar_result:
            # Calculate path overlap
            dijkstra_nodes = set(dijkstra_result['path'])
            astar_nodes = set(astar_result['path'])
            common_nodes = dijkstra_nodes.intersection(astar_nodes)
            overlap_pct = len(common_nodes) / max(len(dijkstra_nodes), len(astar_nodes)) * 100
            
            # Save results
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
                'time_diff': dijkstra_result['travel_time'] - astar_result['travel_time'],
                'improvement_pct': ((dijkstra_result['travel_time'] - astar_result['travel_time']) / 
                                  dijkstra_result['travel_time']) * 100 if dijkstra_result['travel_time'] > 0 else 0
            })
            
            print(f"Dijkstra path length: {len(dijkstra_result['path'])} nodes")
            print(f"A* path length: {len(astar_result['path'])} nodes")
            print(f"Dijkstra travel time: {dijkstra_result['travel_time']:.2f}s")
            print(f"A* travel time: {astar_result['travel_time']:.2f}s")
            print(f"Paths identical: {dijkstra_result['path'] == astar_result['path']}")
            print(f"Path overlap: {overlap_pct:.2f}%")
            print(f"Time difference: {results[-1]['time_diff']:.2f}s")
            print(f"A* improvement: {results[-1]['improvement_pct']:.2f}%")
    
    # Print summary statistics
    if results:
        df = pd.DataFrame(results)
        print("\n=== Summary of Results ===")
        print(f"Tests with different paths: {(~df['paths_identical']).sum()} out of {len(df)}")
        print(f"Average path overlap: {df['path_overlap_pct'].mean():.2f}%")
        print(f"Average time difference: {df['time_diff'].mean():.2f}s")
        print(f"Average A* improvement: {df['improvement_pct'].mean():.2f}%")
        
        # Save results to CSV
        result_file = "data/processed/algorithm_comparison_results.csv"
        df.to_csv(result_file, index=False)
        print(f"\nResults saved to {result_file}")

if __name__ == "__main__":
    run_comparison()