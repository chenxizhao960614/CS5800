# test_algorithms.py
import os
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from datetime import datetime
from shapely.geometry import box
from collections import defaultdict

# Import the new implementations
from data.data_loader import DataLoader
from data.data_processor import DataProcessor
from src.algorithms.dijkstra import DijkstraRouter
from src.algorithms.astar import AStarRouter

def load_graph():
    """Load graph from file or create new one if not exists"""
    data_loader = DataLoader()
    
    # Load cached graph if it exists
    cache_file = "data/processed/road_network.json"
    if os.path.exists(cache_file):
        print("\nLoading cached road network...")
        G = data_loader.load_graph_json(cache_file)
    else:
        print("\nLoading fresh road network from OpenStreetMap...")
        G = data_loader.load_osm_graph("Vancouver, British Columbia, Canada")
        print("Saving road network for future use...")
        data_loader.save_graph_json(G, cache_file)
    
    return G, data_loader

def create_extreme_traffic_scenario(G, data_processor):
    """Create an extreme traffic scenario to force algorithm differences"""
    print("\nCreating extreme traffic scenario with heavy congestion...")
    
    # Create a copy of the graph to avoid modifying the original
    G_extreme = G.copy()
    
    # Get bounding box of the graph
    all_lats = [data['y'] for _, data in G.nodes(data=True) if 'y' in data]
    all_lons = [data['x'] for _, data in G.nodes(data=True) if 'x' in data]
    
    if not all_lats or not all_lons:
        print("Error: Graph nodes don't have coordinates")
        return G
    
    min_lat, max_lat = min(all_lats), max(all_lats)
    min_lon, max_lon = min(all_lons), max(all_lons)
    
    # Create bbox string for the data processor
    bbox = f"{min_lon},{min_lat},{max_lon},{max_lat}"
    
    # Apply extreme traffic conditions to create diverse routing options
    
    # 1. Add extreme congestion to major roads
    for u, v, k, data in G_extreme.edges(keys=True, data=True):
        highway_type = data.get('highway', '')
        
        # Add basic weight if missing
        if 'weight' not in data and 'length' in data:
            length = data['length']
            G_extreme.edges[u, v, k]['weight'] = length / 13.89  # 50 km/h
        
        # Add extreme delays to major roads (forcing algorithms to find alternatives)
        if highway_type in ['motorway', 'trunk', 'primary']:
            if 'weight' in data:
                # Increase travel time by 400-600% on major roads
                multiplier = 4.0 + (hash(str(u) + str(v)) % 200) / 100.0
                G_extreme.edges[u, v, k]['weight'] *= multiplier
                G_extreme.edges[u, v, k]['traffic_speed'] = 50 / multiplier  # Reduced speed
        
        # Add medium congestion to secondary roads
        elif highway_type in ['secondary']:
            if 'weight' in data:
                # Increase travel time by 200-300%
                multiplier = 2.0 + (hash(str(u) + str(v)) % 100) / 100.0
                G_extreme.edges[u, v, k]['weight'] *= multiplier
                G_extreme.edges[u, v, k]['traffic_speed'] = 50 / multiplier  # Reduced speed
    
    # 2. Add random incident blockages
   
   # Create collections for road types
    major_edges = []
    for u, v, k, data in G_extreme.edges(keys=True, data=True):
        highway_type = data.get('highway', 'other')
        
        # Handle case where highway_type is a list
        if isinstance(highway_type, list):
            # If it's a list, check if any element is a major road type
            is_major = any(h in ['motorway', 'trunk', 'primary'] for h in highway_type)
        else:
            # Direct string comparison
            is_major = highway_type in ['motorway', 'trunk', 'primary']
        
        if is_major:
            major_edges.append((u, v, k))



    
    # Block 10 random major road segments
    # major_edges = []
    # for road_type in ['motorway', 'trunk', 'primary']:
    #     major_edges.extend(edges_by_type.get(road_type, []))
    
    if major_edges:
        num_to_block = min(10, len(major_edges))
        blocked_edges = random.sample(major_edges, num_to_block)
        
        for u, v, k in blocked_edges:
            # Effectively block the road
            G_extreme.edges[u, v, k]['weight'] = float('inf')
            G_extreme.edges[u, v, k]['traffic_speed'] = 0.1
            G_extreme.edges[u, v, k]['incident'] = True
            G_extreme.edges[u, v, k]['incident_type'] = 'ROAD_CLOSURE'
    
    return G_extreme

def test_route_algorithms(G, start_node, end_node):
    """Test both routing algorithms on the same graph"""
    print(f"\nTesting route from node {start_node} to node {end_node}")
    
    # Initialize routers
    dijkstra_router = DijkstraRouter(G)
    astar_router = AStarRouter(G)
    
    # Run Dijkstra's algorithm
    print("\nRunning Dijkstra's algorithm...")
    dijkstra_result = dijkstra_router.find_route(start_node, end_node)
    
    # Run A* algorithm
    print("\nRunning A* algorithm...")
    astar_result = astar_router.find_route(start_node, end_node)
    
    if not dijkstra_result or not astar_result:
        print("Error: One or both algorithms failed to find a route")
        return None
    
    # Compare results
    print("\n=== Comparison of Results ===")
    
    # Path stats
    dijkstra_path = dijkstra_result['path']
    astar_path = astar_result['path']
    paths_identical = dijkstra_path == astar_path
    
    # Calculate path overlap percentage
    dijkstra_set = set(dijkstra_path)
    astar_set = set(astar_path)
    intersection = dijkstra_set.intersection(astar_set)
    union = dijkstra_set.union(astar_set)
    overlap_percentage = len(intersection) / len(union) * 100 if union else 100
    
    print(f"Paths identical: {paths_identical}")
    print(f"Path overlap: {overlap_percentage:.1f}%")
    print(f"Dijkstra path length: {len(dijkstra_path)} nodes")
    print(f"A* path length: {len(astar_path)} nodes")
    
    # Performance stats
    dijkstra_time = dijkstra_result['travel_time']
    astar_time = astar_result['travel_time']
    dijkstra_compute = dijkstra_result['computation_time']
    astar_compute = astar_result['computation_time']
    
    time_diff = dijkstra_time - astar_time
    time_pct = (time_diff / dijkstra_time) * 100 if dijkstra_time > 0 else 0
    
    print(f"\nDijkstra travel time: {dijkstra_time:.2f} seconds")
    print(f"A* travel time: {astar_time:.2f} seconds")
    print(f"Difference: {time_diff:.2f} seconds ({time_pct:.1f}%)")
    
    print(f"\nDijkstra computation time: {dijkstra_compute:.5f} seconds")
    print(f"A* computation time: {astar_compute:.5f} seconds")
    
    # Nodes expanded
    dijkstra_nodes = dijkstra_result.get('nodes_expanded', 0)
    astar_nodes = astar_result.get('nodes_expanded', 0)
    
    print(f"\nDijkstra nodes expanded: {dijkstra_nodes}")
    print(f"A* nodes expanded: {astar_nodes}")
    
    # Overall assessment
    print("\n=== Overall Assessment ===")
    if paths_identical:
        print("Both algorithms found the same path.")
        if time_diff == 0:
            print("Travel times are identical.")
        elif time_diff < 0:
            print(f"Dijkstra found a slightly faster route by {abs(time_diff):.2f} seconds.")
        else:
            print(f"A* found a slightly faster route by {time_diff:.2f} seconds.")
    else:
        print("Algorithms found different paths!")
        if time_diff < 0:
            print(f"Dijkstra found a faster route by {abs(time_diff):.2f} seconds.")
        else:
            print(f"A* found a faster route by {time_diff:.2f} seconds.")
    
    # Return results for visualization
    return {
        'dijkstra_result': dijkstra_result,
        'astar_result': astar_result,
        'paths_identical': paths_identical,
        'overlap_percentage': overlap_percentage,
        'time_diff': time_diff,
        'time_pct': time_pct
    }

def visualize_routes(G, results):
    """Visualize both routes on the graph"""
    if not results:
        return
    
    dijkstra_path = results['dijkstra_result']['path']
    astar_path = results['astar_result']['path']
    
    try:
        import osmnx as ox
        
        # Create figure and axes
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot base graph
        edge_args = {
            'node_size': 0,
            'edge_color': '#ccc',
            'edge_linewidth': 0.5,
            'bgcolor': 'white',
            'show': False,
            'ax': ax
        }
        ox.plot_graph(G, **edge_args)
        
        # Plot paths with different styles
        if results['paths_identical']:
            # If paths are identical, make them visible by using different line styles
            dijkstra_args = {
                'route_color': 'blue',
                'route_linewidth': 4,
                'route_alpha': 0.7,
                'ax': ax,
                'show': False
            }
            astar_args = {
                'route_color': 'red',
                'route_linewidth': 2,
                'route_linestyle': '--',
                'route_alpha': 0.7,
                'ax': ax,
                'show': False
            }
        else:
            # Different paths - use different colors
            dijkstra_args = {
                'route_color': 'blue',
                'route_linewidth': 4,
                'route_alpha': 0.7,
                'ax': ax,
                'show': False
            }
            astar_args = {
                'route_color': 'red',
                'route_linewidth': 2,
                'route_alpha': 0.7,
                'ax': ax,
                'show': False
            }
            
        # Plot routes - A* first so Dijkstra is on top
        ox.plot_graph_route(G, astar_path, **astar_args)
        ox.plot_graph_route(G, dijkstra_path, **dijkstra_args)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='blue', linewidth=4, label="Dijkstra's Algorithm"),
            Line2D([0], [0], color='red', linewidth=2, label="A* Algorithm")
        ]
        ax.legend(handles=legend_elements, loc='upper right', title="Routing Algorithms")
        
        # Add title with statistics
        title = "Route Comparison: Dijkstra vs A*\n"
        if results['paths_identical']:
            title += "Identical Paths"
        else:
            title += f"Different Paths (Overlap: {results['overlap_percentage']:.1f}%)"
        
        title += f"\nDijkstra: {results['dijkstra_result']['travel_time']:.1f}s  |  A*: {results['astar_result']['travel_time']:.1f}s"
        title += f"\nDifference: {abs(results['time_diff']):.1f}s ({abs(results['time_pct']):.1f}%)"
        
        ax.set_title(title)
        
        # Show the plot
        plt.tight_layout()
        plt.savefig('route_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        print(f"Error visualizing routes: {e}")
        import traceback
        traceback.print_exc()

def run_multiple_tests(G, num_tests=5):
    """Run multiple tests with random start/end nodes"""
    print(f"\nRunning {num_tests} random route tests...")
    
    # Get a list of valid nodes (those with coordinates)
    valid_nodes = []
    for node, data in G.nodes(data=True):
        if 'x' in data and 'y' in data:
            valid_nodes.append(node)
    
    if len(valid_nodes) < 2:
        print("Error: Not enough valid nodes with coordinates")
        return
    
    # Results container
    test_results = []
    
    for i in range(num_tests):
        print(f"\n--- Test {i+1}/{num_tests} ---")
        
        # Pick random nodes far enough apart
        while True:
            start_node = random.choice(valid_nodes)
            end_node = random.choice(valid_nodes)
            
            # Check if they're far enough apart
            try:
                start_coords = (G.nodes[start_node]['y'], G.nodes[start_node]['x'])
                end_coords = (G.nodes[end_node]['y'], G.nodes[end_node]['x'])
                
                # Calculate rough distance
                from math import radians, sin, cos, sqrt, atan2
                lat1, lon1 = radians(start_coords[0]), radians(start_coords[1])
                lat2, lon2 = radians(end_coords[0]), radians(end_coords[1])
                
                dlon = lon2 - lon1
                dlat = lat2 - lat1
                a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                c = 2 * atan2(sqrt(a), sqrt(1-a))
                distance = 6371000 * c  # Earth radius in meters
                
                # Only accept pairs at least 1km apart
                if distance >= 1000 and start_node != end_node:
                    break
            except:
                # If exception occurs, just pick a new pair
                continue
        
        # Run the test
        result = test_route_algorithms(G, start_node, end_node)
        
        if result:
            # Store the result
            test_data = {
                'test': i,
                'source': start_node,
                'target': end_node,
                'dijkstra_travel_time': result['dijkstra_result']['travel_time'],
                'astar_travel_time': result['astar_result']['travel_time'],
                'dijkstra_path_length': len(result['dijkstra_result']['path']),
                'astar_path_length': len(result['astar_result']['path']),
                'paths_identical': result['paths_identical'],
                'path_overlap_pct': result['overlap_percentage'],
                'time_diff': result['time_diff'],
                'improvement_pct': result['time_pct']
            }
            test_results.append(test_data)
            
            # Visualize only the first result
            if i == 0:
                visualize_routes(G, result)
    
    # Convert to DataFrame and save
    if test_results:
        results_df = pd.DataFrame(test_results)
        results_df.to_csv('extreme_traffic_results.csv', index=False)
        
        # Print summary
        print("\n=== Summary of Tests ===")
        print(f"Tests completed: {len(test_results)}")
        identical_paths = sum(1 for r in test_results if r['paths_identical'])
        print(f"Tests with identical paths: {identical_paths} ({identical_paths/len(test_results)*100:.1f}%)")
        
        avg_time_diff = results_df['time_diff'].mean()
        print(f"Average time difference: {avg_time_diff:.2f} seconds")
        
        avg_improvement = results_df['improvement_pct'].mean()
        print(f"Average improvement percentage: {avg_improvement:.2f}%")
        
        # Which algorithm was better?
        a_star_wins = sum(1 for r in test_results if r['time_diff'] > 0)
        dijkstra_wins = sum(1 for r in test_results if r['time_diff'] < 0)
        ties = sum(1 for r in test_results if r['time_diff'] == 0)
        
        print(f"A* found faster routes: {a_star_wins} times")
        print(f"Dijkstra found faster routes: {dijkstra_wins} times")
        print(f"Both found equally fast routes: {ties} times")
        
        return results_df
    
    return None

if __name__ == "__main__":
    # Load graph
    G, data_loader = load_graph()
    
    # Create data processor
    data_processor = DataProcessor(data_loader)
    
    # Create extreme traffic scenario
    G_extreme = create_extreme_traffic_scenario(G, data_processor)
    
    # Run tests
    results = run_multiple_tests(G_extreme, num_tests=5)