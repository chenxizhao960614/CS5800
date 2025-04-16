# check_weights.py
import os
import networkx as nx
from data.data_loader import DataLoader
from data.data_processor import DataProcessor

def check_graph_weights():
    """Check if graph weights are properly set"""
    print("Checking graph weights...")
    
    # Initialize components
    data_loader = DataLoader()
    
    # Load cached graph
    cache_file = "data/processed/road_network.json"
    if os.path.exists(cache_file):
        print("\nLoading cached road network...")
        G = data_loader.load_graph_json(cache_file)
    else:
        print("\nNo cached graph found.")
        return
    
    # Check edge weights
    total_edges = 0
    edges_with_weight = 0
    zero_weight_edges = 0
    negative_weight_edges = 0
    edges_with_traffic_speed = 0
    
    for u, v, k, data in G.edges(keys=True, data=True):
        total_edges += 1
        
        if 'weight' in data:
            edges_with_weight += 1
            if data['weight'] == 0:
                zero_weight_edges += 1
            elif data['weight'] < 0:
                negative_weight_edges += 1
        
        if 'traffic_speed' in data:
            edges_with_traffic_speed += 1
    
    print(f"\nTotal edges: {total_edges}")
    print(f"Edges with weight: {edges_with_weight} ({edges_with_weight/total_edges*100:.1f}%)")
    print(f"Edges with zero weight: {zero_weight_edges} ({zero_weight_edges/total_edges*100:.1f}%)")
    print(f"Edges with negative weight: {negative_weight_edges} ({negative_weight_edges/total_edges*100:.1f}%)")
    print(f"Edges with traffic speed: {edges_with_traffic_speed} ({edges_with_traffic_speed/total_edges*100:.1f}%)")
    
    # Check a few random edges
    print("\nSample edge data:")
    
    import random
    sample_edges = random.sample(list(G.edges(keys=True, data=True)), min(5, total_edges))
    
    for u, v, k, data in sample_edges:
        print(f"\nEdge ({u}, {v}, {k}):")
        weight = data.get('weight', 'Not set')
        traffic_speed = data.get('traffic_speed', 'Not set')
        length = data.get('length', 'Not set')
        highway = data.get('highway', 'Not set')
        
        print(f"  weight: {weight}")
        print(f"  traffic_speed: {traffic_speed}")
        print(f"  length: {length}")
        print(f"  highway: {highway}")
    
    # Now apply traffic and check again
    print("\nUpdating graph with traffic data...")
    data_processor = DataProcessor(data_loader)
    G = data_processor.update_graph_with_traffic(G)
    
    # Check weights after processing
    print("\nChecking weights after traffic processing:")
    
    zero_weight_after = 0
    for u, v, k, data in G.edges(keys=True, data=True):
        if 'weight' in data and data['weight'] == 0:
            zero_weight_after += 1
    
    print(f"Edges with zero weight after processing: {zero_weight_after}")
    
    # Check same sample edges after processing
    print("\nSame sample edges after processing:")
    
    for u, v, k, data in sample_edges:
        try:
            edge_data = G.get_edge_data(u, v, k)
            if edge_data:
                print(f"\nEdge ({u}, {v}, {k}):")
                weight = edge_data.get('weight', 'Not set')
                traffic_speed = edge_data.get('traffic_speed', 'Not set')
                print(f"  weight: {weight}")
                print(f"  traffic_speed: {traffic_speed}")
        except:
            print(f"Edge ({u}, {v}, {k}) no longer exists")

if __name__ == "__main__":
    check_graph_weights()