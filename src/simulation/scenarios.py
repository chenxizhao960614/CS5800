# src/simulation/scenarios.py
import pandas as pd
import numpy as np
import random

class ScenarioGenerator:
    def __init__(self, G, data_processor, data_loader):
        self.G = G
        self.data_processor = data_processor
        self.data_loader = data_loader
        
    def normal_traffic_scenario(self):
        """Normal traffic conditions scenario"""
        # Use a copy of the original graph
        G_normal = self.G.copy()
        
        # Update with traffic data and ensure higher coverage
        G_normal = self.data_processor.update_graph_with_traffic(G_normal, update_percentage=70)
        
        return G_normal
    
    def rush_hour_scenario(self):
        """Rush hour traffic scenario with heavy congestion"""
        # Use a copy of the original graph
        G_rush = self.G.copy()
        
        # Update with traffic data
        G_rush = self.data_processor.update_graph_with_traffic(G_rush, update_percentage=70)
        
        # Add additional congestion to major roads
        for u, v, k, data in G_rush.edges(keys=True, data=True):
            highway_type = data.get('highway', '')
            
            # Make sure 'weight' exists in the data dictionary
            if 'weight' not in data:
                continue  # Skip this edge if it doesn't have a weight
                
            # Increase travel time for major roads
            if highway_type in ['motorway', 'trunk', 'primary']:
                # Increase travel time by 100% (double)
                data['weight'] *= 2.0
                if 'traffic_speed' in data:
                    data['traffic_speed'] /= 2.0
            elif highway_type in ['secondary', 'tertiary']:
                # Increase travel time by 50%
                data['weight'] *= 1.5
                if 'traffic_speed' in data:
                    data['traffic_speed'] /= 1.5
        
        return G_rush
    
    def incident_scenario(self, num_incidents=5):
        """Scenario with random traffic incidents/accidents"""
        # Use a copy of the original graph
        G_incident = self.G.copy()
        
        # Update with traffic data
        G_incident = self.data_processor.update_graph_with_traffic(G_incident, update_percentage=70)
        
        # Get random edges to simulate incidents
        all_edges = list(G_incident.edges(keys=True))
        incident_edges = random.sample(all_edges, num_incidents)
        
        # Add severe delays or block roads entirely
        for u, v, k in incident_edges:
            edge_data = G_incident.get_edge_data(u, v)[k]
            
            # Make sure 'weight' exists in the data dictionary
            if 'weight' not in edge_data:
                edge_data['weight'] = 60  # Default 1 minute
            
            # Severely increase travel time (5x-10x)
            delay_factor = random.uniform(5, 10)
            edge_data['weight'] *= delay_factor
            
            # Update traffic speed
            if 'traffic_speed' in edge_data:
                edge_data['traffic_speed'] /= delay_factor
            else:
                edge_data['traffic_speed'] = 5  # Very slow speed
        
        return G_incident
    
    def missing_data_scenario(self, missing_percentage=20):
        """Scenario with missing traffic data"""
        # Use a copy of the original graph
        G_missing = self.G.copy()
        
        # Update with traffic data
        G_missing = self.data_processor.update_graph_with_traffic(G_missing, update_percentage=70)
        
        # Get random edges to simulate missing data
        all_edges = list(G_missing.edges(keys=True))
        num_missing = int(len(all_edges) * (missing_percentage / 100))
        missing_edges = random.sample(all_edges, num_missing)
        
        # Reset these edges to use default weights
        for u, v, k in missing_edges:
            edge_data = G_missing.get_edge_data(u, v)[k]
            
            # Reset to default weight based on length and speed limit
            length = edge_data.get('length', 100)  # meters
            speed_limit = edge_data.get('speed_kph', 50)  # km/h
            
            # Calculate default travel time in seconds
            default_time = (length / 1000) / (speed_limit / 3600)
            edge_data['weight'] = default_time
            
            # Remove traffic speed data
            if 'traffic_speed' in edge_data:
                del edge_data['traffic_speed']
        
        return G_missing
    



    def extreme_traffic_scenario(self, num_incidents=15):
        """Scenario with extreme traffic conditions to highlight algorithm differences"""
        # Use a copy of the original graph
        G_extreme = self.G.copy()
        
        # Update with traffic data
        G_extreme = self.data_processor.update_graph_with_traffic(G_extreme, update_percentage=100)
        
        # Apply very different traffic patterns to different road types
        for u, v, k, data in G_extreme.edges(keys=True, data=True):
            highway_type = data.get('highway', '')
            
            # Make sure 'weight' exists
            if 'weight' not in data:
                continue
                
            # Extreme congestion on major roads (motorways/trunks)
            if highway_type in ['motorway', 'trunk']:
                # Increase travel time by 500-700% on major roads
                multiplier = 5.0 + (hash(str(u) + str(v)) % 200) / 100.0  # 5-7x
                data['weight'] *= multiplier
                if 'traffic_speed' in data:
                    data['traffic_speed'] /= multiplier
            
            # High congestion on primary roads
            elif highway_type in ['primary']:
                # Increase travel time by 300-400%
                multiplier = 3.0 + (hash(str(u) + str(v)) % 100) / 100.0  # 3-4x
                data['weight'] *= multiplier
                if 'traffic_speed' in data:
                    data['traffic_speed'] /= multiplier
            
            # Moderate congestion on secondary roads
            elif highway_type in ['secondary']:
                # Increase travel time by 150-200%
                multiplier = 1.5 + (hash(str(u) + str(v)) % 50) / 100.0  # 1.5-2x
                data['weight'] *= multiplier
                if 'traffic_speed' in data:
                    data['traffic_speed'] /= multiplier
            
            # Little change to residential and minor roads - these become the shortcuts
            else:
                # Increase by only 0-30%
                multiplier = 1.0 + (hash(str(u) + str(v)) % 30) / 100.0  # 1-1.3x
                data['weight'] *= multiplier
                if 'traffic_speed' in data:
                    data['traffic_speed'] /= multiplier
        
        # Create random incidents (complete road blocks)
        # Get random edges to simulate incidents
        import random
        random.seed(42)  # For reproducibility
        
        major_edges = []
        for u, v, k, data in G_extreme.edges(keys=True, data=True):
            if 'highway' in data and data['highway'] in ['motorway', 'trunk', 'primary']:
                major_edges.append((u, v, k))
        
        # Block some major roads
        num_to_block = min(len(major_edges), num_incidents)
        incidents = random.sample(major_edges, num_to_block)
        
        # Add severe delays to blocked roads
        for u, v, k in incidents:
            # Either add extreme delay (15-20x) or completely block (infinite weight)
            if random.random() < 0.7:  # 70% chance of extreme delay
                multiplier = 15.0 + (random.random() * 5.0)  # 15-20x slowdown
                G_extreme.edges[u, v, k]['weight'] *= multiplier
                if 'traffic_speed' in G_extreme.edges[u, v, k]:
                    G_extreme.edges[u, v, k]['traffic_speed'] /= multiplier
            else:  # 30% chance of complete block
                G_extreme.edges[u, v, k]['weight'] = float('inf')
                G_extreme.edges[u, v, k]['traffic_speed'] = 0.1  # Almost stopped
        
        return G_extreme
    





    def extreme_traffic_with_forced_blockages(self, num_major_blocks=15, major_road_multiplier=10.0):
        """Create extreme traffic with guaranteed different optimum paths by blocking certain roads"""
        # Use a copy of the original graph
        G_extreme = self.G.copy()
        
        # Apply varying traffic factors based on road type
        print("Applying extreme traffic patterns by road type...")
        for u, v, k, data in G_extreme.edges(keys=True, data=True):
            highway_type = data.get('highway', 'residential')
            
            # Get base weight
            orig_weight = float(data.get('weight', 10.0))
            
            # Different multipliers for different road types
            if highway_type in ['motorway', 'trunk']:
                # Major highways - extremely congested
                multiplier = random.uniform(8.0, 10.0)
            elif highway_type in ['primary']:
                # Primary roads - heavily congested
                multiplier = random.uniform(6.0, 8.0)
            elif highway_type in ['secondary']:
                # Secondary roads - moderately congested
                multiplier = random.uniform(3.0, 5.0)
            elif highway_type in ['tertiary']:
                # Tertiary roads - light congestion
                multiplier = random.uniform(2.0, 3.0)
            else:
                # Residential and other minor roads - minimal congestion
                multiplier = random.uniform(1.0, 1.5)
            
            # Apply multiplier
            G_extreme.edges[u, v, k]['weight'] = orig_weight * multiplier
            
            # Update traffic speed if it exists
            if 'traffic_speed' in data:
                G_extreme.edges[u, v, k]['traffic_speed'] = data['traffic_speed'] / multiplier
        
        # Find edges that connect major roads to minor roads
        # These are critical junctions where the algorithms might decide to take different routes
        critical_junctions = []
        for u in G_extreme.nodes():
            major_neighbors = 0
            minor_neighbors = 0
            
            for v in G_extreme.neighbors(u):
                # Check the highway type of the edge
                is_major = False
                for edge_key in G_extreme[u][v]:
                    highway_type = G_extreme[u][v][edge_key].get('highway', '')
                    if highway_type in ['motorway', 'trunk', 'primary']:
                        is_major = True
                        break
                
                if is_major:
                    major_neighbors += 1
                else:
                    minor_neighbors += 1
            
            # If this node connects major and minor roads, it's a critical junction
            if major_neighbors > 0 and minor_neighbors > 0:
                critical_junctions.append(u)
        
        print(f"Identified {len(critical_junctions)} critical junction nodes where major and minor roads meet")
        
        # Create complete road blockages
        # We'll block major roads near critical junctions to create incentives for different routing
        major_edges_near_junctions = []
        for junction in critical_junctions:
            for u, v, k, data in G_extreme.edges(keys=True, data=True):
                if u == junction or v == junction:
                    if data.get('highway', '') in ['motorway', 'trunk', 'primary']:
                        major_edges_near_junctions.append((u, v, k))
        
        # Block a subset of the major roads near junctions
        num_to_block = min(num_major_blocks, len(major_edges_near_junctions))
        random.seed(42)  # For reproducibility
        blocked_edges = random.sample(major_edges_near_junctions, num_to_block)
        
        print(f"Blocking {num_to_block} major roads near critical junctions...")
        for u, v, k in blocked_edges:
            # Make this road extremely undesirable - effectively blocked
            G_extreme.edges[u, v, k]['weight'] *= major_road_multiplier
            if 'traffic_speed' in G_extreme.edges[u, v, k]:
                G_extreme.edges[u, v, k]['traffic_speed'] /= major_road_multiplier
        
        return G_extreme
