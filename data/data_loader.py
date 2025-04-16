# data/data_loader.py
import osmnx as ox
import networkx as nx
import pandas as pd
import requests
from shapely.geometry import LineString, Point
import time
import os
from dotenv import load_dotenv

class DataLoader:
    def __init__(self, api_key=None):
        # Load environment variables
        load_dotenv()
        # Use provided API key or fall back to environment variable
        self.tomtom_api_key = api_key or os.getenv('TOMTOM_API_KEY')
        
    def load_osm_graph(self, place_name="Vancouver, British Columbia, Canada", network_type='drive'):
        """Load road network from OpenStreetMap"""
        print(f"Loading road network for {place_name}...")
        G = ox.graph_from_place(place_name, network_type=network_type)
        print(f"Loaded graph with {len(G.nodes)} nodes and {len(G.edges)} edges")
        return G
    
    def save_graph(self, G, filename="vancouver_road_network.graphml"):
        """Save the graph to a file"""
        ox.save_graphml(G, filename)
        print(f"Graph saved to {filename}")
    
    def save_graph_json_(self, G, filename="vancouver_road_network.json"):
        """Save the graph to a JSON file"""
        import json
        
        # Convert NetworkX graph to dictionary
        data = nx.node_link_data(G)
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(data, f)
        
        print(f"Graph saved to {filename}")

    def save_graph_json(self, G, filename="vancouver_road_network.json"):
        """Save the graph to a JSON file"""
        import json
        import numpy as np
        from shapely.geometry import LineString
        
        # Create a custom JSON encoder that can handle LineString objects
        class CustomEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, LineString):
                    # Convert LineString to a list of coordinates
                    return {'type': 'LineString', 'coordinates': list(obj.coords)}
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)
        
        # Convert NetworkX graph to dictionary
        data = nx.node_link_data(G)
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(data, f, cls=CustomEncoder)
        
        print(f"Graph saved to {filename}")



    def load_graph_json(self, filename="vancouver_road_network.json"):
        """Load graph from JSON file"""
        import json
        from shapely.geometry import LineString
        
        def object_hook(d):
            if isinstance(d, dict) and d.get('type') == 'LineString' and 'coordinates' in d:
                return LineString(d['coordinates'])
            return d
        
        with open(filename, 'r') as f:
            data = json.load(f, object_hook=object_hook)
        
        G = nx.node_link_graph(data)
        
        # Add initial weights based on length and default speed
        edge_count = 0
        weight_added = 0
        
        for u, v, k, data in G.edges(keys=True, data=True):
            edge_count += 1
            if 'weight' not in data and 'length' in data:
                # Convert length (meters) to travel time (seconds) using 50 km/h as default
                # 50 km/h = 13.89 m/s, so time = length / 13.89
                try:
                    G.edges[u, v, k]['weight'] = float(data['length'] / 13.89)
                    weight_added += 1
                except (ValueError, TypeError):
                    # Fallback if length is not a valid number
                    G.edges[u, v, k]['weight'] = 30.0  # Default 30 seconds
                    weight_added += 1
        
        print(f"Loaded graph with {len(G.nodes)} nodes and {len(G.edges)} edges")
        print(f"Added weights to {weight_added}/{edge_count} edges")
        return G









    def load_graph(self, filename="vancouver_road_network.graphml"):
        """Load graph from file"""
        G = ox.load_graphml(filename)
        print(f"Loaded graph with {len(G.nodes)} nodes and {len(G.edges)} edges")
        return G
    
    def get_tomtom_traffic_(self, lat, lon):
        """Query TomTom Traffic Flow API for a specific location"""
        if not self.tomtom_api_key:
            # Return None instead of raising an error
            return None
            
        url = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
        params = {
            'point': f'{lat},{lon}',
            'key': self.tomtom_api_key
        }
        
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                try:
                    # Extract current travel time in seconds
                    return data['flowSegmentData'].get('currentTravelTime')
                except KeyError:
                    return None
            else:
                print(f"Error {response.status_code}: {response.text}")
                return None
        except Exception as e:
            print(f"Error fetching TomTom data: {e}")
            return None

    def get_tomtom_traffic(self, lat, lon, use_cache=True, cache_expiry=600):
        """Query TomTom Traffic Flow API with caching support
        
        Args:
            lat: Latitude
            lon: Longitude
            use_cache: Whether to use cached data (default: True)
            cache_expiry: Cache expiry time in seconds (default: 10 minutes)
        """
        import os
        import json
        from datetime import datetime, timedelta
        
        # Create cache directory if it doesn't exist
        cache_dir = "data/cache/tomtom"
        os.makedirs(cache_dir, exist_ok=True)
        
        # Generate cache filename based on coordinates
        cache_file = f"{cache_dir}/traffic_{lat:.5f}_{lon:.5f}.json"
        
        # Check if cache file exists and is recent enough
        if use_cache and os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                
            # Check if cache is still valid
            timestamp = cached_data.get('timestamp', 0)
            if datetime.now().timestamp() - timestamp < cache_expiry:
                print(f"Using cached traffic data for {lat}, {lon}")
                return cached_data.get('travelTime')
        
        # If no valid cache, query the API
        if not self.tomtom_api_key:
            return None
            
        url = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
        params = {
            'point': f'{lat},{lon}',
            'key': self.tomtom_api_key
        }
        
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                try:
                    # Extract current travel time in seconds
                    travel_time = data['flowSegmentData'].get('currentTravelTime')
                    
                    # If successful, save to cache
                    if travel_time is not None:
                        cached_data = {
                            'timestamp': datetime.now().timestamp(),
                            'travelTime': travel_time,
                            'raw_response': data  # Store full response for other potential uses
                        }
                        
                        with open(cache_file, 'w') as f:
                            json.dump(cached_data, f)
                    
                    return travel_time
                except KeyError:
                    return None
            else:
                print(f"Error {response.status_code}: {response.text}")
                return None
        except Exception as e:
            print(f"Error fetching TomTom data: {e}")
            return None

    # def update_graph_with_traffic(self, G, update_percentage=100, fallback_speed=50):


    def get_tomtom_incidents(self, lat, lon, radius=1000):
        """Query TomTom Traffic Incidents API"""
        if not self.tomtom_api_key:
            # Return None instead of raising an error
            return None
            
        url = "https://api.tomtom.com/traffic/services/5/incidentDetails"
        params = {
            'key': self.tomtom_api_key,
            'point': f'{lat},{lon}',
            'radius': radius,
            'fields': '{incidents{type,geometry,properties}}',
            'language': 'en-US'
        }
        
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error {response.status_code}: {response.text}")
                return None
        except Exception as e:
            print(f"Error fetching TomTom incidents: {e}")
            return None


   
        """Query TomTom Traffic Flow API with caching support
        
        Args:
            lat: Latitude
            lon: Longitude
            use_cache: Whether to use cached data (default: True)
            cache_expiry: Cache expiry time in seconds (default: 10 minutes)
        """
        import os
        import json
        from datetime import datetime, timedelta
        
        # Create cache directory if it doesn't exist
        cache_dir = "data/cache/tomtom"
        os.makedirs(cache_dir, exist_ok=True)
        
        # Generate cache filename based on coordinates
        cache_file = f"{cache_dir}/traffic_{lat:.5f}_{lon:.5f}.json"
        
        # Check if cache file exists and is recent enough
        if use_cache and os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                
            # Check if cache is still valid
            timestamp = cached_data.get('timestamp', 0)
            if datetime.now().timestamp() - timestamp < cache_expiry:
                print(f"Using cached traffic data for {lat}, {lon}")
                return cached_data.get('travelTime')
        
        # If no valid cache, query the API
        if not self.tomtom_api_key:
            return None
        
        # [Your existing API query code]
        
        # If successful, save to cache
        if travel_time is not None:
            cached_data = {
                'timestamp': datetime.now().timestamp(),
                'travelTime': travel_time,
                'raw_response': data  # Store full response for other potential uses
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cached_data, f)
        
        return travel_time