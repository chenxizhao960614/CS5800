# data/data_processor.py
import networkx as nx
import numpy as np
import random
from datetime import datetime, timedelta
import os
import json
from shapely.geometry import LineString, Point
from collections import defaultdict
import osmnx as ox

class DataProcessor:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.cache_dir = "data/cache"
        os.makedirs(self.cache_dir, exist_ok=True)


    def update_graph_with_traffic(self, G, update_percentage=100, fallback_speed=50):
        """Update graph edge weights with traffic data, using cached data or simulation"""
        print("Updating graph with traffic data...")
        
        # Load traffic cache
        cache_file = os.path.join(self.cache_dir, "traffic_cache.json")
        traffic_cache = self._load_traffic_cache(cache_file)
        
        # Get edges to update
        edges_to_update = list(G.edges(keys=True, data=True))
        if update_percentage < 100:
            sample_size = int(len(edges_to_update) * (update_percentage / 100))
            edges_to_update = random.sample(edges_to_update, sample_size)
        
        cached_count = 0
        simulated_count = 0
        
        # Get current hour for time-based simulation
        current_hour = datetime.now().hour
        
        # Dictionary to store node traffic speeds
        node_speeds = defaultdict(list)
        
        for u, v, k, data in edges_to_update:
            try:
                # Extract coordinates in a safe way
                lat, lon = None, None
                
                if 'geometry' in data:
                    # Handle geometry data safely
                    geom = data['geometry']
                    
                    # Try different methods to extract coordinates from geometry
                    if hasattr(geom, 'xy'):
                        # If it's a shapely LineString
                        try:
                            xs, ys = geom.xy
                            if len(xs) > 0 and len(ys) > 0:
                                # Use midpoint of the line
                                mid_idx = len(xs) // 2
                                lat = float(ys[mid_idx])
                                lon = float(xs[mid_idx])
                        except (AttributeError, IndexError, TypeError):
                            pass
                    
                    # If it's a dictionary with type LineString
                    elif isinstance(geom, dict) and geom.get('type') == 'LineString' and 'coordinates' in geom:
                        try:
                            coords = geom['coordinates']
                            if len(coords) > 0:
                                mid_idx = len(coords) // 2
                                lon, lat = coords[mid_idx]
                        except (IndexError, TypeError, ValueError):
                            pass
                    
                    # If it's a list of coordinates
                    elif isinstance(geom, list) and len(geom) > 0:
                        try:
                            mid_idx = len(geom) // 2
                            if isinstance(geom[mid_idx], (list, tuple)) and len(geom[mid_idx]) >= 2:
                                lon, lat = geom[mid_idx]
                        except (IndexError, TypeError, ValueError):
                            pass
                
                # If geometry processing failed, use node coordinates
                if lat is None or lon is None:
                    try:
                        if 'y' in G.nodes[u] and 'x' in G.nodes[u] and 'y' in G.nodes[v] and 'x' in G.nodes[v]:
                            lat = (G.nodes[u]['y'] + G.nodes[v]['y']) / 2
                            lon = (G.nodes[u]['x'] + G.nodes[v]['x']) / 2
                        else:
                            # Default fallback if no coordinates are available
                            lat, lon = 0, 0
                    except Exception:
                        # Last resort - use arbitrary coordinates
                        lat, lon = 0, 0
                
                # Create a unique key for caching
                cache_key = f"{lat:.5f}_{lon:.5f}"
                
                # Check if we have cached data
                if cache_key in traffic_cache:
                    cached_data = traffic_cache[cache_key]
                    travel_time = cached_data.get('travel_time')
                    if travel_time is not None:
                        travel_time = float(travel_time)
                        # Store traffic speed if available from cache
                        if 'speed' in cached_data:
                            G.edges[u, v, k]['traffic_speed'] = float(cached_data['speed'])
                        cached_count += 1
                    else:
                        travel_time = None
                else:
                    travel_time = None
                
                # If no cached data, simulate traffic
                if travel_time is None:
                    # Simulate traffic based on road type and time of day
                    length = data.get('length', 0)
                    if length > 0:
                        # Base speed depends on road type
                        highway_type = data.get('highway', 'residential')
                        base_speed = self._get_base_speed(highway_type)
                        
                        # Apply time-of-day factor
                        time_factor = self._get_time_factor(current_hour)
                        
                        # Add some randomness (±20%)
                        random_factor = random.uniform(0.8, 1.2)
                        
                        # Calculate final speed (km/h)
                        actual_speed = base_speed * time_factor * random_factor
                        
                        # Calculate travel time (seconds)
                        travel_time = (length / 1000) / (actual_speed / 3600)
                        
                        # Store the speed for this edge
                        G.edges[u, v, k]['traffic_speed'] = float(actual_speed)
                        
                        # Add speed to both nodes
                        node_speeds[u].append(actual_speed)
                        node_speeds[v].append(actual_speed)
                        
                        # Update cache with simulated data
                        traffic_cache[cache_key] = {
                            'timestamp': datetime.now().timestamp(),
                            'travel_time': float(travel_time),
                            'speed': float(actual_speed),
                            'simulated': True,
                            'lat': float(lat),
                            'lon': float(lon)
                        }
                        simulated_count += 1
                    else:
                        # Default travel time if no length (1 minute)
                        travel_time = 60.0
                        G.edges[u, v, k]['traffic_speed'] = 30.0  # Default 30 km/h
                        simulated_count += 1
                
                # IMPORTANT: Update edge weight with travel time

                if travel_time and travel_time > 0:
                    G.edges[u, v, k]['weight'] = float(travel_time)
                    # Make sure we have at least 6 decimal places of precision
                    if G.edges[u, v, k]['weight'] < 0.000001:
                        G.edges[u, v, k]['weight'] = 0.000001  # Minimum non-zero weight
                else:
                    # Calculate fallback weight based on length and speed
                    length = data.get('length', 100)  # Default 100m if no length
                    fallback_time = float(length / (fallback_speed * 1000 / 3600))
                    G.edges[u, v, k]['weight'] = max(fallback_time, 0.000001)  # Minimum weight

            
            except Exception as e:
                # Detailed error message with geometry type information
                if 'geometry' in data:
                    geom_type = type(data['geometry'])
                    print(f"Warning: Error processing edge ({u}, {v}, {k}): {str(e)} - Geometry type: {geom_type}")
                else:
                    print(f"Warning: Error processing edge ({u}, {v}, {k}): {str(e)}")
                
                # Even on error, set a reasonable default weight
                try:
                    length = data.get('length', 100)  # Default 100m if no length
                    default_travel_time = float(length / (fallback_speed * 1000 / 3600))
                    G.edges[u, v, k]['weight'] = max(default_travel_time, 1.0)  # Minimum 1 second
                    G.edges[u, v, k]['traffic_speed'] = float(fallback_speed)
                except Exception as inner_e:
                    # Last resort fallback
                    G.edges[u, v, k]['weight'] = 30.0  # Default 30 seconds
                    G.edges[u, v, k]['traffic_speed'] = 30.0  # Default 30 km/h
                    print(f"  Inner error setting default weight: {str(inner_e)}")
        
        # Update node traffic speeds (average of connected edges)
        for node, speeds in node_speeds.items():
            if speeds:
                G.nodes[node]['traffic_speed'] = sum(speeds) / len(speeds)
        
        # Save updated cache
        self._save_traffic_cache(traffic_cache, cache_file)
        
        print(f"Used {cached_count} cached traffic data points")
        print(f"Simulated traffic data for {simulated_count} road segments")
        
        # Verify weights were properly set
        missing_weights = 0
        for u, v, k, data in G.edges(keys=True, data=True):
            if 'weight' not in data or data['weight'] <= 0:
                # Set a minimum weight
                length = data.get('length', 100)
                default_time = float(length / (fallback_speed * 1000 / 3600))
                G.edges[u, v, k]['weight'] = max(default_time, 1.0)  # Minimum 1 second
                missing_weights += 1
        
        if missing_weights > 0:
            print(f"Fixed {missing_weights} edges with missing or invalid weights")
        
        return G


    def _get_base_speed(self, highway_type):
        """Get base speed based on road type"""
        speed_map = {
            'motorway': 100,
            'trunk': 80,
            'primary': 60,
            'secondary': 50,
            'tertiary': 40,
            'residential': 30,
            'service': 20
        }
        return speed_map.get(highway_type, 30)
    
    def _get_time_factor(self, hour):
        """Get traffic factor based on time of day"""
        # Rush hours: 7-9 AM and 4-6 PM
        if hour in [7, 8, 9] or hour in [16, 17, 18]:
            return random.uniform(0.4, 0.7)  # Heavy traffic
        # Mid-day: 10 AM - 3 PM
        elif hour >= 10 and hour <= 15:
            return random.uniform(0.7, 0.9)  # Medium traffic
        # Night time: 10 PM - 5 AM
        elif hour >= 22 or hour <= 5:
            return random.uniform(0.9, 1.1)  # Light traffic
        # Other times
        else:
            return random.uniform(0.6, 0.8)  # Normal traffic
    
    def _load_traffic_cache(self, cache_file):
        """Load traffic data cache from file"""
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_traffic_cache(self, cache_data, cache_file):
        """Save traffic data cache to file"""
        try:
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
        except Exception as e:
            print(f"Warning: Could not save traffic cache: {e}")

    def process_intersection_data(self, G, intersection_file):
        """Process intersection movement counts from Vancouver Open Data"""
        try:
            # Load intersection data
            import pandas as pd
            imc_data = pd.read_csv(intersection_file)
            
            # Process and match to nodes in the graph
            print("Processing intersection data...")
            
            # For each intersection, find the closest node and add delay
            for idx, row in imc_data.iterrows():
                # This would need to be adapted based on your actual data format
                lat, lon = row.get('Latitude'), row.get('Longitude')
                volume = row.get('Volume', 0)
                
                # Find nearest node in graph
                nearest_node = ox.distance.nearest_nodes(G, lon, lat)
                
                # Add delay attribute based on volume
                if volume > 1000:
                    G.nodes[nearest_node]['delay'] = 10  # 10 seconds delay for busy intersections
                elif volume > 500:
                    G.nodes[nearest_node]['delay'] = 5   # 5 seconds delay for medium traffic
                else:
                    G.nodes[nearest_node]['delay'] = 2   # 2 seconds delay for light traffic
                    
            print(f"Processed {len(imc_data)} intersections")
            return G
        except Exception as e:
            print(f"Error processing intersection data: {e}")
            return G