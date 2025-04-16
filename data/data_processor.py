# data/data_processor.py
import networkx as nx
import numpy as np
import random
from datetime import datetime
import os
import json
from shapely.geometry import LineString, Point
from collections import defaultdict
import requests

class DataProcessor:
    def __init__(self, data_loader, api_key=None):
        self.data_loader = data_loader
        self.cache_dir = "data/cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.api_key = api_key or os.getenv('TOMTOM_API_KEY')
        

    def get_tomtom_traffic(self, lat, lon, use_cache=True, cache_expiry=3600):  # Increase cache expiry to 1 hour
        """Query TomTom Traffic Flow API with caching support"""
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
                return cached_data.get('flow_data')
        
        # If API key is missing or we've hit rate limits recently, use simulation
        if not self.api_key or hasattr(self, 'rate_limited') and self.rate_limited:
            return self._simulate_traffic_data(lat, lon)
        
        # If we get here, try the API but be prepared to handle rate limits
        try:
            url = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
            params = {
                'point': f'{lat},{lon}',
                'key': self.api_key
            }
            
            response = requests.get(url, params=params)
            if response.status_code == 200:
                # API request successful
                data = response.json()
                
                # Extract traffic flow data
                flow_data = {
                    'currentSpeed': data['flowSegmentData'].get('currentSpeed', 50),
                    'freeFlowSpeed': data['flowSegmentData'].get('freeFlowSpeed', 50),
                    'currentTravelTime': data['flowSegmentData'].get('currentTravelTime'),
                    'freeFlowTravelTime': data['flowSegmentData'].get('freeFlowTravelTime'),
                    'confidence': data['flowSegmentData'].get('confidence', 1.0),
                    'roadClosure': data['flowSegmentData'].get('roadClosure', False),
                    'timestamp': datetime.now().timestamp()
                }
                
                # Save to cache
                cached_data = {
                    'timestamp': datetime.now().timestamp(),
                    'flow_data': flow_data,
                    'raw_response': data
                }
                
                with open(cache_file, 'w') as f:
                    json.dump(cached_data, f)
                
                return flow_data
            elif response.status_code == 403:
                # Rate limit hit - switch to simulation mode
                print(f"Error {response.status_code}: {response.text}")
                self.rate_limited = True  # Flag to avoid more API calls
                return self._simulate_traffic_data(lat, lon)
            else:
                print(f"Error {response.status_code}: {response.text}")
                return self._simulate_traffic_data(lat, lon)
        except Exception as e:
            print(f"Error fetching TomTom data: {e}")
            return self._simulate_traffic_data(lat, lon)





    
    def _simulate_traffic_data(self, lat, lon):
        """Simulate traffic data when TomTom API is unavailable"""
        # Use deterministic simulation based on coordinates
        # This ensures the same location always gets the same traffic pattern
        seed = int(abs(hash(f"{lat:.5f}_{lon:.5f}")) % 10000)
        random.seed(seed)
        
        # Get time of day to simulate traffic patterns
        hour = datetime.now().hour
        
        # Base values
        base_speed = 50  # km/h
        
        # Modify speed based on time of day (rush hour effect)
        if 7 <= hour <= 9 or 16 <= hour <= 18:  # Rush hours
            # Reduce speed by 30-70%
            speed_factor = random.uniform(0.3, 0.7)
        elif 10 <= hour <= 15:  # Midday
            # Reduce speed by 10-30%
            speed_factor = random.uniform(0.7, 0.9)
        elif 22 <= hour or hour <= 5:  # Night
            # Almost no reduction
            speed_factor = random.uniform(0.9, 1.0)
        else:  # Other times
            # Moderate reduction
            speed_factor = random.uniform(0.6, 0.8)
        
        # Calculate speeds based on factors
        free_flow_speed = base_speed
        current_speed = base_speed * speed_factor
        
        # Calculate travel times for a standard 100m segment
        segment_length = 100  # meters
        free_flow_time = (segment_length / 1000) / (free_flow_speed / 3600)  # seconds
        current_travel_time = (segment_length / 1000) / (current_speed / 3600)  # seconds
        
        # Random chance for road closure (1%)
        road_closure = random.random() < 0.01
        
        # If closed, set speed to near-zero
        if road_closure:
            current_speed = 1
            current_travel_time = free_flow_time * 100  # 100x normal time
        
        # Return simulated data in the same format as the TomTom API
        return {
            'currentSpeed': current_speed,
            'freeFlowSpeed': free_flow_speed,
            'currentTravelTime': current_travel_time,
            'freeFlowTravelTime': free_flow_time,
            'confidence': random.uniform(0.7, 1.0),
            'roadClosure': road_closure,
            'timestamp': datetime.now().timestamp()
        }
    
    def get_tomtom_incidents(self, bbox, use_cache=True, cache_expiry=600):
        """Query TomTom Traffic Incidents API"""
        if not self.api_key:
            return self._simulate_incidents(bbox)
            
        cache_dir = "data/cache/tomtom"
        os.makedirs(cache_dir, exist_ok=True)
        
        # Generate cache filename based on bbox
        cache_file = f"{cache_dir}/incidents_{bbox.replace(',', '_')}.json"
        
        # Check if cache file exists and is recent enough
        if use_cache and os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                
            # Check if cache is still valid
            timestamp = cached_data.get('timestamp', 0)
            if datetime.now().timestamp() - timestamp < cache_expiry:
                print(f"Using cached incident data for {bbox}")
                return cached_data.get('incidents', [])
        
        url = "https://api.tomtom.com/traffic/services/5/incidentDetails"
        params = {
            'key': self.api_key,
            'bbox': bbox,
            'fields': '{incidents{type,geometry,properties}}',
            'language': 'en-US'
        }
        
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                incidents = data.get('incidents', [])
                
                # Save to cache
                cached_data = {
                    'timestamp': datetime.now().timestamp(),
                    'incidents': incidents,
                    'raw_response': data
                }
                
                with open(cache_file, 'w') as f:
                    json.dump(cached_data, f)
                
                return incidents
            else:
                print(f"Error {response.status_code}: {response.text}")
                return self._simulate_incidents(bbox)
        except Exception as e:
            print(f"Error fetching TomTom incidents: {e}")
            return self._simulate_incidents(bbox)
    
    def _simulate_incidents(self, bbox):
        """Simulate traffic incidents when TomTom API is unavailable"""
        # Parse bbox
        try:
            min_lon, min_lat, max_lon, max_lat = map(float, bbox.split(','))
        except:
            return []
        
        # Use deterministic simulation based on bbox
        seed = int(abs(hash(bbox)) % 10000)
        random.seed(seed)
        
        # Determine number of incidents based on area size
        area = (max_lon - min_lon) * (max_lat - min_lat)
        num_incidents = max(0, min(10, int(area * 10000)))
        
        incidents = []
        for i in range(num_incidents):
            # Generate random location within bbox
            lat = min_lat + random.random() * (max_lat - min_lat)
            lon = min_lon + random.random() * (max_lon - min_lon)
            
            # Random incident types
            incident_types = ['ACCIDENT', 'CONGESTION', 'CONSTRUCTION', 'LANE_RESTRICTION']
            incident_type = random.choice(incident_types)
            
            # Generate incident
            incident = {
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': [lon, lat]
                },
                'properties': {
                    'id': f'sim-{i}-{seed}',
                    'iconCategory': incident_type,
                    'magnitudeOfDelay': random.randint(1, 8),
                    'startTime': datetime.now().timestamp() - random.randint(0, 3600),
                    'endTime': datetime.now().timestamp() + random.randint(3600, 7200),
                    'length': random.randint(50, 500),
                    'delay': random.randint(60, 600),
                    'description': f'Simulated {incident_type.lower()}'
                }
            }
            
            incidents.append(incident)
        
        return incidents
    
    def update_graph_with_traffic(self, G, bbox=None, update_percentage=70, include_incidents=True):
        """Update graph edge weights with traffic data
        
        Args:
            G: NetworkX graph
            bbox: Bounding box string "min_lon,min_lat,max_lon,max_lat" to limit processing
            update_percentage: Percentage of edges to update (0-100)
            include_incidents: Whether to include traffic incidents
            
        Returns:
            Updated graph with traffic-aware edge weights
        """
        print("Updating graph with traffic data...")
        
        # Process a subset of edges based on update_percentage
        all_edges = list(G.edges(keys=True, data=True))
        if update_percentage < 100:
            sample_size = int(len(all_edges) * (update_percentage / 100))
            # Use fixed seed for reproducibility
            random.seed(42)
            edges_to_update = random.sample(all_edges, sample_size)
        else:
            edges_to_update = all_edges
        
        # Get incidents if requested
        incidents = []
        if include_incidents and bbox:
            incidents = self.get_tomtom_incidents(bbox)
            print(f"Found {len(incidents)} incidents in the area")
        
        # Update edges with traffic data
        updated_count = 0
        incident_affected = 0
        
        for u, v, k, data in edges_to_update:
            try:
                # Extract coordinates for the edge
                lat, lon = self._get_edge_coordinates(G, u, v, data)
                
                if lat is None or lon is None:
                    continue
                
                # Get traffic data for this location
                traffic_data = self.get_tomtom_traffic(lat, lon)
                
                if not traffic_data:
                    continue
                
                # Store traffic speed information
                G.edges[u, v, k]['traffic_speed'] = traffic_data.get('currentSpeed', 50)
                
                # Get length of the edge
                length = data.get('length', 0)
                if length == 0:
                    # Calculate length if not available
                    length = self._calculate_edge_length(G, u, v)
                    G.edges[u, v, k]['length'] = length
                
                # Check for incidents at this location
                incident = self._find_nearby_incident(lat, lon, incidents)
                
                # Calculate edge weight (travel time in seconds)
                if incident:
                    # Apply incident delay
                    incident_affected += 1
                    G.edges[u, v, k]['weight'] = self._calculate_incident_weight(traffic_data, incident, length)
                    G.edges[u, v, k]['incident'] = True
                    G.edges[u, v, k]['incident_type'] = incident.get('properties', {}).get('iconCategory', 'UNKNOWN')
                else:
                    # Calculate normal travel time based on current speed
                    current_speed = traffic_data.get('currentSpeed', 50)  # km/h
                    
                    if current_speed < 1:  # Avoid division by zero
                        current_speed = 1
                    
                    # Convert km/h to m/s and calculate time
                    # length in meters, speed in km/h, result in seconds
                    travel_time = (length / 1000) / (current_speed / 3600)
                    G.edges[u, v, k]['weight'] = travel_time
                
                updated_count += 1
                
            except Exception as e:
                # Skip edge on error
                print(f"Error updating edge ({u}, {v}, {k}): {e}")
        
        print(f"Updated {updated_count} edges with traffic data")
        print(f"Found {incident_affected} edges affected by incidents")
        
        # Make sure all edges have weights
        for u, v, k, data in G.edges(keys=True, data=True):
            if 'weight' not in data or data['weight'] <= 0:
                # Default weight based on length and default speed (50 km/h)
                length = data.get('length', 100)  # default 100m
                G.edges[u, v, k]['weight'] = (length / 1000) / (50 / 3600)  # seconds
        
        return G
    
    def _get_edge_coordinates(self, G, u, v, data):
        """Extract coordinates from an edge"""
        lat, lon = None, None
        
        if 'geometry' in data:
            # Use the midpoint of the geometry if available
            geom = data['geometry']
            if hasattr(geom, 'interpolate'):
                # It's a shapely geometry
                midpoint = geom.interpolate(0.5, normalized=True)
                lat, lon = midpoint.y, midpoint.x
        
        if lat is None or lon is None:
            # Fall back to midpoint of nodes
            if 'y' in G.nodes[u] and 'x' in G.nodes[u] and 'y' in G.nodes[v] and 'x' in G.nodes[v]:
                lat = (G.nodes[u]['y'] + G.nodes[v]['y']) / 2
                lon = (G.nodes[u]['x'] + G.nodes[v]['x']) / 2
        
        return lat, lon
    
    def _calculate_edge_length(self, G, u, v):
        """Calculate length of an edge in meters"""
        if 'y' in G.nodes[u] and 'x' in G.nodes[u] and 'y' in G.nodes[v] and 'x' in G.nodes[v]:
            # Use haversine formula for distance calculation
            from math import radians, sin, cos, sqrt, atan2
            
            lat1 = radians(G.nodes[u]['y'])
            lon1 = radians(G.nodes[u]['x'])
            lat2 = radians(G.nodes[v]['y'])
            lon2 = radians(G.nodes[v]['x'])
            
            # Haversine formula
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * atan2(sqrt(a), sqrt(1-a))
            r = 6371000  # Earth radius in meters
            
            return c * r
        else:
            return 100  # Default 100 meters
    
    def _find_nearby_incident(self, lat, lon, incidents, threshold_meters=50):
        """Find if there's an incident near the given coordinates"""
        if not incidents:
            return None
            
        for incident in incidents:
            try:
                # Get incident geometry
                geom = incident.get('geometry', {})
                geom_type = geom.get('type', '')
                coords = geom.get('coordinates', [])
                
                if geom_type == 'Point' and coords:
                    # Calculate distance to point
                    inc_lon, inc_lat = coords
                    dist = self._haversine_distance(lat, lon, inc_lat, inc_lon)
                    
                    if dist <= threshold_meters:
                        return incident
                        
                elif geom_type == 'LineString' and coords:
                    # Find minimum distance to line
                    min_dist = float('inf')
                    for i in range(len(coords) - 1):
                        # Calculate perpendicular distance to line segment
                        p1_lon, p1_lat = coords[i]
                        p2_lon, p2_lat = coords[i+1]
                        
                        dist = self._distance_to_line_segment(lat, lon, p1_lat, p1_lon, p2_lat, p2_lon)
                        min_dist = min(min_dist, dist)
                    
                    if min_dist <= threshold_meters:
                        return incident
            except:
                continue
                
        return None
    
    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate haversine distance between two points in meters"""
        from math import radians, sin, cos, sqrt, atan2
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        r = 6371000  # Earth radius in meters
        
        return c * r
    
    def _distance_to_line_segment(self, lat, lon, lat1, lon1, lat2, lon2):
        """Calculate distance from point to line segment in meters"""
        # Convert to cartesian for simplicity
        from math import radians, sin, cos, sqrt, atan2
        
        # Convert to radians and calculate cartesian coordinates
        lat, lon, lat1, lon1, lat2, lon2 = map(radians, [lat, lon, lat1, lon1, lat2, lon2])
        
        # Project point onto line segment
        # This is an approximation for small distances
        x = lon
        y = lat
        x1 = lon1
        y1 = lat1
        x2 = lon2
        y2 = lat2
        
        # Calculate projection
        dx = x2 - x1
        dy = y2 - y1
        
        # Handle zero-length segments
        if dx == 0 and dy == 0:
            return self._haversine_distance(lat, lon, lat1, lon1)
            
        # Calculate projection parameter
        t = ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy)
        
        if t < 0:
            # Closest to first endpoint
            dist = self._haversine_distance(lat, lon, lat1, lon1)
        elif t > 1:
            # Closest to second endpoint
            dist = self._haversine_distance(lat, lon, lat2, lon2)
        else:
            # Closest to perpendicular projection
            px = x1 + t * dx
            py = y1 + t * dy
            dist = self._haversine_distance(lat, lon, py, px)
            
        return dist
    
    def _calculate_incident_weight(self, traffic_data, incident, length):
        """Calculate edge weight considering incident delay"""
        props = incident.get('properties', {})
        
        # Extract incident properties
        delay = props.get('delay', 0)  # Delay in seconds
        magnitude = props.get('magnitudeOfDelay', 0)  # 1-10 scale
        incident_type = props.get('iconCategory', 'UNKNOWN')
        
        # Base travel time calculation
        current_speed = traffic_data.get('currentSpeed', 50)  # km/h
        
        if current_speed < 1:  # Avoid division by zero
            current_speed = 1
        
        # Base travel time in seconds
        base_time = (length / 1000) / (current_speed / 3600)
        
        # Apply delay based on incident type and magnitude
        if incident_type == 'ACCIDENT':
            # Accidents cause severe delays
            factor = 2.0 + (magnitude / 2)
        elif incident_type == 'CONGESTION':
            # Congestion already reflected in currentSpeed, but add a bit more
            factor = 1.2 + (magnitude / 10)
        elif incident_type == 'CONSTRUCTION':
            # Construction typically causes moderate delays
            factor = 1.5 + (magnitude / 5)
        elif incident_type == 'LANE_RESTRICTION':
            # Lane restrictions cause mild delays
            factor = 1.3 + (magnitude / 10)
        else:
            # Unknown incident type
            factor = 1.5
        
        # Calculate final weight
        # Either apply reported delay or calculate based on factor
        if delay > 0:
            # Use reported delay
            weight = base_time + delay
        else:
            # Use calculated delay
            weight = base_time * factor
        
        # For major incidents with high magnitude, consider blocking the road
        if magnitude >= 8 and (incident_type == 'ACCIDENT' or incident_type == 'CONSTRUCTION'):
            # Very severe - almost impassable
            weight = float('inf')
            
        return weight