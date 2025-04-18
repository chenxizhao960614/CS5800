import requests

TOMTOM_API_KEY = 'RhNwSSzqbC5B6VAieYAAkzDl0i8K5QAb'


# Helper: Compute delay penalty based on magnitude
def compute_proportional_penalty(base_time, magnitude):
    """
    Returns a penalty time (in seconds) proportional to the base time
    based on delay magnitude.

    Parameters:
        base_time (float): travel or free-flow time for the segment
        magnitude (int): delay severity level (1–4)

    Returns:
        float: penalty time in seconds
    """
    multiplier = {
        1: 0.10,
        2: 0.25,
        3: 0.50,
        4: 1.00
    }.get(magnitude, 0)

    return base_time * multiplier


# Get real-time traffic flow for a lat/lon
def get_flow_data(lat, lon):
    """
    Fetches real-time traffic flow data from TomTom's Flow Segment Data API
    for a given latitude and longitude.

    Parameters:
        lat (float): Latitude of the point.
        lon (float): Longitude of the point.

    Returns:
        dict or None: A dictionary with keys:
            - 'confidence' (float): Confidence in the traffic data (0 to 1).
            - 'currentSpeed' (float): Current traffic speed in km/h.
            - 'freeFlowSpeed' (float): Normal traffic speed in km/h under free-flow conditions.
        Returns None if the API call fails.
    """
    url = 'https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json'
    params = {
        'point': f'{lat},{lon}',
        'key': TOMTOM_API_KEY
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json().get('flowSegmentData', {})
        return {
            'confidence': data.get('confidence'),
            'currentSpeed': data.get('currentSpeed'),
            'freeFlowSpeed': data.get('freeFlowSpeed')
        }
    else:
        print(f"Flow API error: {response.status_code}")
        print(response.text)
        return None


# Get incident data for a bounding box
def get_incidents_with_magnitude(bbox):
    """
    Retrieves current traffic incidents within a specified bounding box from
    the TomTom Incident Details API.

    Parameters:
        bbox (str): A string defining the bounding box in the format:
                    "minLon,minLat,maxLon,maxLat"

    Returns:
        list: A list of incident dictionaries, each containing:
            - 'geometry': Type and coordinates of the incident.
            - 'properties': Includes 'iconCategory' (e.g. 8 for road closure) and
              'magnitudeOfDelay' (int 1–4).
        Returns an empty list if the API call fails.
    """
    url = f"https://api.tomtom.com/traffic/services/5/incidentDetails?key={TOMTOM_API_KEY}&bbox={bbox}&fields=%7Bincidents%7Btype%2Cgeometry%7Btype%2Ccoordinates%7D%2Cproperties%7BiconCategory%2CmagnitudeOfDelay%7D%7D%7D&language=en-GB&timeValidityFilter=present"

    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get("incidents", [])
    else:
        print(f"API Error {response.status_code}")
        print(response.text)
        return []


def calculate_edge_weight(flow, incident_props, length=100):
    """
    Calculates the travel time (edge weight) for a road segment based on
    current traffic flow, incident severity, and road length.

    Parameters:
        flow (dict): Traffic flow data with keys 'currentSpeed', 'freeFlowSpeed', and 'confidence'.
        incident_props (dict): Incident data containing 'iconCategory' and 'magnitudeOfDelay'.
        length (float): Length of the road segment in meters. Defaults to 100.

    Returns:
        float: Estimated travel time (in seconds) for the segment.
               Returns float('inf') if the road is closed (iconCategory == 8).
               Applies proportional penalty if traffic confidence is low or incident severity is high.
    """
    icon_category = incident_props.get("iconCategory")
    magnitude = incident_props.get("magnitudeOfDelay")

    # 1. Road closed (e.g. accident, construction)
    if icon_category == 8:
        return float('inf')

    confidence = 1.0
    current_speed = 0
    free_flow_speed = 0

    if flow:
        confidence = flow.get("confidence", 1.0)
        current_speed = flow.get("currentSpeed", 0)  # in km/h
        free_flow_speed = flow.get("freeFlowSpeed", 0)  # in km/h

    # 2. Use current speed if available and valid
    if current_speed > 0:
        current_speed_mps = current_speed * 1000 / 3600
        base_time = length / current_speed_mps
        if confidence >= 0.7:
            return base_time
        else:
            penalty = compute_proportional_penalty(base_time, magnitude)
            return base_time + penalty

    # 3. Fallback to free flow speed with penalty
    if free_flow_speed > 0:
        free_speed_mps = free_flow_speed * 1000 / 3600
        base_time = length / free_speed_mps
        penalty = compute_proportional_penalty(base_time, magnitude)
        return base_time + penalty

    # 4. Final fallback: static speed estimate (including no flow data)
    base_time = length / 8.33  # 30 km/h = 8.33 m/s
    return base_time + compute_proportional_penalty(base_time, magnitude)
