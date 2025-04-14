import requests

TOMTOM_API_KEY = 'nCxojQDHEzOQcIAPVivZX1joMIgr8nsL'


# === Helper: Compute delay penalty based on magnitude ===
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
        1: 0.10,  # Light traffic
        2: 0.25,  # Moderate
        3: 0.50,  # Heavy
        4: 1.00   # Severe
    }.get(magnitude, 0)

    return base_time * multiplier


# Get real-time traffic flow for a lat/lon
def get_flow_data(lat, lon):
    url = 'https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json'
    params = {
        'point': f'{lat},{lon}',
        'key': TOMTOM_API_KEY
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json().get('flowSegmentData', {})
        return {
            'travelTime': data.get('currentTravelTime'),
            'freeFlowTravelTime': data.get('freeFlowTravelTime'),
            'confidence': data.get('confidence')
        }
    else:
        print(f"Flow API error: {response.status_code}")
        print(response.text)
        return None


# Get incident data for a bounding box
def get_incidents_with_magnitude(bbox):
    url = f"https://api.tomtom.com/traffic/services/5/incidentDetails?key={TOMTOM_API_KEY}&bbox={bbox}&fields=%7Bincidents%7Btype%2Cgeometry%7Btype%2Ccoordinates%7D%2Cproperties%7BiconCategory%2CmagnitudeOfDelay%7D%7D%7D&language=en-GB&timeValidityFilter=present"

    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get("incidents", [])
    else:
        print(f"API Error {response.status_code}")
        print(response.text)
        return []


def calculate_edge_weight(flow, incident_props, length=100):
    icon_category = incident_props.get("iconCategory")
    magnitude = incident_props.get("magnitudeOfDelay")

    # 1. Road closed (e.g. accident, construction)
    if icon_category == 8:
        return float('inf')

    # 2. No flow data — fallback to static
    if not flow:
        return length / 13.9  # Assume 13.9 m/s (~50 km/h)

    travel_time = flow.get("travelTime")
    free_flow_time = flow.get("freeFlowTravelTime", 0)
    confidence = flow.get("confidence", 1.0)

    # 4. Use current travel time if available and valid
    if travel_time and travel_time > 0:
        if confidence >= 0.7:
            return travel_time
        else:
            penalty = compute_proportional_penalty(travel_time, magnitude)
            return travel_time + penalty

    # 5. Fallback to free flow time with delay or penalty
    if free_flow_time and free_flow_time > 0:
        penalty = compute_proportional_penalty(free_flow_time, magnitude)
        return free_flow_time + penalty

    # 6. Final fallback: use static length
    return length / 13.9


if __name__ == "__main__":
    VANCOUVER_BBOX = "-123.2247,49.1985,-123.0235,49.3165"
    print("Testing: Fetching incident data for Vancouver bbox...")
    incidents = get_incidents_with_magnitude(VANCOUVER_BBOX)

    print(f" Total incidents retrieved: {len(incidents)}")

    for i, inc in enumerate(incidents[:5]):
        props = inc.get("properties", {})
        coords = inc.get("geometry", {}).get("coordinates", [])

        print(f"\n Incident {i+1}")
        print(f"  - Icon Category: {props.get('iconCategory')}")
        print(f"  - Magnitude of Delay: {props.get('magnitudeOfDelay')}")
        print(f"  - First coordinate: {coords[0] if coords else 'N/A'}")
