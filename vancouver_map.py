import osmnx as ox
import networkx as nx
from shapely.geometry import Point, LineString, box
from tomtom_api import get_flow_data, get_incidents_with_magnitude, calculate_edge_weight
from math import radians, sin, cos, sqrt, atan2, degrees
import matplotlib.pyplot as plt

TOMTOM_API_KEY = 'nCxojQDHEzOQcIAPVivZX1joMIgr8nsL'
GRAPH_PATH = "vancouver.graphml"

# Helper Function
def find_nearby_incident(lat, lon, incidents, threshold_m=25):
    pt = Point(lon, lat)
    for inc in incidents:
        geom = inc.get("geometry", {})
        coords = geom.get("coordinates", [])
        if coords:
            if geom.get("type") == "Point":
                line = Point(coords)
            else:
                line = LineString(coords)
            if pt.distance(line) * 111139 < threshold_m:
                return inc.get("properties", {})
    return {}

# Integration with OSMnx Graph
def load_graph():
    try:
        G = ox.load_graphml(GRAPH_PATH)
    except:
        G = ox.graph_from_place("Vancouver, British Columbia, Canada", network_type='drive')
        ox.save_graphml(G, GRAPH_PATH)

    for u, v, k, data in G.edges(keys=True, data=True):
        if "geometry" in data:
            midpoint = data["geometry"].interpolate(0.5, normalized=True)
            data["lat"] = midpoint.y
            data["lon"] = midpoint.x
        else:
            data["lat"] = (G.nodes[u]["y"] + G.nodes[v]["y"]) / 2
            data["lon"] = (G.nodes[u]["x"] + G.nodes[v]["x"]) / 2
    return G


def update_edge_weights_with_bbox(G, start_lat, start_lon, end_lat, end_lon, buffer_km=1.5):
    """
    Update edge weights in the subgraph defined by the bounding box of start & end locations.

    Parameters:
        G          : Full OSMnx road network graph
        start_lat  : Latitude of start point
        start_lon  : Longitude of start point
        end_lat    : Latitude of end point
        end_lon    : Longitude of end point
        buffer_km  : Padding in kilometers to include roads around the route corridor
    """
    # Step 1: Expand a bounding box around start and end points
    min_lat = min(start_lat, end_lat) - buffer_km / 111  # approx conversion km to deg
    max_lat = max(start_lat, end_lat) + buffer_km / 111
    min_lon = min(start_lon, end_lon) - buffer_km / (111 * cos(radians((start_lat + end_lat) / 2)))
    max_lon = max(start_lon, end_lon) + buffer_km / (111 * cos(radians((start_lat + end_lat) / 2)))

    # Step 2: Define area of interest (AOI)
    bbox = box(min_lon, min_lat, max_lon, max_lat)

    # Step 3: Fetch incidents only once for this bounding box
    bbox_str = f"{min_lon},{min_lat},{max_lon},{max_lat}"
    incidents = get_incidents_with_magnitude(bbox_str)

    flow_cache = {}

    blocked = 0
    for u, v, k, data in G.edges(keys=True, data=True):
        lat, lon = data.get("lat"), data.get("lon")

        # Skip if edge is outside the box
        if not bbox.contains(Point(lon, lat)):
            continue

        key = (round(lat, 4), round(lon, 4))
        flow = flow_cache.get(key)
        if flow is None:
            flow = get_flow_data(lat, lon)
            flow_cache[key] = flow

        length = data.get("length")
        if not length:
            y1, x1 = G.nodes[u]["y"], G.nodes[u]["x"]
            y2, x2 = G.nodes[v]["y"], G.nodes[v]["x"]
            length = haversine_distance(y1, x1, y2, x2)
            data["length"] = length

        incident_props = find_nearby_incident(lat, lon, incidents)
        weight = calculate_edge_weight(flow, incident_props, length)
        data["weight"] = weight

        if weight == float('inf'):
            blocked += 1

    print(f"Total blocked edges: {blocked}")


# === Routing ===
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * sqrt(a) * 1000


def euclidean_heuristic(u, v, G):
    x1, y1 = G.nodes[u]['x'], G.nodes[u]['y']
    x2, y2 = G.nodes[v]['x'], G.nodes[v]['y']
    return sqrt((x1 - x2)**2 + (y1 - y2)**2)


def compute_route(G, start_lat, start_lon, end_lat, end_lon, threshold=1000):
    """
    Compute the shortest path between two coordinates on the graph using
    Dijkstra (for short range) or A* (for long range).

    Parameters:
        G         : The OSMnx graph
        start_lat : Latitude of the start point
        start_lon : Longitude of the start point
        end_lat   : Latitude of the end point
        end_lon   : Longitude of the end point
        threshold : Distance threshold (in meters) to decide between Dijkstra and A*

    Returns:
        path        : List of node IDs forming the path
        start_node  : Nearest graph node to start point
        end_node    : Nearest graph node to end point
    """
    # Find nearest nodes on the graph
    start_node = ox.nearest_nodes(G, X=start_lon, Y=start_lat)
    end_node = ox.nearest_nodes(G, X=end_lon, Y=end_lat)

    # Compute straight-line distance
    dist = haversine_distance(start_lat, start_lon, end_lat, end_lon)
    print(f"Straight-line distance: {int(dist)} meters")

    try:
        # Choose algorithm based on threshold
        if dist < threshold:
            print("Using Dijkstra algorithm (short range)")
            path = nx.shortest_path(G, source=start_node, target=end_node, weight="weight")
        else:
            print("Using A* algorithm (long range)")
            path = nx.astar_path(
                G,
                source=start_node,
                target=end_node,
                heuristic=lambda u, v: euclidean_heuristic(u, v, G),
                weight="weight"
            )
        return path, start_node, end_node

    except nx.NetworkXNoPath:
        print("No path found between the selected nodes.")
        return None, start_node, end_node


def compute_directions(G, path):
    print("\n Turn-by-Turn Directions:")
    i = 0
    while i < len(path) - 2:
        u, v, w = path[i], path[i+1], path[i+2]
        name = G.edges[u, v, 0].get("name", "[Unnamed Road]")
        length = G.edges[u, v, 0].get("length", 0)

        # compute turn direction
        vec1 = (G.nodes[v]['x'] - G.nodes[u]['x'], G.nodes[v]['y'] - G.nodes[u]['y'])
        vec2 = (G.nodes[w]['x'] - G.nodes[v]['x'], G.nodes[w]['y'] - G.nodes[v]['y'])
        dot = vec1[0]*vec2[0] + vec1[1]*vec2[1]
        det = vec1[0]*vec2[1] - vec1[1]*vec2[0]
        angle = atan2(det, dot)
        degrees_turn = degrees(angle)

        if degrees_turn > 30:
            turn = "Turn left"
        elif degrees_turn < -30:
            turn = "Turn right"
        else:
            turn = "Continue straight"

        # group same name + direction
        total_length = length
        j = i + 1
        while j < len(path) - 1:
            next_u, next_v = path[j], path[j+1]
            next_name = G.edges[next_u, next_v, 0].get("name", "[Unnamed Road]")
            next_length = G.edges[next_u, next_v, 0].get("length", 0)
            if next_name == name:
                total_length += next_length
                j += 1
            else:
                break

        print(f" - {turn} on {name}, continue for {int(total_length)} meters")
        i = j


if __name__ == "__main__":
    G = load_graph()

    # Get user input
    start_lat = float(input("Start latitude: "))
    start_lon = float(input("Start longitude: "))
    end_lat = float(input("End latitude: "))
    end_lon = float(input("End longitude: "))

    print("Updating edge weights with TomTom traffic data...")
    update_edge_weights_with_bbox(G, start_lat, start_lon, end_lat, end_lon)

    print("Computing the optimal path...")
    path, start_node, end_node = compute_route(G, start_lat, start_lon, end_lat, end_lon)

    print("\nEdge weight diagnostics along the route:")
    for u, v in zip(path[:-1], path[1:]):
        edge = G.edges[u, v, 0]
        print(f"{u} → {v} | Length: {edge.get('length', 0):.1f} m | Weight: {edge.get('weight', 0):.1f} sec")

    if path:
        # Output summary
        total_dist = sum(G.edges[u, v, 0].get("length", 0) for u, v in zip(path[:-1], path[1:]))
        total_time = sum(G.edges[u, v, 0].get("weight", 0) for u, v in zip(path[:-1], path[1:]))
        print(f"Distance: {int(total_dist)} m, Time: {total_time/60:.1f} min")
        compute_directions(G, path)
    else:
        print("No route found.")

    print("Route found. Plotting with start/end points...")
    fig, ax = ox.plot_graph_route(G, path, route_linewidth=4, node_size=0, show=False, close=False)

    ax.scatter(G.nodes[start_node]['x'], G.nodes[start_node]['y'], c="green", s=80, label="Start Node")
    ax.scatter(G.nodes[end_node]['x'], G.nodes[end_node]['y'], c="green", s=80, label="End Node")
    ax.legend()
    ax.set_title(f"Route (Dist: {int(total_dist)} m, Time: {(total_time/60):.1f} min)", color="black")
    plt.show()
