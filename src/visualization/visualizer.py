# visualization/visualizer.py
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

class RouteVisualizer:
    def __init__(self, G):
        self.G = G
        
    def plot_route(self, path, title="Route Visualization"):
        """Plot a route using OSMNX"""
        # Create route as a list of node pairs
        route_edges = list(zip(path[:-1], path[1:]))
        
        # Plot the graph with the route highlighted
        fig, ax = ox.plot_graph_route(self.G, path, route_linewidth=6, 
                                      node_size=0, bgcolor='white',
                                      figsize=(12, 8))
        plt.title(title)
        return fig, ax
    
    def plot_comparison(self, dijkstra_path, astar_path, title="Route Comparison"):
        """Plot two routes for comparison"""
        # Create a figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot the base graph with better visibility
        ox.plot_graph(self.G, ax=ax, node_size=0, edge_linewidth=0.5,
                     edge_color='#cccccc', bgcolor='white')
        
        # Plot Dijkstra path with thicker line
        ox.plot_graph_routes(self.G, [dijkstra_path], ax=ax, 
                           route_colors=['blue'], 
                           route_linewidth=4, 
                           route_alpha=0.8)
        
        # Plot A* path with thinner line
        ox.plot_graph_routes(self.G, [astar_path], ax=ax, 
                           route_colors=['red'], 
                           route_linewidth=2.5, 
                           route_alpha=0.7)
        
        # Add traffic congestion indicators
        for u, v, k, data in self.G.edges(keys=True, data=True):
            if 'traffic_speed' in data and data['traffic_speed'] < 20:
                y1, x1 = self.G.nodes[u]['y'], self.G.nodes[u]['x']
                y2, x2 = self.G.nodes[v]['y'], self.G.nodes[v]['x']
                mid_y = (y1 + y2) / 2
                mid_x = (x1 + x2) / 2
                ax.plot(mid_x, mid_y, 'o', color='orange', markersize=5, alpha=0.7)
        
        # Enhanced title and legend
        plt.title("Emergency Route Comparison:\nDijkstra's Algorithm vs A* Algorithm", 
                 fontsize=14, pad=20)
        
        # Create custom legend with route information
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='blue', linewidth=4, 
                  label="Dijkstra's Algorithm (optimal time)"),
            Line2D([0], [0], color='red', linewidth=2.5, 
                  label="A* Algorithm (informed search)"),
            Line2D([0], [0], color='orange', marker='o', linestyle='None',
                  label='Traffic Congestion', markersize=8)
        ]
        ax.legend(handles=legend_elements, loc='upper right', 
                 bbox_to_anchor=(1, 1), fontsize=12,
                 title="Routing Methods")
        
        # Adjust layout
        plt.tight_layout()
        return fig, ax



    def plot_comparison_1(self, dijkstra_path, astar_path, title="Route Comparison"):
        """Plot two routes for comparison - more robust version"""
        try:
            # Create a figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot the base graph - make sure edges are visible
            ox.plot_graph(self.G, ax=ax, node_size=0, edge_linewidth=0.5,
                        edge_color='#cccccc', bgcolor='white', show=False)
            
            # Check if paths exist and have nodes
            if dijkstra_path and len(dijkstra_path) > 1:
                # Plot Dijkstra path
                dijkstra_edges = list(zip(dijkstra_path[:-1], dijkstra_path[1:]))
                ox.plot_graph_routes(self.G, [dijkstra_path], ax=ax, 
                                route_colors=['blue'], 
                                route_linewidth=4, 
                                route_alpha=0.8,
                                show=False)
                
                print(f"Plotting Dijkstra path with {len(dijkstra_path)} nodes")
            else:
                print("Warning: Dijkstra path is empty or has only one node")
            
            if astar_path and len(astar_path) > 1:
                # Plot A* path
                astar_edges = list(zip(astar_path[:-1], astar_path[1:]))
                ox.plot_graph_routes(self.G, [astar_path], ax=ax, 
                                route_colors=['red'], 
                                route_linewidth=2.5, 
                                route_alpha=0.7,
                                show=False)
                
                print(f"Plotting A* path with {len(astar_path)} nodes")
            else:
                print("Warning: A* path is empty or has only one node")
            
            # Add traffic congestion indicators
            for u, v, k, data in self.G.edges(keys=True, data=True):
                if 'traffic_speed' in data and data['traffic_speed'] < 20:
                    y1, x1 = self.G.nodes[u]['y'], self.G.nodes[u]['x']
                    y2, x2 = self.G.nodes[v]['y'], self.G.nodes[v]['x']
                    mid_y = (y1 + y2) / 2
                    mid_x = (x1 + x2) / 2
                    ax.plot(mid_x, mid_y, 'o', color='orange', markersize=5, alpha=0.7)
            
            # Enhanced title and legend
            plt.title("Emergency Route Comparison:\nDijkstra's Algorithm vs A* Algorithm", 
                    fontsize=14, pad=20)
            
            # Create custom legend with route information
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='blue', linewidth=4, 
                    label="Dijkstra's Algorithm (optimal time)"),
                Line2D([0], [0], color='red', linewidth=2.5, 
                    label="A* Algorithm (informed search)"),
                Line2D([0], [0], color='orange', marker='o', linestyle='None',
                    label='Traffic Congestion', markersize=8)
            ]
            ax.legend(handles=legend_elements, loc='upper right', 
                    bbox_to_anchor=(1, 1), fontsize=12,
                    title="Routing Methods")
            
            # Adjust layout
            plt.tight_layout()
            
            # Return the figure and axis for further customization or saving
            return fig, ax
        except Exception as e:
            print(f"Error in plot_comparison: {str(e)}")
            # Create a simple fallback figure with error message
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f"Error generating route comparison: {str(e)}", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=12)
            return fig, ax


    def plot_comparison_2(self, dijkstra_path, astar_path, title="Route Comparison"):
        """Plot a comparison of routes using direct matplotlib plotting"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Extract coordinates for Dijkstra path
        dijkstra_coords = []
        for node in dijkstra_path:
            y = self.G.nodes[node]['y']
            x = self.G.nodes[node]['x']
            dijkstra_coords.append((x, y))
        
        # Extract coordinates for A* path
        astar_coords = []
        for node in astar_path:
            y = self.G.nodes[node]['y']
            x = self.G.nodes[node]['x']
            astar_coords.append((x, y))
        
        # Plot routes
        if dijkstra_coords:
            x_vals, y_vals = zip(*dijkstra_coords)
            ax.plot(x_vals, y_vals, 'b-', linewidth=3, label="Dijkstra's Algorithm")
        
        if astar_coords:
            x_vals, y_vals = zip(*astar_coords)
            ax.plot(x_vals, y_vals, 'r-', linewidth=2, label="A* Algorithm")
        
        # Add title and legend
        plt.title(title, fontsize=16)
        plt.xlabel('Longitude', fontsize=12)
        plt.ylabel('Latitude', fontsize=12)
        ax.legend(fontsize=12)
        
        plt.tight_layout()
        return fig, ax



    def plot_comparison_3(self, dijkstra_path, astar_path, title="Route Comparison"):
        """Plot two routes for comparison on the road network"""
        try:
            # Create a figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Extract node coordinates for both paths
            dijkstra_coords = [(self.G.nodes[node]['x'], self.G.nodes[node]['y']) for node in dijkstra_path]
            astar_coords = [(self.G.nodes[node]['x'], self.G.nodes[node]['y']) for node in astar_path]
            
            # Get the bounding box from all coords to ensure map covers the routes
            x_values = [x for x, y in dijkstra_coords + astar_coords]
            y_values = [y for x, y in dijkstra_coords + astar_coords]
            north, south = max(y_values) + 0.005, min(y_values) - 0.005
            east, west = max(x_values) + 0.005, min(x_values) - 0.005
            
            # Plot the base map - explicitly define bounds
            G_projected = ox.project_graph(self.G)
            ox.plot_graph(G_projected, ax=ax, node_size=0, edge_linewidth=0.5,
                        edge_color='#cccccc', bgcolor='white', 
                        show=False, close=False)
            
            # Directly plot the routes using matplotlib
            dijkstra_x, dijkstra_y = zip(*dijkstra_coords)
            astar_x, astar_y = zip(*astar_coords)
            
            ax.plot(dijkstra_x, dijkstra_y, '-', color='blue', linewidth=4, alpha=0.8, 
                    label="Dijkstra's Algorithm")
            ax.plot(astar_x, astar_y, '-', color='red', linewidth=2.5, alpha=0.7, 
                    label="A* Algorithm")
            
            # Add traffic congestion indicators
            for u, v, k, data in self.G.edges(keys=True, data=True):
                if 'traffic_speed' in data and data['traffic_speed'] < 20:
                    y1, x1 = self.G.nodes[u]['y'], self.G.nodes[u]['x']
                    y2, x2 = self.G.nodes[v]['y'], self.G.nodes[v]['x']
                    mid_y = (y1 + y2) / 2
                    mid_x = (x1 + x2) / 2
                    ax.plot(mid_x, mid_y, 'o', color='orange', markersize=5, alpha=0.7)
            
            # Enhanced title and legend
            plt.title("Emergency Route Comparison:\nDijkstra's Algorithm vs A* Algorithm", 
                    fontsize=14, pad=20)
            
            # Create custom legend with route information
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='blue', linewidth=4, 
                    label="Dijkstra's Algorithm"),
                Line2D([0], [0], color='red', linewidth=2.5, 
                    label="A* Algorithm"),
                Line2D([0], [0], color='orange', marker='o', linestyle='None',
                    label='Traffic Congestion', markersize=8)
            ]
            ax.legend(handles=legend_elements, loc='upper right', 
                    bbox_to_anchor=(1, 1), fontsize=12,
                    title="Routing Methods")
            
            # Adjust layout
            plt.tight_layout()
            
            # Return the figure and axis for further customization or saving
            return fig, ax
        except Exception as e:
            print(f"Error in plot_comparison: {str(e)}")
            # Create a simple fallback figure with error message
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f"Error generating route comparison: {str(e)}", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=12)
            return fig, ax


    def plot_comparison_4(self, dijkstra_path, astar_path, title="Route Comparison"):
        """Plot two routes for comparison on the road network using direct plotting"""
        try:
            # Create a figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # First, plot the network using osmnx
            ox.plot_graph(self.G, ax=ax, node_size=0, edge_linewidth=0.5,
                        edge_color='#cccccc', bgcolor='white', show=False)
            
            # Now, directly plot the routes using explicit lat/lon coordinates
            # For Dijkstra path
            dijkstra_xs = []
            dijkstra_ys = []
            for node in dijkstra_path:
                try:
                    dijkstra_xs.append(self.G.nodes[node]['x'])
                    dijkstra_ys.append(self.G.nodes[node]['y'])
                except KeyError as e:
                    print(f"Warning: Node {node} not found in graph: {e}")
            
            # For A* path
            astar_xs = []
            astar_ys = []
            for node in astar_path:
                try:
                    astar_xs.append(self.G.nodes[node]['x'])
                    astar_ys.append(self.G.nodes[node]['y'])
                except KeyError as e:
                    print(f"Warning: Node {node} not found in graph: {e}")
            
            # Debug information
            print(f"Dijkstra path: {len(dijkstra_path)} nodes, coordinates: {len(dijkstra_xs)} points")
            print(f"A* path: {len(astar_path)} nodes, coordinates: {len(astar_xs)} points")
            
            # Plot the routes if we have coordinates
            if dijkstra_xs and dijkstra_ys:
                ax.plot(dijkstra_xs, dijkstra_ys, '-', color='blue', linewidth=4, zorder=3,
                    label="Dijkstra's Algorithm")
                # Add start and end markers for better visibility
                ax.plot(dijkstra_xs[0], dijkstra_ys[0], 'o', color='green', markersize=8, zorder=4)
                ax.plot(dijkstra_xs[-1], dijkstra_ys[-1], 's', color='red', markersize=8, zorder=4)
            
            if astar_xs and astar_ys:
                ax.plot(astar_xs, astar_ys, '-', color='red', linewidth=3, zorder=2, 
                    label="A* Algorithm")
            
            # Add traffic congestion indicators
            congestion_xs = []
            congestion_ys = []
            for u, v, k, data in self.G.edges(keys=True, data=True):
                if 'traffic_speed' in data and data['traffic_speed'] < 20:
                    try:
                        u_y, u_x = self.G.nodes[u]['y'], self.G.nodes[u]['x']
                        v_y, v_x = self.G.nodes[v]['y'], self.G.nodes[v]['x']
                        mid_y = (u_y + v_y) / 2
                        mid_x = (u_x + v_x) / 2
                        congestion_xs.append(mid_x)
                        congestion_ys.append(mid_y)
                    except KeyError:
                        continue  # Skip if nodes don't have coordinates
            
            if congestion_xs and congestion_ys:
                ax.scatter(congestion_xs, congestion_ys, c='orange', s=25, alpha=0.7, zorder=1)
            
            # Enhanced title and legend
            plt.title("Emergency Route Comparison:\nDijkstra's Algorithm vs A* Algorithm", 
                    fontsize=14, pad=20)
            
            # Create custom legend with route information
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='blue', linewidth=4, 
                    label="Dijkstra's Algorithm"),
                Line2D([0], [0], color='red', linewidth=3, 
                    label="A* Algorithm"),
                Line2D([0], [0], color='orange', marker='o', linestyle='None',
                    label='Traffic Congestion', markersize=8)
            ]
            ax.legend(handles=legend_elements, loc='upper right', 
                    bbox_to_anchor=(1, 1), fontsize=12,
                    title="Routing Methods")
            
            # Make sure the axes fit all our data
            plt.tight_layout()
            
            # Return the figure and axis for further customization or saving
            return fig, ax
        except Exception as e:
            print(f"Error in plot_comparison: {str(e)}")
            import traceback
            traceback.print_exc()
            # Create a simple fallback figure with error message
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f"Error generating route comparison: {str(e)}", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=12)
            return fig, ax



    def plot_comparison(self, dijkstra_path, astar_path, title="Route Comparison"):
        """Plot two routes for comparison on the road network with offset for identical paths"""
        try:
            # Create a figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # First, plot the network using osmnx
            ox.plot_graph(self.G, ax=ax, node_size=0, edge_linewidth=0.5,
                        edge_color='#cccccc', bgcolor='white', show=False)
            
            # Check if paths are identical
            identical_paths = dijkstra_path == astar_path
            if identical_paths:
                print("Paths are identical - applying offset for visibility")
            
            # Now, directly plot the routes using explicit lat/lon coordinates
            # For Dijkstra path
            dijkstra_xs = []
            dijkstra_ys = []
            for node in dijkstra_path:
                try:
                    dijkstra_xs.append(self.G.nodes[node]['x'])
                    dijkstra_ys.append(self.G.nodes[node]['y'])
                except KeyError as e:
                    print(f"Warning: Node {node} not found in graph: {e}")
            
            # For A* path
            astar_xs = []
            astar_ys = []
            for node in astar_path:
                try:
                    astar_xs.append(self.G.nodes[node]['x'])
                    astar_ys.append(self.G.nodes[node]['y'])
                except KeyError as e:
                    print(f"Warning: Node {node} not found in graph: {e}")
            
            # Debug information
            print(f"Dijkstra path: {len(dijkstra_path)} nodes, coordinates: {len(dijkstra_xs)} points")
            print(f"A* path: {len(astar_path)} nodes, coordinates: {len(astar_xs)} points")
            
            # Plot the routes if we have coordinates
            if dijkstra_xs and dijkstra_ys:
                # Plot Dijkstra path - solid blue
                ax.plot(dijkstra_xs, dijkstra_ys, '-', color='blue', linewidth=4, zorder=3,
                    label="Dijkstra's Algorithm")
                # Add start and end markers for better visibility
                ax.plot(dijkstra_xs[0], dijkstra_ys[0], 'o', color='green', markersize=8, zorder=4)
                ax.plot(dijkstra_xs[-1], dijkstra_ys[-1], 's', color='red', markersize=8, zorder=4)
            
            if astar_xs and astar_ys:
                if identical_paths:
                    # If paths are identical, use a parallel offset and dashed line for A*
                    # Create a small offset (perpendicular to the path direction)
                    offset_xs = []
                    offset_ys = []
                    
                    # Calculate an offset perpendicular to the path
                    for i in range(len(astar_xs)):
                        if i > 0 and i < len(astar_xs) - 1:
                            # Calculate direction vector
                            dx = astar_xs[i+1] - astar_xs[i-1]
                            dy = astar_ys[i+1] - astar_ys[i-1]
                            # Normalize
                            length = (dx**2 + dy**2)**0.5
                            if length > 0:
                                # Create perpendicular vector with small magnitude
                                offset_factor = 0.0001  # Adjust this value as needed
                                offset_x = -dy * offset_factor / length
                                offset_y = dx * offset_factor / length
                                offset_xs.append(astar_xs[i] + offset_x)
                                offset_ys.append(astar_ys[i] + offset_y)
                            else:
                                offset_xs.append(astar_xs[i])
                                offset_ys.append(astar_ys[i])
                        else:
                            # For first and last points, just use original
                            offset_xs.append(astar_xs[i])
                            offset_ys.append(astar_ys[i])
                    
                    # Plot A* with offset and dashed style
                    ax.plot(offset_xs, offset_ys, '--', color='red', linewidth=3, zorder=2, 
                        label="A* Algorithm (identical path)")
                else:
                    # Normal plot for different paths
                    ax.plot(astar_xs, astar_ys, '-', color='red', linewidth=3, zorder=2, 
                        label="A* Algorithm")
            
            # Add traffic congestion indicators
            congestion_xs = []
            congestion_ys = []
            for u, v, k, data in self.G.edges(keys=True, data=True):
                if 'traffic_speed' in data and data['traffic_speed'] < 20:
                    try:
                        u_y, u_x = self.G.nodes[u]['y'], self.G.nodes[u]['x']
                        v_y, v_x = self.G.nodes[v]['y'], self.G.nodes[v]['x']
                        mid_y = (u_y + v_y) / 2
                        mid_x = (u_x + v_x) / 2
                        congestion_xs.append(mid_x)
                        congestion_ys.append(mid_y)
                    except KeyError:
                        continue  # Skip if nodes don't have coordinates
            
            if congestion_xs and congestion_ys:
                ax.scatter(congestion_xs, congestion_ys, c='orange', s=25, alpha=0.7, zorder=1,
                        marker='o', label='Traffic Congestion')
            
            # Enhanced title 
            if identical_paths:
                plt.title("Emergency Route Comparison:\nBoth Algorithms Found the Same Optimal Path", 
                        fontsize=14, pad=20)
            else:
                plt.title("Emergency Route Comparison:\nDijkstra's Algorithm vs A* Algorithm", 
                        fontsize=14, pad=20)
            
            # Let matplotlib handle the legend automatically
            ax.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize=12)
            
            # Make sure the axes fit all our data
            plt.tight_layout()
            
            # Return the figure and axis for further customization or saving
            return fig, ax
        except Exception as e:
            print(f"Error in plot_comparison: {str(e)}")
            import traceback
            traceback.print_exc()
            # Create a simple fallback figure with error message
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f"Error generating route comparison: {str(e)}", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=12)
            return fig, ax



    def create_interactive_map_0(self, route_data):
        """Create a reliable interactive map with route visualization"""
        import folium
        from folium import plugins
        
        # Get paths and travel times
        dijkstra_path = route_data['dijkstra_path']
        astar_path = route_data['astar_path']
        dijkstra_time = route_data.get('dijkstra_time', 0)
        astar_time = route_data.get('astar_time', 0)
        
        # Get center point from the first path
        center_node = dijkstra_path[len(dijkstra_path)//2]
        center_y = self.G.nodes[center_node]['y']
        center_x = self.G.nodes[center_node]['x']
        
        # Create basic map
        m = folium.Map(location=[center_y, center_x], zoom_start=14)
        
        # Create feature groups
        route_group = folium.FeatureGroup(name='Routes')
        traffic_group = folium.FeatureGroup(name='Traffic')
        
        # Add Dijkstra path
        path_coords = [(self.G.nodes[node]['y'], self.G.nodes[node]['x']) 
                      for node in dijkstra_path]
        folium.PolyLine(
            path_coords,
            color='blue',
            weight=5,
            opacity=0.8,
            popup=folium.Popup(f'Dijkstra Path - Travel Time: {dijkstra_time:.2f}s')
        ).add_to(route_group)
        
        # Add A* path
        path_coords = [(self.G.nodes[node]['y'], self.G.nodes[node]['x']) 
                      for node in astar_path]
        folium.PolyLine(
            path_coords,
            color='red',
            weight=3,
            opacity=0.7,
            popup=folium.Popup(f'A* Path - Travel Time: {astar_time:.2f}s')
        ).add_to(route_group)
        
        # Add start and end markers
        start_node = dijkstra_path[0]
        end_node = dijkstra_path[-1]
        
        folium.Marker(
            [self.G.nodes[start_node]['y'], self.G.nodes[start_node]['x']],
            popup='Starting Point',
            icon=folium.Icon(color='green', icon='play', prefix='glyphicon')
        ).add_to(m)
        
        folium.Marker(
            [self.G.nodes[end_node]['y'], self.G.nodes[end_node]['x']],
            popup='Emergency Location',
            icon=folium.Icon(color='red', icon='flag', prefix='glyphicon')
        ).add_to(m)
        
        # Add traffic congestion points from edges
        processed_points = set()  # To avoid duplicate points
        for u, v, k, data in self.G.edges(keys=True, data=True):
            if 'traffic_speed' in data:
                # Get midpoint of the edge
                y1, x1 = self.G.nodes[u]['y'], self.G.nodes[u]['x']
                y2, x2 = self.G.nodes[v]['y'], self.G.nodes[v]['x']
                mid_y = (y1 + y2) / 2
                mid_x = (x1 + x2) / 2
                
                # Create a unique key for this point to avoid duplicates
                point_key = f"{mid_y:.5f}_{mid_x:.5f}"
                
                if point_key not in processed_points:
                    processed_points.add(point_key)
                    
                    # Determine congestion level
                    speed = data['traffic_speed']
                    if speed < 15:  # Severe congestion
                        color = 'red'
                        radius = 6
                        severity = 'Severe'
                    elif speed < 30:  # Moderate congestion
                        color = 'orange'
                        radius = 5
                        severity = 'Moderate'
                    elif speed < 45:  # Light congestion
                        color = 'yellow'
                        radius = 4
                        severity = 'Light'
                    else:
                        continue  # Skip non-congested segments
                    
                    folium.CircleMarker(
                        location=[mid_y, mid_x],
                        radius=radius,
                        color=color,
                        fill=True,
                        fillColor=color,
                        fillOpacity=0.7,
                        popup=f'{severity} Congestion: {speed:.1f} km/h'
                    ).add_to(traffic_group)

        # Add route information box with enhanced legend
        legend_html = f'''
        <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white;
                    padding: 10px; border-radius: 5px; box-shadow: 0 0 10px rgba(0,0,0,0.3);">
        <h4>Route Comparison</h4>
        <div><i style="background: blue; width: 15px; height: 5px; display: inline-block;"></i> Dijkstra: {dijkstra_time:.2f}s</div>
        <div><i style="background: red; width: 15px; height: 5px; display: inline-block;"></i> A*: {astar_time:.2f}s</div>
        <div style="margin-top: 8px;"><b>Traffic Congestion:</b></div>
        <div><i style="background: red; border-radius: 50%; width: 10px; height: 10px; display: inline-block;"></i> Severe (&lt;15 km/h)</div>
        <div><i style="background: orange; border-radius: 50%; width: 10px; height: 10px; display: inline-block;"></i> Moderate (&lt;30 km/h)</div>
        <div><i style="background: yellow; border-radius: 50%; width: 10px; height: 10px; display: inline-block;"></i> Light (&lt;45 km/h)</div>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Add feature groups to map
        route_group.add_to(m)
        traffic_group.add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        return m


    def create_interactive_map_1(self, route_data):
            """Create an interactive map with accurate travel time calculations"""
            import folium
            from folium.plugins import MarkerCluster
            import numpy as np
            
            # Extract route data
            dijkstra_path = route_data['dijkstra_path']
            astar_path = route_data['astar_path']
            
            # Recalculate travel times to ensure accuracy
            dijkstra_time = 0
            astar_time = 0
            
            # Calculate Dijkstra travel time
            for i in range(len(dijkstra_path) - 1):
                u, v = dijkstra_path[i], dijkstra_path[i+1]
                # Get minimum weight edge
                edge_data = min(self.G.get_edge_data(u, v).values(), 
                            key=lambda x: x.get('weight', float('inf')))
                dijkstra_time += edge_data.get('weight', 0)
                dijkstra_time += self.G.nodes[v].get('delay', 0)
            
            # Calculate A* travel time
            for i in range(len(astar_path) - 1):
                u, v = astar_path[i], astar_path[i+1]
                # Get minimum weight edge
                edge_data = min(self.G.get_edge_data(u, v).values(), 
                            key=lambda x: x.get('weight', float('inf')))
                astar_time += edge_data.get('weight', 0)
                astar_time += self.G.nodes[v].get('delay', 0)
            
            print(f"Verified Dijkstra travel time: {dijkstra_time:.2f}s")
            print(f"Verified A* travel time: {astar_time:.2f}s")
            
            # Create base map
            center_y = (self.G.nodes[dijkstra_path[0]]['y'] + self.G.nodes[dijkstra_path[-1]]['y']) / 2
            center_x = (self.G.nodes[dijkstra_path[0]]['x'] + self.G.nodes[dijkstra_path[-1]]['x']) / 2
            
            m = folium.Map(location=[center_y, center_x], zoom_start=14, control_scale=True)
            
            # Add a legend
            legend_html = '''
                <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white;
                            padding: 10px; border-radius: 5px; box-shadow: 0 0 10px rgba(0,0,0,0.3);">
                <h4>Route Comparison</h4>
                <div><i style="background: blue; width: 15px; height: 5px; display: inline-block;"></i> Dijkstra: {:.2f}s</div>
                <div><i style="background: red; width: 15px; height: 5px; display: inline-block;"></i> A*: {:.2f}s</div>
                <div style="margin-top: 8px;"><b>Traffic Congestion:</b></div>
                <div><i style="background: red; border-radius: 50%; width: 10px; height: 10px; display: inline-block;"></i> Severe (&lt;15 km/h)</div>
                <div><i style="background: orange; border-radius: 50%; width: 10px; height: 10px; display: inline-block;"></i> Moderate (&lt;30 km/h)</div>
                <div><i style="background: yellow; border-radius: 50%; width: 10px; height: 10px; display: inline-block;"></i> Light (&lt;45 km/h)</div>
                </div>
                '''.format(dijkstra_time, astar_time)
            
            m.get_root().html.add_child(folium.Element(legend_html))
            
            # Add markers for start and end
            start_y, start_x = self.G.nodes[dijkstra_path[0]]['y'], self.G.nodes[dijkstra_path[0]]['x']
            end_y, end_x = self.G.nodes[dijkstra_path[-1]]['y'], self.G.nodes[dijkstra_path[-1]]['x']
            
            folium.Marker(
                location=[start_y, start_x],
                popup="Starting Point",
                icon=folium.Icon(color='green', icon='play', prefix='glyphicon')
            ).add_to(m)
            
            folium.Marker(
                location=[end_y, end_x],
                popup="Emergency Location",
                icon=folium.Icon(color='red', icon='flag', prefix='glyphicon')
            ).add_to(m)
            
            # Create a feature group for the routes
            routes = folium.FeatureGroup(name="Routes")
            
            # Convert node IDs to coordinates for Dijkstra path
            dijkstra_coords = []
            for node in dijkstra_path:
                y, x = self.G.nodes[node]['y'], self.G.nodes[node]['x']
                dijkstra_coords.append([y, x])
            
            # Add Dijkstra path
            folium.PolyLine(
                locations=dijkstra_coords,
                color='blue',
                weight=5,
                opacity=0.8,
                popup=f"Dijkstra Path - Travel Time: {dijkstra_time:.2f}s"
            ).add_to(routes)
            
            # Convert node IDs to coordinates for A* path
            astar_coords = []
            for node in astar_path:
                y, x = self.G.nodes[node]['y'], self.G.nodes[node]['x']
                astar_coords.append([y, x])
            
            # Add A* path
            folium.PolyLine(
                locations=astar_coords,
                color='red',
                weight=3,
                opacity=0.7,
                popup=f"A* Path - Travel Time: {astar_time:.2f}s"
            ).add_to(routes)
            
            # Add routes to map
            routes.add_to(m)
            
            # Add traffic congestion points
            traffic = folium.FeatureGroup(name="Traffic")
            
            # Find congested edges
            congested_points = []
            for u, v, data in self.G.edges(data=True):
                if 'traffic_speed' in data:
                    speed = data['traffic_speed']
                    # If speed is very low, mark as congestion
                    if speed < 15: # km/h
                        u_y, u_x = self.G.nodes[u]['y'], self.G.nodes[u]['x']
                        v_y, v_x = self.G.nodes[v]['y'], self.G.nodes[v]['x']
                        # Take midpoint of the edge
                        mid_y, mid_x = (u_y + v_y) / 2, (u_x + v_x) / 2
                        congested_points.append((mid_y, mid_x, speed))
            
            # Add congestion markers
            for y, x, speed in congested_points:
                folium.CircleMarker(
                    location=[y, x],
                    radius=6,
                    color='red',
                    fill=True,
                    fill_color='red',
                    fill_opacity=0.7,
                    popup=f"Severe Congestion: {speed:.1f} km/h"
                ).add_to(traffic)
            
            # Add traffic to map
            traffic.add_to(m)
            
            # Add layer control
            folium.LayerControl().add_to(m)
            
            return m


    def create_interactive_map_2(self, route_data):
        """Create an interactive map with accurate travel time calculations and all traffic congestion levels"""
        import folium
        from folium.plugins import MarkerCluster
        import numpy as np
        
        # Extract route data
        dijkstra_path = route_data['dijkstra_path']
        astar_path = route_data['astar_path']
        
        # Recalculate travel times to ensure accuracy
        dijkstra_time = 0
        astar_time = 0
        
        # Calculate Dijkstra travel time
        for i in range(len(dijkstra_path) - 1):
            u, v = dijkstra_path[i], dijkstra_path[i+1]
            # Get minimum weight edge
            edge_data = min(self.G.get_edge_data(u, v).values(), 
                        key=lambda x: x.get('weight', float('inf')))
            dijkstra_time += edge_data.get('weight', 0)
            dijkstra_time += self.G.nodes[v].get('delay', 0)
        
        # Calculate A* travel time
        for i in range(len(astar_path) - 1):
            u, v = astar_path[i], astar_path[i+1]
            # Get minimum weight edge
            edge_data = min(self.G.get_edge_data(u, v).values(), 
                        key=lambda x: x.get('weight', float('inf')))
            astar_time += edge_data.get('weight', 0)
            astar_time += self.G.nodes[v].get('delay', 0)
        
        print(f"Verified Dijkstra travel time: {dijkstra_time:.2f}s")
        print(f"Verified A* travel time: {astar_time:.2f}s")
        
        # Create base map
        center_y = (self.G.nodes[dijkstra_path[0]]['y'] + self.G.nodes[dijkstra_path[-1]]['y']) / 2
        center_x = (self.G.nodes[dijkstra_path[0]]['x'] + self.G.nodes[dijkstra_path[-1]]['x']) / 2
        
        m = folium.Map(location=[center_y, center_x], zoom_start=14, control_scale=True)
        
        # Add a legend
        legend_html = '''
            <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white;
                        padding: 10px; border-radius: 5px; box-shadow: 0 0 10px rgba(0,0,0,0.3);">
            <h4>Route Comparison</h4>
            <div><i style="background: blue; width: 15px; height: 5px; display: inline-block;"></i> Dijkstra: {:.2f}s</div>
            <div><i style="background: red; width: 15px; height: 5px; display: inline-block;"></i> A*: {:.2f}s</div>
            <div style="margin-top: 8px;"><b>Traffic Congestion:</b></div>
            <div><i style="background: red; border-radius: 50%; width: 10px; height: 10px; display: inline-block;"></i> Severe (&lt;15 km/h)</div>
            <div><i style="background: orange; border-radius: 50%; width: 10px; height: 10px; display: inline-block;"></i> Moderate (&lt;30 km/h)</div>
            <div><i style="background: yellow; border-radius: 50%; width: 10px; height: 10px; display: inline-block;"></i> Light (&lt;45 km/h)</div>
            </div>
            '''.format(dijkstra_time, astar_time)
        
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Add markers for start and end
        start_y, start_x = self.G.nodes[dijkstra_path[0]]['y'], self.G.nodes[dijkstra_path[0]]['x']
        end_y, end_x = self.G.nodes[dijkstra_path[-1]]['y'], self.G.nodes[dijkstra_path[-1]]['x']
        
        folium.Marker(
            location=[start_y, start_x],
            popup="Starting Point",
            icon=folium.Icon(color='green', icon='play', prefix='glyphicon')
        ).add_to(m)
        
        folium.Marker(
            location=[end_y, end_x],
            popup="Emergency Location",
            icon=folium.Icon(color='red', icon='flag', prefix='glyphicon')
        ).add_to(m)
        
        # Create a feature group for the routes
        routes = folium.FeatureGroup(name="Routes")
        
        # Convert node IDs to coordinates for Dijkstra path
        dijkstra_coords = []
        for node in dijkstra_path:
            y, x = self.G.nodes[node]['y'], self.G.nodes[node]['x']
            dijkstra_coords.append([y, x])
        
        # Add Dijkstra path
        folium.PolyLine(
            locations=dijkstra_coords,
            color='blue',
            weight=5,
            opacity=0.8,
            popup=f"Dijkstra Path - Travel Time: {dijkstra_time:.2f}s"
        ).add_to(routes)
        
        # Convert node IDs to coordinates for A* path
        astar_coords = []
        for node in astar_path:
            y, x = self.G.nodes[node]['y'], self.G.nodes[node]['x']
            astar_coords.append([y, x])
        
        # Add A* path
        folium.PolyLine(
            locations=astar_coords,
            color='red',
            weight=3,
            opacity=0.7,
            popup=f"A* Path - Travel Time: {astar_time:.2f}s"
        ).add_to(routes)
        
        # Add routes to map
        routes.add_to(m)
        
        # Add traffic congestion points
        traffic = folium.FeatureGroup(name="Traffic")
        
        # Process all edges to find congestion points at different levels
        # Use a set to prevent duplicate points that are too close together
        processed_points = set()
        point_spacing = 0.0005  # Approximately 50 meters spacing
        
        # Find congested edges for all levels
        for u, v, k, data in self.G.edges(keys=True, data=True):
            if 'traffic_speed' in data:
                speed = data['traffic_speed']
                
                # Get edge coordinates
                u_y, u_x = self.G.nodes[u]['y'], self.G.nodes[u]['x']
                v_y, v_x = self.G.nodes[v]['y'], self.G.nodes[v]['x']
                
                # Take midpoint of the edge
                mid_y, mid_x = (u_y + v_y) / 2, (u_x + v_x) / 2
                
                # Create a grid cell key to avoid too many markers in the same area
                grid_y = round(mid_y / point_spacing) * point_spacing
                grid_x = round(mid_x / point_spacing) * point_spacing
                point_key = f"{grid_y:.5f}_{grid_x:.5f}"
                
                if point_key in processed_points:
                    continue
                    
                processed_points.add(point_key)
                
                # Categorize congestion level
                if speed < 15:  # Severe congestion
                    color = 'red'
                    radius = 6
                    severity = 'Severe'
                    folium.CircleMarker(
                        location=[mid_y, mid_x],
                        radius=radius,
                        color=color,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.7,
                        popup=f"{severity} Congestion: {speed:.1f} km/h"
                    ).add_to(traffic)
                elif speed < 30:  # Moderate congestion
                    color = 'orange'
                    radius = 6
                    severity = 'Moderate'
                    folium.CircleMarker(
                        location=[mid_y, mid_x],
                        radius=radius,
                        color=color,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.7,
                        popup=f"{severity} Congestion: {speed:.1f} km/h"
                    ).add_to(traffic)
                elif speed < 45:  # Light congestion
                    color = 'yellow'
                    radius = 6
                    severity = 'Light'
                    folium.CircleMarker(
                        location=[mid_y, mid_x],
                        radius=radius,
                        color=color,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.7,
                        popup=f"{severity} Congestion: {speed:.1f} km/h"
                    ).add_to(traffic)
        
        # Add traffic to map
        traffic.add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        return m



    def create_interactive_map(self, route_data):
        """Create an interactive map with separate traffic congestion layers"""
        import folium
        from folium.plugins import MarkerCluster
        import numpy as np
        
        # Extract route data
        dijkstra_path = route_data['dijkstra_path']
        astar_path = route_data['astar_path']
        
        # Recalculate travel times to ensure accuracy
        dijkstra_time = 0
        astar_time = 0
        
        # Calculate Dijkstra travel time
        for i in range(len(dijkstra_path) - 1):
            u, v = dijkstra_path[i], dijkstra_path[i+1]
            # Get minimum weight edge
            edge_data = min(self.G.get_edge_data(u, v).values(), 
                        key=lambda x: x.get('weight', float('inf')))
            dijkstra_time += edge_data.get('weight', 0)
            dijkstra_time += self.G.nodes[v].get('delay', 0)
        
        # Calculate A* travel time
        for i in range(len(astar_path) - 1):
            u, v = astar_path[i], astar_path[i+1]
            # Get minimum weight edge
            edge_data = min(self.G.get_edge_data(u, v).values(), 
                        key=lambda x: x.get('weight', float('inf')))
            astar_time += edge_data.get('weight', 0)
            astar_time += self.G.nodes[v].get('delay', 0)
        
        print(f"Verified Dijkstra travel time: {dijkstra_time:.2f}s")
        print(f"Verified A* travel time: {astar_time:.2f}s")
        
        # Create base map
        center_y = (self.G.nodes[dijkstra_path[0]]['y'] + self.G.nodes[dijkstra_path[-1]]['y']) / 2
        center_x = (self.G.nodes[dijkstra_path[0]]['x'] + self.G.nodes[dijkstra_path[-1]]['x']) / 2
        
        m = folium.Map(location=[center_y, center_x], zoom_start=14, control_scale=True)
        
        # Add a legend
        legend_html = '''
            <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white;
                        padding: 10px; border-radius: 5px; box-shadow: 0 0 10px rgba(0,0,0,0.3);">
            <h4>Route Comparison</h4>
            <div><i style="background: blue; width: 15px; height: 5px; display: inline-block;"></i> Dijkstra: {:.2f}s</div>
            <div><i style="background: red; width: 15px; height: 5px; display: inline-block;"></i> A*: {:.2f}s</div>
            <div style="margin-top: 8px;"><b>Traffic Congestion:</b></div>
            <div><i style="background: red; border-radius: 50%; width: 10px; height: 10px; display: inline-block;"></i> Severe (&lt;15 km/h)</div>
            <div><i style="background: orange; border-radius: 50%; width: 10px; height: 10px; display: inline-block;"></i> Moderate (&lt;30 km/h)</div>
            <div><i style="background: yellow; border-radius: 50%; width: 10px; height: 10px; display: inline-block;"></i> Light (&lt;45 km/h)</div>
            </div>
            '''.format(dijkstra_time, astar_time)
        
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Add markers for start and end
        start_y, start_x = self.G.nodes[dijkstra_path[0]]['y'], self.G.nodes[dijkstra_path[0]]['x']
        end_y, end_x = self.G.nodes[dijkstra_path[-1]]['y'], self.G.nodes[dijkstra_path[-1]]['x']
        
        folium.Marker(
            location=[start_y, start_x],
            popup="Starting Point",
            icon=folium.Icon(color='green', icon='play', prefix='glyphicon')
        ).add_to(m)
        
        folium.Marker(
            location=[end_y, end_x],
            popup="Emergency Location",
            icon=folium.Icon(color='red', icon='flag', prefix='glyphicon')
        ).add_to(m)
        
        # Create a feature group for the routes
        routes = folium.FeatureGroup(name="Routes")
        
        # Convert node IDs to coordinates for Dijkstra path
        dijkstra_coords = []
        for node in dijkstra_path:
            y, x = self.G.nodes[node]['y'], self.G.nodes[node]['x']
            dijkstra_coords.append([y, x])
        
        # Add Dijkstra path
        folium.PolyLine(
            locations=dijkstra_coords,
            color='blue',
            weight=5,
            opacity=0.8,
            popup=f"Dijkstra Path - Travel Time: {dijkstra_time:.2f}s"
        ).add_to(routes)
        
        # Convert node IDs to coordinates for A* path
        astar_coords = []
        for node in astar_path:
            y, x = self.G.nodes[node]['y'], self.G.nodes[node]['x']
            astar_coords.append([y, x])
        
        # Add A* path
        folium.PolyLine(
            locations=astar_coords,
            color='red',
            weight=3,
            opacity=0.7,
            popup=f"A* Path - Travel Time: {astar_time:.2f}s"
        ).add_to(routes)
        
        # Add routes to map
        routes.add_to(m)
        
        # Create separate feature groups for each traffic congestion level
        severe_traffic = folium.FeatureGroup(name="Severe Traffic")
        moderate_traffic = folium.FeatureGroup(name="Moderate Traffic", show=False)  # Hidden by default
        light_traffic = folium.FeatureGroup(name="Light Traffic", show=False)  # Hidden by default
        
        # Process all edges to find congestion points at different levels
        # Use a set to prevent duplicate points that are too close together
        processed_points = {
            'severe': set(),
            'moderate': set(),
            'light': set()
        }
        point_spacing = 0.0005  # Approximately 50 meters spacing
        
        # Find congested edges for all levels
        for u, v, k, data in self.G.edges(keys=True, data=True):
            if 'traffic_speed' in data:
                speed = data['traffic_speed']
                
                # Get edge coordinates
                u_y, u_x = self.G.nodes[u]['y'], self.G.nodes[u]['x']
                v_y, v_x = self.G.nodes[v]['y'], self.G.nodes[v]['x']
                
                # Take midpoint of the edge
                mid_y, mid_x = (u_y + v_y) / 2, (u_x + v_x) / 2
                
                # Create a grid cell key to avoid too many markers in the same area
                grid_y = round(mid_y / point_spacing) * point_spacing
                grid_x = round(mid_x / point_spacing) * point_spacing
                
                # Categorize congestion level
                if speed < 15:  # Severe congestion
                    point_key = f"severe_{grid_y:.5f}_{grid_x:.5f}"
                    if point_key not in processed_points['severe']:
                        processed_points['severe'].add(point_key)
                        folium.CircleMarker(
                            location=[mid_y, mid_x],
                            radius=6,
                            color='red',
                            fill=True,
                            fill_color='red',
                            fill_opacity=0.7,
                            popup=f"Severe Congestion: {speed:.1f} km/h"
                        ).add_to(severe_traffic)
                elif speed < 30:  # Moderate congestion
                    point_key = f"moderate_{grid_y:.5f}_{grid_x:.5f}"
                    if point_key not in processed_points['moderate']:
                        processed_points['moderate'].add(point_key)
                        folium.CircleMarker(
                            location=[mid_y, mid_x],
                            radius=6,
                            color='orange',
                            fill=True,
                            fill_color='orange',
                            fill_opacity=0.7,
                            popup=f"Moderate Congestion: {speed:.1f} km/h"
                        ).add_to(moderate_traffic)
                elif speed < 45:  # Light congestion
                    point_key = f"light_{grid_y:.5f}_{grid_x:.5f}"
                    if point_key not in processed_points['light']:
                        processed_points['light'].add(point_key)
                        folium.CircleMarker(
                            location=[mid_y, mid_x],
                            radius=6,
                            color='yellow',
                            fill=True,
                            fill_color='yellow',
                            fill_opacity=0.7,
                            popup=f"Light Congestion: {speed:.1f} km/h"
                        ).add_to(light_traffic)
        
        # Add traffic layers to map (severe is visible by default)
        severe_traffic.add_to(m)
        moderate_traffic.add_to(m)
        light_traffic.add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        return m


    def plot_comparison(self, dijkstra_path, astar_path, title="Route Comparison"):
        """Plot two routes for comparison on the road network with guaranteed visibility"""
        try:
            # Create a figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # First, plot the network using osmnx
            ox.plot_graph(self.G, ax=ax, node_size=0, edge_linewidth=0.5,
                        edge_color='#cccccc', bgcolor='white', show=False)
            
            # Extract coordinates
            dijkstra_xs = []
            dijkstra_ys = []
            for node in dijkstra_path:
                try:
                    dijkstra_xs.append(self.G.nodes[node]['x'])
                    dijkstra_ys.append(self.G.nodes[node]['y'])
                except KeyError as e:
                    print(f"Warning: Node {node} not found in graph: {e}")
            
            astar_xs = []
            astar_ys = []
            for node in astar_path:
                try:
                    astar_xs.append(self.G.nodes[node]['x'])
                    astar_ys.append(self.G.nodes[node]['y'])
                except KeyError as e:
                    print(f"Warning: Node {node} not found in graph: {e}")
            
            # Debug information
            print(f"Dijkstra path: {len(dijkstra_path)} nodes, coordinates: {len(dijkstra_xs)} points")
            print(f"A* path: {len(astar_path)} nodes, coordinates: {len(astar_xs)} points")
            
            # Plot A* path FIRST (so Dijkstra can be on top)
            if astar_xs and astar_ys:
                # Plot A* as a thicker red line
                ax.plot(astar_xs, astar_ys, '-', color='red', linewidth=7, alpha=0.6, zorder=2, 
                    label="A* Algorithm")
            
            # Then plot Dijkstra path ON TOP with a thinner blue line
            if dijkstra_xs and dijkstra_ys:
                # Plot Dijkstra as a thinner blue line on top
                ax.plot(dijkstra_xs, dijkstra_ys, '-', color='blue', linewidth=3, zorder=3,
                    label="Dijkstra's Algorithm")
                
                # Add start and end markers
                ax.plot(dijkstra_xs[0], dijkstra_ys[0], 'o', color='green', markersize=8, zorder=4)
                ax.plot(dijkstra_xs[-1], dijkstra_ys[-1], 's', color='red', markersize=8, zorder=4)
            
            # Add traffic congestion indicators
            congestion_xs = []
            congestion_ys = []
            for u, v, k, data in self.G.edges(keys=True, data=True):
                if 'traffic_speed' in data and data['traffic_speed'] < 20:
                    try:
                        u_y, u_x = self.G.nodes[u]['y'], self.G.nodes[u]['x']
                        v_y, v_x = self.G.nodes[v]['y'], self.G.nodes[v]['x']
                        mid_y = (u_y + v_y) / 2
                        mid_x = (u_x + v_x) / 2
                        congestion_xs.append(mid_x)
                        congestion_ys.append(mid_y)
                    except KeyError:
                        continue  # Skip if nodes don't have coordinates
            
            if congestion_xs and congestion_ys:
                ax.scatter(congestion_xs, congestion_ys, c='orange', s=25, alpha=0.7, zorder=1)
            
            # Enhanced title
            if dijkstra_path == astar_path:
                plt.title("Emergency Route Comparison:\nBoth Algorithms Found the Same Optimal Path", 
                        fontsize=14, pad=20)
            else:
                plt.title("Emergency Route Comparison:\nDijkstra's Algorithm vs A* Algorithm", 
                        fontsize=14, pad=20)
            
            # Create custom legend with route information
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='blue', linewidth=3, 
                    label="Dijkstra's Algorithm"),
                Line2D([0], [0], color='red', linewidth=7, alpha=0.6,
                    label="A* Algorithm"),
                Line2D([0], [0], color='orange', marker='o', linestyle='None',
                    label='Traffic Congestion', markersize=8)
            ]
            ax.legend(handles=legend_elements, loc='upper right', 
                    bbox_to_anchor=(1, 1), fontsize=12,
                    title="Routing Methods")
            
            # Make sure the axes fit all our data
            plt.tight_layout()
            
            # Return the figure and axis for further customization or saving
            return fig, ax
        except Exception as e:
            print(f"Error in plot_comparison: {str(e)}")
            import traceback
            traceback.print_exc()
            # Create a simple fallback figure with error message
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f"Error generating route comparison: {str(e)}", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=12)
            return fig, ax


    def _get_bearing(self, p1, p2):
            """Calculate bearing between two points for arrow rotation"""
            import math
            
            lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
            lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
            
            dlon = lon2 - lon1
            
            y = math.sin(dlon) * math.cos(lat2)
            x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
            
            bearing = math.atan2(y, x)
            bearing = math.degrees(bearing)
            bearing = (bearing + 360) % 360
            
            return bearing




    
    def plot_performance_comparison(self, dijkstra_result, astar_result, title="Algorithm Performance Comparison"):
        """Create a bar chart comparing performance metrics of Dijkstra vs A*"""
        try:
            # Extract metrics
            metrics = {
                'Travel Time (s)': [dijkstra_result['travel_time'], astar_result['travel_time']],
                'Computation Time (s)': [dijkstra_result['computation_time'], astar_result['computation_time']],
                'Path Length (nodes)': [len(dijkstra_result['path']), len(astar_result['path'])]
            }
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Calculate positions for grouped bars
            bar_width = 0.35
            x = np.arange(len(metrics))
            
            # Create bars
            ax.bar(x - bar_width/2, [metrics[m][0] for m in metrics], bar_width, 
                label='Dijkstra', color='blue', alpha=0.7)
            ax.bar(x + bar_width/2, [metrics[m][1] for m in metrics], bar_width,
                label='A*', color='red', alpha=0.7)
            
            # Add labels and legend
            ax.set_title(title, fontsize=16)
            ax.set_xticks(x)
            ax.set_xticklabels(metrics.keys())
            ax.legend()
            
            # Add value labels on top of each bar
            for i, metric in enumerate(metrics):
                for j, algorithm in enumerate(['Dijkstra', 'A*']):
                    value = metrics[metric][j]
                    position = x[i] - bar_width/2 if j == 0 else x[i] + bar_width/2
                    if metric == 'Travel Time (s)' or metric == 'Path Length (nodes)':
                        # Format as integer for path length and travel time
                        ax.text(position, value + (max(metrics[metric]) * 0.02), 
                            f"{value:.0f}", ha='center', va='bottom')
                    else:
                        # Format with more precision for computation time
                        ax.text(position, value + (max(metrics[metric]) * 0.02), 
                            f"{value:.5f}", ha='center', va='bottom')
            
            # Calculate improvement percentages
            if metrics['Travel Time (s)'][0] > 0:
                travel_improvement = ((metrics['Travel Time (s)'][0] - metrics['Travel Time (s)'][1]) / 
                                    metrics['Travel Time (s)'][0]) * 100
                ax.text(0, metrics['Travel Time (s)'][0] * 1.15, 
                    f"Improvement: {travel_improvement:.1f}%", 
                    ha='center', color='green', fontweight='bold')
            
            if metrics['Computation Time (s)'][0] > 0:
                comp_improvement = ((metrics['Computation Time (s)'][0] - metrics['Computation Time (s)'][1]) / 
                                metrics['Computation Time (s)'][0]) * 100
                ax.text(1, metrics['Computation Time (s)'][0] * 1.15, 
                    f"Improvement: {comp_improvement:.1f}%", 
                    ha='center', color='green', fontweight='bold')
            
            plt.tight_layout()
            return fig, ax
        except Exception as e:
            print(f"Error in plot_performance_comparison: {str(e)}")
            # Create a simple fallback figure with error message
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f"Error generating performance comparison: {str(e)}", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=12)
            return fig, ax