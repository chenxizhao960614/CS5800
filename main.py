import os
import sys
from dotenv import load_dotenv
from data.data_loader import DataLoader
from data.data_processor import DataProcessor
from src.algorithms.dijkstra import DijkstraRouter
from src.algorithms.astar import AStarRouter
from src.visualization.visualizer import RouteVisualizer
from src.visualization.point_selector import PointSelector
from src.simulation.scenarios import ScenarioGenerator
import matplotlib.pyplot as plt
import customtkinter

def find_free_port(start_port=5500, max_attempts=10):
    """Find an available port starting from start_port"""
    import socket
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    return None

def main():
    # Set appearance mode
    customtkinter.set_appearance_mode("light")
    customtkinter.set_default_color_theme("blue")
    
    # Load environment variables (if any)
    load_dotenv()
    
    print("Emergency Route Finder")
    print("=====================")
    
    try:
        # Initialize components
        data_loader = DataLoader()
        
        # Load a cached graph if it exists, otherwise create new one
        cache_file = "data/processed/road_network.json"
        if os.path.exists(cache_file):
            print("\nLoading cached road network...")
            G = data_loader.load_graph_json(cache_file)
        else:
            print("\nLoading fresh road network from OpenStreetMap...")
            G = data_loader.load_osm_graph("UBC, Vancouver, Canada")
            print("Saving road network for future use...")
            data_loader.save_graph_json(G, cache_file)
        
        # Initialize data processor and scenario generator
        data_processor = DataProcessor(data_loader)
        scenario_gen = ScenarioGenerator(G, data_processor, data_loader)
        
        print("\nSimulating rush hour traffic conditions...")
        # Use rush hour scenario for more visible traffic congestion
        G = scenario_gen.rush_hour_scenario()
        
        # Add some random incidents to create more congestion points
        G = scenario_gen.incident_scenario(num_incidents=10)
        
        # Initialize routers with the traffic-updated graph
        dijkstra_router = DijkstraRouter(G)
        astar_router = AStarRouter(G)
        
        # Create visualizer for the network
        visualizer = RouteVisualizer(G)
        
        # Create point selector and get user input
        print("\nOpening map for point selection...")
        print("Please follow these steps:")
        print("1. Wait for the map window to load completely")
        print("2. Click once to set your starting point")
        print("3. Click again to set your destination")
        print("4. Click 'Confirm Selection' when done")
        
        selector = PointSelector(G)
        source, target = selector.show_map_selector()
        
        if source is None or target is None:
            print("\nError: Start and end points must be selected.")
            return
        
        # Run both algorithms
        print("\nCalculating optimal routes...")
        dijkstra_result = dijkstra_router.find_route(source, target)
        astar_result = astar_router.find_route(source, target)
        
        if dijkstra_result and astar_result:
            # Print results
            print("\n=== Route Comparison ===")
            print("\nDijkstra's Algorithm:")
            print(f"- Path length: {len(dijkstra_result['path'])} nodes")
            print(f"- Estimated travel time: {dijkstra_result['travel_time']:.2f} seconds")
            print("\nA* Algorithm:")
            print(f"- Path length: {len(astar_result['path'])} nodes")
            print(f"- Estimated travel time: {astar_result['travel_time']:.2f} seconds")
            
            # Create visualization
            print("\nGenerating route visualization...")
            
            # Create comparison plot
            # fig, ax = visualizer.plot_comparison(
            #     dijkstra_result['path'], 
            #     astar_result['path'],
            #     "Emergency Route Comparison"
            # )
            # plt.savefig('data/processed/route_comparison.png', bbox_inches='tight', dpi=300)
            # plt.show()
            # plt.close(fig)


            # ============= Create a detailed route visualization============

            # 1. Create route map visualization
            try:
                fig_route, ax_route = visualizer.plot_comparison(
                    dijkstra_result['path'], 
                    astar_result['path'],
                    "Emergency Route Comparison"
                )
                plt.figure(fig_route.number)  # Ensure this figure is active
                plt.savefig('data/processed/route_comparison.png', bbox_inches='tight', dpi=300)
                plt.show()
                plt.close(fig_route)
            except Exception as e:
                print(f"Error creating route visualization: {e}")
            
            # 2. Create performance comparison chart
            try:
                fig_perf, ax_perf = visualizer.plot_performance_comparison(
                    dijkstra_result,
                    astar_result,
                    "Algorithm Performance Comparison"
                )
                plt.figure(fig_perf.number)  # Ensure this figure is active
                plt.savefig('data/processed/performance_comparison.png', bbox_inches='tight', dpi=300)
                plt.show()
                plt.close(fig_perf)
            except Exception as e:
                print(f"Error creating performance visualization: {e}")

            # ============= Create a detailed route visualization============




            
            # Create interactive map
            route_data = {
                'dijkstra_path': dijkstra_result['path'],
                'astar_path': astar_result['path'],
                'dijkstra_time': dijkstra_result['travel_time'],
                'astar_time': astar_result['travel_time']
            }
            
            interactive_map = visualizer.create_interactive_map(route_data)
            
            # Save and serve the interactive map
            map_file = 'data/processed/interactive_map.html'
            interactive_map.save(map_file)
            
            # Start a simple HTTP server to serve the map
            import http.server
            import socketserver
            import threading
            import webbrowser
            import socket
            
            # Find an available port
            PORT = find_free_port()
            if PORT is None:
                print("\nError: Could not find an available port. Please try again later.")
                return 1
            
            server = None
            def start_server():
                nonlocal server
                Handler = http.server.SimpleHTTPRequestHandler
                try:
                    server = socketserver.TCPServer(("", PORT), Handler)
                    print(f"\nServing map at http://localhost:{PORT}/data/processed/interactive_map.html")
                    server.serve_forever()
                except Exception as e:
                    print(f"\nServer error: {e}")
            
            # Start server in a separate thread
            server_thread = threading.Thread(target=start_server, daemon=True)
            server_thread.start()
            
            # Give the server a moment to start
            import time
            time.sleep(1)
            
            # Open in browser
            try:
                webbrowser.open(f'http://localhost:{PORT}/data/processed/interactive_map.html')
            except Exception as e:
                print(f"\nError opening browser: {e}")
            
            print("\nRoute planning completed successfully!")
            print("\nVisualizations have been saved to:")
            print("- data/processed/route_comparison.png")
            print("- data/processed/interactive_map.html")
            
            # Wait for user input before closing
            input("\nPress Enter to exit...")
            
            # Clean up server if it was started
            if server:
                try:
                    server.shutdown()
                    server.server_close()
                except Exception as e:
                    print(f"\nError shutting down server: {e}")
            
            return 0
        else:
            print("\nError: Could not find valid routes between the selected points.")
            return 1
            
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)