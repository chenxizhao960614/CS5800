import tkinter as tk
from tkinter import ttk
import osmnx as ox
import networkx as nx
import customtkinter
from PIL import Image
import tkintermapview
import threading
import queue
import time

class PointSelector:
    def __init__(self, G):
        self.G = G
        self.start_point = None
        self.end_point = None
        self.markers = []
        self.thread_running = False
        self.window_open = False
        self.message_queue = queue.Queue()
        self.root = None
        self.map_widget = None
        self.status_label = None
        self.confirm_btn = None
        self.reset_btn = None
        self.load_thread = None
        customtkinter.set_appearance_mode("light")
        customtkinter.set_default_color_theme("blue")
        
    def show_map_selector(self):
        try:
            # Create the main window
            self.root = customtkinter.CTk()
            self.root.title("Emergency Route Selection")
            self.root.geometry("1000x700")
            self.window_open = True
            
            # Prevent window from being destroyed accidentally
            self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
            
            # Create main frame
            main_frame = customtkinter.CTkFrame(self.root)
            main_frame.pack(fill="both", expand=True, padx=10, pady=10)
            
            # Add instructions
            instructions = customtkinter.CTkLabel(
                main_frame,
                text="Select Start and End Points",
                font=customtkinter.CTkFont(size=16, weight="bold")
            )
            instructions.pack(pady=5)
            
            # Create map frame
            map_frame = customtkinter.CTkFrame(main_frame)
            map_frame.pack(fill="both", expand=True, padx=5, pady=5)
            
            # Create map widget
            self.map_widget = tkintermapview.TkinterMapView(
                map_frame, 
                width=900,
                height=500,
                corner_radius=0
            )
            self.map_widget.pack(fill="both", expand=True)
            
            # Get center point of the graph
            center_y = sum(nx.get_node_attributes(self.G, 'y').values()) / len(self.G)
            center_x = sum(nx.get_node_attributes(self.G, 'x').values()) / len(self.G)
            
            # Set initial map position
            self.map_widget.set_position(center_y, center_x)
            self.map_widget.set_zoom(14)
            
            # Status label
            self.status_label = customtkinter.CTkLabel(
                main_frame,
                text="Loading map...",
                font=customtkinter.CTkFont(size=12)
            )
            self.status_label.pack(pady=5)
            
            # Button frame
            button_frame = customtkinter.CTkFrame(main_frame)
            button_frame.pack(pady=10)
            
            # Add buttons
            self.confirm_btn = customtkinter.CTkButton(
                button_frame,
                text="Confirm Selection",
                command=self._on_confirm,
                width=120,
                state="disabled"  # Initially disabled
            )
            self.confirm_btn.pack(side="left", padx=5)
            
            self.reset_btn = customtkinter.CTkButton(
                button_frame,
                text="Reset",
                command=self._reset_points,
                width=120,
                state="disabled"  # Initially disabled
            )
            self.reset_btn.pack(side="left", padx=5)
            
            # Bind map click event
            self.map_widget.add_left_click_map_command(self._on_map_click)
            
            # Start loading roads in background
            self.thread_running = True
            self.load_thread = threading.Thread(target=self._add_major_roads)
            self.load_thread.daemon = True
            self.load_thread.start()
            
            # Start periodic updates
            self._check_queue()
            
            # Start GUI main loop
            self.root.mainloop()
            
            return self.start_point, self.end_point
            
        except Exception as e:
            print(f"Error initializing map window: {e}")
            if self.root:
                self.root.destroy()
            return None, None
    
    def _check_queue(self):
        """Check message queue and update UI"""
        if not self.window_open:
            return
            
        try:
            while True:
                message = self.message_queue.get_nowait()
                if message.get('type') == 'status':
                    self.status_label.configure(text=message['text'])
                elif message.get('type') == 'enable_buttons':
                    self.confirm_btn.configure(state="normal")
                    self.reset_btn.configure(state="normal")
                elif message.get('type') == 'add_road':
                    self.map_widget.set_path(
                        message['path'],
                        width=message['width'],
                        color=message['color']
                    )
        except queue.Empty:
            pass
        
        if self.window_open and self.root:
            self.root.after(100, self._check_queue)
    
    def _add_major_roads(self):
        """Add major roads to the map in a background thread"""
        try:
            added = 0
            total_roads = sum(1 for _, _, data in self.G.edges(data=True) 
                            if 'highway' in data and data['highway'] in ['motorway', 'trunk', 'primary', 'secondary'])
            
            for u, v, data in self.G.edges(data=True):
                if not self.thread_running:
                    break
                    
                if 'highway' in data and data['highway'] in ['motorway', 'trunk', 'primary', 'secondary']:
                    if 'geometry' in data:
                        xs, ys = data['geometry'].xy
                        path = [(y, x) for x, y in zip(xs, ys)]
                    else:
                        u_y = self.G.nodes[u]['y']
                        u_x = self.G.nodes[u]['x']
                        v_y = self.G.nodes[v]['y']
                        v_x = self.G.nodes[v]['x']
                        path = [(u_y, u_x), (v_y, v_x)]
                    
                    width = 3 if data['highway'] in ['motorway', 'trunk'] else 2
                    
                    # Queue the road addition
                    self.message_queue.put({
                        'type': 'add_road',
                        'path': path,
                        'width': width,
                        'color': "#404040"
                    })
                    
                    added += 1
                    if added % 10 == 0:
                        self.message_queue.put({
                            'type': 'status',
                            'text': f"Loading road network... ({added}/{total_roads})"
                        })
                        time.sleep(0.01)  # Give the UI time to update
            
            # Enable buttons and update status when done
            if self.thread_running:
                self.message_queue.put({
                    'type': 'status',
                    'text': "Click on the map to set starting point"
                })
                self.message_queue.put({'type': 'enable_buttons'})
            
        except Exception as e:
            print(f"Error loading roads: {e}")
            if self.window_open:
                self.message_queue.put({
                    'type': 'status',
                    'text': f"Error loading roads: {str(e)}"
                })
        finally:
            self.thread_running = False
    
    def _on_map_click(self, coords):
        """Handle map clicks"""
        if not self.window_open or len(self.markers) >= 2:
            return
        
        lat, lon = coords
        
        try:
            if len(self.markers) == 0:
                # First click - set start point
                marker = self.map_widget.set_marker(lat, lon, text="Start")
                self.markers.append(marker)
                self.status_label.configure(text="Click to set destination point")
            else:
                # Second click - set end point
                marker = self.map_widget.set_marker(lat, lon, text="End")
                self.markers.append(marker)
                self.status_label.configure(text="Points selected. Click Confirm when ready.")
        except Exception as e:
            print(f"Error setting marker: {e}")
            self.status_label.configure(text="Error setting point. Please try again.")
    
    def _reset_points(self):
        """Reset selected points"""
        if not self.window_open:
            return
            
        for marker in self.markers:
            marker.delete()
        self.markers = []
        self.status_label.configure(text="Click to set starting point")
    
    def _on_confirm(self):
        """Handle confirmation"""
        if len(self.markers) != 2:
            self.status_label.configure(text="Please select both points first")
            return
        
        try:
            # Get coordinates from markers
            start_coords = self.markers[0].position
            end_coords = self.markers[1].position
            
            # Find nearest nodes in the graph
            start_node = ox.nearest_nodes(self.G, start_coords[1], start_coords[0])
            end_node = ox.nearest_nodes(self.G, end_coords[1], end_coords[0])
            
            self.start_point = start_node
            self.end_point = end_node
            
            # Clean up any existing route paths
            if hasattr(self, 'route_paths'):
                for path in self.route_paths:
                    self.map_widget.delete(path)
            self.route_paths = []
            
            # Close the window and return selected points
            self.window_open = False
            self.root.quit()
            self.root.destroy()
            
        except Exception as e:
            print(f"Error confirming selection: {e}")
            self.status_label.configure(text="Error confirming points. Please try again.")
            return
    
    def _on_closing(self):
        """Handle window closing"""
        self.thread_running = False
        self.window_open = False
        if self.root:
            self.root.destroy()