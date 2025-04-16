# src/simulation/simulator.py
import random
import time
import pandas as pd
import numpy as np
import networkx as nx
import osmnx as ox

class EmergencySimulator:
    def __init__(self, G, emergency_stations=None):
        self.G = G
        
        # If emergency stations are not provided, use random nodes
        if emergency_stations is None:
            # Randomly select 5 nodes as emergency stations
            self.emergency_stations = random.sample(list(G.nodes()), 5)
        else:
            self.emergency_stations = emergency_stations
            
        self.scenarios = []
        self.results = pd.DataFrame()
        
    def generate_scenarios(self, num_scenarios=10, seed=42):
        """Generate random emergency scenarios"""
        random.seed(seed)
        np.random.seed(seed)
        
        scenarios = []
        
        for i in range(num_scenarios):
            # Pick a random emergency station as the source
            source = random.choice(self.emergency_stations)
            
            # Pick a random node as the emergency location
            # Make sure it's not an emergency station itself
            target_candidates = list(set(self.G.nodes()) - set(self.emergency_stations))
            target = random.choice(target_candidates)
            
            # Create a scenario
            scenario = {
                'id': i,
                'source': source,
                'target': target,
                'source_name': self.G.nodes[source].get('name', f"node_{source}"),
                'target_name': self.G.nodes[target].get('name', f"node_{target}")
            }
            
            scenarios.append(scenario)
            
        self.scenarios = scenarios
        return scenarios
    
    def run_simulation(self, dijkstra_router, astar_router):
        """Run simulation comparing Dijkstra and A* on all scenarios"""
        results = []
        
        for scenario in self.scenarios:
            source = scenario['source']
            target = scenario['target']
            
            # Run Dijkstra's algorithm
            dijkstra_result = dijkstra_router.find_route(source, target)
            
            # Run A* algorithm
            astar_result = astar_router.find_route(source, target)
            
            if dijkstra_result and astar_result:
                # Record results
                result = {
                    'scenario_id': scenario['id'],
                    'source': source,
                    'target': target,
                    'source_name': scenario['source_name'],
                    'target_name': scenario['target_name'],
                    'dijkstra_distance': dijkstra_result['distance'],
                    'dijkstra_travel_time': dijkstra_result['travel_time'],
                    'dijkstra_computation_time': dijkstra_result['computation_time'],
                    'astar_distance': astar_result['distance'],
                    'astar_travel_time': astar_result['travel_time'],
                    'astar_computation_time': astar_result['computation_time'],
                    'dijkstra_path': dijkstra_result['path'],
                    'astar_path': astar_result['path'],
                    'same_path': dijkstra_result['path'] == astar_result['path']
                }
                
                results.append(result)
        
        # Convert results to DataFrame
        self.results = pd.DataFrame(results)
        return self.results
    
    def analyze_results(self):
        """Analyze simulation results"""
        if self.results.empty:
            print("No results to analyze. Run simulation first.")
            return {}
        
        analysis = {
            'total_scenarios': len(self.results),
            'avg_dijkstra_travel_time': self.results['dijkstra_travel_time'].mean(),
            'avg_astar_travel_time': self.results['astar_travel_time'].mean(),
            'avg_dijkstra_computation_time': self.results['dijkstra_computation_time'].mean(),
            'avg_astar_computation_time': self.results['astar_computation_time'].mean(),
            'same_path_percentage': (self.results['same_path'].sum() / len(self.results)) * 100,
            'astar_improvement_percentage': 
                ((self.results['dijkstra_travel_time'].mean() - self.results['astar_travel_time'].mean()) / 
                 self.results['dijkstra_travel_time'].mean()) * 100
        }
        
        print("\n=== Simulation Results ===")
        print(f"Total scenarios: {analysis['total_scenarios']}")
        print(f"Average Dijkstra travel time: {analysis['avg_dijkstra_travel_time']:.2f} seconds")
        print(f"Average A* travel time: {analysis['avg_astar_travel_time']:.2f} seconds")
        print(f"Average Dijkstra computation time: {analysis['avg_dijkstra_computation_time']:.5f} seconds")
        print(f"Average A* computation time: {analysis['avg_astar_computation_time']:.5f} seconds")
        print(f"Same path percentage: {analysis['same_path_percentage']:.2f}%")
        print(f"A* improvement: {analysis['astar_improvement_percentage']:.2f}%")
        
        return analysis
    


    def run_extreme_traffic_simulation(self, dijkstra_router, astar_router, num_scenarios=5):
        """Run simulation in extreme traffic conditions to show algorithm differences"""
        print("\nRunning extreme traffic simulation...")
        
        # Save original graph
        orig_graph = self.G
        
        # Create extreme traffic scenario
        from src.simulation.scenarios import ScenarioGenerator
        scenario_gen = ScenarioGenerator(self.G, None, None)  # We don't need data_processor and data_loader here
        
        # Add the extreme_traffic_scenario method to ScenarioGenerator if not present
        if not hasattr(scenario_gen, 'extreme_traffic_scenario'):
            # Use the updated extreme traffic scenario code here
            # [Copy the extreme_traffic_scenario method code here]
            print("WARNING: extreme_traffic_scenario method not available")
            return pd.DataFrame()
        
        # Generate extreme traffic graph
        extreme_G = scenario_gen.extreme_traffic_scenario(num_incidents=15)
        
        # Update routers with extreme traffic graph
        dijkstra_router.G = extreme_G
        astar_router.G = extreme_G
        
        # If no scenarios exist, generate some
        if not self.scenarios:
            self.generate_scenarios(num_scenarios)
        
        results = []
        
        # Use at most num_scenarios scenarios
        scenarios_to_run = self.scenarios[:num_scenarios]
        
        for scenario in scenarios_to_run:
            source = scenario['source']
            target = scenario['target']
            
            print(f"\nScenario {scenario['id']}: {source} → {target}")
            
            # Run Dijkstra's algorithm
            print("Running Dijkstra algorithm...")
            dijkstra_result = dijkstra_router.find_route(source, target)
            
            # Run A* algorithm
            print("Running A* algorithm...")
            astar_result = astar_router.find_route(source, target)
            
            if dijkstra_result and astar_result:
                # Calculate path overlap
                dijkstra_set = set(dijkstra_result['path'])
                astar_set = set(astar_result['path'])
                common_nodes = dijkstra_set.intersection(astar_set)
                overlap_pct = (len(common_nodes) / max(len(dijkstra_set), len(astar_set))) * 100
                
                # Record results
                result = {
                    'scenario_id': scenario['id'],
                    'source': source,
                    'target': target,
                    'dijkstra_travel_time': dijkstra_result['travel_time'],
                    'astar_travel_time': astar_result['travel_time'],
                    'dijkstra_path_length': len(dijkstra_result['path']),
                    'astar_path_length': len(astar_result['path']),
                    'path_overlap_pct': overlap_pct,
                    'time_diff': dijkstra_result['travel_time'] - astar_result['travel_time'],
                    'improvement_pct': ((dijkstra_result['travel_time'] - astar_result['travel_time']) / 
                                    dijkstra_result['travel_time']) * 100 if dijkstra_result['travel_time'] > 0 else 0
                }
                
                print(f"Dijkstra time: {dijkstra_result['travel_time']:.2f}s")
                print(f"A* time: {astar_result['travel_time']:.2f}s")
                print(f"Improvement: {result['improvement_pct']:.2f}%")
                print(f"Path overlap: {overlap_pct:.2f}%")
                
                results.append(result)
        
        # Restore original graph to routers
        dijkstra_router.G = orig_graph
        astar_router.G = orig_graph
        
        # Convert results to DataFrame
        extreme_results = pd.DataFrame(results)
        
        # Print summary
        print("\n=== Extreme Traffic Simulation Results ===")
        print(f"Average Dijkstra travel time: {extreme_results['dijkstra_travel_time'].mean():.2f}s")
        print(f"Average A* travel time: {extreme_results['astar_travel_time'].mean():.2f}s")
        print(f"Average improvement: {extreme_results['improvement_pct'].mean():.2f}%")
        print(f"Average path overlap: {extreme_results['path_overlap_pct'].mean():.2f}%")
        
        return extreme_results