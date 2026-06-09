import pandas as pd
import numpy as np
import heapq
from collections import deque

class METRALARoutingGraph:
    def __init__(self, csv_path):
        self.adj = {}
        self.load_la_topology(csv_path)
        
    def load_la_topology(self, csv_path):
        df = pd.read_csv(csv_path)
        df.columns = ['from', 'to', 'cost']
        for _, row in df.iterrows():
            try:
                u, v, cost = int(row['from']), int(row['to']), float(row['cost'])
                if u not in self.adj: self.adj[u] = []
                self.adj[u].append((v, cost))
            except ValueError:
                continue
        print(f"[Graph System] Mapped LA Highway Topology. Active Junction Vertices: {len(self.adj)}")

def load_sensor_locations(csv_path='graph_sensor_locations.csv'):
    print("[Graph System] Ingesting latitude and longitude metrics...")
    df_locs = pd.read_csv(csv_path)
    la_node_coords = {}
    for i in range(len(df_locs)):
        row = df_locs.iloc[i]
        try:
            sensor_id, latitude, longitude = int(row.iloc[1]), float(row.iloc[2]), float(row.iloc[3])
            la_node_coords[sensor_id] = (latitude, longitude)
        except (ValueError, IndexError, TypeError):
            try:
                sensor_id, latitude, longitude = int(row.iloc[0]), float(row.iloc[1]), float(row.iloc[2])
                la_node_coords[sensor_id] = (latitude, longitude)
            except:
                continue
    print(f"-> Successfully mapped coordinates for {len(la_node_coords)} highway sensor hubs.")
    return la_node_coords

def execute_static_dijkstra(graph, source, target, la_node_coords):
    pq = [(0.0, source)]
    best_costs = {source: 0.0}
    routing_trace = {source: None}
    while pq:
        curr_cost, u = heapq.heappop(pq)
        if u == target: break
        if curr_cost > best_costs.get(u, float('inf')): continue
        for neighbor, base_dist in graph.adj.get(u, []):
            if neighbor not in la_node_coords: continue
            projected_cost = curr_cost + base_dist
            if projected_cost < best_costs.get(neighbor, float('inf')):
                best_costs[neighbor] = projected_cost
                routing_trace[neighbor] = u
                heapq.heappush(pq, (projected_cost, neighbor))
    if target not in best_costs: return None, float('inf')
    path, trace_node = [], target
    while trace_node is not None:
        path.append(trace_node)
        trace_node = routing_trace[trace_node]
    return path[::-1], best_costs[target]

def execute_time_dependent_route(graph, ml_model, source, target, departure_hour, la_node_coords):
    pq = [(departure_hour, source)]
    best_arrival_times = {source: departure_hour}
    routing_trace = {source: None}
    sensor_id_list = sorted(list(la_node_coords.keys()))
    sensor_to_idx = {sid: idx for idx, sid in enumerate(sensor_id_list)}
    
    while pq:
        curr_hour, u = heapq.heappop(pq)
        if u == target: break
        if curr_hour > best_arrival_times.get(u, float('inf')): continue
        for neighbor, base_dist in graph.adj.get(u, []):
            if neighbor not in la_node_coords: continue
            current_cyclic_hour = curr_hour % 24
            s_idx = sensor_to_idx.get(u, 0)
            
            h_sin = np.sin(2 * np.pi * current_cyclic_hour / 24.0)
            h_cos = np.cos(2 * np.pi * current_cyclic_hour / 24.0)
            day_of_week_val = 0 
            
            pred_features = np.array([[h_sin, h_cos, day_of_week_val, s_idx, 55.0, 55.0]])
            predicted_speed = ml_model.predict(pred_features,validate_features=False)[0]
            
            congestion_penalty = 60.0 / max(predicted_speed, 8.0)
            travel_time_hours = (base_dist / 1000.0 / 1.609) * (congestion_penalty / 60.0)
            projected_arrival_hour = curr_hour + travel_time_hours
            
            if projected_arrival_hour < best_arrival_times.get(neighbor, float('inf')):
                best_arrival_times[neighbor] = projected_arrival_hour
                routing_trace[neighbor] = u
                heapq.heappush(pq, (projected_arrival_hour, neighbor))
                
    if target not in best_arrival_times: return None, float('inf')
    path, trace_node = [], target
    while trace_node is not None:
        path.append(trace_node)
        trace_node = routing_trace[trace_node]
    return path[::-1], best_arrival_times[target]

def discover_complex_la_paths(graph, la_node_coords):
    verified_available_nodes = sorted(list(set(graph.adj.keys()).intersection(set(la_node_coords.keys()))))
    for start in verified_available_nodes:
        visited = {start}
        queue = deque([start])
        while queue:
            curr = queue.popleft()
            if len(visited) > 12 and curr in verified_available_nodes:
                return start, curr
            for neighbor, _ in graph.adj.get(curr, []):
                if neighbor not in visited and neighbor in verified_available_nodes:
                    visited.add(neighbor)
                    queue.append(neighbor)
    return verified_available_nodes[0], verified_available_nodes[min(10, len(verified_available_nodes)-1)]
