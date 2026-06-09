import sys
import os
import gradio as gr
import folium

# Ensure local imports work when run directly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from routing import METRALARoutingGraph, load_sensor_locations, execute_static_dijkstra, execute_time_dependent_route, discover_complex_la_paths
from xgboost import XGBRegressor

MODEL_PATH = 'traffic_xgb_model.json'
GRAPH_PATH = 'distances_la_2012.csv'
LOCATIONS_PATH = 'graph_sensor_locations.csv'

def initialize_system():
    print("Loading ML Model...")
    ml_model = XGBRegressor()
    if os.path.exists(MODEL_PATH):
        ml_model.load_model(MODEL_PATH)
    else:
        print("Model not found. Please run model_training.py first.")
        
    print("Loading Graph Topology...")
    la_network = METRALARoutingGraph(GRAPH_PATH)
    
    print("Loading Sensor Coordinates...")
    la_node_coords = load_sensor_locations(LOCATIONS_PATH)
    
    p_src, p_dest = discover_complex_la_paths(la_network, la_node_coords)
    verified_available_nodes = sorted(list(set(la_network.adj.keys()).intersection(set(la_node_coords.keys()))))
    
    return ml_model, la_network, la_node_coords, verified_available_nodes, p_src, p_dest

ml_model, la_network, la_node_coords, verified_available_nodes, p_src, p_dest = initialize_system()

def run_la_dashboard(start_node, end_node, departure_hour):
    start_node, end_node, departure_hour = int(start_node), int(end_node), float(departure_hour)
    
    static_p, static_cost = execute_static_dijkstra(la_network, start_node, end_node, la_node_coords)
    ml_p, ml_arrival = execute_time_dependent_route(la_network, ml_model, start_node, end_node, departure_hour, la_node_coords)
    
    if static_p is None or ml_p is None:
        return "### ❌ Error\nSelected LA Highway sensor segments are disconnected or missing coordinates.", None
        
    static_duration_mins = (static_cost / 1000.0 / 1.609) * (60.0 / 55.0)  
    ml_duration_mins = (ml_arrival - departure_hour) * 60.0
    
    time_saved = max(0.0, static_duration_mins - ml_duration_mins)
    pct_improvement = (time_saved / max(1.0, static_duration_mins)) * 100.0 if static_duration_mins > ml_duration_mins else 0.0
    
    m_lat, m_lon = la_node_coords[start_node]
    mymap = folium.Map(location=[m_lat, m_lon], zoom_start=12, tiles="CartoDB positron")
    
    static_pts = [la_node_coords[n] for n in static_p if n in la_node_coords]
    folium.PolyLine(static_pts, color="#1f77b4", weight=6, opacity=0.7, tooltip="Static Baseline Route").add_to(mymap)
    
    ml_pts = [la_node_coords[n] for n in ml_p if n in la_node_coords]
    folium.PolyLine(ml_pts, color="#d62728", weight=3, opacity=0.9, tooltip="ML Congestion Avoidance Route", dash_array="6, 6").add_to(mymap)
    
    folium.Marker(la_node_coords[start_node], icon=folium.Icon(color="green", icon="play")).add_to(mymap)
    folium.Marker(la_node_coords[end_node], icon=folium.Icon(color="black", icon="flag")).add_to(mymap)
    
    report_md = f"""
    ### 📊 Los Angeles Freeway System Analytics
    
    | Routing Architecture Model | Trajectory Sensor Route Sequence | Total Expected Travel Time |
    | :--- | :--- | :--- |
    | **Standard Static Baseline** | `{static_p}` | **{static_duration_mins:.2f} Minutes** |
    | **Predictive ML-Optimized Routing** | `{ml_p}` | **{ml_duration_mins:.2f} Minutes** |
    
    ### ⚡ Decision Optimization Diagnostics
    * **Calculated Travel Time Saved:** `{time_saved:.2f} Minutes`
    * **Estimated Time Efficiency Improvement:** `{pct_improvement:.1f}%`
    
    *💡 **System Verification:** The LA freeway loop connections support active detours. Set the slider to peak evening rush hour (**17.5 / 5:30 PM**) to observe the predictive model branching onto alternative freeway corridors to avoid localized gridlock.*
    """
    return report_md, mymap._repr_html_()

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚗 METR-LA Spatio-Temporal Predictive Routing Engine")
    gr.Markdown("### Integrating Spatio-Temporal Predictive Speed Modeling with Time-Dependent Graph Search")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Constraints Manager")
            src_input = gr.Dropdown(choices=verified_available_nodes, value=p_src, label="Origin Sensor ID")
            dest_input = gr.Dropdown(choices=verified_available_nodes, value=p_dest, label="Destination Sensor ID")
            time_slider = gr.Slider(minimum=0.0, maximum=23.5, step=0.5, value=17.5, label="Departure Time Window (24h)")
            submit_btn = gr.Button("Compute Optimal Trajectories", variant="primary")
            
        with gr.Column(scale=2):
            metrics_panel = gr.Markdown("### 📊 Metrics Output\n*Trigger computation to stream telemetry matrices.*")
            map_panel = gr.HTML(value="<div style='text-align:center; padding:100px; color:gray;'>Map loading...</div>")
            
    submit_btn.click(fn=run_la_dashboard, inputs=[src_input, dest_input, time_slider], outputs=[metrics_panel, map_panel])

if __name__ == "__main__":
    demo.launch(debug=True, share=True)
