# METR-LA Spatio-Temporal Predictive Routing Engine

A modular, production-ready routing engine that integrates XGBoost-based spatio-temporal speed prediction with time-dependent Dijkstra graph search algorithms.

## Project Structure
- `src/model_training.py`: Feature engineering and XGBoost regressor training pipeline.
- `src/routing.py`: Graph topology loading, static Dijkstra, and time-dependent ML routing algorithms.
- `src/app.py`: Professional Gradio dashboard for interactive route visualization and analytics.

## Setup
1. Create a virtual environment: `python -m venv venv`
2. Activate it and install dependencies: `pip install -r requirements.txt`
3. Place your data files (`metr-la.h5`, `distances_la_2012.csv`, `graph_sensor_locations.csv`) in the root directory.
4. Train the model (if not already present): `python -c "from src.model_training import train_metr_la_model; train_metr_la_model()"`
5. Launch the dashboard: `python src/app.py`
