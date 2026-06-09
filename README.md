# Predictive Traffic Routing via Time-Dependent Dijkstra Search

## Overview

This project combines **Machine Learning, Graph Algorithms, and Transportation Intelligence** to perform congestion-aware route optimization.

Traditional routing systems assume static edge costs and compute shortest paths using fixed distances. In reality, traffic conditions vary continuously, making static routes suboptimal during peak hours. This project extends classical **Dijkstra's Algorithm** by introducing **time-dependent edge weights** generated from a traffic forecasting model.

The system learns temporal traffic dynamics from historical observations and predicts future traffic conditions for a given departure time. These predictions are then used to dynamically adjust traversal costs across the road network, enabling the routing engine to compute routes that minimize expected travel time rather than physical distance.

---

## Dynamic Routing Framework

### Traffic Forecasting Layer

A machine learning model is trained on historical traffic observations to capture spatio-temporal traffic patterns and predict expected traffic speeds under varying conditions throughout the day.

### Time-Dependent Graph Search

Predicted traffic speeds are transformed into dynamic edge traversal costs, converting the static road network into a time-varying weighted graph. A modified Dijkstra search then computes the optimal route by minimizing expected arrival time instead of geographic distance.

This enables the routing engine to adapt route selection based on anticipated congestion levels, producing different routes for identical origin-destination pairs when departure times change.

---

## Key Technical Contributions

* Developed a spatio-temporal traffic prediction pipeline for forecasting future traffic conditions.
* Implemented a time-dependent variant of Dijkstra's Algorithm with dynamically generated edge weights.
* Modeled transportation networks using graph-based representations for route optimization.
* Integrated predictive traffic intelligence with shortest-path search to generate congestion-aware routes.
* Built an interactive dashboard for route simulation, visualization, and comparison against classical static routing approaches.
