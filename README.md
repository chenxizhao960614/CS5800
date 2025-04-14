# 🚗 Vancouver Real-Time Routing with TomTom API & OSMnx

This project computes optimal driving routes in Vancouver using real-time traffic and incident data from the TomTom API. It visualizes the computed route using OSMnx and NetworkX.

---

## 🔧 Features

- ⛽ Real-time traffic-based edge weight updates via TomTom Flow API
- ⚠️ Incident-aware routing (accidents, closures)
- 🧠 Smart routing: chooses between Dijkstra (short) and A* (long)
- 🗺️ Interactive route visualization with matplotlib
- 📍 Turn-by-turn direction generation

---

## 📦 Requirements

Install required Python packages:

```bash
pip install -r requirements.txt
```

## 🚀 How to Run
```bash
python vancouver_map.py
```
