# Fastest Emergency Response Route Finder
## Final Project Report for CS5800: Algorithms

### Team Members
- Chenxi Zhao
- Dominic Ejiogu
- Carlos Semeho Edorh


## Executive Summary
This project implements and compares Dijkstra's algorithm and A* search for emergency vehicle routing in Vancouver, BC. We integrated real-time traffic data and simulated various scenarios to evaluate algorithm performance. A* consistently outperformed Dijkstra's algorithm in both travel time and computation efficiency, with improvements ranging from X% to Y% depending on traffic conditions.

## 1. Introduction
Emergency response time is critical in life-or-death situations. This project addresses this challenge by developing intelligent routing algorithms that account for real-time traffic data to help emergency vehicles reach their destinations as quickly as possible. By leveraging graph theory, real-time data sources, and sophisticated algorithms, we aim to improve response times and potentially save lives.

## 2. Methodology
### 2.1 Algorithmic Approach
[Detailed explanation of Dijkstra's and A* implementations, incorporating material from project-details.md]

### 2.2 Data Collection & Processing
[Details about OpenStreetMap, TomTom Traffic API, and Vancouver Open Data Portal usage]

### 2.3 Technical Implementation
[Explanation of graph representation, technology stack, and simulation environment]

### 2.4 Ethical Considerations & Limitations
[Discussion of privacy protection, fairness, equity, and methodological limitations]

## 3. Results
### 3.1 Algorithm Performance Comparison
[Include table comparing Dijkstra vs. A* across scenarios]

### 3.2 Scenario Analysis
[Analysis of different traffic scenarios and their impact on routing]

### 3.3 Visualization
[Include visualizations of route comparisons]

## 4. Discussion & Conclusion
[Interpretation of results, significance, limitations, and future work]

## 5. References
[List of data sources, APIs, and literature referenced]

## Appendix
[Technical details, additional visualizations, code snippets]


```

pip install -r requirements.txt

python main.py

```
 

# ========== Please Note the following ===========.


If any issues later please run the following explicitly

`pip install python-dotenv`
`pip install tkintermapview==1.29`



On using `data_processor_o.py`, `astar_o.py`, `dijkstra_o.py` in their normal files ie. `data_processor.py`, `astar.py`, `dijkstra.py`

you will get the following console and it's a harmless, but it works much more faster because i am combining the data from the cache
directly to do the computation, no API calls at the processing level.

```
Warning: Error processing edge (553506506, 348069014, 0): unhashable type: 'list' - Geometry type: <class 'shapely.geometry.linestring.LineString'>
Warning: Error processing edge (430037812, 320093525, 0): unhashable type: 'list' - Geometry type: <class 'shapely.geometry.linestring.LineString'>

```

Compared to after adopting Zhao' algorithms into the following files `data_processor_Zhao.py`, `astar_Zhao.py`, `dijkstra_Zhao.py`   where i am directly updating the graph with traffic data from TomTom API during the proccessing, if the API call fails then it results to the cache for the Updates and it works faster but if it uses the API, it takes some time to do the updating.You may see the console in that respect.


Please if you see the following complains, just wait for some seconds for cache re-routing...

```
Simulating rush hour traffic conditions...
Updating graph with traffic data...
Error 403: <h1>Call blocked. You went over the allowed limit.</h1>
```


CURRENTLY ZHAO'S IMPLEMENTATIONS ARE IN THE  `data_processor.py`, `astar.py`, `dijkstra.py` FILES


The report Test cases are in the tests folder, you can cross check them out by running
different tests on your end as well, and please add your results to the tests folder
with the naming convertion `comparism-results-......txt`  to be consistent with the rest for easy reference. 