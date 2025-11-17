# A Multiscale Hexagonal Framework for Urban Walkability Analysis using Open Geospatial Data

[![Python 3.12.5](https://img.shields.io/badge/python-3.12.5-blue.svg)](https://www.python.org/downloads/)

[![IEEE Big Data 2025](https://img.shields.io/badge/IEEE-Big%20Data%202025-red.svg)](https://bigdataieee.org/)

## Installation

### Prerequisites

- Python 3.12.5

- pip package manager

- Virtual environment (recommended)

### Setup

```bash

# Clone the repository

git  clone  https://github.com/yourusername/urban-walkability.git

cd  urban-walkability



# Create and activate virtual environment

python  -m  venv  venv

source  venv/bin/activate  # On Windows: venv\Scripts\activate



# Install dependencies

pip  install  -r  requirements.txt

```

### System Requirements

- **RAM:** 4GB minimum, 16GB recommended

- **Internet:** Required for first-time OSM data download

---

## Output Data

### 1. H3 Hexagon Metrics

File: `data/csv/h3_info/{location}_h3_hexagons_res{resolution}_{profile}.csv`

**Columns:**

- `sequential_id`: Hexagon sequential identifier

- `h3_id`: Uber H3 hexagon ID (global unique)

- `latitude`, `longitude`: Centroid coordinates (WGS84)

- `coord_x_proj`, `coord_y_proj`: Projected coordinates

- `resolution`: H3 resolution level

- `area_m2`: Hexagon area (square meters)

- `area_vegetation_m2`: Vegetation coverage area

- `area_water_m2`: Water body coverage area

- `percent_vegetation`: Vegetation percentage

- `percent_water`: Water body percentage

- `count_pois_accessible`: Total accessible POIs (≤20 min)

- `pois_restaurant`, `pois_hospital`, ... : Count by category

- `avg_attractiveness_threshold_pois`: Average POI attractiveness

- `count_crosswalks_accessible`: Accessible crosswalks

- `count_traffic_signals_accessible`: Accessible traffic signals

- `node_time_travel`: Travel time to nearest network node

- `edge_travel_time`: Travel time on intersecting edge

- `min_time_access`: Minimum access time (best of node/edge)

### 2. Accessible POIs Report

File: `data/csv/report/{location}_pois_acessiveis.csv`

**Columns:**

- `sequential_id`: POI identifier

- `name`: POI name (from OSM)

- `poi_type`: POI category

- `real_time_poi`: Total walking time (minutes)

- `attractiveness_threshold`: Attractiveness score [0-1]

- `access_interval`: Time interval category (0-5 min, 5-10 min, ...)

- `addr:street`, `addr:housenumber`: Address information

- `latitude`, `longitude`: POI coordinates

### 3. Visualization Outputs

Generated maps in `data/visualizations/`:

- `basic_map_{location}.png`: Base map with street network

- `map_isochrones_{location}_{profile}.png`: Isochrone visualization

- `map_isochrones_pois_{location}_{profile}.png`: Isochrones + POIs

- `map_isochrones_pois_h3_{location}_{profile}.png`: Full analysis map

---

## Performance Optimization

The system implements several optimization techniques for large-scale analysis:

### 1. Persistent Caching

OSM data is cached to disk after first download:

```python

cache_file =  f"data/cache/pois_{lat}_{lon}_{dist}_{crs}.pkl"

```

**Benefits:**

- 40-80x speedup on subsequent runs

- 95% reduction in API requests

- Enables reproducible research

### 2. Spatial Indexing

R-tree spatial indexes accelerate geometric queries:

```python

sindex = gdf_pois.sindex # O(log n) queries

possible_idx = sindex.query(hexagon, predicate='intersects')

```

**Benefits:**

- 8-15x speedup for spatial queries

- O(log n + k) vs O(n) complexity

- Essential for large datasets (1000+ features)

### 3. LRU Caching

In-memory cache for repeated conversions:

```python

@lru_cache(maxsize=2048)

def  h3_to_polygon_cached(h3_index:  str, target_crs:  str) ->  str:

# Cached H3 → Polygon conversion

```

**Benefits:**

- 15-30% speedup in hexagon processing loop

- Automatic memory management

- Thread-safe implementation

---

## Citation

If you use this code in your research, please cite:

```bibtex

@inproceedings{XXXXX,

title={A Multiscale Hexagonal Framework for Urban Walkability Analysis using Open Geospatial Data},

author={Anderson Tadeu de Oliveira Vicente and Flavio Soares Correa da Silva},

booktitle={XXXXX},

year={2025},

organization={IEEE}

}

```

**Related Publications:**

- Boeing, G. (2017). OSMnx: New methods for acquiring, constructing, analyzing, and visualizing complex street networks. _Computers, Environment and Urban Systems_, 65, 126-139.

- Uber Technologies, Inc. (2018). H3: A Hexagonal Hierarchical Geospatial Indexing System. [https://h3geo.org/](https://h3geo.org/)

---

## Contact

**Principal Investigator:** Anderson Tadeu de Oliveira Vicente (PhD Student)

**Email:** ander.oliveira@ime.usp.br

**Institution:** University of São Paulo (USP)

**Conference:** IEEE Big Data 2025

**Project Links:**

- [GitHub Repository](https://github.com/ander-oliveira/urban-walkability)
