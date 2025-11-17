import csv
import json
import math
import os
import pickle
from functools import lru_cache
from multiprocessing import Pool, cpu_count
from typing import Tuple, Optional

import geopandas as gpd
import h3
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import rasterio
from matplotlib.lines import Line2D
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely import wkt
from shapely.geometry import Point, Polygon
from shapely.strtree import STRtree
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


def ensure_directory_exists(filepath: str) -> None:
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Directory created: {directory}")


def ensure_data_directories() -> None:
    """
    Creates all necessary standard directories for the project.
    """
    directories = [
        'data/csv/result',
        'data/csv/gdfs', 
        'data/csv/h3_info',
        'data/csv/report',
        'data/visualizations'
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"Directory created: {directory}")


def select_location() -> Tuple[Tuple[float, float], str, Optional[str]]:
    entries = []
    with open('data/csv/locations.csv', 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f, delimiter=';')
        next(csv_reader)
        for row in csv_reader:
            entries.append({
                    'address': row[0],
                    'location': row[1],
                    'coordinates': row[2],
                    'dem_file': row[3]
                })
            
    print("\nSelect a location:")
    for i, entry in enumerate(entries, 1):
        print(f"{i} - {entry['location'].upper()} - {entry['address']}")

    while True:
        try:
            choice = int(input("\nEnter option number: "))
            if 1 <= choice <= len(entries):
                break
            print(f"Please enter between 1-{len(entries)}")
        except ValueError:
            print("Invalid number. Try again.")

    selected = entries[choice-1]
    coords = selected['coordinates'].strip('()').split(',')
    central_point = (float(coords[0]), float(coords[1]))
    
    return central_point, selected['location'], selected.get('dem_file')


def save_nodes_and_edges(graph, location: str, id: str) -> None:
    """
    Saves the graph node and edge data to CSV files.
    """
    nodes, edges = ox.graph_to_gdfs(graph)
    
    nodes_file = f"data/csv/result/nodes_{location}_{id}.csv"
    edges_file = f"data/csv/result/edges_{location}_{id}.csv"

    ensure_directory_exists(nodes_file)
    ensure_directory_exists(edges_file)

    nodes.to_csv(nodes_file, index_label='node_id')
    edges.to_csv(edges_file, index_label='edge_id')
    
    print(f"Nodes saved in: {nodes_file}")
    print(f"Edges saved in: {edges_file}")


def save_features_to_csv(gdf: gpd.GeoDataFrame, location: str, feature_type: str) -> None:
    """
    Saves a GeoDataFrame of features to CSV with latitude and longitude coordinates.
    """
    if gdf.empty:
        print(f"No features of type '{feature_type}' found to save.")
        return
    
    gdf_to_save = gdf.copy()
    
    # Converting coordinates to lat/lon (EPSG:4326)
    if gdf_to_save.crs != 'EPSG:4326':
        gdf_coords = gdf_to_save.to_crs('EPSG:4326')
    else:
        gdf_coords = gdf_to_save
    
    gdf_to_save['latitude'] = gdf_coords.geometry.y
    gdf_to_save['longitude'] = gdf_coords.geometry.x
    
    # The geometry column is not compatible with CSV
    gdf_to_save = gdf_to_save.drop(columns=['geometry'])

    csv_file = f"data/csv/gdfs/{feature_type}_{location}.csv"
    
    ensure_directory_exists(csv_file)
    gdf_to_save.to_csv(csv_file, index=False, encoding='utf-8')


def reproject_dem(dem_path: str, target_crs) -> str:
    """
    Checks the DEM CRS and reprojects if necessary.
    Must be in the same form as the graph.
    """
    
    if not os.path.isabs(dem_path):
        dem_path = os.path.abspath(dem_path)
    
    try:
        with rasterio.open(dem_path) as src:
            dem_crs = src.crs
        
        if dem_crs != target_crs:
            dst_crs = target_crs
            with rasterio.open(dem_path) as src:
                transform, width, height = calculate_default_transform(
                    src.crs, dst_crs, src.width, src.height, *src.bounds
                )
                kwargs = src.meta.copy()
                kwargs.update({
                    'crs': dst_crs, 
                    'transform': transform, 
                    'width': width, 
                    'height': height
                })
                dem_dir = os.path.dirname(dem_path)
                dem_filename = os.path.basename(dem_path)
                reprojected_dem_path = os.path.join(dem_dir, f'reprojected_{dem_filename}')
                
                with rasterio.open(reprojected_dem_path, 'w', **kwargs) as dst:
                    for i in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, i), 
                            destination=rasterio.band(dst, i),
                            src_transform=src.transform, 
                            src_crs=src.crs,
                            dst_transform=transform, 
                            dst_crs=dst_crs,
                            resampling=Resampling.nearest
                        )
                
                print(f"DEM reprojected to graph CRS: {reprojected_dem_path}")
                return reprojected_dem_path
        else:
            print("DEM is already in the graph CRS.")
            return dem_path
    except Exception as e:
        print(f"ERROR processing DEM file '{dem_path}': {type(e).__name__}: {e}")
        return None


def get_green_and_water_areas(place: tuple, distance: float, target_crs) -> tuple:
    
    # Check cache first
    cache_dir = 'data/cache'
    os.makedirs(cache_dir, exist_ok=True)
    cache_key = f"green_water_{place[0]:.6f}_{place[1]:.6f}_{distance}_{target_crs}"
    cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
    
    if os.path.exists(cache_file):
        print("Loading green/water areas from cache...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    tags_green = {
        'landuse': ['allotments', 'farmland', 'meadow', 'greenfield', 'forest', 'grass', 'park', 'greenery'], 
        'leisure': ['park', 'garden', 'nature_reserve'],
        'natural': ['fell', 'tree', 'wood', 'scrub', 'grassland']
    }
    tags_water = {
        'natural': ['bay', 'beach', 'spring', 'water', 'wetland'], 
        'water': ['river', 'lake', 'canal', 'ditch', 'reservoir', 'lagoon'],
        'waterway': ['river', 'riverbank', 'stream', 'canal', 'waterfall']
    }
    
    gdf_green = ox.features_from_point(place, tags=tags_green, dist=distance).to_crs(target_crs)
    gdf_water = ox.features_from_point(place, tags=tags_water, dist=distance).to_crs(target_crs)
    
    # Save to cache
    with open(cache_file, 'wb') as f:
        pickle.dump((gdf_green, gdf_water), f)
    
    return gdf_green, gdf_water


def get_pois(place: tuple, distance: float, target_crs) -> gpd.GeoDataFrame:
    
    # Check cache first
    cache_dir = 'data/cache'
    os.makedirs(cache_dir, exist_ok=True)
    cache_key = f"pois_{place[0]:.6f}_{place[1]:.6f}_{distance}_{target_crs}"
    cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
    
    if os.path.exists(cache_file):
        print("Loading POIs from cache...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    tags_pois = {
        'amenity': [
            'bar', 'cafe', 'fast_food', 'restaurant', 'college', 'kindergarten', 
            'library', 'school', 'university', 'bicycle_parking', 'bank', 
            'clinic', 'dentist', 'doctors', 'hospital', 'pharmacy', 'cinema', 
            'theatre', 'gym', 'fitness_centre', 'fitness_center'
        ],
        'building': ['hotel', 'hospital', 'kindergarten', 'school', 'university', 'supermarket'],
        'shop': ['supermarket', 'convenience', 'bakery', 'greengrocer', 'grocery']
    }
    
    gdf_pois = ox.features_from_point(
        place, tags=tags_pois, dist=distance
    ).to_crs(target_crs)
    gdf_pois = gdf_pois[gdf_pois.geometry.geom_type == 'Point'].copy()
    
    # Save to cache
    with open(cache_file, 'wb') as f:
        pickle.dump(gdf_pois, f)
    
    return gdf_pois


def get_crosswalks(place: tuple, distance: float, target_crs) -> gpd.GeoDataFrame:
    
    # Check cache first
    cache_dir = 'data/cache'
    os.makedirs(cache_dir, exist_ok=True)
    cache_key = f"crosswalks_{place[0]:.6f}_{place[1]:.6f}_{distance}_{target_crs}"
    cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
    
    if os.path.exists(cache_file):
        print("Loading crosswalks from cache...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    tags_crosswalks = {
        'highway': 'crossing',
        'footway': 'crossing'
    }
    
    gdf_crosswalks = ox.features_from_point(
        place, tags=tags_crosswalks, dist=distance
    ).to_crs(target_crs)
    gdf_crosswalks = gdf_crosswalks[
        gdf_crosswalks.geometry.geom_type == 'Point'
    ].copy()
    
    # Save to cache
    with open(cache_file, 'wb') as f:
        pickle.dump(gdf_crosswalks, f)
    
    return gdf_crosswalks


def get_traffic_signals(place: tuple, distance: float, target_crs) -> gpd.GeoDataFrame:

    # Check cache first
    cache_dir = 'data/cache'
    os.makedirs(cache_dir, exist_ok=True)
    cache_key = f"signals_{place[0]:.6f}_{place[1]:.6f}_{distance}_{target_crs}"
    cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
    
    if os.path.exists(cache_file):
        print("Loading traffic signals from cache...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    tags_signals = {
        'highway': ['traffic_signals', 'crossing', 'stop'],
        'traffic_signals': True,
        'crossing': True
    }
    
    try:
        gdf_signals = ox.features_from_point(place, tags=tags_signals, dist=distance).to_crs(target_crs)
        gdf_signals = gdf_signals[gdf_signals.geometry.geom_type == 'Point'].copy()
        
        # Save to cache
        with open(cache_file, 'wb') as f:
            pickle.dump(gdf_signals, f)
        
        return gdf_signals
    except Exception as e:
        print(f"Error obtaining traffic signals: {e}")
        return gpd.GeoDataFrame()


def plot_basic_map(graph, center_node: int, gdf_green: gpd.GeoDataFrame, 
                   gdf_water: gpd.GeoDataFrame, 
                   title: str, filename: str) -> None:
    
    edges_gdf = ox.graph_to_gdfs(graph, nodes=False, fill_edge_geometry=True)
    
    fig, ax = plt.subplots(figsize=(14, 14), dpi=300, facecolor='white')
    ax.set_facecolor('white')
    
    # Create geographic boundary from graph edges
    map_boundary = edges_gdf.unary_union.convex_hull

    if not gdf_green.empty:
        gdf_green_clipped = gpd.clip(gdf_green, map_boundary)
        if not gdf_green_clipped.empty:
            gdf_green_clipped.plot(ax=ax, color='green', alpha=0.5, zorder=1)
    
    if not gdf_water.empty:
        gdf_water_clipped = gpd.clip(gdf_water, map_boundary)
        if not gdf_water_clipped.empty:
            gdf_water_clipped.plot(ax=ax, color='blue', alpha=0.5, zorder=1)
    
    edges_gdf.plot(ax=ax, color="black", linewidth=0.5, alpha=0.8, zorder=3)

    center_coords = graph.nodes[center_node]
    ax.scatter(center_coords['x'], center_coords['y'], c='red', s=50, zorder=5)

    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                             markersize=8, label='Central Point')]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, 
             frameon=True, fancybox=True, shadow=False)
    
    plt.title(title, fontsize=16)
    plt.axis("off")
    
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    #print(f"\n-> Basic map saved as '{filename}'")


def get_center_node(graph, central_point: tuple):
    
    central_point_gdf = gpd.GeoDataFrame(
        [1], geometry=[Point(central_point[1], central_point[0])], 
        crs="EPSG:4326"
    ).to_crs(graph.graph['crs'])
    
    central_point_projected = central_point_gdf.geometry.iloc[0]
    # start_node = ox.nearest_nodes(graph, start_lon, start_lat)
    return ox.distance.nearest_nodes(graph, central_point_projected.x, central_point_projected.y)


def select_user_profile(available_profiles: dict) -> tuple:
   
    print("Available walking profiles:")
    for key, value in available_profiles.items():
        print(f"  - {key} ({value['name']})")
    print("-" * 30)

    while True:
        profile_key = input("Enter the profile key you want to use: ")
        if profile_key in available_profiles:
            return available_profiles[profile_key], profile_key
        else:
            print("Invalid profile key! Please try again.")


def compute_edge_tobler(graph: nx.MultiGraph, profile: dict) -> nx.MultiGraph:
    """
    Calculates crossing time using a specific pedestrian profile.
    """
    speed_walk = profile['speed_walk']
    uphill_factor = profile['uphill_factor']
    downhill_factor = profile['downhill_factor']

    edge_times = {}
    elevations = nx.get_node_attributes(graph, 'elevation')
    
    for u, v, k, data in graph.edges(data=True, keys=True):
        length = data.get('length', 0)
        if length == 0:
            edge_times[(u, v, k)] = 0
            continue

        if u in elevations and v in elevations:
            slope = (elevations[v] - elevations[u]) / length
            
            base_v_kmh = speed_walk * math.exp(-3.5 * abs(slope + 0.05))

            if slope > 0:
                v_kmh = base_v_kmh * uphill_factor
            elif slope < 0:
                v_kmh = base_v_kmh * downhill_factor
            else:
                v_kmh = base_v_kmh
        else:
            v_kmh = speed_walk
            
        v_ms = v_kmh / 3.6
        time_min = (length / v_ms) / 60 if v_ms > 0 else float('inf')
        edge_times[(u, v, k)] = time_min

    nx.set_edge_attributes(graph, edge_times, 'time')
    return graph


def compute_node_travel_times(graph, center_node: int) -> dict:
    """
    Calculates minimum times to reach each node from the central node.
    """
    return nx.single_source_dijkstra_path_length(graph, center_node, weight="time")


def assign_edge_colors(graph, node_travel_times: dict, iso_intervals: list, 
                      iso_colors: list) -> None:
    
    for u, v, k, data in graph.edges(data=True, keys=True):
        t_u = node_travel_times.get(u, float("inf"))
        t_v = node_travel_times.get(v, float("inf"))
        edge_time = min(t_u, t_v)
        data["travel_time"] = edge_time
        
        if edge_time <= iso_intervals[0]: 
            data["edge_color"], data["edge_lw"] = iso_colors[0], 2
        elif edge_time <= iso_intervals[1]: 
            data["edge_color"], data["edge_lw"] = iso_colors[1], 2
        elif edge_time <= iso_intervals[2]: 
            data["edge_color"], data["edge_lw"] = iso_colors[2], 2
        elif edge_time <= iso_intervals[3]: 
            data["edge_color"], data["edge_lw"] = iso_colors[3], 2
        else: 
            data["edge_color"], data["edge_lw"] = "black", 0.5


def plot_base_map(ax, graph, gdf_green: gpd.GeoDataFrame, 
                  gdf_water: gpd.GeoDataFrame, iso_colors: list,
                  iso_intervals: list, center_node: int) -> tuple:
    """
    Plots the base map with green areas, water, edges, and isochrones.
    Returns edges_gdf for further use.
    
    Args:
        ax: Matplotlib axis object
        graph: NetworkX MultiDiGraph with travel_time and edge_color attributes
        gdf_green: GeoDataFrame with green areas
        gdf_water: GeoDataFrame with water bodies
        iso_colors: List of colors for isochrone intervals
        iso_intervals: List of time intervals in minutes
        center_node: Origin node ID
        
    Returns:
        tuple: (edges_gdf, colored_edges, black_edges)
    """
    edges_gdf = ox.graph_to_gdfs(graph, nodes=False, fill_edge_geometry=True)
    colored_edges = edges_gdf[edges_gdf["travel_time"] <= iso_intervals[-1]]
    black_edges = edges_gdf[edges_gdf["travel_time"] > iso_intervals[-1]]
    
    # Create geographic boundary from graph edges
    map_boundary = edges_gdf.unary_union.convex_hull
    
    # Plot green areas
    if not gdf_green.empty:
        gdf_green_clipped = gpd.clip(gdf_green, map_boundary)
        if not gdf_green_clipped.empty:
            gdf_green_clipped.plot(ax=ax, color='green', alpha=0.5, zorder=1)
    
    # Plot water bodies
    if not gdf_water.empty:
        gdf_water_clipped = gpd.clip(gdf_water, map_boundary)
        if not gdf_water_clipped.empty:
            gdf_water_clipped.plot(ax=ax, color='blue', alpha=0.5, zorder=1)

    # Plot colored edges (within isochrone)
    for color, group in colored_edges.groupby("edge_color"):
        group.plot(ax=ax, color=color, linewidth=2, alpha=0.8, zorder=3)

    # Plot black edges (outside isochrone)
    if not black_edges.empty:
        black_edges.plot(ax=ax, color="black", linewidth=0.5, alpha=0.5, zorder=3)
    
    # Plot center point
    center_coords = graph.nodes[center_node]
    ax.scatter(center_coords['x'], center_coords['y'], c='red', s=50, zorder=5)
    
    return edges_gdf, colored_edges, black_edges


def plot_map_with_isochrones(graph, center_node: int, iso_colors: list, 
                             iso_intervals: list, gdf_green: gpd.GeoDataFrame, 
                             gdf_water: gpd.GeoDataFrame, place: tuple, 
                             distance: float,
                             title: str, filename: str) -> None:
    """
    Plots a map with isochrones only (no POIs or H3 grid).
    """
    fig, ax = plt.subplots(figsize=(14, 14), dpi=300, facecolor='white')
    ax.set_facecolor('white')
    
    # Plot base map elements
    plot_base_map(ax, graph, gdf_green, gdf_water, iso_colors, iso_intervals, center_node)

    # Create time legend
    legend_elements_time = [
        Line2D([0], [0], color=iso_colors[0], lw=2, label="0 - 5 min"),
        Line2D([0], [0], color=iso_colors[1], lw=2, label="5 - 10 min"),
        Line2D([0], [0], color=iso_colors[2], lw=2, label="10 - 15 min"),
        Line2D([0], [0], color=iso_colors[3], lw=2, label="15 - 20 min"),
        Line2D([0], [0], color="black", lw=0.5, label="> 20 min")
    ]
    
    ax.legend(handles=legend_elements_time, loc="lower left", title="Access Time")
    
    plt.title(title, fontsize=16)
    plt.axis("off")
    
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    #print(f"\n-> Map with isochrones saved as '{filename}'")


def plot_map_with_pois_isochrones(graph, center_node: int, iso_colors: list, 
                                  iso_intervals: list, gdf_green: gpd.GeoDataFrame, 
                                  gdf_water: gpd.GeoDataFrame, 
                                  gdf_pois: gpd.GeoDataFrame,
                                  title: str, filename: str) -> None:
    """
    Plots a map with isochrones and POIs (no H3 grid).
    """
    fig, ax = plt.subplots(figsize=(14, 14), dpi=300, facecolor='white')
    ax.set_facecolor('white')
    
    # Plot base map elements
    plot_base_map(ax, graph, gdf_green, gdf_water, iso_colors, iso_intervals, center_node)
    
    # Plot POIs found within isochrone limits
    if not gdf_pois.empty:
        for color, group in gdf_pois.groupby('color'):
            group.plot(
                ax=ax, color=color, marker='o', markersize=40, 
                edgecolor='black', zorder=6
            )

    legend_elements_time = [
        Line2D([0], [0], color=iso_colors[0], lw=2, label="0 - 5 min"),
        Line2D([0], [0], color=iso_colors[1], lw=2, label="5 - 10 min"),
        Line2D([0], [0], color=iso_colors[2], lw=2, label="10 - 15 min"),
        Line2D([0], [0], color=iso_colors[3], lw=2, label="15 - 20 min"),
        Line2D([0], [0], color="black", lw=0.5, label="> 20 min")
    ]
    
    time_legend = ax.legend(
        handles=legend_elements_time, loc="lower left", title="Access Time"
    )
    ax.add_artist(time_legend)
    
    if not gdf_pois.empty:
        legend_elements_poi = []
        unique_pois = gdf_pois[['poi_type', 'color']].drop_duplicates().sort_values('poi_type')

        for _, row in unique_pois.iterrows():
            label = row['poi_type'].replace('_', ' ').title()
            legend_elements_poi.append(
                Line2D([0], [0], marker='o', color='w', 
                       label=label, markerfacecolor=row['color'], 
                       markeredgecolor='black', markersize=8)
            )
        
        if legend_elements_poi:
            ax.legend(
                handles=legend_elements_poi, loc='upper right', 
                title="POIs", fontsize=9
            )
    
    plt.title(title, fontsize=16)
    plt.axis("off")
    
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    #print(f"\n-> Map with POIs and isochrones saved as '{filename}'")


@lru_cache(maxsize=2048)
def h3_to_polygon_cached(h3_index: str, target_crs_str: str = None) -> str:
    """
    Cached version that returns WKT string for serialization.
    """
    coords = h3.cell_to_boundary(h3_index)
    polygon = Polygon([(lon, lat) for lat, lon in coords])
    
    if target_crs_str:
        gdf = gpd.GeoDataFrame(geometry=[polygon], crs='EPSG:4326')
        polygon = gdf.to_crs(target_crs_str).geometry[0]
    return polygon.wkt


def h3_to_polygon(h3_index: str, crs_source: str = 'EPSG:4326', 
                  target_crs: str = None) -> Polygon:
    """
    Converts an H3 index to a Shapely polygon.
    
    Returns:
        Polygon: Shapely polygon representing the H3 hexagon.
    """
    wkt_str = h3_to_polygon_cached(h3_index, target_crs)
    return wkt.loads(wkt_str)


def _process_hexagon_wrapper(args):
    """
    Wrapper for multiprocessing hexagon processing.
    Handles deserialization and delegates to processing logic.
    Returns hexagon data dictionary or None if outside boundary.
    """
    (i, hex_id, map_boundary_wkt, target_crs, node_coords_times,
     accessible_features_dict, gdf_green_dict, gdf_water_dict, 
     edges_gdf_dict, h3_resolution) = args
    
    try:
        # Deserialize boundary and hexagon
        map_boundary = wkt.loads(map_boundary_wkt)
        hex_poly = h3_to_polygon(hex_id, target_crs=target_crs)
        
        if not hex_poly.intersects(map_boundary):
            return None
        
        # Process hexagon geometry
        centroid_lat, centroid_lon = h3.cell_to_latlng(hex_id)
        centroid_gdf = gpd.GeoDataFrame(
            [1], geometry=[Point(centroid_lon, centroid_lat)], 
            crs="EPSG:4326"
        ).to_crs(target_crs)
        centroid_proj = centroid_gdf.geometry.iloc[0]
        
        # Find nearest node and its travel time
        if node_coords_times:
            nearest_coord = min(node_coords_times.keys(), 
                              key=lambda n: ((n[0] - centroid_proj.x)**2 + (n[1] - centroid_proj.y)**2)**0.5)
            nearest_node, node_time_travel = node_coords_times[nearest_coord]
        else:
            node_time_travel = None
        
        # Initialize counters
        count_pois_accessible = 0
        count_crosswalks_accessible = 0
        count_traffic_signals_accessible = 0
        
        poi_types_count = {
            'restaurant': 0, 'bar_cafe': 0, 'hospital': 0, 'school': 0,
            'bank': 0, 'pharmacy': 0, 'doctors': 0, 'cinema': 0,
            'hotel': 0, 'library': 0, 'bicycle_parking': 0, 'gym': 0,
            'supermarket': 0, 'grocery': 0, 'bakery': 0, 'other_pois': 0
        }
        
        area_vegetation_m2 = 0.0
        area_water_m2 = 0.0
        percent_vegetation = 0.0
        percent_water = 0.0
        avg_attractiveness_threshold_pois = 0.0
        edge_travel_time = None
        
        hex_area_total = h3.cell_area(hex_id, unit='m^2')
        
        # Process vegetation
        if gdf_green_dict:
            gdf_green = gpd.GeoDataFrame.from_dict(gdf_green_dict)
            try:
                vegetation_intersection = gdf_green.geometry.intersection(hex_poly)
                valid_intersections = vegetation_intersection[
                    vegetation_intersection.is_valid & (~vegetation_intersection.is_empty)
                ]
                if not valid_intersections.empty:
                    area_vegetation_m2 = valid_intersections.area.sum()
                    percent_vegetation = (area_vegetation_m2 / hex_area_total) * 100
            except Exception:
                pass
        
        # Process water
        if gdf_water_dict:
            gdf_water = gpd.GeoDataFrame.from_dict(gdf_water_dict)
            try:
                water_intersection = gdf_water.geometry.intersection(hex_poly)
                valid_intersections = water_intersection[
                    water_intersection.is_valid & (~water_intersection.is_empty)
                ]
                if not valid_intersections.empty:
                    area_water_m2 = valid_intersections.area.sum()
                    percent_water = (area_water_m2 / hex_area_total) * 100
            except Exception:
                pass
        
        # Process accessible features
        if accessible_features_dict:
            for feature_type, feature_data in accessible_features_dict.items():
                if feature_data:
                    gdf_feature = gpd.GeoDataFrame.from_dict(feature_data)
                    features_in_hex = gdf_feature[gdf_feature.geometry.within(hex_poly)]
                    
                    if feature_type == 'pois':
                        count_pois_accessible = len(features_in_hex)
                        if count_pois_accessible > 0 and 'poi_type' in features_in_hex.columns:
                            for _, poi in features_in_hex.iterrows():
                                poi_type = poi.get('poi_type', 'other_pois')
                                if poi_type in ['restaurant', 'fast_food']:
                                    poi_types_count['restaurant'] += 1
                                elif poi_type in ['bar', 'cafe']:
                                    poi_types_count['bar_cafe'] += 1
                                elif poi_type in ['hospital', 'clinic']:
                                    poi_types_count['hospital'] += 1
                                elif poi_type in ['school', 'university', 'college', 'kindergarten']:
                                    poi_types_count['school'] += 1
                                elif poi_type == 'bank':
                                    poi_types_count['bank'] += 1
                                elif poi_type == 'pharmacy':
                                    poi_types_count['pharmacy'] += 1
                                elif poi_type in ['doctors', 'dentist']:
                                    poi_types_count['doctors'] += 1
                                elif poi_type in ['cinema', 'theatre']:
                                    poi_types_count['cinema'] += 1
                                elif poi_type == 'hotel':
                                    poi_types_count['hotel'] += 1
                                elif poi_type == 'library':
                                    poi_types_count['library'] += 1
                                elif poi_type == 'bicycle_parking':
                                    poi_types_count['bicycle_parking'] += 1
                                elif poi_type in ['gym', 'fitness_centre', 'fitness_center']:
                                    poi_types_count['gym'] += 1
                                elif poi_type == 'supermarket':
                                    poi_types_count['supermarket'] += 1
                                elif poi_type in ['convenience', 'grocery', 'greengrocer']:
                                    poi_types_count['grocery'] += 1
                                elif poi_type == 'bakery':
                                    poi_types_count['bakery'] += 1
                                else:
                                    poi_types_count['other_pois'] += 1
                                    
                    elif feature_type == 'crosswalks':
                        count_crosswalks_accessible = len(features_in_hex)
                    elif feature_type == 'traffic_signals':
                        count_traffic_signals_accessible = len(features_in_hex)
        
        # Edge travel time calculation
        if edges_gdf_dict:
            edges_gdf = gpd.GeoDataFrame.from_dict(edges_gdf_dict)
            try:
                edges_in_hex = edges_gdf[edges_gdf.geometry.intersects(hex_poly)]
                if not edges_in_hex.empty:
                    intersec = edges_in_hex.geometry.intersection(hex_poly)
                    lengths = intersec.length
                    idx_dom = lengths.idxmax()
                    tt = edges_in_hex.loc[idx_dom, 'travel_time'] if 'travel_time' in edges_in_hex.columns else None
                    if pd.isna(tt) or tt is None:
                        tt = float(edges_in_hex['travel_time'].min())
                    edge_travel_time = round(float(tt), 2)
            except Exception:
                pass
        
        return {
            'sequential_id': i,
            'h3_id': hex_id,
            'latitude': centroid_lat,
            'longitude': centroid_lon,
            'coord_x_proj': centroid_proj.x,
            'coord_y_proj': centroid_proj.y,
            'resolution': h3_resolution,
            'area_m2': h3.cell_area(hex_id, unit='m^2'),
            'area_vegetation_m2': round(area_vegetation_m2, 2),
            'area_water_m2': round(area_water_m2, 2),
            'percent_vegetation': round(percent_vegetation, 2),
            'percent_water': round(percent_water, 2),
            'count_pois_accessible': count_pois_accessible,
            'count_crosswalks_accessible': count_crosswalks_accessible,
            'count_traffic_signals_accessible': count_traffic_signals_accessible,
            'total_accessible_features': count_pois_accessible + count_crosswalks_accessible + count_traffic_signals_accessible,
            'pois_restaurant': poi_types_count['restaurant'],
            'pois_bar_cafe': poi_types_count['bar_cafe'],
            'pois_hospital': poi_types_count['hospital'],
            'pois_school': poi_types_count['school'],
            'pois_bank': poi_types_count['bank'],
            'pois_pharmacy': poi_types_count['pharmacy'],
            'pois_doctors': poi_types_count['doctors'],
            'pois_cinema': poi_types_count['cinema'],
            'pois_hotel': poi_types_count['hotel'],
            'pois_library': poi_types_count['library'],
            'pois_bicycle_parking': poi_types_count['bicycle_parking'],
            'pois_gym': poi_types_count['gym'],
            'pois_supermarket': poi_types_count['supermarket'],
            'pois_grocery': poi_types_count['grocery'],
            'pois_bakery': poi_types_count['bakery'],
            'pois_other': poi_types_count['other_pois'],
            'avg_attractiveness_threshold_pois': avg_attractiveness_threshold_pois,
            'node_time_travel': round(node_time_travel, 2) if node_time_travel is not None else None,
            'edge_travel_time': edge_travel_time
        }
    
    except Exception as e:
        # Log error but don't crash the entire process
        print(f"Error processing hexagon {hex_id}: {str(e)}")
        return None


def plot_h3_grid(ax, graph, center_lat_lon, radius_m, h3_resolution: int):
    
    target_crs = graph.graph['crs']
    
    # Create geographic boundary from graph nodes convex hull
    gdf_nodes = ox.graph_to_gdfs(graph, edges=False)
    map_boundary = gdf_nodes.union_all().convex_hull
    
    # Calculate hexagon IDs that cover the analysis area
    hex_center = h3.latlng_to_cell(center_lat_lon[0], center_lat_lon[1], h3_resolution)
    hex_radius = math.ceil(radius_m / h3.average_hexagon_edge_length(h3_resolution, unit='m'))
    hex_ids = h3.grid_disk(hex_center, hex_radius)
    
    # Plot each hexagon that intersects the map boundary
    for hex_id in hex_ids:
        hex_poly = h3_to_polygon(hex_id, target_crs=target_crs)
        
        if hex_poly.intersects(map_boundary):
            patch = plt.Polygon(
                list(hex_poly.exterior.coords), 
                edgecolor='grey', 
                facecolor='none',
                linewidth=0.4, 
                alpha=0.7, 
                zorder=2
            )
            ax.add_patch(patch)


def save_h3_hexagons_to_csv(graph, center_lat_lon: tuple, radius_m: float, 
                           location: str, 
                           node_travel_times: dict,
                           h3_resolution: int,
                           accessible_features: dict = None,
                           gdf_green: gpd.GeoDataFrame = None,
                           gdf_water: gpd.GeoDataFrame = None,
                           profile_key: str = None,
                           use_multiprocessing: bool = True) -> None:
   
    print("Preparing spatial data structures...")
    gdf_nodes = ox.graph_to_gdfs(graph, edges=False)
    map_boundary = gdf_nodes.union_all().convex_hull
    target_crs = graph.graph['crs']

    # Prepare edges with travel_time to check hexagon intersections
    edges_gdf = ox.graph_to_gdfs(graph, nodes=False, fill_edge_geometry=True)
    if not edges_gdf.empty:
        # keep only what we need
        keep_cols = ['travel_time', 'geometry']
        edges_gdf = edges_gdf[[c for c in keep_cols if c in edges_gdf.columns]].copy()
        # Create spatial index for faster edge intersection queries
        if len(edges_gdf) > 0:
            edges_sindex = edges_gdf.sindex

    # Calculate hexagon IDs that cover the analysis area
    hex_center = h3.latlng_to_cell(center_lat_lon[0], center_lat_lon[1], h3_resolution)
    hex_radius = math.ceil(radius_m / h3.average_hexagon_edge_length(h3_resolution, unit='m'))
    hex_ids = list(h3.grid_disk(hex_center, hex_radius))
    
    # Create spatial indexes for features (speeds up within/intersection queries)
    if accessible_features:
        if 'pois' in accessible_features and not accessible_features['pois'].empty:
            pois_sindex = accessible_features['pois'].sindex
        if 'crosswalks' in accessible_features and not accessible_features['crosswalks'].empty:
            crosswalks_sindex = accessible_features['crosswalks'].sindex
        if 'traffic_signals' in accessible_features and not accessible_features['traffic_signals'].empty:
            signals_sindex = accessible_features['traffic_signals'].sindex
    
    total_hexagons = len(hex_ids)
    print(f"Processing {total_hexagons} hexagons...")
    
    # MULTIPROCESSING IMPLEMENTATION
    if use_multiprocessing and total_hexagons > 100:
        print(f"Using multiprocessing with {cpu_count()} CPU cores...")
        
        # Serialize data for worker processes
        map_boundary_wkt = map_boundary.wkt
        
        # Prepare node_travel_times as dict with (x,y) keys for faster lookup
        node_coords_times = {}
        for node_id, travel_time in node_travel_times.items():
            node_data = graph.nodes[node_id]
            node_coords_times[(node_data['x'], node_data['y'])] = (node_id, travel_time)
        
        # Serialize GeoDataFrames to dict (picklable)
        gdf_green_dict = gdf_green.to_dict() if gdf_green is not None and not gdf_green.empty else None
        gdf_water_dict = gdf_water.to_dict() if gdf_water is not None and not gdf_water.empty else None
        edges_gdf_dict = edges_gdf.to_dict() if not edges_gdf.empty else None
        
        # Serialize accessible features
        accessible_features_dict = {}
        if accessible_features:
            for key, gdf in accessible_features.items():
                if gdf is not None and not gdf.empty:
                    accessible_features_dict[key] = gdf.to_dict()
        
        # Create argument tuples for each hexagon
        args_list = [
            (i+1, hex_id, map_boundary_wkt, target_crs, node_coords_times,
             accessible_features_dict, gdf_green_dict, gdf_water_dict,
             edges_gdf_dict, h3_resolution)
            for i, hex_id in enumerate(hex_ids)
        ]
        
        # Process hexagons in parallel
        num_processes = max(1, cpu_count() - 1)  # Leave 1 core free
        with Pool(processes=num_processes) as pool:
            results = pool.map(_process_hexagon_wrapper, args_list)
        
        # Filter out None results (hexagons outside boundary)
        hexagon_data = [r for r in results if r is not None]
        print(f"  Completed: {len(hexagon_data)}/{total_hexagons} hexagons processed")
        
    else:
        # SERIAL PROCESSING (fallback for small datasets or disabled multiprocessing)
        print("Using serial processing...")
        hexagon_data = []
        
        # Process each hexagon that intersects the map boundary
        for i, hex_id in enumerate(hex_ids, 1):
            if i % 50 == 0:
                print(f"  Processed {i}/{total_hexagons} hexagons ({i*100//total_hexagons}%)")
        
        target_crs = graph.graph['crs']
        hex_poly = h3_to_polygon(hex_id, target_crs=target_crs)
        
        if hex_poly.intersects(map_boundary):
            centroid_lat, centroid_lon = h3.cell_to_latlng(hex_id)
            
            centroid_gdf = gpd.GeoDataFrame(
                [1], geometry=[Point(centroid_lon, centroid_lat)], 
                crs="EPSG:4326"
            ).to_crs(target_crs)
            centroid_proj = centroid_gdf.geometry.iloc[0]
            
            nearest_node = ox.distance.nearest_nodes(graph, centroid_proj.x, centroid_proj.y)
            node_time_travel = node_travel_times.get(nearest_node, None)
            
            count_pois_accessible = 0
            count_crosswalks_accessible = 0
            count_traffic_signals_accessible = 0
            
            poi_types_count = {
                'restaurant': 0,  # restaurant, fast_food
                'bar_cafe': 0,    # bar, cafe
                'hospital': 0,    # hospital, clinic
                'school': 0,      # school, university, college, kindergarten
                'bank': 0,        # bank
                'pharmacy': 0,    # pharmacy
                'doctors': 0,     # doctors, dentist
                'cinema': 0,      # cinema, theatre
                'hotel': 0,       # hotel
                'library': 0,     # library
                'bicycle_parking': 0,  # bicycle_parking
                'gym': 0,         # gym, fitness_centre, fitness_center
                'supermarket': 0, # supermarket
                'grocery': 0,     # convenience, grocery, greengrocer
                'bakery': 0,      # bakery
                'other_pois': 0  # other uncategorized types
            }
            
            area_vegetation_m2 = 0.0
            area_water_m2 = 0.0
            percent_vegetation = 0.0
            percent_water = 0.0

            avg_attractiveness_threshold_pois = 0.0
            edge_travel_time = None
            
            hex_area_total = h3.cell_area(hex_id, unit='m^2')
            
            # Vegetation intersections
            if gdf_green is not None and not gdf_green.empty:
                try:
                    vegetation_intersection = gdf_green.geometry.intersection(hex_poly)
                    valid_intersections = vegetation_intersection[
                        vegetation_intersection.is_valid & 
                        (~vegetation_intersection.is_empty)
                    ]
                    if not valid_intersections.empty:
                        area_vegetation_m2 = valid_intersections.area.sum()
                        percent_vegetation = (area_vegetation_m2 / hex_area_total) * 100
                except Exception:
                    pass
            
            # Water intersections
            if gdf_water is not None and not gdf_water.empty:
                try:
                    water_intersection = gdf_water.geometry.intersection(hex_poly)
                    valid_intersections = water_intersection[
                        water_intersection.is_valid & 
                        (~water_intersection.is_empty)
                    ]
                    if not valid_intersections.empty:
                        area_water_m2 = valid_intersections.area.sum()
                        percent_water = (area_water_m2 / hex_area_total) * 100
                except Exception:
                    pass
            
            # Accessible features within hexagon (using spatial index for speed)
            if accessible_features:
                if 'pois' in accessible_features and not accessible_features['pois'].empty:
                    pois_df = accessible_features['pois']
                    # for safety, if there's a time column, keep only ≤20 min
                    if 'real_time' in pois_df.columns:
                        pois_df = pois_df[pois_df['real_time'] <= 20].copy()
                    
                    # Use spatial index for faster query
                    possible_matches_idx = list(pois_sindex.query(hex_poly, predicate='intersects'))
                    if possible_matches_idx:
                        possible_matches = pois_df.iloc[possible_matches_idx]
                        pois_in_hex = possible_matches[possible_matches.geometry.within(hex_poly)]
                    else:
                        pois_in_hex = pois_df.iloc[0:0]  # empty dataframe
                    
                    count_pois_accessible = len(pois_in_hex)

                    if count_pois_accessible > 0:
                        if 'poi_type' not in pois_in_hex.columns:
                            pois_in_hex = pois_in_hex.copy()
                            pois_in_hex['poi_type'] = pois_in_hex['amenity'].fillna(
                                pois_in_hex['building'].fillna(pois_in_hex['shop'])
                            )
                            
                        for _, poi in pois_in_hex.iterrows():
                            poi_type = poi.get('poi_type', 'other_pois')
                            
                            if poi_type in ['restaurant', 'fast_food']:
                                poi_types_count['restaurant'] += 1
                            elif poi_type in ['bar', 'cafe']:
                                poi_types_count['bar_cafe'] += 1
                            elif poi_type in ['hospital', 'clinic']:
                                poi_types_count['hospital'] += 1
                            elif poi_type in ['school', 'university', 'college', 'kindergarten']:
                                poi_types_count['school'] += 1
                            elif poi_type == 'bank':
                                poi_types_count['bank'] += 1
                            elif poi_type == 'pharmacy':
                                poi_types_count['pharmacy'] += 1
                            elif poi_type in ['doctors', 'dentist']:
                                poi_types_count['doctors'] += 1
                            elif poi_type in ['cinema', 'theatre']:
                                poi_types_count['cinema'] += 1
                            elif poi_type == 'hotel':
                                poi_types_count['hotel'] += 1
                            elif poi_type == 'library':
                                poi_types_count['library'] += 1
                            elif poi_type == 'bicycle_parking':
                                poi_types_count['bicycle_parking'] += 1
                            elif poi_type in ['gym', 'fitness_centre', 'fitness_center']:
                                poi_types_count['gym'] += 1
                            elif poi_type == 'supermarket':
                                poi_types_count['supermarket'] += 1
                            elif poi_type in ['convenience', 'grocery', 'greengrocer']:
                                poi_types_count['grocery'] += 1
                            elif poi_type == 'bakery':
                                poi_types_count['bakery'] += 1
                            else:
                                poi_types_count['other_pois'] += 1

                    # Average POI attractiveness threshold weighted by all categories
                    # Example: 2 cinema (0.9 + 0.73) + 5 hospital (0.85+0.77+0.65+0.56+0.38) + 10 empty categories (0)
                    # avg = (0.9 + 0.73 + 0.85 + 0.77 + 0.65 + 0.56 + 0.38 + 0 + ... + 0) / 17
                    # where 17 = 7 individual POIs + 10 categories without POIs
                    if count_pois_accessible > 0:
                        if 'attractiveness_threshold' not in pois_in_hex.columns:
                            pois_in_hex = pois_in_hex.copy()
                            if 'real_time' in pois_in_hex.columns:
                                pois_in_hex['attractiveness_threshold'] = pois_in_hex['real_time'].apply(calculate_attractiveness_threshold)
                            else:
                                pois_in_hex['attractiveness_threshold'] = np.nan
                        
                        threshold_sum = pois_in_hex['attractiveness_threshold'].dropna().sum()
                        num_individual_pois = len(pois_in_hex['attractiveness_threshold'].dropna())
                        total_categories = 16
                        
                        # Denominator: individual POIs + empty categories
                        # Example: 7 POIs + (12 - 7) = 7 + 5 = 12? No, that doesn't make sense
                        # Actually: 7 POIs + 10 empty categories = 17
                        # But 10 empty categories only makes sense if 2 categories have POIs
                        # Let's count how many unique categories have POIs
                        categories_with_pois = len(set(pois_in_hex['poi_type']))
                        empty_categories = total_categories - categories_with_pois
                        denominator = num_individual_pois + empty_categories
                        avg_attractiveness_threshold_pois = round(float(threshold_sum / denominator), 3)

                if 'crosswalks' in accessible_features and not accessible_features['crosswalks'].empty:
                    # Use spatial index for faster query
                    possible_idx = list(crosswalks_sindex.query(hex_poly, predicate='intersects'))
                    if possible_idx:
                        possible = accessible_features['crosswalks'].iloc[possible_idx]
                        crosswalks_in_hex = possible[possible.geometry.within(hex_poly)]
                    else:
                        crosswalks_in_hex = accessible_features['crosswalks'].iloc[0:0]
                    count_crosswalks_accessible = len(crosswalks_in_hex)

                if 'traffic_signals' in accessible_features and not accessible_features['traffic_signals'].empty:
                    # Use spatial index for faster query
                    possible_idx = list(signals_sindex.query(hex_poly, predicate='intersects'))
                    if possible_idx:
                        possible = accessible_features['traffic_signals'].iloc[possible_idx]
                        signals_in_hex = possible[possible.geometry.within(hex_poly)]
                    else:
                        signals_in_hex = accessible_features['traffic_signals'].iloc[0:0]
                    count_traffic_signals_accessible = len(signals_in_hex)

            # Edge travel time that the hexagon covers:
            # we choose the edge whose INTERSECTION with the hex has LARGEST LENGTH
            try:
                if edges_gdf is not None and not edges_gdf.empty:
                    # Use spatial index for faster edge query
                    possible_edges_idx = list(edges_sindex.query(hex_poly, predicate='intersects'))
                    if possible_edges_idx:
                        edges_in_hex = edges_gdf.iloc[possible_edges_idx]
                    else:
                        edges_in_hex = edges_gdf.iloc[0:0]
                    
                    if not edges_in_hex.empty:
                        # 2) measure how much of each edge crosses the hex
                        intersec = edges_in_hex.geometry.intersection(hex_poly)
                        lengths = intersec.length
                        # 3) get the dominant edge (largest overlap)
                        idx_dom = lengths.idxmax()
                        tt = edges_in_hex.loc[idx_dom, 'travel_time'] if 'travel_time' in edges_in_hex.columns else None
                        # fallback: if travel_time is empty, use the minimum
                        if pd.isna(tt) or tt is None:
                            tt = float(edges_in_hex['travel_time'].min())
                        edge_travel_time = round(float(tt), 2)
            except Exception:
                pass

            hexagon_data.append({
                'sequential_id': i,
                'h3_id': hex_id,
                'latitude': centroid_lat,
                'longitude': centroid_lon,
                'coord_x_proj': centroid_proj.x,
                'coord_y_proj': centroid_proj.y,
                'resolution': h3_resolution,
                'area_m2': h3.cell_area(hex_id, unit='m^2'),
                'area_vegetation_m2': round(area_vegetation_m2, 2),
                'area_water_m2': round(area_water_m2, 2),
                'percent_vegetation': round(percent_vegetation, 2),
                'percent_water': round(percent_water, 2),
                'count_pois_accessible': count_pois_accessible,
                'count_crosswalks_accessible': count_crosswalks_accessible,
                'count_traffic_signals_accessible': count_traffic_signals_accessible,
                'total_accessible_features': count_pois_accessible + count_crosswalks_accessible + count_traffic_signals_accessible,
                # POI counters by type (≤20min)
                'pois_restaurant': poi_types_count['restaurant'],
                'pois_bar_cafe': poi_types_count['bar_cafe'],
                'pois_hospital': poi_types_count['hospital'],
                'pois_school': poi_types_count['school'],
                'pois_bank': poi_types_count['bank'],
                'pois_pharmacy': poi_types_count['pharmacy'],
                'pois_doctors': poi_types_count['doctors'],
                'pois_cinema': poi_types_count['cinema'],
                'pois_hotel': poi_types_count['hotel'],
                'pois_library': poi_types_count['library'],
                'pois_bicycle_parking': poi_types_count['bicycle_parking'],
                'pois_gym': poi_types_count['gym'],
                'pois_supermarket': poi_types_count['supermarket'],
                'pois_grocery': poi_types_count['grocery'],
                'pois_bakery': poi_types_count['bakery'],
                'pois_other': poi_types_count['other_pois'],
                'avg_attractiveness_threshold_pois': avg_attractiveness_threshold_pois,
                'node_time_travel': round(node_time_travel, 2) if node_time_travel is not None else None,
                'edge_travel_time': edge_travel_time
            })
    
    if hexagon_data:
        df_hexagons = pd.DataFrame(hexagon_data)
        
        # Calculate minimum access time (prioritizes best available access)
        df_hexagons['min_time_access'] = df_hexagons.apply(lambda row: 
            min(
                row['node_time_travel'] if pd.notna(row['node_time_travel']) else float('inf'),
                row['edge_travel_time'] if pd.notna(row['edge_travel_time']) else float('inf')
            ), axis=1
        )
        
        df_hexagons['min_time_access'] = df_hexagons['min_time_access'].replace([float('inf')], None)
        
        df_hexagons['min_time_access'] = df_hexagons['min_time_access'].apply(
            lambda x: round(x, 2) if pd.notna(x) else None
        )
        
        profile_suffix = f"_{profile_key}" if profile_key else ""
        csv_file = f"data/csv/h3_info/{location}_h3_hexagons_res{h3_resolution}{profile_suffix}.csv"
        ensure_directory_exists(csv_file)
        
        df_hexagons.to_csv(csv_file, index=False, encoding='utf-8')
        
        print(f"\n-> H3 hexagons saved in: {csv_file}")
        print(f"   Total hexagons: {len(df_hexagons)}")
        print(f"   H3 Resolution: {h3_resolution}")
        

def plot_map_with_pois_isochrones_h3(graph, center_node: int, iso_colors: list, 
                                     iso_intervals: list, gdf_green: gpd.GeoDataFrame, 
                                     gdf_water: gpd.GeoDataFrame, 
                                     gdf_pois: gpd.GeoDataFrame,
                                     place: tuple, distance: float,
                                     title: str, filename: str,
                                     h3_resolution: int) -> None:
    """
    Plots a map with isochrones, POIs, and H3 hexagonal grid.
    """
    fig, ax = plt.subplots(figsize=(14, 14), dpi=300, facecolor='white')
    ax.set_facecolor('white')
    
    # Plot base map elements
    plot_base_map(ax, graph, gdf_green, gdf_water, iso_colors, iso_intervals, center_node)
    
    # Plot H3 grid overlay
    plot_h3_grid(ax, graph, place, distance, h3_resolution)
    
    if not gdf_pois.empty:
        for color, group in gdf_pois.groupby('color'):
            group.plot(
                ax=ax, color=color, marker='o', markersize=40, 
                edgecolor='black', zorder=6
            )
    
    legend_elements_time = [
        Line2D([0], [0], color=iso_colors[0], lw=2, label="0 - 5 min"),
        Line2D([0], [0], color=iso_colors[1], lw=2, label="5 - 10 min"),
        Line2D([0], [0], color=iso_colors[2], lw=2, label="10 - 15 min"),
        Line2D([0], [0], color=iso_colors[3], lw=2, label="15 - 20 min"),
        Line2D([0], [0], color="black", lw=0.5, label="> 20 min")
    ]
    
    time_legend = ax.legend(
        handles=legend_elements_time, loc="lower left", title="Access Time"
    )
    ax.add_artist(time_legend)
    
    if not gdf_pois.empty:
        legend_elements_poi = []
        unique_pois = gdf_pois[['poi_type', 'color']].drop_duplicates().sort_values('poi_type')

        for _, row in unique_pois.iterrows():
            label = row['poi_type'].replace('_', ' ').title()
            legend_elements_poi.append(
                Line2D([0], [0], marker='o', color='w', 
                       label=label, markerfacecolor=row['color'], 
                       markeredgecolor='black', markersize=8)
            )
        
        if legend_elements_poi:
            ax.legend(
                handles=legend_elements_poi, loc='upper right', 
                title="POIs", fontsize=9
            )
    
    plt.title(title, fontsize=16)
    plt.axis("off")
    
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\n-> Map with POIs and isochrones saved as '{filename}'")
    

def map_poi_colors_and_types(gdf_pois: gpd.GeoDataFrame, 
                            color_map: dict) -> gpd.GeoDataFrame:

    if gdf_pois.empty: 
        return gdf_pois
    
    gdf_pois['poi_type'] = gdf_pois['amenity'].fillna(
        gdf_pois['building'].fillna(gdf_pois['shop'])
    )
    gdf_pois['color'] = gdf_pois['poi_type'].map(color_map).fillna('grey')
    
    return gdf_pois


def calculate_feature_travel_time(gdf_features: gpd.GeoDataFrame, graph,
                                  node_travel_times: dict, profile: dict) -> gpd.GeoDataFrame:
    """
    Calculates the actual walking time to each feature (POI, crosswalk, etc.).
    This function calculates time in two ways:
    1. Uses Dijkstra travel time on the street network to the nearest node.
    2. Calculates time from nearest node to the feature using the profile's walking speed.
    """
    if gdf_features.empty:
        return gdf_features

    # Find nearest node for each feature
    nearest_nodes = ox.distance.nearest_nodes(
        graph, gdf_features.geometry.x, gdf_features.geometry.y
    )
    gdf_features['nearest_node'] = nearest_nodes
    gdf_features['time_to_node'] = gdf_features['nearest_node'].map(node_travel_times)

    # Calculate straight-line distance from node to feature
    nodes_geom = {
        node: Point(data['x'], data['y'])
        for node, data in graph.nodes(data=True)
    }
    
    # Map nearest node geometries
    gdf_features['node_geom'] = gdf_features['nearest_node'].map(nodes_geom)
    
    # Calculate distances using vectorized operation
    gdf_features['dist_from_node_to_feature'] = gdf_features.apply(
        lambda row: row.geometry.distance(row.node_geom) if row.node_geom else 0, axis=1
    )

    # Calculate time from node to feature using profile's walking speed
    walking_speed_kmh = profile['speed_walk']  # km/h from profile
    walking_speed_m_per_min = walking_speed_kmh * 1000 / 60  # convert to meters per minute
    gdf_features['time_from_node_to_feature'] = (
        gdf_features['dist_from_node_to_feature'] / walking_speed_m_per_min
    )

    # Calculate total time
    gdf_features['real_time'] = gdf_features['time_to_node'] + gdf_features['time_from_node_to_feature']

    # Remove intermediate columns for cleanup
    gdf_features.drop(columns=['dist_from_node_to_feature', 'time_from_node_to_feature', 'node_geom'], inplace=True)
    return gdf_features


def calculate_attractiveness_threshold(walking_time: float) -> float:
    """
    Calculates the Attractiveness Threshold based on walking time.

    This model uses a cosine curve for smooth decay between 0 and 20 minutes,
    and defines the threshold as 0 for any time equal to or greater than 20 minutes.
    """
    if walking_time < 0:
        return 0.0
    if walking_time >= 20:
        return 0.0
    threshold = (1 + math.cos(math.pi * walking_time / 20)) / 2
    return threshold


def add_isochrone_interval_column(df: pd.DataFrame, intervals: list, 
                                 time_column_name: str) -> pd.DataFrame:
    
    if time_column_name not in df.columns: 
        return df
    
    def get_interval(time):
        if time <= intervals[0]: 
            return f"0-{intervals[0]} min"
        elif time <= intervals[1]: 
            return f"{intervals[0]}-{intervals[1]} min"
        elif time <= intervals[2]: 
            return f"{intervals[1]}-{intervals[2]} min"
        elif time <= intervals[3]: 
            return f"{intervals[2]}-{intervals[3]} min"
        else: 
            return f">{intervals[3]} min"
    
    df['access_interval'] = df[time_column_name].apply(get_interval)
    return df

    
def process_features(features_dict: dict, graph, node_travel_times: dict, location: str, profile: dict) -> dict:
    
    processed_features = {}
    
    for feature_name, gdf in features_dict.items():
        print(f"\nCalculating real walking time for {feature_name}...")
        processed_gdf = calculate_feature_travel_time(gdf, graph, node_travel_times, profile)
        processed_features[feature_name] = processed_gdf
        
        # Save to CSV
        save_features_to_csv(processed_gdf, location, feature_name.lower())
    
    return processed_features


def filter_accessible_features(features_dict: dict, max_time: float) -> dict:
    """
    Filters accessible features within maximum time.
    """
    accessible_features = {}
    
    for feature_name, gdf in features_dict.items():
        accessible_gdf = gdf[gdf['real_time'] <= max_time].copy()
        accessible_features[feature_name] = accessible_gdf
        
        print(f"Found {len(accessible_gdf)} {feature_name} "
              f"within {max_time} minutes walking distance.")
    
    return accessible_features


def process_and_save_pois_report(pois_gdf: gpd.GeoDataFrame, iso_intervals: list, location: str) -> None:
    """
    Processes POIs and saves detailed report to CSV.
    
    """
    if pois_gdf.empty:
        print("No accessible POIs found.")
        return
    
    pois_gdf['attractiveness_threshold'] = pois_gdf['real_time'].apply(calculate_attractiveness_threshold)
    
    # print("\nPOI count by type:")
    # print(pois_gdf['poi_type'].value_counts())

    df_to_save = add_isochrone_interval_column(
        pois_gdf.copy(), iso_intervals, time_column_name='real_time'
    )
    df_to_save['coord_x_proj'] = df_to_save.geometry.x
    df_to_save['coord_y_proj'] = df_to_save.geometry.y
    
    if df_to_save.crs != 'EPSG:4326':
        gdf_coords = df_to_save.to_crs('EPSG:4326')
    else:
        gdf_coords = df_to_save
    
    df_to_save['latitude'] = gdf_coords.geometry.y
    df_to_save['longitude'] = gdf_coords.geometry.x
    
    df_to_save.insert(0, 'sequential_id', range(1, 1 + len(df_to_save)))
    
    # Select and organize columns
    columns_of_interest = [
        'sequential_id', 'name', 'poi_type', 'real_time', 
        'attractiveness_threshold', 'access_interval', 'addr:street', 
        'addr:housenumber', 'latitude', 'longitude', 'coord_x_proj', 'coord_y_proj'
    ]
    
    existing_columns = [col for col in columns_of_interest if col in df_to_save.columns]
    df_final = df_to_save[existing_columns]
    df_final = df_final.rename(columns={'real_time': 'real_time_poi'})
    df_final = df_final.round({'real_time_poi': 2, 'attractiveness_threshold': 3})

    file_name = f'data/csv/report/{location}_pois_acessiveis.csv'
    
    ensure_directory_exists(file_name)
    df_final.to_csv(file_name, index=False, encoding='utf-8-sig')
    
    # print(f"\n-> File '{file_name}' saved with {len(df_final)} records.")
    # print(f"   Columns: {df_final.columns.to_list()}")


def process_and_save_crosswalks_report(crosswalks_gdf: gpd.GeoDataFrame, iso_intervals: list, location: str) -> None:
    
    if crosswalks_gdf.empty:
        print("No accessible crosswalks found.")
        return
    
    # print(f"\nTotal accessible crosswalks: {len(crosswalks_gdf)}")

    df_to_save = add_isochrone_interval_column(
        crosswalks_gdf.copy(), iso_intervals, time_column_name='real_time'
    )
    df_to_save['coord_x_proj'] = df_to_save.geometry.x
    df_to_save['coord_y_proj'] = df_to_save.geometry.y
    
    if df_to_save.crs != 'EPSG:4326':
        gdf_coords = df_to_save.to_crs('EPSG:4326')
    else:
        gdf_coords = df_to_save
    
    df_to_save['latitude'] = gdf_coords.geometry.y
    df_to_save['longitude'] = gdf_coords.geometry.x
    
    df_to_save.insert(0, 'sequential_id', range(1, 1 + len(df_to_save)))
    
    columns_of_interest = [
        'sequential_id', 'highway', 'footway', 'crossing', 'real_time', 
        'access_interval', 'latitude', 'longitude', 'coord_x_proj', 'coord_y_proj'
    ]
    
    existing_columns = [col for col in columns_of_interest if col in df_to_save.columns]
    df_final = df_to_save[existing_columns]
    df_final = df_final.rename(columns={'real_time': 'real_time_crosswalk'})
    df_final = df_final.round({'real_time_crosswalk': 2})

    file_name = f'data/csv/report/{location}_crosswalks_acessiveis.csv'

    ensure_directory_exists(file_name)
    df_final.to_csv(file_name, index=False, encoding='utf-8-sig')
    
    # print(f"\n-> File '{file_name}' saved with {len(df_final)} records.")
    # print(f"   Columns: {df_final.columns.to_list()}")


def process_and_save_traffic_signals_report(traffic_signals_gdf: gpd.GeoDataFrame, iso_intervals: list, location: str) -> None:
    
    if traffic_signals_gdf.empty:
        print("No accessible traffic signals found.")
        return
    
    # print(f"\nTotal accessible traffic signals: {len(traffic_signals_gdf)}")

    df_to_save = add_isochrone_interval_column(
        traffic_signals_gdf.copy(), iso_intervals, time_column_name='real_time'
    )
    df_to_save['coord_x_proj'] = df_to_save.geometry.x
    df_to_save['coord_y_proj'] = df_to_save.geometry.y

    if df_to_save.crs != 'EPSG:4326':
        gdf_coords = df_to_save.to_crs('EPSG:4326')
    else:
        gdf_coords = df_to_save
    
    df_to_save['latitude'] = gdf_coords.geometry.y
    df_to_save['longitude'] = gdf_coords.geometry.x
    
    df_to_save.insert(0, 'sequential_id', range(1, 1 + len(df_to_save)))

    columns_of_interest = [
        'sequential_id', 'highway', 'traffic_signals', 'crossing', 'real_time',
        'access_interval', 'latitude', 'longitude', 'coord_x_proj', 'coord_y_proj'
    ]
    
    existing_columns = [col for col in columns_of_interest if col in df_to_save.columns]
    df_final = df_to_save[existing_columns]
    df_final = df_final.rename(columns={'real_time': 'real_time_traffic_signal'})
    df_final = df_final.round({'real_time_traffic_signal': 2})

    file_name = f'data/csv/report/{location}_traffic_signals_acessiveis.csv'
    
    ensure_directory_exists(file_name)
    df_final.to_csv(file_name, index=False, encoding='utf-8-sig')
    
    # print(f"\n-> File '{file_name}' saved with {len(df_final)} records.")
    # print(f"   Columns: {df_final.columns.to_list()}")


if __name__ == '__main__':
    
    ensure_data_directories()
    central_point, location, dem_path = select_location()
    
    NETWORK_TYPE = "walk"
    ISO_INTERVALS = [5, 10, 15, 20]
    DISTANCE = 1800
    H3_RESOLUTION = 11
    ISO_COLORS = ox.plot.get_colors(n=len(ISO_INTERVALS), cmap="plasma", start=0)  
    
    POI_COLORS = {
        'bar': '#FDBF6F', 'cafe': '#A6761D', 'fast_food': '#FF7F00', 
        'restaurant': '#E41A1C', 'college': '#377EB8', 'kindergarten': '#984EA3', 
        'library': '#4DAF4A', 'school': '#377EB8', 'university': '#377EB8', 
        'bicycle_parking': '#CCCCCC', 'bank': '#FFFF33', 'clinic': '#FB8072', 
        'dentist': '#BEBADA', 'doctors': '#FB8072', 'hospital': '#E31A1C',
        'pharmacy': '#8DD3C7', 'cinema': '#BC80BD', 'theatre': '#BC80BD', 
        'hotel': '#B3DE69',
        # New POIs with viridis/cividis palette colors
        'gym': '#440154', 'fitness_centre': '#440154', 'fitness_center': '#440154',
        'supermarket': '#31688E', 'convenience': '#35B779', 
        'bakery': '#FDE724', 'greengrocer': '#6DCD59', 'grocery': '#35B779'
    }
    
    WALKING_PROFILES = {
    "average_adult": {
        "name": "Average Adult / Tourist",
        "speed_walk": 5.0,
        "uphill_factor": 0.80,
        "downhill_factor": 1.15
    },
    "elderly": {
        "name": "Elderly / Reduced Mobility",
        "speed_walk": 4.0,
        "uphill_factor": 0.65,
        "downhill_factor": 0.95
    },
    "athlete": {
        "name": "Young / Athlete",
        "speed_walk": 6.0,
        "uphill_factor": 0.90,
        "downhill_factor": 1.20
    }}

    print(f"\nCentral point: {central_point}\nLocation: {location}\nDEM file: {dem_path}")
    
    useful_tags_way = ox.settings.useful_tags_way.copy()
    useful_tags_way.append('sidewalk:width')
    useful_tags_way.append('sidewalk:left:width')
    useful_tags_way.append('sidewalk:right:width')
    useful_tags_way.append('sidewalk')

    ox.settings.useful_tags_way = useful_tags_way
    
    # Data acquisition and processing
    print("Obtaining walkable street network graph...")
    G = ox.graph_from_point(central_point, dist=DISTANCE, network_type=NETWORK_TYPE)
    print("Graph obtained successfully!")

    save_nodes_and_edges(G, location, 'before_projection')
    G = ox.projection.project_graph(G)
    dem_path = reproject_dem(dem_path, G.graph['crs'])
    save_nodes_and_edges(G, location, 'before_elevation')
    ox.elevation.add_node_elevations_raster(G, filepath=dem_path)
    save_nodes_and_edges(G, location, 'after_elevation')

    center_node = get_center_node(G, central_point)
    
    print("Obtaining green and water areas...")
    gdf_green, gdf_water = get_green_and_water_areas(central_point, 
                                                     DISTANCE, G.graph['crs'])

    basic_title = "Basic Map - " + location
    basic_map_filename = f"data/visualizations/basic_map_{location}"
    plot_basic_map(G, center_node, gdf_green, gdf_water,
                   basic_title, basic_map_filename)

    selected_profile, profile_key = select_user_profile(WALKING_PROFILES)
    print(f"\n--- Simulating with profile: {selected_profile['name']} ---")
    
    G = compute_edge_tobler(G, profile=selected_profile)
    save_nodes_and_edges(G, location, 'after_tobler')

    node_travel_times = compute_node_travel_times(G, center_node)
    assign_edge_colors(G, node_travel_times, ISO_INTERVALS, ISO_COLORS)
    save_nodes_and_edges(G, location, 'after_edge_colors')
    
    isochrone_title = "Map with Isochrones - " + location
    isochrone_map_filename = f"data/visualizations/map_isochrones_{location}_{profile_key}"
    plot_map_with_isochrones(G, center_node, ISO_COLORS, ISO_INTERVALS, gdf_green, 
                            gdf_water, central_point, DISTANCE,
                            title=isochrone_title, filename=isochrone_map_filename)

    print("\nObtaining features...")
    raw_features = {
        'pois': map_poi_colors_and_types(get_pois(central_point, DISTANCE, G.graph['crs']), POI_COLORS),
        'crosswalks': get_crosswalks(central_point, DISTANCE, G.graph['crs']),
        'traffic_signals': get_traffic_signals(central_point, DISTANCE, G.graph['crs'])
    }
    
    processed_features = process_features(raw_features, G, node_travel_times, location, selected_profile)
    
    max_time = ISO_INTERVALS[-1]
    accessible_features = filter_accessible_features(processed_features, max_time)
    
    process_and_save_pois_report(accessible_features['pois'], ISO_INTERVALS, location)
    process_and_save_crosswalks_report(accessible_features['crosswalks'], ISO_INTERVALS, location)
    process_and_save_traffic_signals_report(accessible_features['traffic_signals'], ISO_INTERVALS, location)

    isochrone_pois_title = "Map with Isochrones and POIs - " + location
    isochrone_pois_map_filename = f"data/visualizations/map_isochrones_pois_{location}_{profile_key}"
    plot_map_with_pois_isochrones(G, center_node, ISO_COLORS, ISO_INTERVALS, gdf_green, 
                                  gdf_water, accessible_features['pois'], title=isochrone_pois_title,
                                  filename=isochrone_pois_map_filename)

    isochrone_pois_h3_title = "Map with Isochrones, POIs and H3 Grid - " + location
    isochrone_pois_h3_map_filename = f"data/visualizations/map_isochrones_pois_h3_{location}_{profile_key}"
    plot_map_with_pois_isochrones_h3(G, center_node, ISO_COLORS, ISO_INTERVALS, gdf_green, 
                                     gdf_water, accessible_features['pois'], central_point, DISTANCE, 
                                     title=isochrone_pois_h3_title,
                                     filename=isochrone_pois_h3_map_filename,
                                     h3_resolution=H3_RESOLUTION)

    print("\nSaving H3 hexagon information...")
    df_hexagons = save_h3_hexagons_to_csv(G, central_point, DISTANCE, location, 
                                         node_travel_times, H3_RESOLUTION, accessible_features, gdf_green, gdf_water, profile_key)