#!/usr/bin/env python
# coding: utf-8

# # BiciMAD interactive network visualization dashboard

# In[2]:


from typing import Union, List

from geopy.distance import geodesic
import networkx as nx
from scipy import stats
import numpy as np
import json
import pandas as pd


# In[3]:


# Type aliases
Series = pd.core.series.Series
DataFrame = pd.core.frame.DataFrame
Graph = nx.classes.graph.Graph


# In[4]:


def distance_calc(row: Series, start: tuple = (40.4166, -3.70384)) -> float:
    """Calculate the distance between two points.

    Arguments:
        row(pandas.core.series.Series): row of a DataFrame containing
            at least 'Latitud' and 'Longitud' columns
        start(tuple): tuple containing longitude and latitude

    Returns:
        float: distance in meters
    """
    stop = (row["Latitud"], row["Longitud"])

    return geodesic(start, stop).meters


# In[5]:


def load_stations(stations_path: str = "data/bases_bicimad.csv") -> DataFrame:
    """Load the stations CSV merging duplicated stations and
    adding a column with the distance of the station to the
    center of the city.

    Arguments:
        stations_path(str): path to the stations CSV file

    Returns:
        pandas.core.frame.DataFrame: resulting DataFrame
    """
    stations = pd.read_csv(stations_path, sep=";", index_col=False)
    # Remove duplicated stations (added manually)
    stations.drop(
        stations.loc[
            stations["Número"].isin(
                [
                    "001 b",
                    "020 ampliacion",
                    "025 b",
                    "080 b",
                    "090 ampliacion",
                    "106 b",
                    "111 b",
                    "116 b",
                    "128 ampliacion",
                    "140 ampliación",
                    "161 ampliacion",
                ]
            )
        ].index,
        inplace=True,
    )

    # Replace names of some of the original stations that were
    # duplicated (added manually)
    stations["Número"] = stations["Número"].replace(
        {"001 a": 1, "025 a": 25, "080 a": 80, "106 a": 106, "111 a": 111, "116 a": 116}
    )

    # Convert 'Número' (station id) to numeric type
    stations["Número"] = pd.to_numeric(stations["Número"])
    # Add a columun with the distance of the station to the
    # center of the city.
    stations["dist_km0"] = stations.apply(lambda row: distance_calc(row), axis=1)

    return stations


# In[6]:


def load_trips(
    stations: DataFrame,
    trips_path: str = "data/201906_Usage_Bicimad.json",
    calc_ratios: bool = True,
) -> DataFrame:
    """Load the trips CSV and add the following columns (ratios if
    `calc_ratios` is `True`):
        `od_time_ratio`: travel time divided by the mean travel
            time of trips with the same origin and destination,
        `o_time_ratio`: travel time divided by the mean travel
            time of trips with the same origin
        `d_time_ratio`: travel time divided by the mean travel
            time of trips with the same destination
        `hour`: only the hour when the trip began (no minutes),
        `day_type`: boolean variable, True if the trip started on a
            Saturday or a Sunday
        `o_dist_km0`: distance of the trip origin station to the
            center of the city
        `d_dist_km0`: distance of the trip destination station to the
            center of the city


    Arguments:
        trips_path(str): path to the trips CSV file
        stations(pandas.core.frame.DataFrame): stations DataFrame generated
            from `load_stations()`

    Returns:
        pandas.core.frame.DataFrame: resulting DataFrame
    """

    trips = pd.read_json(
        trips_path,
        lines=True,
        encoding="latin-1",
        dtype={"zip_code": "Int64"},
        convert_dates="unplug_hourTime",
    ).drop(["track"], axis=1, errors="ignore")

    trips["_id"] = trips["_id"].apply(lambda x: x["$oid"])
    trips["zip_code"] = pd.to_numeric(
        trips["zip_code"], errors="coerce", downcast="integer"
    )
    trips["unplug_hourTime"] = pd.to_datetime(
        trips["unplug_hourTime"].apply(
            lambda x: x["$date"] if isinstance(x, dict) else x
        ),
        errors="coerce",
    )
    # trips = trips[trips['travel_time'] < 60 * 60 * 8]
    if calc_ratios:
        trips["od_time_ratio"] = trips["travel_time"] / trips.groupby(
            ["idunplug_station", "idplug_station"]
        )["travel_time"].transform("mean")
        trips["o_time_ratio"] = trips["travel_time"] / trips.groupby(
            ["idunplug_station"]
        )["travel_time"].transform("mean")
        trips["d_time_ratio"] = trips["travel_time"] / trips.groupby(
            ["idplug_station"]
        )["travel_time"].transform("mean")
    trips["hour"] = trips["unplug_hourTime"].apply(lambda x: x.hour)
    trips["day_type"] = trips["unplug_hourTime"].apply(
        lambda x: int(x.weekday in [5, 6])
    )
    trips = (
        trips.merge(
            stations[["Número", "dist_km0"]],
            how="left",
            left_on="idunplug_station",
            right_on="Número",
        )
        .rename(columns={"dist_km0": "o_dist_km0"})
        .drop(columns="Número")
    )
    trips = (
        trips.merge(
            stations[["Número", "dist_km0"]],
            how="left",
            left_on="idplug_station",
            right_on="Número",
        )
        .rename(columns={"dist_km0": "d_dist_km0"})
        .drop(columns="Número")
    )
    #     trips = trips.fillna(trips.mean())

    return trips


# In[7]:


def load_situations(situations_path: str) -> DataFrame:
    """Load the situations json and add the following columns:
        `hour`: only the hour of the situation (no minutes),
        `day_type`: boolean variable, True if
            Saturday or a Sunday

    Arguments:
        situations_path(str): path to the situations json file

    Returns:
        pandas.core.frame.DataFrame: resulting DataFrame
    """
    data = []
    with open(situations_path, encoding="latin-1") as json_file:
        for line in json_file:
            data.append(json.loads(line))

    L = []
    for l in data:
        for s in l["stations"]:
            d = {"date": l["_id"]}
            for k, v in s.items():
                d[k] = v
            L.append(d)

    situations = pd.DataFrame(L)

    situations["longitude"] = pd.to_numeric(
        situations["longitude"].apply(lambda x: x.replace(",", ".")), errors="coerce"
    )
    situations["latitude"] = pd.to_numeric(
        situations["latitude"].apply(lambda x: x.replace(",", ".")), errors="coerce"
    )
    situations["date"] = pd.to_datetime(situations["date"], errors="coerce")
    situations["hour"] = situations["date"].apply(lambda x: x.hour)
    situations["day_type"] = situations["date"].apply(
        lambda x: int(x.weekday in [5, 6])
    )

    return situations


# In[8]:


def get_trips_grouped(trips: DataFrame) -> DataFrame:
    """Group the trips DataFrame by origin and destiny and
    aggregate by count and mean of the travel tiem. This
    represents the number of trips and mean travel time
    between pairs of stations (order matters).

    Arguments:
        trips(pandas.core.frame.DataFrame): trips DataFrame generated
            from `load_trips()`

    Returns:
        pandas.core.frame.DataFrame: resulting DataFrame
    """
    trips_grouped = (
        trips.groupby(["idunplug_station", "idplug_station"])
        .agg(weight=("travel_time", "count"), cost=("travel_time", "mean"))
        .reset_index()
    )
    # Convert the mean travel time to int64
    trips_grouped["cost"] = trips_grouped["cost"].astype("int64")

    return trips_grouped


# In[9]:


def generate_graph(trips_grouped: DataFrame, stations: DataFrame) -> Graph:
    """Generate BiciMAD NetworkX graph.

    Nodes attributes:
        size: degree of the node
        title: address of the station (Dirección)
        lon: logitude of the station
        lat: latitude of the station

    Edge attributes:
        title: cost of the edge

    Arguments:
        trips_grouped(pandas.core.frame.DataFrame): grouped trips DataFrame
            generated from `get_trips_grouped()`
        stations(pandas.core.frame.DataFrame): stations DataFrame generated
            from `load_stations()`

    Returns:
        networkx.classes.graph.Graph: NetworkX graph
    """
    # Generate graph
    G = nx.from_pandas_edgelist(
        trips_grouped,
        source="idunplug_station",
        target="idplug_station",
        edge_attr=["weight", "cost"],
        create_using=nx.DiGraph(),
    )

    pr = nx.algorithms.link_analysis.pagerank(G)

    # Add size, title, lon, and lat to the nodes
    for n in G.nodes:
        G.nodes[n]["degree"] = G.degree(n)
        G.nodes[n]["in_degree"] = G.in_degree(n)
        G.nodes[n]["out_degree"] = G.out_degree(n)
        G.nodes[n]["degree_w"] = G.degree(n, weight="weight")
        G.nodes[n]["in_degree_w"] = G.in_degree(n, weight="weight")
        G.nodes[n]["out_degree_w"] = G.out_degree(n, weight="weight")
        G.nodes[n]["pagerank"] = pr[n]
        G.nodes[n]["title"] = (
            stations[stations["Número"] == n]["Direccion"]
            .reset_index(drop=True)
            .get(0, None)
        )
        G.nodes[n]["lon"] = (
            stations[stations["Número"] == n]["Longitud"]
            .reset_index(drop=True)
            .get(0, None)
        )
        G.nodes[n]["lat"] = (
            stations[stations["Número"] == n]["Latitud"]
            .reset_index(drop=True)
            .get(0, None)
        )

    # Add title to the edges
    for t in G.edges:
        e = G.edges[t]
        e["title"] = "Cost: " + str(round(e["cost"] / 60, 3)) + " min."

    return G


# In[10]:


def get_edges(G: Graph) -> 2 * (list,):
    """Get lists of edges properties for representation with Plotly.

    The lists size is three times the number of edges.
    The lists contain:
        * origin longitude, destination longitude and None for each edge
        * origin latitude, destination latitude and None for each edge

    Arguments:
        G(networkx.classes.graph.Graph): NetworkX graph

    Retrurns:
        tuple: tuple containing the lists

    """
    # Initialization of the empty lists
    edge_x = []  # longitudes
    edge_y = []  # latitudes

    # Loop over the edges
    for edge in G.edges(data=True):
        # Get lon, lat of the edges
        x0, y0 = [G.nodes[edge[0]][key] for key in ["lon", "lat"] if edge[0] in G.nodes]
        x1, y1 = [G.nodes[edge[1]][key] for key in ["lon", "lat"] if edge[1] in G.nodes]
        # Append to their corresponding lists in the Plotly format
        # (origin longitude/latitude, destination lonigtude/latitude, None)
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    return (edge_x, edge_y)


# In[11]:


def get_nodes(G: Graph) -> 4 * (list,):
    """Get lists of nodes properties for representation with Plotly.

    The lists size is equal to the number of nodes.
    The lists contain:
        * node origin latitude
        * node origin longitude
        * node text
        * node id

    Arguments:
        G(networkx.classes.graph.Graph): NetworkX graph

    Retrurns:
        tuple: tuple containing the lists

    """
    # Initialization of the empty lists
    node_x = []  # longitudes
    node_y = []  # latitudes
    node_text = []  # texts
    node_id = []  # IDs

    # Loop over the nodes
    for node in G.nodes():
        # Get the longitude and latitude of the node
        x, y = [G.nodes[node][key] for key in ["lon", "lat"]]
        node_x.append(x)
        node_y.append(y)
        node_text.append(
            str(G.nodes[node]["title"])
            + ", # of connections: "
            + str(G.nodes[node]["size"])
        )
        node_id.append(node)

    return (node_x, node_y, node_text, node_id)


# In[12]:


def generate_graphs(
    trips: DataFrame, stations: DataFrame, by: str
) -> (List[int], List[Graph]):
    """Generate BiciMAD NetworkX graphs per hour, day...

    Arguments:
        trips(pandas.core.frame.DataFrame): trips DataFrame generated from `get_trips()`
        stations(pandas.core.frame.DataFrame): stations DataFrame generated
            from `load_stations()`
        by(str): grouping method, either "hour" or "day"

    Returns:
        List[int]: list of group keys (list of hours or days)
        List[networkx.classes.graph.Graph]: list of NetworkX graphs
    """
    indexes = []
    graphs = []

    if by == "hour":
        gby = trips["unplug_hourTime"].dt.hour
    elif by == "day":
        # gby = [trips["unplug_hourTime"].dt.day, trips["unplug_hourTime"].dt.strftime("%A")]  # untested
        gby = [
            d.strftime("%d") + ("*" if d.weekday() in [5, 6] else "")
            for d in trips["unplug_hourTime"]
        ]
    elif by == "weekday":
        gby = trips["unplug_hourTime"].dt.strftime("%w%A")
    else:
        raise

    for k, g in trips.groupby(gby):
        indexes.append(k)
        graphs.append(generate_graph(get_trips_grouped(g), stations))

    return indexes, graphs


# In[15]:


def generate_graphs(
    trips_gby: DataFrame, stations: DataFrame
) -> (List[int], List[Graph]):
    """Generate BiciMAD NetworkX graphs per hour, day...

    Arguments:
        trips_gby(pandas.core.frame.DataFrame): trips DataFrame grouped by `gby`
            with a `gby` column
        stations(pandas.core.frame.DataFrame): stations DataFrame generated
            from `load_stations()`

    Returns:
        List[int]: list of group keys (list of hours or days)
        List[networkx.classes.graph.Graph]: list of NetworkX graphs
    """
    indexes = []
    graphs = []

    for k, g in trips_gby.groupby("gby"):
        indexes.append(str(int(k)))
        graphs.append(generate_graph(g, stations))

    return indexes, graphs


# In[16]:


stations = load_stations()
