from typing import Union, List
import pickle
import json

from dash.dependencies import Input, Output
import dash
import dash_core_components as dcc
import dash_html_components as html
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from functions import *

# Type aliases
Series = pd.core.series.Series
DataFrame = pd.core.frame.DataFrame
Graph = nx.classes.graph.Graph
Figure = go.Figure

stations = load_stations()

G = nx.read_gpickle("data/dashboard/G.p")

ndays = 761

indexes_day = pickle.load(open("data/dashboard/indexes_day.p", "rb"))
graphs_day = pickle.load(open("data/dashboard/graphs_day.p", "rb"))
indexes_hour = pickle.load(open("data/dashboard/indexes_hour.p", "rb"))
graphs_hour = pickle.load(open("data/dashboard/graphs_hour.p", "rb"))
indexes_weekday = pickle.load(open("data/dashboard/indexes_weekday.p", "rb"))
graphs_weekday = pickle.load(open("data/dashboard/graphs_weekday.p", "rb"))
indexes_month = pickle.load(open("data/dashboard/indexes_month.p", "rb"))
graphs_month = pickle.load(open("data/dashboard/graphs_month.p", "rb"))

# Build App
app = dash.Dash(__name__)
app.config.update(
    {
        "routes_pathname_prefix": "/bicimad/",
        # "requests_pathname_prefix": "",
    }
)
server = app.server

node_size = "pagerank"  # One of "pagerank", "degree", "in_degree", "out_degree"...
max_size = max([n[1][node_size] for n in G.nodes(data=True)])

app.layout = html.Div(
    [
        html.Link(
            href="https://fonts.googleapis.com/css?family=Fira+Sans", rel="stylesheet"
        ),
        html.H1("BiciMAD network dashboard"),
        # Map row
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.H2(
                            "Visualization parameters", style={"text-align": "center"}
                        ),
                        html.Div(
                            id="visualization_parameters",
                            children=[
                                html.H3("Node size metric"),
                                dcc.Dropdown(
                                    id="node_size_metric",
                                    options=[
                                        {"label": "PageRank", "value": "pagerank"},
                                        {"label": "Degree", "value": "degree"},
                                        {"label": "In degree", "value": "in_degree"},
                                        {"label": "Out degree", "value": "out_degree"},
                                        {
                                            "label": "Weighted degree",
                                            "value": "degree_w",
                                        },
                                        {
                                            "label": "Weighted in degree",
                                            "value": "in_degree_w",
                                        },
                                        {
                                            "label": "Weighted out degree",
                                            "value": "out_degree_w",
                                        },
                                    ],
                                    value=node_size,
                                    clearable=False,
                                ),
                                html.H3("Slider aggregation"),
                                dcc.Dropdown(
                                    id="slider_aggregation",
                                    options=[
                                        {"label": "Hour", "value": "hour"},
                                        {"label": "Day", "value": "day"},
                                        {"label": "Weekday", "value": "weekday"},
                                        {"label": "Month", "value": "month"},
                                    ],
                                    value="hour",
                                    clearable=False,
                                ),
                                html.H3("Node size histogram"),
                                dcc.Graph(id="node_size", style={"padding": "1vw"}),
                            ],
                        ),
                    ],
                    style={
                        "flex": "1 0 20%",
                        "min-height": "100px",
                    },
                ),
                dcc.Graph(
                    id="map",
                    style={
                        "flex": "1 0 50%",
                    },
                ),
            ],
            style={
                "display": "flex",
                "width": "100%",
                "padding": "1vw",
            },
        ),
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.H2("Node info", style={"text-align": "center"}),
                        html.Div(id="node_info"),
                    ],
                    style={
                        "display": "true",
                        "flex": "1 0 20%",
                        "min-height": "100px",
                    },
                )
            ],
            style={"margin": "1vw"},
        ),
    ]
)

# Define callbacks


@app.callback(
    [Output("node_size", "figure"), Output("node_size", "selectedData")],
    [
        Input("node_size_metric", "value"),
        Input("slider_aggregation", "value"),
    ],
)
def node_size_histogram(node_size, slider_aggregation) -> Figure:
    #     max_size = max([n[1][node_size] for n in G.nodes(data=True)])
    if slider_aggregation == "hour":
        indexes, graphs = indexes_hour, graphs_hour
    elif slider_aggregation == "day":
        indexes, graphs = indexes_day, graphs_day
    elif slider_aggregation == "weekday":
        indexes, graphs = indexes_weekday, graphs_weekday
    elif slider_aggregation == "month":
        indexes, graphs = indexes_month, graphs_month
    else:
        raise
    return {
        "data": [
            {
                #             "x": [n[1][node_size] for n in G.nodes(data=True)],
                "x": [
                    (
                        np.mean(
                            [
                                g.nodes.get(n[0], {node_size: None})[node_size]
                                for g in graphs
                                if g.nodes.get(n[0], {node_size: None})[node_size]
                                != None
                            ]
                        )
                    )
                    for n in G.nodes(data=True)
                ],
                "customdata": list(G.nodes(data=True)),
                "type": "histogram",
            }
        ],
        "layout": {
            "dragmode": "select",
            "title": None,
            "xaxis": {"automargin": True, "title": {"text": "Size metric"}},
            "yaxis": {"automargin": True, "title": {"text": "Count"}},
            "height": 200,
            "margin": {"t": 10, "l": 10, "r": 10},
        },
    }, None


@app.callback(Output("node_info", "children"), [Input("map", "hoverData")])
def display_selected_data(selectedData):
    if selectedData != None:
        if False and "customdata" in selectedData["points"][0]:
            #             size = ((selectedData["points"][0]["marker.size"])**2) * max_size / size_factor
            id = selectedData["points"][0]["customdata"][0]
            name = selectedData["points"][0]["customdata"][1]["title"]
            degree = selectedData["points"][0]["customdata"][1]["degree"]
            in_degree = selectedData["points"][0]["customdata"][1]["in_degree"]
            out_degree = selectedData["points"][0]["customdata"][1]["out_degree"]
            pagerank = selectedData["points"][0]["customdata"][1]["pagerank"]
            return [
                #                 html.P("Current metric size: " + str(size)),
                html.P("ID: " + str(id)),
                html.P("Name: " + str(name)),
                html.P("Degree: " + str(degree)),
                html.P("In degree: " + str(in_degree)),
                html.P("Out degree: " + str(out_degree)),
                html.P("PageRank: " + "{:.2e}".format(pagerank)),
            ]
        else:
            return [html.P(json.dumps(selectedData["points"][0], indent=2))]


@app.callback(
    Output("map", "figure"),
    [
        Input("node_size_metric", "value"),
        Input("slider_aggregation", "value"),
        Input("node_size", "selectedData"),
        Input("map", "clickData"),
    ],
)
def update_figure(node_size, slider_aggregation, selection, click_data) -> Figure:
    """Generate the network graph of BiciMAD and return the plot.

    Returns:
        plotly.graph_objs._figure.Figure: representation of the network
    """
    if slider_aggregation == "hour":
        indexes, graphs = indexes_hour, graphs_hour
    elif slider_aggregation == "day":
        indexes, graphs = indexes_day, graphs_day
    elif slider_aggregation == "weekday":
        indexes, graphs = indexes_weekday, graphs_weekday
    elif slider_aggregation == "month":
        indexes, graphs = indexes_month, graphs_month
    else:
        raise

    ids = range(len(indexes))  # indices of the selected agroupation

    # calculate max size of any node at anytime
    max_size = max([max([n[1][node_size] for n in g.nodes(data=True)]) for g in graphs])
    # max_size_all = max([n[1][node_size] for n in G.nodes(data=True)])
    size_factor = 20 ** 2

    top_size = 10

    if click_data == None:
        clicked_nodes = []
    else:
        clicked_nodes = [
            p.get("customdata", ["error"])[0]
            for p in click_data.get("points", [{"customdata": [1]}])
            if p.get("customdata", ["error"])[0] != "error"
        ]

    def filter_edge_out(a: int, b: int) -> bool:
        if a not in clicked_nodes:
            return False
        else:
            node, other = (a, b)

            return other in [e[1] for e in top_edges_out[node][0 : top_size - 1]]

    def filter_edge_in(a: int, b: int) -> bool:
        if b not in clicked_nodes:
            return False
        else:
            node, other = (b, a)

            return other in [e[0] for e in top_edges_in[node][0 : top_size - 1]]

    selected_nodes = []
    frames = []
    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": slider_aggregation.title() + ": ",
            "visible": True,
            "xanchor": "right",
        },
        "transition": {"duration": 300, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": [],
    }

    duration = 800 + (5000 / len(indexes))

    for i in ids:
        top_edges_out = {}

        for n in clicked_nodes:
            top_edges_out[n] = sorted(
                graphs[i].out_edges([n], data=True),
                key=lambda x: x[2]["weight"],
                reverse=True,
            )

        top_edges_in = {}

        for n in clicked_nodes:
            top_edges_in[n] = sorted(
                graphs[i].in_edges([n], data=True),
                key=lambda x: x[2]["weight"],
                reverse=True,
            )

        def filter_node(a: int) -> bool:
            if selection == None or a in clicked_nodes:
                return True
            else:
                range_x = selection["range"]["x"]
                return range_x[0] < graphs[i].nodes[a][node_size] < range_x[1]

        view = nx.subgraph_view(graphs[i], filter_node=filter_node)

        edges_in = get_edges(nx.subgraph_view(view, filter_edge=filter_edge_in))
        edges_out = get_edges(nx.subgraph_view(view, filter_edge=filter_edge_out))

        selected_nodes = list(view.nodes(data=True))

        frames.append(
            {
                "name": str(indexes[i]),
                "data": [
                    {
                        "name": "Stations",
                        "lon": [n[1]["lon"] for n in selected_nodes],
                        "lat": [n[1]["lat"] for n in selected_nodes],
                        "customdata": selected_nodes,
                        "type": "scattermapbox",
                        "hoverinfo": "text",
                        "hovertext": [n[1]["title"] for n in selected_nodes],
                        "marker": {
                            "size": [
                                (n[1][node_size] * size_factor / max_size) ** (1 / 2)
                                for n in selected_nodes
                            ],
                            "color": "blue",
                        },
                    },
                    {
                        "name": "Incoming",
                        "mode": "lines",
                        "lon": edges_in[0],
                        "lat": edges_in[1],
                        "type": "scattermapbox",
                        "line": dict(dash="dash", width=4, color="green"),
                    },
                    {
                        "name": "Outgoing",
                        "mode": "lines",
                        "lon": edges_out[0],
                        "lat": edges_out[1],
                        "type": "scattermapbox",
                        "line": dict(dash="dash", width=2, color="red"),
                    },
                ],
            }
        )
        sliders_dict["steps"].append(
            {
                "args": [
                    [indexes[i]],
                    {
                        "frame": {"duration": 300, "redraw": True},
                        "mode": "immediate",
                        "transition": {"duration": 300},
                    },
                ],
                "label": str(indexes[i]),
                "style": {"color": "#77b0b1"},  # not working
                "method": "animate",
            }
        )

    def filter_node(a: int) -> bool:
        if selection == None or a in clicked_nodes or a not in node_sizes:
            return True
        else:
            range_x = selection["range"]["x"]
            return range_x[0] < node_sizes[a] < range_x[1]

    view = nx.subgraph_view(G, filter_node=filter_node)

    node_sizes = {}

    for n in view.nodes():
        node_sizes[n] = np.mean(
            [
                g.nodes.get(n, {node_size: None})[node_size]
                for g in graphs
                if g.nodes.get(n, {node_size: None})[node_size] != None
            ]
        )

    top_edges_out = {}

    for n in clicked_nodes:
        top_edges_out[n] = sorted(
            view.out_edges([n], data=True), key=lambda x: x[2]["weight"], reverse=True
        )

    top_edges_in = {}

    for n in clicked_nodes:
        top_edges_in[n] = sorted(
            view.in_edges([n], data=True), key=lambda x: x[2]["weight"], reverse=True
        )

    #     if selection == None:
    #         all_nodes = list(G.nodes(data=True))
    #     else:
    #         range_x = selection["range"]["x"]
    #         all_nodes = [
    #             n for n in G.nodes(data=True)
    #             if range_x[0] < n[1][node_size] < range_x[1]
    #         ]

    #     nodes = get_nodes(view)
    edges_in = get_edges(nx.subgraph_view(view, filter_edge=filter_edge_in))
    edges_out = get_edges(nx.subgraph_view(view, filter_edge=filter_edge_out))

    all_nodes = list(view.nodes(data=True))

    return {
        "data": [
            {
                "name": "Stations",
                "lon": [n[1]["lon"] for n in all_nodes],
                "lat": [n[1]["lat"] for n in all_nodes],
                "customdata": all_nodes,
                "type": "scattermapbox",
                "hoverinfo": "text",
                "hovertext": [n[1]["title"] for n in all_nodes],
                "marker": {
                    "size": [
                        (node_sizes[n[0]] * size_factor / max_size) ** (1 / 2)
                        for n in all_nodes
                    ],
                    "color": "blue",
                },
            },
            {
                "name": "Incoming",
                "mode": "lines",
                "lon": edges_in[0],
                "lat": edges_in[1],
                "type": "scattermapbox",
                "line": dict(dash="dash", width=4, color="green"),
            },
            {
                "name": "Outgoing",
                "mode": "lines",
                "lon": edges_out[0],
                "lat": edges_out[1],
                "type": "scattermapbox",
                "line": dict(dash="dash", width=2, color="red"),
            },
        ],
        "frames": frames,
        "layout": {
            "showlegend": True,
            "dragmode": "pan",
            "title": "Map",
            "xaxis": {
                "automargin": True,
            },
            "yaxis": {
                "automargin": True,
            },
            "height": 600,
            "margin": {"t": 10, "l": 10, "r": 10},
            "mapbox": {
                "center": {"lon": -3.7, "lat": 40.4277},
                "style": "open-street-map",
                "zoom": 11,
            },
            "sliders": [sliders_dict],
            "updatemenus": [
                {
                    "buttons": [
                        {
                            "args": [
                                None,
                                {
                                    "frame": {"duration": duration, "redraw": True},
                                    "fromcurrent": True,
                                    "transition": {
                                        "duration": 0.8 * duration,
                                        "easing": "quadratic-in-out",
                                    },
                                },
                            ],
                            "label": "Play",
                            "method": "animate",
                        },
                        {
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                            "label": "Pause",
                            "method": "animate",
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 87},
                    "showactive": False,
                    "type": "buttons",
                    "x": 0.1,
                    "xanchor": "right",
                    "y": 0,
                    "yanchor": "top",
                }
            ],
        },
    }


if __name__ == "__main__":
    app.run_server(debug=False)
