import folium
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from sklearn.utils import resample
from plotly.subplots import make_subplots

def rand_jitter(list):
    scale_factor = max(list) - min(list)
    if scale_factor != 0:
        stdev = 0.001 * (scale_factor)
    else:
        stdev = 0.0003
    return list + np.random.randn(len(list)) * stdev

def get_travel_time(solution_path, graph, weight):
    travel_time = 0
    for i in range(len(solution_path)-1):
        sp = solution_path[i]
        ep = solution_path[i+1]
        travel_time += graph.get_edge_data(sp, ep)[weight]
    return travel_time

def get_minimum_distances(df):
    return df.groupby("target").min().reset_index()

def control_polygon(lon, lat,  width, height):
    # defines the bezier control points for a location pin
    # width is the width of region bounded by the B curve, between the points where the tangents are vertical
    # height is the height of curve between the first control point and intersection point with the vertical through it
    return  [[lon, lat], 
             [lon - np.sqrt(3)*width, lat + 4*height/3], 
             [lon + np.sqrt(3)*width, lat + 4*height/3], 
             [lon, lat] ]

def BezierCv(b, nr=30):# compute nr points on the Bezier curve of control points in list bom
    t = np.linspace(0, 1, nr)
    return [[deCasteljau(b, t[k]) for k in range(nr)]] #the points on a Bezier curve define here the coordinates
                                                       #of a Polygon in a geojson type dict

def facilities_on_map(fls, extra_text=None, title_pad_l=50):
    mapping = {}
    if extra_text is None:
        extra_text = [""]*len(fls)
        
    for i in range(len(fls)):
        if "StochasticFacilityLocation" in str(type(fls[i])):
            mapping[f"stochastic_{extra_text[i]}_{fls[i].n_of_locations_to_choose}"] = fls[i]
            print(mapping)
        else:
            mapping[f"deterministic_{extra_text[i]}_{fls[i].n_of_locations_to_choose}"] = fls[i]
    
    lats = {}
    lons = {}
    
    lat_global = fls[0].coordinates.geometry.y
    lon_global = fls[0].coordinates.geometry.x

        
    for k, fl in mapping.items():
        if "deterministic" in k:
            lats[k] = [p.geometry.y for p in fl.locations_coordinates]
            lons[k] = [p.geometry.x for p in fl.locations_coordinates]
        elif "stochastic" in k:
            if fl.n_of_locations_to_choose == 1:
                idx = int(pd.Series([k2 if fl.first_stage_solution[k2] != 0 else None 
                        for k2 in fl.first_stage_solution.keys()]).dropna().iloc[0])

                stochastic_locations_coordinates = fl.coordinates.loc[idx]
                    
                lats[k] = [p.y for p in stochastic_locations_coordinates]
                lons[k] = [p.x for p in stochastic_locations_coordinates]
            else:
                lats[k] = [p.geometry.y for p in fl.locations_coordinates]
                lons[k] = [p.geometry.x for p in fl.locations_coordinates]
    fig = go.Figure()
        
    fig.add_trace(go.Scattermapbox(
            lat=lat_global,
            lon=lon_global,
            mode='markers',
            marker=dict(
                color=["grey"]*fls[0].coordinates.shape[0],
                size=4.5,
            ),
            hovertemplate='<extra></extra>',
            showlegend=False,
        ))
    
    is_s = False
    for k in mapping.keys():
        if "stochastic" in k:
            is_s = True
    colors = ["red", "black", "blue", "purple", "green", "orange", "pink", "brown", "black", "grey"]
    colors_mapping = {k: colors[i] for i, k in enumerate(mapping.keys())}
    geojd = {"type": "FeatureCollection"}
    geojd['features'] = []
    size = 10
    for k, fl in mapping.items():
        c=colors_mapping[k]
        if not is_s:
            n=k[len("deterministic")+1:]
        else:
            n=k
        
        fig.add_trace(go.Scattermapbox(
                lat=rand_jitter(lats[k]),
                lon=rand_jitter(lons[k]),
                mode='markers',
                marker=dict(
                    color=[c]*fl.n_of_locations_to_choose,
                    size=size,
                ),
                hovertemplate=f'<br>solution value: {round(fl.solution_value/60,2)} minutes<extra></extra>',
                name=n,
                showlegend=True,
            ))

        for lon, lat in zip(rand_jitter(lons[k]), rand_jitter(lats[k])):
            b = control_polygon(lon, lat, width=0.24, height= 0.22) #The width and height of a location pin 
                                                                    # are chosen by trial and error
            bez = BezierCv(b, nr=30)
            geojd['features'].append({ "type": "Feature",
                                    "geometry": {"type": "Polygon",
                                                    "coordinates": bez }})
                                
                                
    layers=[dict(sourcetype='geojson',
                    source=geojd,
                    below=' ', 
                    type='fill',   
                    color = c,
                    opacity=0.9
    )]


    if is_s:
        title = "deterministic and stochastic solution comparison"
    else:
        title = "deterministic solution comparison"
        
    fig.update_layout(title=f"<b>{title}</b><br>             number of facilities: "+str(fls[0].n_of_locations_to_choose),
                        mapbox=dict(
                            style="open-street-map",
                            center=dict(lat=fls[0].coordinates.geometry.y.mean(), lon=fls[0].coordinates.geometry.x.mean()),
                            layers=layers,
                            zoom=9
                            ),
                        legend=dict(
                            orientation='h',  # Set the orientation to 'h' for horizontal
                            yanchor='bottom',  # Anchor the legend to the bottom
                            y=-0.1,  # Adjust the y position to place the legend below the figure
                            xanchor='left',  # Anchor the legend to the left
                            x=0  # Adjust the x position if necessary
                        ),
                        title_pad_l=title_pad_l,
                        height=700,
                        width=500,
                        xaxis_title="time of the day",)

    return fig

def visualize_longest_paths(dfs, average_graphs):
    # ------------------------------ prepare the data ----------------------------------#
    dfs_min = {}
    for key, df in dfs.items():
        dfs_min[key] = get_minimum_distances(df).sort_values(by="travel_time", ascending=False)
    
    sources = {}
    destinations = {}
    solution_paths = {}

    for key, df in dfs_min.items():
        try:
            if key[0] == key[1] or (key[0] == "all-day-free-flow" and key[1] == "all-day" and key[2] == "weight2"):
                sources[key] = df.iloc[0]["source"]
                destinations[key] = df.iloc[0]["target"]
                solution_paths[key] = nx.dijkstra_path(G=average_graphs[key[1].replace("-", "_")],
                                                    source=sources[key],
                                                    target=destinations[key],
                                                    weight=key[2])
            else:
                continue
        except:
            print("Skipping: ", key[0])
        
    travel_time = {}
    for key in solution_paths.keys():
        travel_time[key] = {}
        for time in ['all_day_free_flow', 'all_day', 'morning', 'midday', 'afternoon']:
            if key[0].replace("-", "_") == time:
                if time == "all_day_free_flow":
                    travel_time[key][time] = nx.dijkstra_path_length(G=average_graphs["all_day"], source=sources[key], target=destinations[key], weight=key[2])
                else:
                    # print(key[0].replace("-", "_"), time)
                    travel_time[key][time] = nx.dijkstra_path_length(G=average_graphs[time], source=sources[key], target=destinations[key], weight=key[2])
            else:
                if time == "all_day_free_flow":
                    travel_time[key][time] = get_travel_time(solution_paths[key], average_graphs["all_day"], "weight2")
                else:
                    travel_time[key][time] = get_travel_time(solution_paths[key], average_graphs[time], "weight")
            minutes = int(travel_time[key][time]/60)
            seconds = int(travel_time[key][time]%60)
            travel_time[key][time] = str(minutes) + " min" + " " + str(seconds) + " sec"
    
    #----------------------------------- design the map --------------------------------#
    center_pt = [60.41, 5.32415]
    color_mapping = {
        "all-day-free-flow":"red",
        "all-day":"black",
        "morning":"blue",
        "midday":"purple",
        "afternoon":"green",
    }
    map = folium.Map(location=center_pt, tiles="OpenStreetMap", zoom_start=11)

    tooltip_targets = {}
    for key in solution_paths.keys():
        target = solution_paths[key][-1]
        if target not in tooltip_targets.keys():
            tooltip_targets[target] = f"<b>Farthest location for:</b>" 
        
        tooltip_targets[target] += f"<br>- {key[0].upper()}"

    for key in solution_paths.keys():
        tooltip_source =f"<b>{key[0].upper()}</b><br>(opt locations)<br><br>"+"<br>- ".join(["<b>Travel time</b>:"]+[rf"{time}: " + travel_time[key][time] for time in 
                                ['all_day_free_flow', 'all_day', 'morning', 'midday', 'afternoon']])

        start_marker = folium.Marker(location=(solution_paths[key][0][1]+np.random.normal(0, 0.0003, 1),
                                            solution_paths[key][0][0]+np.random.normal(0, 0.0003, 1)),
                    icon=folium.Icon(color=color_mapping[key[0]], prefix='fa',icon='car'),
                    tooltip=tooltip_source)
        start_marker.add_to(map)
        
        folium.Marker(location=(solution_paths[key][-1][1], solution_paths[key][-1][0]),
                    tooltip=tooltip_targets[solution_paths[key][-1]],
                    icon=folium.Icon(color='gray', prefix='fa',icon='crosshairs'),).add_to(map)

        path = folium.PolyLine(locations=[(node[1], node[0]) for node in solution_paths[key]], 
                        color=color_mapping[key[0]],
                        tooltip=key[0],
                        weight=2,)

        path.add_to(map)
    
    return map
        
def compute_rel_diff(fls_exact, dfs, dfs_worst, time):
    df_min = get_minimum_distances(dfs[("all-day-free-flow", time.replace("_", "-"), "weight")])
    df_worst_min = get_minimum_distances(dfs_worst[("all-day-free-flow", time.replace("_","-"), "weight")])

    a = round(fls_exact[time].solution_value/60, 3)

    b = df_min.sort_values(by="travel_time", ascending=False).iloc[0].travel_time
    b_worst = df_worst_min.sort_values(by="travel_time", ascending=False).iloc[0].travel_time

    return a, b, b_worst

def objective_function_value_under_different_cases(a, b, b_worst):
    
    plot_data = []
    
    for i in range(len(a)):
        plot_data.append(a[i])
        plot_data.append(b[i])
        plot_data.append(b_worst[i])
    
    fig = make_subplots(rows=1, cols=1,)
    fig.update_layout(title="<b>Outsample evaluation of free flow solution<b>",
                        title_pad_l=175,
                        height=500,
                        width=1200,
                        yaxis_title="time (minutes)")

    fig.add_trace(go.Bar(y=plot_data,
                        x=["op sol all_day", "ff sol in all_day scenario", "ff sol in all_day worst scenario", 
                           "op sol morning", "ff sol in morning", "ff sol in morning worst scenario",
                           "op sol midday", "ff sol in midday", "ff sol in midday worst scenario",
                           "op sol afternoon", "ff sol in afternoon", "ff sol in afternoon worst scenario"],
                        marker=dict(
                            color=["lightblue", "blue", "navy"]*len(plot_data),
                            )), row=1, col=1)
    
    return fig

def outsample_evaluation_relative_differences(a, b, b_worst):
    rel_diffs = [round(abs(a_-b_)/a_ * 100,3) for a_, b_ in zip(a,b)]
    rel_diffs_worst = [round(abs(a_-b_)/a_ * 100,3) for a_, b_ in zip(a,b_worst)]
    
    fig = make_subplots(rows=1, cols=1,)
    fig.update_layout(title="<b>Outsample evaluation, relative differences<b>",
                  title_pad_l=200,
                  height=500,
                  width=600,
                  yaxis_title="relative difference [%]")

    fig.update_yaxes(range=[0, 100])

    fig.add_trace(go.Bar(y=rel_diffs, 
                     name="average scenario",
                     marker=dict(color=["blue"]*len(rel_diffs)),
                     x=["all_day", "morning", "midday", "afternoon"],), row=1, col=1)

    fig.add_trace(go.Bar(y=rel_diffs_worst,
                     name="average worst scenario",
                     marker=dict(color=["navy"]*len(rel_diffs_worst)),
                     x=["all_day", "morning", "midday", "afternoon"],), row=1, col=1)

    fig.update_layout(legend=dict(
                            orientation='h',  # Set the orientation to 'h' for horizontal
                            yanchor='bottom',  # Anchor the legend to the bottom
                            y=1.02,  # Adjust the y position to place the legend below the figure
                            xanchor='left',  # Anchor the legend to the left
                            x=0  # Adjust the x position if necessary
                        ),)
    return fig

def compute_min_distance_df(dfs, dfs_worst):
    dfs_min_list = []
    dfs_min_worst_list = []
    
    for time in ["all_day", "morning", "midday", "afternoon"]:
        df_min = get_minimum_distances(dfs[('all-day-free-flow', time.replace("_","-"), "weight")])
        df_min_worst = get_minimum_distances(dfs_worst[('all-day-free-flow', time.replace("_","-"), "weight")])

        dfs_min_list.append(df_min)
        dfs_min_worst_list.append(df_min_worst)
    
    df_min = get_minimum_distances(dfs[('all-day-free-flow', "all-day", "weight2")])[["target", "travel_time"]]
    
    for df, name in zip(dfs_min_list+
                        dfs_min_worst_list, 
                        ["all_day", "morning", "midday", "afternoon"]+
                        ["worst_all_day", "worst_morning", "worst_midday", "worst_afternoon"]):
        
        df_min = df_min.merge(df[["target", "travel_time"]], 
                            on="target", 
                            suffixes=(None, "_"+name),
                            how="outer")

    df_min = df_min.rename(columns={"travel_time": "travel_time_free_flow"})
    return df_min

def travel_times_distribution_under_different_cases(df_min):
    fig = go.Figure()

    show_legend = [True]+[False]*len(df_min.columns[1:])

    fig.update_layout(title="<b>Distribution for free flow travel times solution across average scenarios<b>",
                    title_pad_l=150,
                    height=500,
                    width=1200,)

    for i, name in enumerate(["free_flow", "all_day", "morning", "midday", "afternoon",
                            "worst_all_day", "worst_morning", "worst_midday", "worst_afternoon"]):
        fig.add_trace(go.Violin(y=df_min["travel_time_free_flow"],
                                name=name,
                                box_visible=True,
                                meanline_visible=False,
                                hoverinfo="none",
                                side="negative",
                                line_color="lightseagreen",
                                showlegend=show_legend[i]))
        
        fig.add_trace(go.Violin(y=df_min["travel_time_"+name],
                                name=name,
                                box_visible=True,
                                meanline_visible=False,
                                hoverinfo="none",
                                side="positive",
                                line_color="mediumpurple",
                                showlegend=show_legend[-1]))
        
    return fig

def compute_CI(df_min):
    mean_ci = pd.DataFrame({"mean": None, "lower_bound": None, "upper_bound": None}, 
                       index=df_min.columns[1:])

    for col in df_min.columns[1:]:
        # Number of bootstrap iterations
        n_iterations = 1000

        # Confidence level (e.g., 95%)
        confidence_level = 0.95

        # Array to store bootstrap sample statistics
        bootstrap_means = []

        # Perform bootstrap iterations
        for _ in range(n_iterations):
            bootstrap_sample = resample(df_min[col], replace=True, n_samples=len(df_min))
            bootstrap_mean = np.mean(bootstrap_sample)
            bootstrap_means.append(bootstrap_mean)

        # Compute confidence interval
        lower_bound = np.percentile(bootstrap_means, (1 - confidence_level) / 2 * 100)
        upper_bound = np.percentile(bootstrap_means, (1 + confidence_level) / 2 * 100)

        # Add to dataframe
        mean_ci.loc[col] = [df_min[col].mean(), lower_bound, upper_bound]
        
    # Print the confidence interval
    mean_ci = mean_ci.sort_values(by="mean", ascending=False).round(3)
    
    return mean_ci

def average_travel_time_across_under_different_cases(df_min):
    fig = go.Figure()

    mean_ci = compute_CI(df_min)
    
    fig.update_layout(title="<b>Average travel time for free flow solution across average scenarios<b>",
                    title_pad_l=130,
                    height=600,
                    width=1100,
                    yaxis_title="mean travel time [min]")

    fig.add_trace(go.Bar(x=mean_ci.index, 
                        y = mean_ci["mean"],
                        width=0.5,
                        name='mean'))

    # Add the vertical line
    for col in df_min.columns[1:]:
            fig.add_shape(type='line',
                    x0=col, y0=mean_ci.loc[col]["lower_bound"],
                    x1=col, y1=mean_ci.loc[col]["upper_bound"],
                    xref='x', yref='y',
                    line=dict(color='red', width=10))

    fig.update_yaxes(range=[0, mean_ci["upper_bound"].max()+1])

    return fig
