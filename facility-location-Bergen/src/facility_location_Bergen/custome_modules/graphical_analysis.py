import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.utils import resample
from plotly.subplots import make_subplots

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
    colors = ["red", "blue", "green", "orange", "purple", "yellow", "pink", "brown", "black", "grey"]
    colors_mapping = {k: colors[i] for i, k in enumerate(mapping.keys())}
    for k, fl in mapping.items():
        c=colors_mapping[k]
        if not is_s:
            n=k[len("deterministic")+1:]
        else:
            n=k
        
        fig.add_trace(go.Scattermapbox(
                lat=lats[k],
                lon=lons[k],
                mode='markers',
                marker=dict(
                    color=[c]*fl.n_of_locations_to_choose,
                    size=7,
                ),
                hovertemplate=f'<br>solution value: {round(fl.solution_value/60,2)} minutes<extra></extra>',
                name=n,
                showlegend=True,
            ))

    if is_s:
        title = "deterministic and stochastic solution comparison"
    else:
        title = "deterministic solution comparison"
        
    fig.update_layout(title=f"<b>{title}</b><br>             number of facilities: "+str(fls[0].n_of_locations_to_choose),
                        mapbox=dict(
                            style="open-street-map",
                            center=dict(lat=fls[0].coordinates.geometry.y.mean(), lon=fls[0].coordinates.geometry.x.mean()),
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


def get_minimum_distances(df):
    return df.groupby("target").min().reset_index()

def compute_rel_diff(fls_exact, dfs, dfs_worst, time):
    df_min = get_minimum_distances(dfs[(time, "weight")])
    df_worst_min = get_minimum_distances(dfs_worst[(time, "weight")])

    a = round(fls_exact[time].solution_value/60, 3)

    b = df_min.sort_values(by="travel_time", ascending=False).iloc[0].travel_time
    b_worst = df_worst_min.sort_values(by="travel_time", ascending=False).iloc[0].travel_time

    rel_difference = round(abs(a-b)/a * 100,3)
    rel_difference_worst = round(abs(a-b_worst)/a * 100,3)
    return rel_difference, rel_difference_worst


def objective_function_value_under_different_cases(rel_diffs, rel_diffs_worst):
    fig = make_subplots(rows=1, cols=2,)
    fig.update_layout(title="<b>Relative difference between the exact solution and the free flow approximation<b>",
                        title_pad_l=150,
                        height=500,
                        width=1200,
                        yaxis_title="relative difference [%]")

    fig.update_yaxes(range=[0, 100])

    fig.add_trace(go.Bar(y=rel_diffs, 
                        name="average scenario",
                        x=["all_day", "morning", "midday", "afternoon"],), row=1, col=1)

    fig.add_trace(go.Bar(y=rel_diffs_worst,
                            name="average worst scenario",
                            x=["all_day", "morning", "midday", "afternoon"],), row=1, col=2)
    
    return fig


def compute_min_distance_df(dfs, dfs_worst):
    dfs_min_list = []
    dfs_min_worst_list = []
    
    for time in ["all_day", "morning", "midday", "afternoon"]:
        df_min = get_minimum_distances(dfs[(time, "weight")])
        df_min_worst = get_minimum_distances(dfs_worst[(time, "weight")])

        dfs_min_list.append(df_min)
        dfs_min_worst_list.append(df_min_worst)
    
    df_min = get_minimum_distances(dfs[("all_day", "weight2")])[["target", "travel_time"]]
    
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