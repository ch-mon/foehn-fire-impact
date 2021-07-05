import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from plotly.subplots import make_subplots
from plotly import graph_objects as go
from scipy.stats import ranksums
import datetime


def save_figure(fig_name, save_figures_bool=False):
    """
    Save figures when bool is set to True.
    :param fig_name: Name of the figure.
    :param save_figures_bool: Whether to save this figure.
    """
    if save_figures_bool:
        fig_path = f'data/08_reporting/{fig_name}.pdf'
        plt.savefig(fig_path, bbox_inches='tight', dpi=200)
        print(f"Saved figure at: {fig_path}")


def test_multiple_bins_against_no_foehn(df, hours):
    """
    Test multiple foehn minute bins against the no foehn bin.
    :param df: Fire dataframe
    :param hours: Time period to consider.
    """
    
    # Define bins and calculate intervals
    bins = [-0.001, 0.001] + [hours * 10 * i for i in range(1, 6 + 1)]
    intervals = pd.cut(df[f'foehn_minutes_during_{hours}_hours_after_start_of_fire'], bins=bins)

    # Filter all non foehn fires
    no_foehn_fires = df.loc[intervals == pd.Interval(-0.001, 0.001), "total [ha]"]

    # Loop over all other foehn minute bins
    for interval in intervals.unique().categories[1:]:
        foehn_fires = df.loc[interval == intervals, "total [ha]"]

        # Print results of Wilcoxon test and median increase between bins
        print("No-foehn vs.", interval, "\t",
              np.round(ranksums(no_foehn_fires, foehn_fires).pvalue, 6), "\t",
              np.round(foehn_fires.median() / no_foehn_fires.median(), 2)
              )


def test_binary_bins(df, hours, variable="after", control_var="", categories=""):
    """
    Run a statistical test for non-foehn vs. foehn fires for a given control variable.
    :param df: Fire dataframe
    :param hours: Time period to consider
    :param control_var: Variable as decade or fire-regime
    :param categories: Manifestations of the control variable
    """
    
    # Which aspect to look at
    if variable=="after":
        variable=f'foehn_minutes_during_{hours}_hours_after_start_of_fire'
    elif variable=="before":
        variable=f'foehn_minutes_{hours}_hour_before'

    # Create the binary non-foehn and foehn intervals
    intervals = pd.cut(df[variable], bins=[-0.001, 0.001, 60 * hours])

    # Control whether a control variable is given
    if control_var:
        # Test for each category within the control variable.
        for category in categories:
            filter_category = df[control_var] == category

            no_foehn_values = df.loc[(intervals == pd.Interval(-0.001, 0.001)) & filter_category, "total [ha]"]
            foehn_values = df.loc[(intervals == pd.Interval(0.001, 60 * hours)) & filter_category, "total [ha]"]

            # Print results of Wilcoxon test and median increase between non-foehn and foehn bin
            print(f"({hours}h) non-foehn vs. foehn for", category, "\t",
                  np.round(ranksums(no_foehn_values, foehn_values).pvalue, 6), "\t",
                  np.round(foehn_values.median() / no_foehn_values.median(), 2))
    else:
        no_foehn_values = df.loc[intervals == pd.Interval(-0.001, 0.001), "total [ha]"]
        foehn_values = df.loc[intervals == pd.Interval(0.001, 60 * hours), "total [ha]"]

        # Print results of Wilcoxon test and median increase between non-foehn and foehn bin
        pvalue = ranksums(no_foehn_values, foehn_values).pvalue
        median_increase_factor = np.round(foehn_values.median() / no_foehn_values.median(), 2)

        return pvalue, median_increase_factor


def test_foehn_within_variable(df, hours, control_var, categories):
    """
    Test fires which showed foehn activity for each of the manifestations of the control variables.
    :param df: Fire dataframe
    :param hours: time period to consider
    :param control_var: Variable such as decade or fire-regime
    :param categories: Manifestations of the control variable
    """

    # Create the binary non-foehn and foehn intervals
    intervals = pd.cut(df[f'foehn_minutes_during_{hours}_hours_after_start_of_fire'], bins=[-0.001, 0.001, 60*hours])
    foehn_mask = (intervals == pd.Interval(0.001, 60*hours))

    # Loop over all possible combination of bins
    categories_dummies = categories.copy()
    for category in categories:
        categories_dummies.remove(category)
        foehn_values_1 = df.loc[(df[control_var] == category) & foehn_mask, "total [ha]"]
        for category_dummy in categories_dummies:
            foehn_values_2 = df.loc[(df[control_var] == category_dummy) & foehn_mask, "total [ha]"]

            # Print results of Wilcoxon test and median increase between different foehn bins
            print(f"({hours}h) ", category, " vs. ", category_dummy, "\t",
                  np.round(ranksums(foehn_values_1, foehn_values_2).pvalue, 6), "\t",
                  np.round(foehn_values_1.median() / foehn_values_2.median(), 3))


def test_foehn_strength(df, hours, strength_var, bins, foehn_bool=True):
    """
    Test different foehn strength bins against each other
    :param df: Fire dataframe
    :param hours: Time period to consider
    :param strength_var: FF or FFX
    :param bins: How to bin the different wind strengths
    """

    # Define column of interest
    strength_col = f"{strength_var}_q75_during_{hours}_hours_after_start_of_fire"
    plt.figure()
    plt.xlabel(f"{strength_var} [km/h]")
    df.loc[df[f'foehn_minutes_during_{hours}_hours_after_start_of_fire'] > 0, strength_col].hist(alpha=0.5, bins=range(0,120))
    df.loc[df[f'foehn_minutes_during_{hours}_hours_after_start_of_fire'] == 0, strength_col].hist(alpha=0.5, bins=range(0,120))
    plt.legend(["Foehn fires", "no-foehn fires"])
    
    # Reduce dataframe to fires which show foehn activity
    if foehn_bool:
        foehn_mask = df[f'foehn_minutes_during_{hours}_hours_after_start_of_fire'] > 0
        print("Amount of foehn-influenced fires: ", foehn_mask.sum())
    else:
        foehn_mask = df[f'foehn_minutes_during_{hours}_hours_after_start_of_fire'] == 0
        print("Amount of non-foehn-influenced fires: ", foehn_mask.sum())
    df = df.loc[foehn_mask, :].reset_index(drop=True)
    

    # Plot the different bins
    plt.figure()
    g = sns.boxplot(x=pd.cut(df[strength_col], bins=bins), y=df["total [ha]"], color="tab:blue")
    plt.xlabel(f"{strength_var} [km/h]")
    plt.ylabel("Total burned area [ha]")
    g.set_yscale("log")
    plt.grid(True, which="both", ls="--", c='gray', alpha=0.5)
    save_figure(f"BoxplotBurnedAreaOver{strength_var}ForFirst{hours}HoursAfterStart")

    # Loop over all possible combinations of strength bins
    intervals = pd.cut(df[strength_col], bins=bins).unique().sort_values(ascending=True).to_list()
    if np.NaN in intervals:
        intervals.remove(np.NaN)
    intervals_dummies = intervals.copy()
    for interval in intervals:
        intervals_dummies.remove(interval)
        burned_values_1 = df.loc[(pd.cut(df[strength_col], bins=bins) == interval), "total [ha]"]
        for interval_dummy in intervals_dummies:
            burned_values_2 = df.loc[(pd.cut(df[strength_col], bins=bins) == interval_dummy), "total [ha]"]

            # Print results of Wilcoxon test and median increase between different foehn strength bins
            print(f"({hours}h) {interval} ({len(burned_values_1)}) vs. {interval_dummy} ({len(burned_values_2)}) \t",
                  np.round(ranksums(burned_values_1, burned_values_2).pvalue, 6), "\t",
                  np.round(burned_values_2.median() / burned_values_1.median(), 3))


def test_foehn_strength_foehn_nofoehn(df, hours, strength_var, bins, quantile="mean"):
    """
    Test different foehn strength bins against each other
    :param df: Fire dataframe
    :param hours: Time period to consider
    :param strength_var: FF or FFX
    :param bins: How to bin the different wind strengths
    """

    # Define column of interest
    strength_col = f"{strength_var}_{quantile}_during_{hours}_hours_after_start_of_fire"
    plt.figure()


    plt.xlabel(f"{strength_var} [km/h]")
    df.loc[df[f'foehn_minutes_during_{hours}_hours_after_start_of_fire'] > 0, strength_col].hist(alpha=0.5,
                                                                                                 bins=range(0, 80))
    df.loc[df[f'foehn_minutes_during_{hours}_hours_after_start_of_fire'] == 0, strength_col].hist(alpha=0.5,
                                                                                                  bins=range(0, 80))
    plt.legend(["Foehn fires", "no-foehn fires"])
    plt.vlines(bins[0], ymin=0, ymax=50, color="r", linewidth=3)
    plt.vlines(bins[-1], ymin=0, ymax=50, color="r", linewidth=3)
    plt.ylabel("Count of fires")


    # Reduce dataframe to fires which show foehn activity
    print(f"({hours}h) Amount of foehn-influenced fires: ", (df[f'foehn_minutes_during_{hours}_hours_after_start_of_fire'] > 0).sum())
    print(f"({hours}h) Amount of non-foehn-influenced fires: ", (df[f'foehn_minutes_during_{hours}_hours_after_start_of_fire'] == 0).sum())


    df.loc[(df[f'foehn_minutes_during_{hours}_hours_after_start_of_fire'] > 0), "type"] = "foehn"
    df.loc[df[f'foehn_minutes_during_{hours}_hours_after_start_of_fire'] == 0, "type"] = "nofoehn"

    df["strength_bins"] = pd.cut(df[strength_col], bins=bins)

    # Plot the different bins
    plt.figure()
    g = sns.boxplot(x="strength_bins", y="total [ha]", data=df, hue="type", color="tab:blue")
    plt.xlabel(f"{strength_var} [km/h]")
    plt.ylabel("Total burned area [ha]")
    g.set_yscale("log")
    plt.grid(True, which="both", ls="--", c='gray', alpha=0.5)
    save_figure(f"BoxplotBurnedAreaOver{strength_var}ForFirst{hours}HoursAfterStartFoehnVsNofoehn")

    # Loop over all strength bins
    intervals = df["strength_bins"].unique().sort_values(ascending=True).to_list()
    if np.NaN in intervals:
        intervals.remove(np.NaN)

    for interval in intervals:
        burned_values_1 = df.loc[(df["strength_bins"] == interval) & (df["type"] == "foehn"), "total [ha]"]
        burned_values_2 = df.loc[(df["strength_bins"] == interval) & (df["type"] == "nofoehn"), "total [ha]"]
        # Print results of Wilcoxon test and median increase between different foehn strength bins
        if len(burned_values_1)>9 and len(burned_values_2)>9:
            print(f"({hours}h) {interval} (foehnfires: {len(burned_values_1)}), (nofoehnfires: {len(burned_values_2)}) \t",
                  np.round(ranksums(burned_values_1, burned_values_2).pvalue, 6))#, "\t",
                  #np.round(burned_values_2.median() / burned_values_1.median(), 3))


def plot_multiple_binned_burned_area_after_fire_start(df, hours, variable="after", control_var=""):
    """
    Plot multiple boxes for each of the six intervals and potentially separated by hue for control variables.
    :param df: Fire dataframe
    :param hours: Time period to consider
    :param control_var: Variable such as decade or fire-regime
    """
    
    # Which aspect to look at
    if variable=="after":
        variable=f'foehn_minutes_during_{hours}_hours_after_start_of_fire'
    elif variable=="before":
        variable=f'foehn_minutes_{hours}_hour_before'

    # Specify the bins to be plotted
    bins = [-0.001, 0.001] + [hours * 10 * i for i in range(1, 6 + 1)]
    kwargs = dict(x=pd.cut(df[variable], bins=bins),
                  y=df['total [ha]'],
                  color="tab:blue")
    if control_var:  # If control variable is specified
        kwargs.update(dict(hue=df[control_var]))

    # Plot figure and make labels pretty
    plt.figure()
    g = sns.boxplot(**kwargs)
    plt.ylabel("Burned area [ha]")
    plt.xlabel("Foehn minutes")
    g.set_yscale("log")
    plt.grid(True, which="both", ls="--", c='gray', alpha=0.5)
    xticks, xlabels = plt.xticks()
    xlabels = [label.get_text() for label in xlabels]
    xlabels[0] = "No foehn"
    plt.xticks(xticks, xlabels)

    save_figure(f"BoxplotBurnedAreaOverFoehnMinutesForFirst{hours}HoursAfterStart")


def plot_binary_binned_burned_area_after_fire_start(df, hours, variable="after", control_var=""):
    """
    Create binary non-foehn vs. foehn comparison plots (potentially for each control variable).
    :param df: Fire dataframe
    :param hours: Time period to consider
    :param control_var: Variable such as decade or fire-regime
    """
    
    # Which aspect to look at
    if variable=="after":
        variable=f'foehn_minutes_during_{hours}_hours_after_start_of_fire'
    elif variable=="before":
        variable=f'foehn_minutes_{hours}_hour_before'

    # Specify the bins to be plotted
    bins = [-0.001, 0.001, 60 * hours]
    kwargs = dict(x=pd.cut(df[variable], bins=bins),
                  y=df['total [ha]'],
                  color="tab:blue")
    if control_var:  # If control variable is specified
        kwargs.update(dict(x=df[control_var],
                           hue=pd.cut(df[variable], bins=bins)))

    # Plot variable but label a bit differently depending on a control variable
    plt.figure()
    g = sns.boxplot(**kwargs)

    if control_var:
        L = plt.legend()
        L.get_texts()[0].set_text('Non-foehn fires')
        L.get_texts()[1].set_text('Foehn fires')
    else:
        xticks, xlabels = plt.xticks()
        xlabels = [label.get_text() for label in xlabels]
        xlabels[0] = "Non-foehn fires"
        xlabels[1] = "Foehn fires"
        plt.xticks(xticks, xlabels)

    plt.ylabel("Burned area [ha]")
    plt.xlabel("")
    g.set_yscale("log")
    plt.grid(True, which="both", ls="--", c='gray', alpha=0.5)

    save_figure(f"NonVsFoehnBurnedAreaOverFoehnMinutesForFirst{hours}HoursAfterStart")


def plot_binned_fire_count_before_fire_start(df, df_foehn, hours, stations_in_region):
    """
    Plot the count of fires normalized by the general occurrence of this foehn length for a specified time period
    :param df: Fire dataframe
    :param df_foehn: Foehn dataframe
    :param hours:  Time period to consider (24 or 48 hours)
    """

    # Create plot in multiple and binary bin form
    for bin_amount in [[-1,1, 240, 480, 720, 960, 1200, 1440], [-0.001, 0.001, hours * 60]]:
        df_binned = df.groupby(pd.cut(df[f'foehn_minutes_{hours}_hour_before'], bins=bin_amount)).count()
    
        # Loop over all stations and get general occurrence of a certain foehn length at each station
        foehn_minutes_in_interval = np.zeros(len(df_binned.index))
        for station in stations_in_region:
            foehn_minutes_during_timeframe = df_foehn[f"{station}_foehn"].rolling(6 * hours).sum().loc[(df_foehn["date"].dt.hour == 7) & (df_foehn["date"].dt.minute == 0) & (df_foehn[f"{station}_rainavg"] <=10)].reset_index(drop=True) * 10
            
            foehn_minutes_in_interval += np.array([((i.left < foehn_minutes_during_timeframe) &
                                                    (foehn_minutes_during_timeframe <= i.right)).sum() for i in
                                                   df_binned.index])

        # Weigh them all equally
        print(foehn_minutes_in_interval)
        df_binned["foehn_minutes_in_interval"] = foehn_minutes_in_interval / len(stations_in_region)
        print(df_binned["foehn_minutes_in_interval"])
        print(df_binned['total [ha]'])
        df_binned["normalized_count"] = df_binned['total [ha]'] / df_binned["foehn_minutes_in_interval"]

        # Plot figure (normalize y-axis by first bin) and make labels pretty
        plt.figure()
        sns.barplot(x=df_binned.index, y=df_binned["normalized_count"], #/ df_binned["normalized_count"][0],
                    color="tab:blue")
        plt.ylabel("Normalized count of fires")
        plt.xlabel("")
        xticks, xlabels = plt.xticks()
        xlabels = [label.get_text() for label in xlabels]
        if len(bin_amount) == 8:  # In multiple bin case
            plt.xlabel("Foehn minutes")
            xlabels[0] = "Non-foehn"
            xlabels[1] = "(0.0," + xlabels[1][3:]
            figname = f"NormalizedFireCountOverFoehnMinutesForThe{hours}HoursBeforeStart"
        else:  # In the binary bin case
            xlabels[0] = "Non-foehn presence"
            xlabels[1] = "Foehn presence"
            figname = f"NonFoehnVsFoehnOverFoehnMinutesForThe{hours}HoursBeforeStart"

        plt.xticks(xticks, xlabels)
        save_figure(figname)


# noinspection PyUnresolvedReferences,PyTypeChecker
def plot_fire_count_over_foehn_days(df_fires, df_foehn, stations, bins, hours=24):
    """
    Plot the count of fires normalized by the general occurrence foehn days for a specific foehn duration
    :param df_fires: Fire dataframe
    :param df_foehn: Foehn dataframe
    :param stations: List of stations to plot
    :param bins: Bin boundaries for the foehn intervals
    :param hours:  Time period to consider (24 hours)
    """

    # For easier readability and to ensure that count data later is correct
    df_fires["fire_occurrence"] = 1

    # Make list of pd.Intervals for foehn minutes
    intervals = [pd.Interval(bins[n], bins[n + 1]) for n in range(len(bins) - 1)]

    # Make Facet Plot figure
    fig = make_subplots(rows=int(np.ceil(len(stations) / 3)), cols=3, subplot_titles=stations)

    # Get general occurrence of foehn for a daily window for all stations
    df_foehn_day = df_foehn.filter(regex="_foehn").rolling(6 * hours).sum()*10
    df_foehn_day = pd.concat([df_foehn_day, df_foehn.filter(regex="_rainavg")], axis=1)

    # Filter for one time in the day to get consecutive days (needs to overlap with rain dataframe)
    df_foehn_day = df_foehn_day.loc[df_foehn["date"].dt.time == datetime.time(14, 0), :].reset_index(drop=True)

    # Loop over all stations
    for st_nr, station in enumerate(stations):
        # Col and row coordinates for Facet Plot
        fig_kwargs = dict(col=(st_nr % 3) + 1, row=int(np.ceil((st_nr + 1) / 3)))

        # Filter fires at station
        df_station = df_fires.loc[df_fires["abbreviation"] == station, [f'foehn_minutes_{hours}_hour_before', "fire_occurrence"]].reset_index(drop=True)
        df_foehn_station = df_foehn_day[[f"{station}_foehn", f"{station}_rainavg"]]

        # Get fire occurrence for each interval
        fires_per_interval = df_station.groupby(pd.cut(df_station[f'foehn_minutes_{hours}_hour_before'], bins=bins)).count()["fire_occurrence"].values

        # Get foehn days with certain foehn length for each interval
        foehn_days_per_interval = np.array(
            [((i.left < df_foehn_station[f"{station}_foehn"]) &
              (df_foehn_station[f"{station}_foehn"] <= i.right) &
              (df_foehn_station[f"{station}_rainavg"] < 10)  # Ensure that rain days are excluded from analysis
             ).sum() for i in intervals]
        )

        # Calculate relative occurrence of fires and foehn days per interval
        fires_per_foehn = fires_per_interval / foehn_days_per_interval

        # Plot results
        fig.add_trace(
            go.Bar(
                x=["Non-foehn"] + [str(i) for i in intervals[1:]], y=fires_per_foehn,
                name="#(fires|foehn)", showlegend=False, opacity=0.75, marker_color="#1f77b4",
                hovertemplate='Rel. fire occurrence: %{y:.2f} %{customdata}',
                customdata=[f"<br>Fires: {fires_per_interval[i]} " +
                            f"<br>Foehn days: {foehn_days_per_interval[i]}" for i in range(len(intervals))]
            ),
            **fig_kwargs
        )

        # Set title for sub-figure
        fig.layout.annotations[st_nr].update(text=station + f" (N_fires={int(fires_per_interval.sum())})")

        # Add annotations for fire and foehn day amounts
        for i, row in enumerate(intervals):
            fig.add_annotation(x=i, y=fires_per_foehn[i], **fig_kwargs,
                               text=int(fires_per_interval[i]),
                               showarrow=False,
                               font={"color": "orange"},
                               yshift=10)
            fig.add_annotation(x=i, y=fires_per_foehn[i], **fig_kwargs,
                               text=int(foehn_days_per_interval[i]),
                               showarrow=False,
                               font={"color": "green"},
                               yshift=-10)

    # Set some global figure properties
    fig.update_layout(height=400 * int(np.ceil(len(stations) / 3)))
    fig.update_yaxes(title="Fires per foehn day")
    fig.show()


# noinspection PyTypeChecker,PyUnresolvedReferences
def plot_fire_day_count_over_foehn_days(df_fires, df_foehn, stations, bins, hours=24):
    """
    Plot the count of fire days which showed prior foehn presence normalized by the general occurrence foehn days for a specific foehn duration
    :param df_fires: Fire dataframe
    :param df_foehn: Foehn dataframe
    :param stations: List of stations to plot
    :param bins: Bin boundaries for the foehn intervals
    :param hours:  Time period to consider (24 hours)
    """

    # Shift fires so that a new day "begins" at 14pm (peak of fire occurrence)
    df_fires["date"] = (df_fires["start_date_min"] - pd.Timedelta("14h")).dt.date.astype(np.datetime64)

    # Define a fire day if at least one fire occurs at a station on a day
    df_fires = df_fires.drop_duplicates(subset=["date", "abbreviation"]).reset_index(drop=True)
    df_fires["fire_day"] = 1

    # Make list of pd.Intervals for foehn minutes
    intervals = [pd.Interval(bins[n], bins[n+1]) for n in range(len(bins)-1)]

    # Make Facet Plot figure
    fig = make_subplots(rows=int(np.ceil(len(stations) / 3)), cols=3, subplot_titles=stations)

    # Get general occurrence of foehn for a daily window for all stations
    df_foehn_day = df_foehn.filter(regex="_foehn").rolling(6 * hours).sum()*10
    df_foehn_day = pd.concat([df_foehn_day, df_foehn.filter(regex="_rainavg")], axis=1)
    df_foehn_day["date"] = df_foehn["date"].dt.date.astype(np.datetime64)

    # Filter for one time in the day to get consecutive days (needs to overlap with rain dataframe)
    df_foehn_day = df_foehn_day.loc[df_foehn["date"].dt.time == datetime.time(14, 0), :].reset_index(drop=True)

    # Loop over all stations
    for st_nr, station in enumerate(stations):
        # Col and row coordinates for Facet Plot
        fig_kwargs = dict(col=(st_nr%3)+1, row=int(np.ceil((st_nr+1)/3)))

        # Filter fires at station
        df_station = df_fires.loc[df_fires["abbreviation"] == station, ["date", "fire_day"]].reset_index(drop=True)
        df_foehn_station = df_foehn_day[["date", f"{station}_foehn", f"{station}_rainavg"]]

        # Merge foehn values and days with fire occurrence
        df_foehn_fire = pd.merge(df_foehn_station, df_station, on="date", how="left")

        # Count foehn days for each interval
        foehn_days_per_interval = np.array(
            [((i.left < df_foehn_fire[f"{station}_foehn"]) &
              (df_foehn_fire[f"{station}_foehn"] <= i.right) &
              (df_foehn_fire[f"{station}_rainavg"] < 10)  # Ensure that rain days are excluded from analysis
             ).sum() for i in intervals]
        )

        # Count foehn days which showed fire the day after for each interval
        foehn_and_fire_days_per_interval = np.array(
            [((i.left < df_foehn_fire[f"{station}_foehn"]) &
              (df_foehn_fire[f"{station}_foehn"] <= i.right) &
              (df_foehn_fire[f"{station}_rainavg"] < 10) &  # Ensure that rain days are excluded from analysis
              (df_foehn_fire["fire_day"] == 1)).sum() for i in intervals]
        )

        # Calculate relative occurrence of foehn and fires
        rel_fire_days_per_interval = foehn_and_fire_days_per_interval / foehn_days_per_interval

        # Plot results
        fig.add_trace(
            go.Bar(
                x=["Non-foehn"] + [str(i) for i in intervals[1:]], y=rel_fire_days_per_interval,
                name="P(fire=1|foehn)", showlegend=False, opacity=0.75, marker_color="#1f77b4",
                hovertemplate='Rel. fire day occurrence: %{y:.2f} %{customdata}',
                customdata=[f"<br>Fire days: {foehn_and_fire_days_per_interval[i]} " +
                            f"<br>Foehn days: {foehn_days_per_interval[i]}" for i in range(len(intervals))]
            ),
            **fig_kwargs
        )

        # Set title for sub-figure
        fig.layout.annotations[st_nr].update(text=station + f" (N_fire_days={int(foehn_and_fire_days_per_interval.sum())})")

        # Add annotations for fire and foehn day amounts
        for i, row in enumerate(intervals):
            fig.add_annotation(x=i, y=rel_fire_days_per_interval[i], **fig_kwargs,
                               text=int(foehn_and_fire_days_per_interval[i]),
                               showarrow=False,
                               font={"color": "orange"},
                               yshift=10)
            fig.add_annotation(x=i, y=rel_fire_days_per_interval[i], **fig_kwargs,
                               text=int(foehn_days_per_interval[i]),
                               showarrow=False,
                               font={"color": "green"},
                               yshift=-10)

    # Set some global figure properties
    fig.update_layout(height=400*int(np.ceil(len(stations) / 3)))
    fig.update_yaxes(title="Fire occurrence on days after foehn")
    fig.show()

def plot_binned_fire_count_before_fire_start_temperature(df, df_foehn, hours, stations):
    """
    Plot the count of fires normalized by the general occurrence of this foehn temperature increase for a specified time period
    :param df: Fire dataframe
    :param df_foehn: Foehn dataframe with foehn and temperature values
    :param hours: Time period to consider (24 or 48 hours)
    :param stations: List of stations to consider
    """
    # Filter fires which showed foehn activity before ignition and were to the south of the Alps
    df = df.loc[(df[f'foehn_minutes_{hours}_hour_before'] > 0) & (df["potential_foehn_species"] == "North foehn"), :]

    # Bin by foehn temperature increase (as obtained from a histogram investigation)
    bins = [0, 3, 6, 9, 12, 15]
    df_binned = df.groupby(pd.cut(df[f"TT_mean_{hours}_hour_before"], bins=bins)).count()

    # Loop over all north foehn stations and identify where foehn had which temperature increase
    foehn_conditions_in_interval = np.zeros(len(df_binned.index))
    for station in stations:
        temp_mean_foehn = df_foehn[f"{station}_TT"].where(df_foehn[f"{station}_foehn"] == 1.0).rolling(6 * hours,
                                                                                                       min_periods=1).mean()
        temp_mean_no_foehn = df_foehn[f"{station}_TT"].where(df_foehn[f"{station}_foehn"] == 0.0).rolling(6 * hours,
                                                                                                          min_periods=1).mean()
        strength_difference = temp_mean_foehn - temp_mean_no_foehn

        foehn_conditions_in_interval += np.array([((i.left < strength_difference) &
                                                   (strength_difference < i.right)).sum() for i in df_binned.index])

    df_binned["normalized_count"] = df_binned['total [ha]'] / foehn_conditions_in_interval

    # Plot figure (normalize y-axis by first bin) and make labels pretty
    plt.figure()
    sns.barplot(x=df_binned.index, y=df_binned["normalized_count"] / df_binned["normalized_count"][0], color="tab:blue")
    plt.ylabel("Normalized count of fires")
    plt.xlabel("Foehn temperature increase [K]")

    save_figure(f'NormalizedFireCountOverTemperatureForThe{hours}HoursBeforeStart')
    

def plot_binned_burned_area_before_and_after_fire_start(df, hours_before, hours_after):
    """
    Create binary non-foehn vs. foehn comparison plots (potentially for each control variable).
    :param df: Fire dataframe
    :param hours: Time period to consider
    """
    
    
    mask = (df[f'foehn_minutes_{hours_before}_hour_before'] == 0) & (df[f'foehn_minutes_during_{hours_after}_hours_after_start_of_fire'] == 0)
    df.loc[mask, "no-foehn"] = df.loc[mask, "total [ha]"]
    print("No foehn: ", mask.sum())
    mask = (df[f'foehn_minutes_{hours_before}_hour_before'] > 0)
    df.loc[mask, "before-foehn"] = df.loc[mask, "total [ha]"]
    print("Before foehn: ", mask.sum())
    mask = (df[f'foehn_minutes_during_{hours_after}_hours_after_start_of_fire'] > 0)
    df.loc[mask, "after-foehn"] = df.loc[mask, "total [ha]"]
    print("After foehn: ", mask.sum())
    mask = (df[f'foehn_minutes_{hours_before}_hour_before'] > 0) & (df[f'foehn_minutes_during_{hours_after}_hours_after_start_of_fire'] > 0)
    df.loc[mask, "both-foehn"] = df.loc[mask, "total [ha]"]
    print("Before and after foehn: ", mask.sum())
    
    df = df[["no-foehn", "before-foehn", "after-foehn", "both-foehn"]]
    # Specify the bins to be plotted
    kwargs = dict(x="variable",
                  y="value",
                  data=pd.melt(df).dropna(),
                  color="tab:blue",
                 )

    # Plot variable but label a bit differently depending on a control variable
    plt.figure()
    g = sns.boxplot(**kwargs)

    #if control_var:
    #    L = plt.legend()
    #    L.get_texts()[0].set_text('Non-foehn fires')
    #    L.get_texts()[1].set_text('Foehn fires')
    #else:
    #    xticks, xlabels = plt.xticks()
    #    xlabels = [label.get_text() for label in xlabels]
    #    xlabels[0] = "Non-foehn fires"
    #    xlabels[1] = "Foehn fires"
    #    plt.xticks(xticks, xlabels)

    plt.ylabel("Burned area [ha]")
    plt.xlabel("")
    g.set_yscale("log")
    plt.grid(True, which="both", ls="--", c='gray', alpha=0.5)

    save_figure(f"BurnedAreaOverFoehnMinutes{hours_before}HoursBeforeAnd{hours_after}AfterStart")