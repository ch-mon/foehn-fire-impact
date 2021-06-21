import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.stats import ranksums


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
        print(f"({hours}h) non-foehn vs. foehn ",
              ranksums(no_foehn_values, foehn_values).pvalue, "\t",
              np.round(foehn_values.median() / no_foehn_values.median(), 2))


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


def plot_binned_fire_count_before_fire_start_single_station(df, df_foehn, hours, stations_in_region):
    """
    Plot the count of fires normalized by the general occurrence of this foehn length for a specified time period
    :param df: Fire dataframe
    :param df_foehn: Foehn dataframe
    :param hours:  Time period to consider (24 or 48 hours)
    """

    # Create plot in multiple and binary bin form
    for bins in [[-1, 1, 240, 480, 720, 960, 1200, 1440], [-0.001, 0.001, 60, hours * 60]]:

        plt.figure(figsize=(16*2,9*2))
        plt.rcParams.update({'font.size': 10})
        # Loop over all stations and get general occurrence of a certain foehn length at each station
        
        for st_nr, station in enumerate(stations_in_region):
            
            print(station)
            df_station = df.loc[df["abbreviation"] == station, :]
            df_binned = df_station.groupby(pd.cut(df_station[f'foehn_minutes_{hours}_hour_before'], bins=bins)).count()
            foehn_minutes_in_interval = np.zeros(len(bins)-1)
            
            if len(bins) == 4:
                df_binned = df_binned.loc[df_binned.index != pd.Interval(0.001, 60), :].copy()
                df_binned.index = df_binned.index.categories[0:(2+1):2]
    
                foehn_minutes_in_interval = np.zeros(len(bins)-2)
                
                
            
            foehn_minutes_during_timeframe = df_foehn[f"{station}_foehn"].rolling(6 * hours).sum().loc[
                                                 (df_foehn["date"].dt.hour == 14) & (df_foehn["date"].dt.minute == 0) & (
                                                             df_foehn[f"{station}_rainavg"] <= 10)].reset_index(drop=True)*10

            foehn_minutes_in_interval += np.array([((i.left < foehn_minutes_during_timeframe) &
                                                    (foehn_minutes_during_timeframe <= i.right)).sum() for i in
                                                   df_binned.index])

            df_binned["foehn_minutes_in_interval"] = foehn_minutes_in_interval
            
            #print(df_binned["foehn_minutes_in_interval"])
            #print(df_binned['total [ha]'])
            df_binned["normalized_count"] = df_binned['total [ha]'] / df_binned["foehn_minutes_in_interval"]
            # Plot figure (normalize y-axis by first bin) and make labels pretty
            ax = plt.subplot(int(round(len(stations_in_region) / 3,0)), 3, st_nr+1, )
            sns.barplot(x=df_binned.index, y=df_binned["normalized_count"],  # / df_binned["normalized_count"][0],
                        color="tab:blue")
            for index, row in df_binned.reset_index(drop=True).iterrows():
                ax.text(index, row["normalized_count"], int(row["total [ha]"]), color='black', ha="center", fontsize=14)
                ax.text(index, row["normalized_count"], "\n" + str(int(row["foehn_minutes_in_interval"])), color='white', ha="center",va="top", fontsize=14)
            plt.ylabel("Normalized count of fires", fontsize=14)
            plt.xlabel("")
            plt.title(station + f" (N_fires={df_binned['total [ha]'].sum()})", fontsize=14)
            xticks, xlabels = plt.xticks(fontsize=10)
            xlabels = [label.get_text() for label in xlabels]
            if len(bins) == 8:  # In multiple bin case
                #plt.xlabel("Foehn minutes")
                xlabels[0] = "Non-foehn"
                xlabels[1] = "(0.0," + xlabels[1][3:]
                figname = f"NormalizedFireCountOverFoehnMinutesForThe{hours}HoursBeforeStart"
            else:  # In the binary bin case
                xlabels[0] = "Non-foehn presence"
                xlabels[1] = "Foehn presence"
                figname = f"NonFoehnVsFoehnOverFoehnMinutesForThe{hours}HoursBeforeStart"

            plt.xticks(xticks, xlabels)
            #save_figure(figname)

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