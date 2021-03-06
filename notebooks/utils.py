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


def test_binary_bins(df, hours, control_var="", categories=""):
    """
    Run a statistical test for non-foehn vs. foehn fires for a given control variable.
    :param df: Fire dataframe
    :param hours: Time period to consider
    :param control_var: Variable as decade or fire-regime
    :param categories: Manifestations of the control variable
    """

    # Create the binary non-foehn and foehn intervals
    intervals = pd.cut(df[f'foehn_minutes_during_{hours}_hours_after_start_of_fire'], bins=[-0.001, 0.001, 60 * hours])

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


def test_foehn_strength(df, hours, strength_var, bins):
    """
    Test different foehn strength bins against each other
    :param df: Fire dataframe
    :param hours: Time period to consider
    :param strength_var: FF or FFX
    :param bins: How to bin the different wind strengths
    """

    # Reduce dataframe to fires which show foehn activity
    foehn_mask = df[f'foehn_minutes_during_{hours}_hours_after_start_of_fire'] > 0
    df = df.loc[foehn_mask, :].reset_index(drop=True)
    print("Amount of fires with foehn activity: ", len(df.index))

    # Define column of interest
    strength_col = f"{strength_var}_mean_during_{hours}_hours_after_start_of_fire"
    # plt.figure()
    # df[strength_col].hist(alpha=0.5, bins=30)

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
    intervals_dummies = intervals.copy()
    for interval in intervals:
        intervals_dummies.remove(interval)
        burned_values_1 = df.loc[(pd.cut(df[strength_col], bins=bins) == interval), "total [ha]"]
        for interval_dummy in intervals_dummies:
            burned_values_2 = df.loc[(pd.cut(df[strength_col], bins=bins) == interval_dummy), "total [ha]"]

            # Print results of Wilcoxon test and median increase between different foehn strength bins
            print(f"({hours}h) ", interval, " vs. ", interval_dummy, "\t",
                  np.round(ranksums(burned_values_1, burned_values_2).pvalue, 6), "\t",
                  np.round(burned_values_2.median() / burned_values_1.median(), 3))



def plot_multiple_binned_burned_area_after_fire_start(df, hours, control_var=""):
    """
    Plot multiple boxes for each of the six intervals and potentially separated by hue for control variables.
    :param df: Fire dataframe
    :param hours: Time period to consider
    :param control_var: Variable such as decade or fire-regime
    """

    # Specify the bins to be plotted
    bins = [-0.001, 0.001] + [hours * 10 * i for i in range(1, 6 + 1)]
    kwargs = dict(x=pd.cut(df[f'foehn_minutes_during_{hours}_hours_after_start_of_fire'], bins=bins),
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


def plot_binary_binned_burned_area_after_fire_start(df, hours, control_var=""):
    """
    Create binary non-foehn vs. foehn comparison plots (potentially for each control variable).
    :param df: Fire dataframe
    :param hours: Time period to consider
    :param control_var: Variable such as decade or fire-regime
    """

    # Specify the bins to be plotted
    bins = [-0.001, 0.001, 60 * hours]
    kwargs = dict(x=pd.cut(df[f'foehn_minutes_during_{hours}_hours_after_start_of_fire'], bins=bins),
                  y=df['total [ha]'],
                  color="tab:blue")
    if control_var:  # If control variable is specified
        kwargs.update(dict(x=df[control_var],
                           hue=pd.cut(df[f'foehn_minutes_during_{hours}_hours_after_start_of_fire'], bins=bins)))

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


def plot_binned_fire_count_before_fire_start(df, df_foehn, hours):
    """
    Plot the count of fires normalized by the general occurrence of this foehn length for a specified time period
    :param df: Fire dataframe
    :param df_foehn: Foehn dataframe
    :param hours:  Time period to consider (24 or 48 hours)
    """

    # Crate plot in multiple and binary bin form
    for bin_amount in [6, [-0.001, 0.001, hours * 60]]:
        df_binned = df.groupby(pd.cut(df[f'foehn_minutes_{hours}_hour_before'], bins=bin_amount)).count()

        # Loop over all stations and get general occurrence of a certain foehn length at each station
        foehn_minutes_in_interval = np.zeros(len(df_binned.index))
        for station in df_foehn:
            foehn_minutes_during_timeframe = df_foehn[station].rolling(6 * hours).sum() * 10
            foehn_minutes_in_interval += np.array([((i.left < foehn_minutes_during_timeframe) &
                                                    (foehn_minutes_during_timeframe <= i.right)).sum() for i in
                                                   df_binned.index])

        # Weigh them all equally
        df_binned["foehn_minutes_in_interval"] = foehn_minutes_in_interval / len(df_foehn.columns)
        df_binned["normalized_count"] = df_binned['total [ha]'] / df_binned["foehn_minutes_in_interval"]

        # Plot figure (normalize y-axis by first bin) and make labels pretty
        plt.figure()
        sns.barplot(x=df_binned.index, y=df_binned["normalized_count"] / df_binned["normalized_count"][0],
                    color="tab:blue")
        plt.ylabel("Normalized count of fires")
        plt.xlabel("")
        xticks, xlabels = plt.xticks()
        xlabels = [label.get_text() for label in xlabels]
        if bin_amount == 6:  # In multiple bin case
            plt.xlabel("Foehn minutes")
            xlabels[0] = "[0.0" + xlabels[0][6:]
            figname = f"NormalizedFireCountOverFoehnMinutesForThe{hours}HoursBeforeStart"
        else:  # In the binary bin case
            xlabels[0] = "Non-foehn presence"
            xlabels[1] = "Foehn presence"
            figname = f"NonFoehnVsFoehnOverFoehnMinutesForThe{hours}HoursBeforeStart"

        plt.xticks(xticks, xlabels)
        save_figure(figname)


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
