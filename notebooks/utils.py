import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import scipy as sci

def save_figure(fig_name, save_figures_bool=False):
    """
    Save figures when bool is set to True
    :param fig_name:
    :param save_figures_bool:
    :return:
    """
    if save_figures_bool:
        fig_path = f'data/08_reporting/{fig_name}.pdf'
        plt.savefig(fig_path, bbox_inches='tight', dpi=200)
        print(f"Saved figure at: {fig_path}")

    return

def test_multiple_bins_against_no_foehn(df, hours):
    bins = [-0.001, 0.001] + [hours * 10 * i for i in range(1, 6 + 1)]
    intervals = pd.cut(df[f'foehn_minutes_during_{hours}_hours_after_start_of_fire'], bins=bins)

    no_foehn_mask = (intervals == pd.Interval(-0.001, 0.001))
    no_foehn_fires = df.loc[no_foehn_mask, "total [ha]"]

    foehn_intervals = sorted(intervals.value_counts().index[1:])

    for interval in foehn_intervals:
        foehn_mask = (intervals == interval)
        foehn_fires = df.loc[foehn_mask, "total [ha]"]

        print("No-foehn vs.", interval, "\t",
              np.round(sci.stats.ranksums(no_foehn_fires, foehn_fires).pvalue, 6), "\t",
              np.round(foehn_fires.median() / no_foehn_fires.median(), 2)
              )

    return

def test_binary_bins(df, hours, control_var="", categories=""):
    intervals = pd.cut(df[f'foehn_minutes_during_{hours}_hours_after_start_of_fire'], bins=[-0.001, 0.001, 60*hours])

    if control_var:
        for category in categories:
            filter_category = df[control_var] == category

            no_foehn_values = df.loc[(intervals == pd.Interval(-0.001, 0.001)) & filter_category, "total [ha]"]
            foehn_values = df.loc[(intervals == pd.Interval(0.001, 60*hours)) & filter_category, "total [ha]"]

            print(f"({hours}h) non-foehn vs. foehn for", category, "\t",
                  np.round(sci.stats.ranksums(no_foehn_values, foehn_values).pvalue, 6), "\t",
                  np.round(foehn_values.median()/no_foehn_values.median(), 2))
    else:
        no_foehn_values = df.loc[intervals == pd.Interval(-0.001, 0.001), "total [ha]"]
        foehn_values = df.loc[intervals == pd.Interval(0.001, 60 * hours), "total [ha]"]

        print(f"({hours}h) non-foehn vs. foehn ",
              sci.stats.ranksums(no_foehn_values, foehn_values).pvalue, "\t",
              np.round(foehn_values.median() / no_foehn_values.median(), 2))

def test_foehn_within_variable(df, hours, control_var, categories):
    bins= [-0.001, 0.001, 60 * hours]

    categories_dummies = categories.copy()
    intervals = pd.cut(df[f'foehn_minutes_during_{hours}_hours_after_start_of_fire'], bins=bins)

    for category in categories:
        categories_dummies.remove(category)
        foehn_values_1 = df.loc[(df[control_var] == category) & (intervals == pd.Interval(0.001, 60 * hours)), "total [ha]"]
        for category_dummy in categories_dummies:
            foehn_values_2 = df.loc[
                (df[control_var] == category_dummy) & (intervals == pd.Interval(0.001, 60 * hours)), "total [ha]"]

            print(f"({hours}h) ", category, " vs. " , category_dummy, "\t",
                  np.round(sci.stats.ranksums(foehn_values_1, foehn_values_2).pvalue, 6), "\t",
                  np.round(foehn_values_1.median() / foehn_values_2.median(), 3))


def plot_multiple_binned_burned_area_after_fire_start(df, hours, control_var=""):
    """
    Plot multiple boxes for each of the six intervals and potentially separated by hue for control variables.
    :param df:
    :param hours:
    :param control_var:
    :return:
    """
    bins = [-0.001, 0.001] + [hours * 10 * i for i in range(1, 6 + 1)]
    kwargs = dict(x=pd.cut(df[f'foehn_minutes_during_{hours}_hours_after_start_of_fire'], bins=bins),
                  y=df['total [ha]'],
                  color="tab:blue")
    if control_var:
        kwargs.update(dict(hue=df[control_var]))

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
    return

def plot_binary_binned_burned_area_after_fire_start(df, hours, control_var=""):
    """
    Create binary (non-foehn vs. foehn comparison plots (potentially for each control variable).
    :param df:
    :param hours:
    :param control_var:
    :return:
    """

    bins = [-0.001, 0.001, 60 * hours]
    kwargs = dict(x=pd.cut(df[f'foehn_minutes_during_{hours}_hours_after_start_of_fire'], bins= bins),
                  y=df['total [ha]'],
                  color="tab:blue")
    if control_var:
        kwargs.update(dict(x=df[control_var], hue=pd.cut(df[f'foehn_minutes_during_{hours}_hours_after_start_of_fire'], bins=bins)))

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
    return

def plot_binned_burned_area_before_fire_start(df, hours):
    bin_amount=6
    amount_of_fires = df[f'foehn_minutes_{hours}_hour_before'].count()
    df_binned = df.groupby(pd.cut(df[f'foehn_minutes_{hours}_hour_before'], bins=bin_amount)).count()

    foehn_minutes_in_interval = np.zeros(bin_amount)
    foehn_stations = df.columns.tolist()
    foehn_stations.remove("date")

    for station in foehn_stations:
        foehn_minutes_during_timeframe = df[station].rolling(6*hours).sum()*10
        foehn_minutes_in_interval += np.array([((df_binned.index[i].left < foehn_minutes_during_timeframe) &
                                                 (foehn_minutes_during_timeframe <df_binned.index[i].right)).sum() for i in range(len(df_binned.index))])

    foehn_minutes_in_interval = foehn_minutes_in_interval/len(foehn_stations)
    df_binned["foehn_minutes_in_interval"] =foehn_minutes_in_interval

    plt.figure()
    sns.barplot(x=df_binned.index, y=df_binned['total [ha]']/df_binned["foehn_minutes_in_interval"], color="tab:blue")
    plt.ylabel("Normalized count of fires")
    plt.xlabel("Foehn minutes")
    xticks, xlabels = plt.xticks()
    xlabels = [label.get_text() for label in xlabels]
    xlabels[0] = "[0.0" + xlabels[0][6:]
    plt.xticks(xticks, xlabels)
    save_figure(f"NormalizedFireCountOverFoehnMinutesForThe{hours}HoursBeforeStart")

    amount_of_fires = df[f'foehn_minutes_{hours}_hour_before'].count()
    df_binned = df.groupby(pd.cut(df[f'foehn_minutes_{hours}_hour_before'], bins=[-0.001,0.001,hours*60])).count()
    foehn_minutes_in_interval = np.zeros(2)
    foehn_stations = df.columns.tolist()
    foehn_stations.remove("date")

    for station in foehn_stations:
        foehn_minutes_during_timeframe = df[station].rolling(6*hours).sum()*10
        foehn_minutes_in_interval += np.array([((df_binned.index[i].left < foehn_minutes_during_timeframe) &
                                                 (foehn_minutes_during_timeframe <df_binned.index[i].right)).sum() for i in range(len(df_binned.index))])

    df_binned["foehn_minutes_in_interval"] =foehn_minutes_in_interval/len(foehn_stations)
    plt.figure(figsize=(16,9))
    sns.barplot(x=df_binned.index, y=df_binned['total [ha]']/df_binned["foehn_minutes_in_interval"], color="tab:blue")
    plt.ylabel("Normalized count of fires")
    plt.xlabel("")
    xticks, xlabels = plt.xticks()
    xlabels = [label.get_text() for label in xlabels]
    xlabels[0] = "Non-Foehn fires"
    xlabels[1] = "Foehn fires"
    plt.xticks(xticks, xlabels)
    save_figure(f"NonFoehnVsFoehnOverFoehnMinutesForThe{hours}HoursBeforeStart")
