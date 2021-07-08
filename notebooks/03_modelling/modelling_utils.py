import numpy as np
from scipy.integrate import simps
from sklearn.metrics import *
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import mlflow


def evaluate_regression(y_train_true, y_train_pred, y_test_true, y_test_pred):
    """
    Automatically evaluate the performance of a regressor (model-agnostic)
    """

    # Little preprocessing to more conveniently use the variables
    train = pd.DataFrame()
    train["True value"] = y_train_true
    train["Predicted value"] = y_train_pred
    train["dataset"] = "train"

    test = pd.DataFrame()
    test["True value"] = y_test_true
    test["Predicted value"] = y_test_pred
    test["dataset"] = "test"

    df = pd.concat([train, test], axis=0, ignore_index=True)

    # Make scatter plot
    fig = px.scatter(data_frame=df, x="True value", y="Predicted value", facet_col="dataset")

    # Insert y=x line
    min_val = min(train.drop(columns="dataset").min())
    max_val = max(train.drop(columns="dataset").max())

    # Add y=x lines
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=min_val, x1=max_val, y0=min_val, y1=max_val, row=1, col=1
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=min_val, x1=max_val, y0=min_val, y1=max_val, row=1, col=2
    )

    # Set some layout parameters    
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')
    fig.update_layout(height=500)

    # Calculate regression metrics and annotate plot
    digits = 3
    for i, df in enumerate([train, test]):
        mae = round(mean_absolute_error(df["True value"], df["Predicted value"]), digits)
        mse = round(mean_squared_error(df["True value"], df["Predicted value"]), digits)
        rmse = round(np.sqrt(mse), digits)
        r2 = round(r2_score(df["True value"], df["Predicted value"]), digits)

        fig.add_annotation(
            text=f"(MAE) Mean absolute error: {mae} <br>(RMSE) Root mean squared error:  {rmse} <br>(R2) R2-score: {r2}",
            xref="x domain", yref="y domain", align="left",
            x=0, y=1, showarrow=False, row=1, col=i + 1)

        # Log metrics if wanted
        # set_ = df["dataset"].unique()[0]
        # mlflow.log_metrics({f"mae_{set_}": mae, "rmse_{set_}": rmse, "r2_{set_}": r2})

    fig.show()


def evaluate_binary_classification(y_train_true, y_train_proba, y_test_true, y_test_proba, threshold_optimized=None):
    """
    Automatically evaluate the performance of a binary classifier (model-agnostic)
    """

    def calc_aucbg(y_true: np.array, y_proba: np.array) -> (np.array, np.array, float):
        """
        Calculate the AUC.bg curve and score.
        :param y_true: True labels
        :type y_true: np.array
        :param y_proba: Predicted probability for label
        :type y_proba: np.array
        :return: Background rate (bgr), True positive rate (tpr) and AUC.bg (aucbg)
        :rtype: tuple
        """
        # Define array of thresholds
        ths = np.linspace(1, 0, 5000)

        # Make mask for positive labels
        positives = (y_true == 1)

        # Loop over all thresholds and calculate tpr and bgr
        tpr, bgr = [], []
        for c in ths:
            # Binarize probability according to threshold
            y_pred = (c <= y_proba)

            # Calculate tp und bg
            tp = y_pred[positives].sum()
            bg = y_pred.sum()
            tpr.append(tp)
            bgr.append(bg)

        # Calculate rates
        tpr = np.array(tpr) / positives.sum()
        bgr = np.array(bgr) / len(y_proba)

        # Calculate prevalence and maximum possible AUC.bg
        prevalence = positives.sum() / len(y_proba)
        max_aucbg = 1 - prevalence / 2
        # plt.plot(bgr, tpr, "x")
        # plt.xlabel("Background proportion")
        # plt.ylabel("True positive rate")

        # Calculate AUC.bg
        index = np.unique(bgr, return_index=True)[1]
        aucbg = simps(y=tpr[index], x=bgr[index])
        return bgr, tpr, aucbg, max_aucbg

    # Little preprocessing to more conveniently use the variables
    train = pd.DataFrame()
    train["True value"] = y_train_true
    train["Predicted value"] = y_train_proba
    train["dataset"] = "train"
    test = pd.DataFrame()
    test["True value"] = y_test_true
    test["Predicted value"] = y_test_proba
    test["dataset"] = "test"
    df = pd.concat([train, test], axis=0, ignore_index=True)

    # Prediction density plot
    fig = px.histogram(data_frame=df, x="Predicted value", color="True value", facet_col="dataset", log_y=True,
                       nbins=30, marginal="box", opacity=0.75)
    fig.update_layout(barmode="overlay", height=500, hovermode="x")
    fig.show()

    # ROC.bg curves
    train_met = pd.DataFrame()
    train_met["bgr"], train_met["tpr"], aucbg_train, max_aucbg_train = calc_aucbg(train["True value"],
                                                                                  train["Predicted value"])
    train_met["dataset"] = "train"
    test_met = pd.DataFrame()
    test_met["bgr"], test_met["tpr"], aucbg_test, max_aucbg_test = calc_aucbg(test["True value"],
                                                                              test["Predicted value"])
    test_met["dataset"] = "test"
    df_met = pd.concat([train_met, test_met], axis=0, ignore_index=True)

    fig = px.area(data_frame=df_met,
                  x="bgr", y="tpr", facet_col="dataset",
                  labels=dict(tpr='True Positive Rate', bgr='Background Rate'), range_y=(0, 1), range_x=(0, 1),
                  height=500,
                  )

    # Insert y=x line
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1, col=1, row=1
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1, col=2, row=1
    )

    # Set some more plot parameters
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')
    fig.update_layout(hovermode="x")
    fig.layout.annotations[0]['text'] = fig.layout.annotations[0][
                                            'text'] + f" (AUC.bg: {round(aucbg_train, 4)}, AUC.bg.max: {round(max_aucbg_train, 4)})"
    fig.layout.annotations[1]['text'] = fig.layout.annotations[1][
                                            'text'] + f" (AUC.bg: {round(aucbg_test, 4)}, AUC.bg.max: {round(max_aucbg_test, 4)})"
    fig.show()

    # ROC curves
    train_met = pd.DataFrame()
    train_met["fpr"], train_met["tpr"], thresholds = roc_curve(train["True value"], train["Predicted value"])
    train_met["dataset"] = "train"
    test_met = pd.DataFrame()
    test_met["fpr"], test_met["tpr"], thresholds = roc_curve(test["True value"], test["Predicted value"])
    test_met["dataset"] = "test"
    df_met = pd.concat([train_met, test_met], axis=0, ignore_index=True)

    fig = px.area(data_frame=df_met,
                  x="fpr", y="tpr", facet_col="dataset",
                  labels=dict(tpr='True Positive Rate', fpr='False Positive Rate'), range_y=(0, 1), range_x=(0, 1),
                  height=500,
                  )

    # Insert y=x line
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1, col=1, row=1
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1, col=2, row=1
    )

    # Set some more plot parameters
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')
    fig.update_layout(hovermode="x")
    fig.layout.annotations[0]['text'] = fig.layout.annotations[0][
                                            'text'] + f" (AUC: {round(roc_auc_score(train['True value'], train['Predicted value']), 4)})"
    fig.layout.annotations[1]['text'] = fig.layout.annotations[1][
                                            'text'] + f" (AUC: {round(roc_auc_score(test['True value'], test['Predicted value']), 4)})"
    fig.show()

    # Precision-Recall (PR) curves
    train_met = pd.DataFrame()
    train_met["pre"], train_met["rec"], thresholds = precision_recall_curve(train["True value"],
                                                                            train["Predicted value"])
    train_met["dataset"] = "train"
    auc_train = auc(train_met["rec"], train_met["pre"])
    test_met = pd.DataFrame()
    test_met["pre"], test_met["rec"], thresholds = precision_recall_curve(test["True value"], test["Predicted value"])
    test_met["dataset"] = "test"
    auc_test = auc(test_met["rec"], test_met["pre"])
    df_met = pd.concat([train_met, test_met], axis=0, ignore_index=True)

    fig = px.area(data_frame=df_met,
                  x="rec", y="pre", facet_col="dataset",
                  labels=dict(rec='Recall', pre='Precision'), range_y=(0, 1), range_x=(0, 1),
                  height=500,
                  )

    # Set some more plot parameters
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')
    fig.update_layout(hovermode="x")
    fig.layout.annotations[0]['text'] = fig.layout.annotations[0]['text'] + f" (AUC: {round(auc_train, 4)})"
    fig.layout.annotations[1]['text'] = fig.layout.annotations[1]['text'] + f" (AUC: {round(auc_test, 4)})"
    fig.show()

    # Precision, Recall, and F1 over threshold plot
    train_met = pd.DataFrame()
    train_met["pre"], train_met["rec"], thresholds = precision_recall_curve(train["True value"],
                                                                            train["Predicted value"])
    train_met = train_met.iloc[:-1, :]
    train_met["thresholds"] = thresholds
    train_met["f1"] = 2 * train_met["pre"] * train_met["rec"] / (train_met["pre"] + train_met["rec"])
    train_met["dataset"] = "train"
    test_met = pd.DataFrame()
    test_met["pre"], test_met["rec"], thresholds = precision_recall_curve(test["True value"], test["Predicted value"])
    test_met = test_met.iloc[:-1, :]
    test_met["thresholds"] = thresholds
    test_met["f1"] = 2 * test_met["pre"] * test_met["rec"] / (test_met["pre"] + test_met["rec"])
    test_met["dataset"] = "test"
    df_met = pd.concat([train_met, test_met], axis=0, ignore_index=True)
    df_met = df_met.melt(id_vars=["dataset", "thresholds"], var_name="metric")

    fig = px.line(data_frame=df_met,
                  x="thresholds", y="value", color="metric", facet_col="dataset",
                  labels=dict(rec='Recall', pre='Precision', f1="F1-Score"), range_y=(0, 1), range_x=(0, 1),
                  height=500,
                  )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')
    fig.update_layout(hovermode="x")
    fig.show()

    # Optimize threshold for objective (here F1)
    if not threshold_optimized:
        threshold_optimized = train_met.loc[train_met["f1"].argmax(), "thresholds"]
    y_train_pred = (y_train_proba > threshold_optimized).astype(int)
    y_test_pred = (y_test_proba > threshold_optimized).astype(int)

    # Write out metrics into Table
    fig = make_subplots(rows=1, cols=2)

    metrics = {
        "AUROC": [roc_auc_score(y_train_true, y_train_proba), roc_auc_score(y_test_true, y_test_proba)],
        "Log.loss": [log_loss(y_train_true, y_train_proba), log_loss(y_test_true, y_test_proba)],
        "Precision": [precision_score(y_train_true, y_train_pred), precision_score(y_test_true, y_test_pred)],
        "Recall": [recall_score(y_train_true, y_train_pred), recall_score(y_test_true, y_test_pred)],
        "F1": [f1_score(y_train_true, y_train_pred), f1_score(y_test_true, y_test_pred)]
    }
    metrics = pd.DataFrame.from_dict(metrics, orient="index").reset_index().rename(
        columns={"index": "Metric", 0: "Train set", 1: "Test set"})

    fig.add_trace(
        go.Table(
            cells={"values": [metrics["Metric"], np.round(metrics["Train set"], 4), np.round(metrics["Test set"], 4)]},
            header={"values": metrics.columns}),
    )

    fig.show()
