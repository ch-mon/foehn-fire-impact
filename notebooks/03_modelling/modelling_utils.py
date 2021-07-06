import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
        
        fig.add_annotation(text=f"(MAE) Mean absolute error: {mae} <br>(RMSE) Root mean squared error:  {rmse} <br>(R2) R2-score: {r2}",
                  xref="x domain", yref="y domain", align="left",
                  x=0, y=1, showarrow=False, row=1,col=i+1)
        
        # Log metrics if wanted
        # set_ = df["dataset"].unique()[0]
        # mlflow.log_metrics({f"mae_{set_}": mae, "rmse_{set_}": rmse, "r2_{set_}": r2})

    fig.show() 
    


def evaluate_binary_classification(y_train_true, y_train_pred, y_test_true, y_test_pred, threshold_optimized=None):
    """
    Automatically evaluate the performance of a binary classifier (model-agnostic)
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

    # Prediction density plot
    fig = px.histogram(data_frame=df, x="Predicted value", color="True value", facet_col="dataset", log_y=True, nbins=30, marginal="box", opacity=0.75)
    fig.update_layout(barmode="overlay", height=500, hovermode="x")
    fig.show()

    # ROC curves
    train_roc = pd.DataFrame()
    train_roc["fpr"], train_roc["tpr"], thresholds = roc_curve(train["True value"], train["Predicted value"])
    train_roc["dataset"] = "train"

    test_roc = pd.DataFrame()
    test_roc["fpr"], test_roc["tpr"], thresholds = roc_curve(test["True value"], test["Predicted value"])
    test_roc["dataset"] = "test"

    df_roc = pd.concat([train_roc, test_roc], axis=0, ignore_index=True)

    fig = px.area(data_frame=df_roc,
        y="tpr", x="fpr", facet_col="dataset",
        labels=dict(tpr='False Positive Rate', fpr='True Positive Rate'), range_y=(0, 1), range_x=(0, 1),
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

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')
    fig.update_layout(hovermode="x")
    fig.layout.annotations[0]['text'] = fig.layout.annotations[0]['text'] + f" (AUC: {round(roc_auc_score(train['True value'], train['Predicted value']),4)})"
    fig.layout.annotations[1]['text'] = fig.layout.annotations[1]['text'] + f" (AUC: {round(roc_auc_score(test['True value'], test['Predicted value']),4)})"
    fig.show()

    # PR curves
    train_roc = pd.DataFrame()
    train_roc["pre"], train_roc["rec"], thresholds = precision_recall_curve(train["True value"], train["Predicted value"])
    train_roc["dataset"] = "train"
    auc_train = auc(train_roc["rec"], train_roc["pre"])

    test_roc = pd.DataFrame()
    test_roc["pre"], test_roc["rec"], thresholds = precision_recall_curve(test["True value"], test["Predicted value"])
    test_roc["dataset"] = "test"
    auc_test = auc(test_roc["rec"], test_roc["pre"])

    df_roc = pd.concat([train_roc, test_roc], axis=0, ignore_index=True)

    fig = px.area(data_frame=df_roc,
                  x="rec", y="pre", facet_col="dataset",
                  labels=dict(rec='Recall', pre='Precision'), range_y=(0, 1), range_x=(0, 1),
                  height=500,
                  )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')
    fig.update_layout(hovermode="x")
    fig.layout.annotations[0]['text'] = fig.layout.annotations[0]['text'] + f" (AUC: {round(auc_train, 4)})"
    fig.layout.annotations[1]['text'] = fig.layout.annotations[1]['text'] + f" (AUC: {round(auc_test, 4)})"
    fig.show()


    return

    # Precision, Recall, and F1 over threshold
    plt.figure()
    f1 = np.nan_to_num(2 * precision * recall / (precision + recall))
    sns.lineplot(x=thresholds, y=precision[:-1])
    sns.lineplot(x=thresholds, y=recall[:-1])
    sns.lineplot(x=thresholds, y=f1[:-1])
    plt.axis('square')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend(["Precision", "Recall", "F1"])
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Precision, Recall, and F1 vs. threshold")

    # Optimize threshold for objective
    if not threshold_optimized:
        threshold_optimized = thresholds[f1.argmax()]
    y_pred = (y_proba > threshold_optimized).astype(int)

    # Confusion matrix
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred)).plot()

    print("AUROC: ", roc_auc_score(y_true, y_proba))
    print("Log. loss: ", log_loss(y_true, y_proba))
    print("Best threshold with F1-score: ", threshold_optimized)
    print("Precision: ", precision_score(y_true, y_pred))
    print("Recall: ", recall_score(y_true, y_pred))
    print("F1: ", f1_score(y_true, y_pred))

    return threshold_optimized

def calc_AUCbg(y_true, y_proba):
    thresholds = np.linspace(0,1,100)
    positives = (y_true == 1)
    
    for c in thresholds:
        y_pred = int(c < y_proba)
        TP = (y_true[positives] == y_pred[positives]).sum()
        
        (~y_pred).sum()/len(y_proba)
        
    return AUCbg