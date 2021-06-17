import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import *

def evaluate_regression(y_true, y_pred):
    """
    Automatically evaluate the performance of a regressor (model-agnostic)
    """
    # Calculate maximum range
    min_val, max_val = min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))

    # Prediction density plot
    plt.figure()
    sns.histplot(y_true, kde=True, bins=np.linspace(min_val, max_val, 20))
    sns.histplot(y_pred, kde=True, bins=np.linspace(min_val, max_val, 20), color="orange")
    plt.legend(["True", "Pred"])
    plt.title("Prediction histogram")

    # Scatter plot
    plt.figure()
    sns.scatterplot(x=y_true, y=y_pred)
    plt.plot([min_val, max_val], [min_val, max_val], "r-")

    # Regression metrics
    digits = 4
    mae = round(mean_absolute_error(y_true, y_pred), digits)
    mse = round(mean_squared_error(y_true, y_pred), digits)
    rmse = round(np.sqrt(mse), digits)
    r2 = round(r2_score(y_true, y_pred), digits)
    
    print('(MAE) Mean absolute error: ', mae)
    #print('(MSE) Mean squared error: ', mse)
    print('(RMSE) Root mean squared error: ', rmse)  # Gives a relatively high weight to large errors (compared to MAE)
    #print('(RMSLE) Root mean squared log error: ', round(mean_squared_log_error(y_true, y_pred),4))  # Punishes underprediction harder, robuster towards outliers, also considers a relative error
    #print('(MAPE) Mean absolute percentage error: ', round(mean_absolute_percentage_error(y_true, y_pred), 4))  # sensitive to relative errors, scale-invariant
    #print('(MedAE) Median absolute error: ', round(median_absolute_error(y_true, y_pred), 4))  # Robust to outliers
    print('(R2) R2-score: ', r2)
    # print('(EV) Explained_variance: ', round(explained_variance_score(y_true, y_pred), 4))  # Equivalent to R2 if mean error/residuals == 0, otherwise bias in residuals
    return mae, rmse, r2

def evaluate_binary_classification(y_true, y_proba, threshold_optimized=None):
    """
    Automatically evaluate the performance of a binary classifier (model-agnostic)
    """
    # Prediction density plot
    plt.figure()
    sns.histplot(y_proba[y_true == 1], kde=True, binrange=[0, 1], bins=25)
    sns.histplot(y_proba[y_true == 0], kde=True, binrange=[0, 1], bins=25, color="orange")
    plt.legend(["class-1", "class-0"])
    plt.title("Prediction density plot")

    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    plt.axis('square')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title("ROC Curve")

    # Precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    PrecisionRecallDisplay(precision, recall).plot()
    plt.axis('square')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title("Precision Recall Curve")

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