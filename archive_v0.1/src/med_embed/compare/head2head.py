import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def compare_models(df, models=None, metrics=None):
    """
    Perform head-to-head comparison of models across all tasks and metrics.
    
    :param df: DataFrame with columns 'task_name', 'model', and various metrics
    :param models: List of models to compare. If None, use all models in df
    :param metrics: List of metrics to use. If None, use all numeric columns except 'task_name' and 'model'
    :return: Tuple of (win_matrix, win_percentages)
    """
    if models is None:
        models = df['model'].unique()
    else:
        df = df[df['model'].isin(models)]
    
    if metrics is None:
        metrics = df.select_dtypes(include=[np.number]).columns.drop(['task_name', 'model'], errors='ignore')
    
    n_models = len(models)
    win_matrix = np.zeros((n_models, n_models))
    
    for task in df['task_name'].unique():
        task_data = df[df['task_name'] == task].set_index('model')
        for metric in metrics:
            for i, model1 in enumerate(models):
                for j, model2 in enumerate(models):
                    if i != j:
                        if task_data.loc[model1, metric] > task_data.loc[model2, metric]:
                            win_matrix[i, j] += 1

    total_comparisons = len(df['task_name'].unique()) * len(metrics) * (n_models - 1)
    win_percentages = win_matrix.sum(axis=1) / total_comparisons * 100

    return win_matrix, win_percentages

def plot_comparison(win_matrix, win_percentages, models):
    """
    Plot heatmap of win matrix and bar plot of win percentages.
    
    :param win_matrix: Matrix of head-to-head wins
    :param win_percentages: Array of win percentages for each model
    :param models: List of model names
    """
    # Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(win_matrix, annot=True, fmt='.0f', cmap='YlGnBu',
                xticklabels=models, yticklabels=models)
    plt.title('Head-to-Head Comparison of Models (Number of Wins Across All Metrics)')
    plt.xlabel('Model (Columns)')
    plt.ylabel('Model (Rows)')
    plt.tight_layout()
    plt.show()

    # Bar plot
    plt.figure(figsize=(12, 6))
    plt.bar(models, win_percentages)
    plt.title('Model Performance: Percentage of Head-to-Head Wins Across All Metrics')
    plt.xlabel('Model')
    plt.ylabel('Win Percentage')
    plt.ylim(0, 100)  # Set y-axis limit from 0 to 100%
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()