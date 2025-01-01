# model_performance_chart
import matplotlib.pyplot as plt
import numpy as np

def plot_model_performance(train_metrics, test_metrics):

    models = ['XGBoost/LSTM', 'LightGBM/LSTM', 'XGBoost/CNN', 'LightGBM/CNN']
    
    # Prepare training data
    train_mae = [train_metrics[model][0] for model in models]
    train_rmse = [train_metrics[model][1] for model in models]
    train_r2 = [train_metrics[model][2] for model in models]
    
    # Prepare testing data
    test_mae = [test_metrics[model][0] for model in models]
    test_rmse = [test_metrics[model][1] for model in models]
    test_r2 = [test_metrics[model][2] for model in models]

    # Helper function for adding value labels
    def autolabel(rects, ax):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(rect.get_x() + rect.get_width()/2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=8)

    # Training MAE and RMSE plot
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, train_mae, width, label='MAE', color='lightcoral')
    rects2 = ax.bar(x + width/2, train_rmse, width, label='RMSE', color='lightblue')

    ax.set_ylabel('Error Value')
    ax.set_title('Training Performance (MAE and RMSE)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45)
    ax.legend()

    autolabel(rects1, ax)
    autolabel(rects2, ax)

    plt.tight_layout()
    plt.savefig('training_performance_mae_rmse.png')
    plt.close()

    # Testing MAE and RMSE plot
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, test_mae, width, label='MAE', color='red')
    rects2 = ax.bar(x + width/2, test_rmse, width, label='RMSE', color='blue')

    ax.set_ylabel('Error Value')
    ax.set_title('Testing Performance (MAE and RMSE)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45)
    ax.legend()

    autolabel(rects1, ax)
    autolabel(rects2, ax)

    plt.tight_layout()
    plt.savefig('testing_performance_mae_rmse.png')
    plt.close()

    # Training R² plot
    fig, ax = plt.subplots(figsize=(12, 6))
    rects = ax.bar(models, train_r2, color='lightgreen')

    ax.set_ylabel('R² Value')
    ax.set_title('Training Performance (R²)')
    plt.xticks(rotation=45)

    autolabel(rects, ax)

    plt.tight_layout()
    plt.savefig('training_performance_r2.png')
    plt.close()

    # Testing R² plot
    fig, ax = plt.subplots(figsize=(12, 6))
    rects = ax.bar(models, test_r2, color='darkgreen')

    ax.set_ylabel('R² Value')
    ax.set_title('Testing Performance (R²)')
    plt.xticks(rotation=45)

    autolabel(rects, ax)

    plt.tight_layout()
    plt.savefig('testing_performance_r2.png')
    plt.close()