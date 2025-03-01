import matplotlib.pyplot as plt
import seaborn as sns

def plot_accuracy_distribution(data):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='source', y='accuracy', data=data)
    plt.title('User Accuracy Distribution by Source')
    plt.xlabel('Source')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.show()

def plot_boxplots(data):
    metrics = ['accuracy', 'ndcg@3', 'ndcg@20'] # Parameters
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='source', y=metric, data=data)
        plt.title(f'User {metric.capitalize()} Distribution by Source')
        plt.xlabel('Source')
        plt.ylabel(metric.capitalize())
        plt.xticks(rotation=45)
        plt.show()