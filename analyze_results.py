import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ndcg_score, confusion_matrix
from scipy.stats import spearmanr, kendalltau
import os
from tqdm import tqdm


LABEL_MAP = {0.5 : 0,
         1.0 : 1,
         1.5 : 2,
         2.0 : 3,
         2.5 : 4,
         3.0 : 5,
         3.5 : 6,
         4.0 : 7,
         4.5 : 8,
         5.0 : 9}

# Load a single dataset
def load_data(predictions_path, movies_path, ratings_path):
    predictions = pd.read_csv(predictions_path)
    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)
    
    # Merge data
    df = predictions.merge(movies, on='movieId', how='left')
    df = df.merge(ratings.drop(columns=['rating']), on=['userId', "movieId"], how='left')
    return df

# Compute per-user and per-source statistics
def compute_user_metrics(df):
    def gini_index(labels):
        possible_labels = np.arange(0.5, 5.5, 0.5)
        counts = labels.value_counts(normalize=True).reindex(possible_labels, fill_value=0)
        return 1 - np.sum(counts ** 2)
    
    user_stats = df.groupby(['userId', 'source']).agg(
        accuracy=('predicted_label', lambda x: np.mean(x == df.loc[x.index, 'true_label'])),
        num_movies=('movieId', 'count'),
        num_unique_labels=('true_label', 'nunique'),
        median_label=('true_label', 'median'),
        most_frequent_label=('true_label', lambda x: x.mode()[0] if not x.mode().empty else np.nan),
        std_label=('true_label', 'std'),
        gini_index=('true_label', gini_index)
    ).reset_index()
    return user_stats

# Compute NDCG per user and per source
def compute_ndcg(df, user_stats):
    ndcg_scores = []
    k_values = [3, 5, 10, 15, 20, 50, 100]
    
    for (user_id, source), user_data in tqdm(df.groupby(['userId', 'source']), desc="Computing NDCG"):
        num_movies = len(user_data)
        
        # Sort movies by predicted relevance
        user_data = user_data.sort_values(by='predicted_label', ascending=False)
        
        # Compute ground truth and predicted relevance
        true_relevance = user_data['true_label'].values.reshape(1, -1)
        predicted_relevance = user_data['predicted_label'].values.reshape(1, -1)
        
        ndcg_result = {'userId': user_id, 'source': source}
        
        for k in k_values:
            if num_movies >= k:
                ndcg_result[f'NDCG@{k}'] = ndcg_score(true_relevance, predicted_relevance, k=k)
            else:
                ndcg_result[f'NDCG@{k}'] = np.nan  # Not enough movies for this k value
        
        ndcg_scores.append(ndcg_result)
    
    return pd.DataFrame(ndcg_scores)

# Compute confusion matrix per source
def compute_confusion_matrices(final_df, output_path):
    final_df['true_label'] = final_df['true_label'].astype(str) #.map(LABEL_MAP)
    final_df['predicted_label'] = final_df['predicted_label'].astype(str) #.map(LABEL_MAP)
    sources = final_df['source'].unique()
    for source in sources:
        subset = final_df[final_df['source'] == source]
        cm = confusion_matrix(subset['true_label'], subset['predicted_label'])
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(f"Confusion Matrix for {source}")
        plt.savefig(f"{output_path}_confusion_matrix_{source}.png")

# Compute correlation metrics
def compute_correlations(final_user_stats):
    final_user_stats['avg_NDCG'] = final_user_stats.filter(like='NDCG@').mean(axis=1)
    correlations = {
        'accuracy_num_movies': final_user_stats[['accuracy', 'num_movies']].corr().iloc[0, 1],
        'accuracy_num_unique_labels': final_user_stats[['accuracy', 'num_unique_labels']].corr().iloc[0, 1],
        'accuracy_avg_NDCG': final_user_stats[['accuracy', 'avg_NDCG']].corr().iloc[0, 1],
        'std_label_accuracy': final_user_stats[['std_label', 'accuracy']].corr().iloc[0, 1],
        'gini_index_accuracy': final_user_stats[['gini_index', 'accuracy']].corr().iloc[0, 1]
    }
    spearman_corr = {
        'spearman_std_label_accuracy': spearmanr(final_user_stats['std_label'], final_user_stats['accuracy'])[0],
        'spearman_gini_index_accuracy': spearmanr(final_user_stats['gini_index'], final_user_stats['accuracy'])[0]
    }
    kendall_corr = {
        'kendall_std_label_accuracy': kendalltau(final_user_stats['std_label'], final_user_stats['accuracy'])[0],
        'kendall_gini_index_accuracy': kendalltau(final_user_stats['gini_index'], final_user_stats['accuracy'])[0]
    }
    
    return {**correlations, **spearman_corr, **kendall_corr}

# Plot boxplots for Gini index, Most Frequent Label, Accuracy, and NDCG scores
def plot_boxplots(final_user_stats, output_path):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='source', y='gini_index', data=final_user_stats)
    plt.title("Gini Index Distribution per Source")
    plt.ylabel("Gini Index")
    plt.xlabel("Source")
    plt.grid()
    plt.savefig(f"{output_path}_gini_index.png")
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='source', y='most_frequent_label', data=final_user_stats)
    plt.title("Most Frequent Label Distribution per Source")
    plt.ylabel("Most Frequent Label")
    plt.xlabel("Source")
    plt.grid()
    plt.savefig(f"{output_path}_most_frequent_label.png")

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='source', y='accuracy', data=final_user_stats)
    plt.title("Accuracy Distribution per Source")
    plt.ylabel("Accuracy")
    plt.xlabel("Source")
    plt.grid()
    plt.savefig(f"{output_path}_accuracy.png")

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='source', y='avg_NDCG', data=final_user_stats)
    plt.title("Average NDCG Distribution per Source")
    plt.ylabel("Average NDCG")
    plt.xlabel("Source")
    plt.grid()
    plt.savefig(f"{output_path}_avg_NDCG.png")

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate movie rating predictions.")
    parser.add_argument("--results_dir", default="src/data/results/", type=str, required=False, help="Directory containing predictions CSV files")
    parser.add_argument("--movies", default="src/data/movies.csv", type=str, required=False, help="Path to movies CSV file")
    parser.add_argument("--ratings", default="src/data/ratings.csv", type=str, required=False, help="Path to ratings CSV file")
    parser.add_argument("--balance_users", action="store_true", help="Balance the numbers of users among all sets")
    args = parser.parse_args()
    
    all_files = [f for f in os.listdir(args.results_dir) if f.endswith('.csv')]
    final_df = pd.DataFrame()
    final_user_stats = pd.DataFrame()
    final_ndcg_df = pd.DataFrame()

    for file in all_files:
        file_path = os.path.join(args.results_dir, file)
        df = load_data(file_path, args.movies, args.ratings)
        df['source'] = os.path.splitext(file)[0]

        if os.path.splitext(file)[0] == "agents_prediction":
            usersId = df["userId"].unique()

        user_stats = compute_user_metrics(df)
        ndcg_df = compute_ndcg(df, user_stats)

        final_df = pd.concat([final_df, df], ignore_index=True)
        final_user_stats = pd.concat([final_user_stats, user_stats], ignore_index=True)
        final_ndcg_df = pd.concat([final_ndcg_df, ndcg_df], ignore_index=True)
    
    # Merge user statistics with NDCG scores
    final_user_stats = final_user_stats.merge(final_ndcg_df, on=['userId', 'source'], how='left')
    
    # Compute correlations
    correlations = compute_correlations(final_user_stats)
    print("Correlation Results:", correlations)
    
    # Save final dataset
    final_user_stats.to_csv("final_user_metrics.csv", index=False)
    
    # Compute confusion matrices and save heatmaps
    compute_confusion_matrices(final_df, "confusion_matrix")
    
    # Plot boxplots for Gini Index and Most Frequent Label
    if args.balance_users:
        # Select the users in usersId for all final_user_stat
        final_user_stats = final_user_stats[final_user_stats["userId"].isin(usersId)]
        output_file = "balanced_boxplot"
    else:
        output_file = "boxplot"

    plot_boxplots(final_user_stats, output_file)
