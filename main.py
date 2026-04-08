import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

from src.data.preprocess import load_data, preprocess, build_user_item_matrix
from src.models.recommender import CollaborativeFilteringModel
from src.evaluation.metrics import precision_at_k, recall_at_k, ndcg_at_k

# --- Config ---
DATA_PATH = "data/raw/interactions.csv"
N_COMPONENTS = 20
TOP_N = 10
EXPERIMENT_NAME = "recommendation_system"

# --- MLFlow setup ---
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run():

    # Log parameters
    mlflow.log_param("n_components", N_COMPONENTS)
    mlflow.log_param("top_n", TOP_N)

    # Load & preprocess data
    df = load_data(DATA_PATH)
    df = preprocess(df)
    matrix, user_index, item_index = build_user_item_matrix(df)

    # Train model
    model = CollaborativeFilteringModel(n_components=N_COMPONENTS)
    model.fit(matrix)

    # Evaluate on a sample user
    sample_user_idx = 0
    recommendations = model.recommend(sample_user_idx, top_n=TOP_N).tolist()

    # Dummy relevant items for illustration (replace with real ground truth)
    relevant_items = list(range(5))

    p_at_k = precision_at_k(recommendations, relevant_items, k=TOP_N)
    r_at_k = recall_at_k(recommendations, relevant_items, k=TOP_N)
    ndcg = ndcg_at_k(recommendations, relevant_items, k=TOP_N)

    # Log metrics
    mlflow.log_metric("precision_at_k", p_at_k)
    mlflow.log_metric("recall_at_k", r_at_k)
    mlflow.log_metric("ndcg_at_k", ndcg)

    # Log model
    mlflow.sklearn.log_model(model.model, "svd_model")

    print(f"Precision@{TOP_N}: {p_at_k:.4f}")
    print(f"Recall@{TOP_N}:    {r_at_k:.4f}")
    print(f"NDCG@{TOP_N}:      {ndcg:.4f}")
    print("Run logged to MLFlow.")
