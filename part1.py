
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Save plots when running without Jupyter.
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix


# Ranking metrics from the notebook.
def ranking_metrics_at_k(recommendations, relevant_items, k=10):
    recommendations = list(recommendations)[:k]
    relevant_items = set(relevant_items)
    hits = len(set(recommendations) & relevant_items)
    precision = hits / k
    recall = hits / len(relevant_items) if relevant_items else 0.0
    dcg = 0.0
    for rank, movie_id in enumerate(recommendations, start=1):
        if movie_id in relevant_items:
            dcg += 1 / np.log2(rank + 1)
    ideal_hits = min(k, len(relevant_items))
    idcg = sum(1 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return precision, recall, ndcg


def recommend_popular(user_id, popularity_scores, history, n=10):
    seen_items = set(history.loc[history["user_id"] == user_id, "movie_id"])
    unseen_mask = ~popularity_scores.index.isin(seen_items)
    candidates = popularity_scores[unseen_mask]
    return candidates.head(n)


def recommendation_titles(recommendations, movies):
    return (recommendations.rename("score").reset_index()
            .merge(movies[["movie_id", "title"]], on="movie_id", how="left"))


def evaluate_popular(popularity_scores, validation_truth, train_seen, catalogue, k=10):
    precisions, recalls, ndcgs = [], [], []
    all_recommended = set()
    for user, relevant in validation_truth.items():
        seen = train_seen.get(user, set())
        recommendations = []
        for movie_id in popularity_scores.index:
            if movie_id not in seen:
                recommendations.append(movie_id)
            if len(recommendations) == k:
                break
        if len(recommendations) != k:
            raise ValueError(f"Only {len(recommendations)} recommendations for user {user}")
        if set(recommendations) & seen:
            raise AssertionError("A seen movie was recommended")
        p, r, n = ranking_metrics_at_k(recommendations, relevant, k)
        precisions.append(p)
        recalls.append(r)
        ndcgs.append(n)
        all_recommended.update(recommendations)
    if not validation_truth:
        raise ValueError("No eligible evaluation users")
    return {
        "Users": len(validation_truth),
        "Precision@10": float(np.mean(precisions)),
        "Recall@10": float(np.mean(recalls)),
        "NDCG@10": float(np.mean(ndcgs)),
        "Coverage": len(all_recommended) / len(catalogue)
    }


def build_knn_weights(similarity, k):
    n_items = similarity.shape[0]
    if not 1 <= k < n_items:
        raise ValueError(f"k={k} must be smaller than catalogue size {n_items}")
    sim = similarity.copy()
    np.fill_diagonal(sim, -np.inf)
    neighbors = np.argsort(-sim, axis=1, kind="stable")[:, :k]
    weights = np.take_along_axis(similarity, neighbors, axis=1)
    W = csr_matrix(
        (weights.ravel(),
         (np.repeat(np.arange(n_items), k), neighbors.ravel())),
        shape=(n_items, n_items)
    )
    W.eliminate_zeros()
    return W.T.tocsr()


def recommend_knn(user_id, weights_T, user_to_row, X, train_seen,
                  movie_ids, popularity_tiebreak, n=10):
    scores = np.zeros(len(movie_ids), dtype=np.float32)
    if user_id in user_to_row:
        scores = (X[user_to_row[user_id]] @ weights_T).toarray().ravel()
    seen = train_seen.get(user_id, set())
    seen_positions = movie_ids.get_indexer(list(seen))
    scores[seen_positions[seen_positions >= 0]] = -np.inf
    ranked = np.lexsort((movie_ids.to_numpy(), -popularity_tiebreak, -scores))
    top = ranked[np.isfinite(scores[ranked])][:n]
    return pd.Series(scores[top], index=movie_ids[top], name="knn_score")


def evaluate_knn(weights_T, validation_truth, user_to_row, X,
                 train_seen, movie_ids, popularity_tiebreak, catalogue):
    precisions, recalls, ndcgs = [], [], []
    all_recommended = set()
    for user, relevant in validation_truth.items():
        recommendations = recommend_knn(
            user, weights_T, user_to_row, X, train_seen, movie_ids,
            popularity_tiebreak, n=10
        )
        if len(recommendations) != 10:
            raise ValueError(f"Not enough recommendations for user {user}")
        if set(recommendations.index) & train_seen.get(user, set()):
            raise AssertionError("A seen movie was recommended")
        p, r, ndcg = ranking_metrics_at_k(recommendations.index, relevant, k=10)
        precisions.append(p)
        recalls.append(r)
        ndcgs.append(ndcg)
        all_recommended.update(recommendations.index)
    return {
        "Users": len(validation_truth),
        "Precision@10": float(np.mean(precisions)),
        "Recall@10": float(np.mean(recalls)),
        "NDCG@10": float(np.mean(ndcgs)),
        "Coverage": len(all_recommended) / len(catalogue)
    }


def get_similarities(X):
    # Two similarity measures from the notebook.
    item_similarity = cosine_similarity(X.T).astype(np.float32)
    np.fill_diagonal(item_similarity, 0)
    item_counts = np.asarray(X.sum(axis=0)).ravel().astype(np.float32)
    intersection = (X.T @ X).toarray().astype(np.float32)
    union = item_counts[:, None] + item_counts[None, :] - intersection
    jaccard_similarity = np.divide(
        intersection, union, out=np.zeros_like(intersection), where=union > 0
    )
    np.fill_diagonal(jaccard_similarity, 0)
    return item_similarity, jaccard_similarity


def save_plot(output_dir, filename):
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=160)
    plt.close()


def main(data_dir=".", output_dir="part1_outputs"):
    np.random.seed(42)
    DATA_DIR = Path(data_dir)
    OUTPUT_DIR = Path(output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. MovieLens data.
    for name in ("ratings.dat", "movies.dat", "users.dat"):
        if not (DATA_DIR / name).is_file():
            raise FileNotFoundError(f"Missing {name} in {DATA_DIR.resolve()}")
    ratings = pd.read_csv(
        DATA_DIR / "ratings.dat", sep="::", engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"]
    )
    movies = pd.read_csv(
        DATA_DIR / "movies.dat", sep="::", engine="python", encoding="latin-1",
        names=["movie_id", "title", "genres"]
    )
    users = pd.read_csv(
        DATA_DIR / "users.dat", sep="::", engine="python",
        names=["user_id", "gender", "age", "occupation", "zip_code"],
        dtype={"zip_code": str}
    )
    assert ratings.notna().all().all()
    assert movies["movie_id"].is_unique
    print("Ratings shape:", ratings.shape)
    print("Movies shape:", movies.shape)
    print("Users shape:", users.shape)

    n_users = ratings["user_id"].nunique()
    n_items = ratings["movie_id"].nunique()
    n_interactions = len(ratings)
    ratings["datetime"] = pd.to_datetime(ratings["timestamp"], unit="s")
    possible_interactions = n_users * n_items
    density = n_interactions / possible_interactions
    print("Users:", n_users, "Movies rated:", n_items)
    print("Interactions:", n_interactions, "Density %:", round(density * 100, 2))
    print("Start:", ratings["datetime"].min(), "End:", ratings["datetime"].max())

    # 2. Data analysis plots.
    user_interactions = ratings.groupby("user_id").size()
    plt.figure(figsize=(8, 5))
    plt.hist(user_interactions.values, bins=20)
    plt.xlabel("Number of ratings per user")
    plt.ylabel("Number of users")
    plt.title("Distribution of interactions per user")
    save_plot(OUTPUT_DIR, "interactions_per_user.png")

    item_interactions = ratings.groupby("movie_id").size()
    sorted_item_interactions = item_interactions.sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(sorted_item_interactions)), sorted_item_interactions.values)
    plt.xlabel("Movies sorted by popularity")
    plt.ylabel("Number of ratings")
    plt.title("Long-tail distribution of movie popularity")
    save_plot(OUTPUT_DIR, "item_long_tail.png")

    top_users = user_interactions.nlargest(int(np.ceil(len(user_interactions) * 0.10)))
    user_share = top_users.sum() / len(ratings) * 100
    top_movies = item_interactions.nlargest(int(np.ceil(len(item_interactions) * 0.10)))
    movie_share = top_movies.sum() / len(ratings) * 100
    print(f"Top 10% of users: {user_share:.2f}% of ratings")
    print(f"Top 10% of movies: {movie_share:.2f}% of ratings")

    rating_distribution = ratings["rating"].value_counts().sort_index()
    plt.figure(figsize=(7, 5))
    rating_distribution.plot(kind="bar")
    plt.xlabel("Rating")
    plt.ylabel("Number of ratings")
    plt.title("Rating distribution")
    save_plot(OUTPUT_DIR, "rating_distribution.png")

    rating_percentage = ratings["rating"].value_counts(normalize=True).sort_index() * 100
    ratings["month"] = ratings["datetime"].dt.to_period("M")
    monthly_activity = ratings.groupby("month").size()
    plt.figure(figsize=(12, 5))
    plt.plot(monthly_activity.index.to_timestamp(), monthly_activity.values)
    plt.xlabel("Month")
    plt.ylabel("Number of ratings")
    plt.title("Rating activity over time")
    save_plot(OUTPUT_DIR, "monthly_activity.png")

    # 3. Relevance and temporal split.
    POSITIVE_THRESOLD = 4  # Notebook's original variable name.
    ratings["relevant"] = ratings["rating"] >= POSITIVE_THRESOLD
    ratings = ratings.sort_values("timestamp").reset_index(drop=True)
    train_cut = ratings["timestamp"].quantile(0.80)
    val_cut = ratings["timestamp"].quantile(0.90)
    train_cut_date = pd.to_datetime(train_cut, unit="s")
    val_cut_date = pd.to_datetime(val_cut, unit="s")
    train = ratings[ratings["timestamp"] <= train_cut].copy()
    validation = ratings[
        (ratings["timestamp"] > train_cut) & (ratings["timestamp"] <= val_cut)
    ].copy()
    test = ratings[ratings["timestamp"] > val_cut].copy()
    assert min(len(train), len(validation), len(test)) > 0
    assert train["timestamp"].max() < validation["timestamp"].min()
    assert validation["timestamp"].max() < test["timestamp"].min()
    assert len(train) + len(validation) + len(test) == len(ratings)
    print("Train/validation/test rows:", len(train), len(validation), len(test))
    print("Cut dates:", train_cut_date, val_cut_date)

    train_users = set(train["user_id"])
    validation_users = set(validation["user_id"])
    test_users = set(test["user_id"])
    train_items = set(train["movie_id"])
    validation_items = set(validation["movie_id"])
    test_items = set(test["movie_id"])
    print("New users in validation/test:",
          len(validation_users - train_users), len(test_users - train_users))
    print("New items in validation/test:",
          len(validation_items - train_items), len(test_items - train_items))
    catalogue = pd.Index(movies["movie_id"].drop_duplicates(), name="movie_id")
    catalogue_set = set(catalogue)
    assert set(ratings["movie_id"]).issubset(catalogue_set)

    # 4. Validation ground truth; correctly extract movie IDs.
    train_seen = train.groupby("user_id")["movie_id"].apply(set).to_dict()
    validation_truth = (
        validation[validation["relevant"]]
        .groupby("user_id")["movie_id"].apply(set).to_dict()
    )
    validation_truth = {
        user: (items - train_seen.get(user, set())) & catalogue_set
        for user, items in validation_truth.items()
    }
    validation_truth = {user: items for user, items in validation_truth.items() if items}
    assert validation_truth, "No eligible validation users"
    print("Eligible validation users:", len(validation_truth))

    # 5. Popularity baselines.
    popular_all = (train.groupby("movie_id").size()
                   .reindex(catalogue, fill_value=0)
                   .sort_values(ascending=False))
    train_positive = train[train["relevant"]]
    popular_positive = (train_positive.groupby("movie_id").size()
                        .reindex(catalogue, fill_value=0)
                        .sort_values(ascending=False))
    result_all = evaluate_popular(popular_all, validation_truth, train_seen, catalogue, k=10)
    result_positive = evaluate_popular(popular_positive, validation_truth, train_seen, catalogue, k=10)
    results = pd.DataFrame(
        [result_all, result_positive],
        index=["Most Popular - All Ratings", "Most Popular - Positive Ratings"]
    )
    print("\nMost Popular validation results:\n", results.round(4).to_string())

    # 6. Binary user-item matrix.
    positive_train = train.loc[train["relevant"], ["user_id", "movie_id"]].drop_duplicates()
    user_ids = pd.Index(train["user_id"].unique(), name="user_id")
    movie_ids = catalogue
    user_positions = user_ids.get_indexer(positive_train["user_id"])
    movie_positions = movie_ids.get_indexer(positive_train["movie_id"])
    assert (user_positions >= 0).all() and (movie_positions >= 0).all()
    X = csr_matrix(
        (np.ones(len(positive_train), dtype=np.float32), (user_positions, movie_positions)),
        shape=(len(user_ids), len(movie_ids))
    )
    assert X.nnz == len(positive_train)
    user_to_row = {uid: i for i, uid in enumerate(user_ids)}
    popularity_tiebreak = popular_all.reindex(movie_ids, fill_value=0).to_numpy()
    item_similarity, jaccard_similarity = get_similarities(X)

    # 7. Validation-only k and similarity tuning.
    similarities = {"Cosine": item_similarity, "Jaccard": jaccard_similarity}
    k_values = [10, 20, 50]
    experiments = []
    for sim_name, sim_matrix in similarities.items():
        for k in k_values:
            print("Evaluating:", sim_name, "k =", k)
            weights_T = build_knn_weights(sim_matrix, k)
            metrics = evaluate_knn(
                weights_T, validation_truth, user_to_row, X,
                train_seen, movie_ids, popularity_tiebreak, catalogue
            )
            experiments.append({"Similarity": sim_name, "k": k, **metrics})
    knn_results = pd.DataFrame(experiments).sort_values(
        ["NDCG@10", "Precision@10"], ascending=False
    ).reset_index(drop=True)
    knn_comparison = knn_results.copy()
    knn_comparison["Model"] = (
        "kNN " + knn_comparison["Similarity"] + " k=" + knn_comparison["k"].astype(str)
    )
    popular_comparison = results.reset_index().rename(columns={"index": "Model"})
    comparison = pd.concat([popular_comparison, knn_comparison], ignore_index=True)
    comparison.to_csv(OUTPUT_DIR / "part1_validation_results.csv", index=False)
    knn_results.to_csv(OUTPUT_DIR / "part1_knn_validation_results.csv", index=False)
    print("\nAll validation results:\n", comparison[
        ["Model", "Precision@10", "Recall@10", "NDCG@10", "Coverage"]
    ].round(4).to_string(index=False))

    # 8. Select hyperparameters and refit on Train + Validation.
    best = knn_results.iloc[0]
    best_similarity = best["Similarity"]
    best_k = int(best["k"])
    train_val = pd.concat([train, validation], ignore_index=True)
    train_val["relevant"] = train_val["rating"] >= POSITIVE_THRESOLD
    final_seen = train_val.groupby("user_id")["movie_id"].apply(set).to_dict()
    test_truth = (test[test["relevant"]]
                  .groupby("user_id")["movie_id"].apply(set).to_dict())
    test_truth = {
        user: (items - final_seen.get(user, set())) & catalogue_set
        for user, items in test_truth.items()
    }
    test_truth = {user: items for user, items in test_truth.items() if items}
    assert test_truth, "No eligible test users"
    assert train_val["timestamp"].max() < test["timestamp"].min()
    popular_all_final = (train_val.groupby("movie_id").size()
                         .reindex(catalogue, fill_value=0)
                         .sort_values(ascending=False, kind="stable"))
    popular_positive_final = (
        train_val[train_val["relevant"]].groupby("movie_id").size()
        .reindex(catalogue, fill_value=0).sort_values(ascending=False, kind="stable")
    )
    final_users = pd.Index(train_val["user_id"].unique())
    final_user_to_row = {user: i for i, user in enumerate(final_users)}
    positive_train_val = (train_val.loc[train_val["relevant"], ["user_id", "movie_id"]]
                          .drop_duplicates())
    rows = final_users.get_indexer(positive_train_val["user_id"])
    cols = catalogue.get_indexer(positive_train_val["movie_id"])
    assert (rows >= 0).all() and (cols >= 0).all()
    X_final = csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(final_users), len(catalogue))
    )
    if best_similarity == "Cosine":
        sim_final = cosine_similarity(X_final.T).astype(np.float32)
    elif best_similarity == "Jaccard":
        counts = np.asarray(X_final.sum(axis=0)).ravel().astype(np.float32)
        intersection = (X_final.T @ X_final).toarray().astype(np.float32)
        union = counts[:, None] + counts[None, :] - intersection
        sim_final = np.divide(
            intersection, union, out=np.zeros_like(intersection), where=union > 0
        )
    else:
        raise ValueError("Unknown similarity: " + best_similarity)
    np.fill_diagonal(sim_final, 0)
    W_final_T = build_knn_weights(sim_final, best_k)
    final_popularity = popular_all_final.reindex(catalogue, fill_value=0).to_numpy()

    # 9. Final test recommender/evaluation.
    def recommend_knn_final(user, n=10):
        scores = np.zeros(len(catalogue), dtype=np.float32)
        if user in final_user_to_row:
            scores = (X_final[final_user_to_row[user]] @ W_final_T).toarray().ravel()
        seen_positions = catalogue.get_indexer(list(final_seen.get(user, set())))
        scores[seen_positions[seen_positions >= 0]] = -np.inf
        ranked = np.lexsort((catalogue.to_numpy(), -final_popularity, -scores))
        top = ranked[np.isfinite(scores[ranked])][:n]
        return list(catalogue[top])

    def evaluate_knn_final():
        precisions, recalls, ndcgs = [], [], []
        all_recommended = set()
        for user, relevant in test_truth.items():
            recs = recommend_knn_final(user)
            assert len(recs) == 10
            assert len(set(recs)) == 10
            assert not set(recs) & final_seen.get(user, set())
            p, r, ndcg = ranking_metrics_at_k(recs, relevant, k=10)
            precisions.append(p)
            recalls.append(r)
            ndcgs.append(ndcg)
            all_recommended.update(recs)
        return {
            "Users": len(test_truth),
            "Precision@10": float(np.mean(precisions)),
            "Recall@10": float(np.mean(recalls)),
            "NDCG@10": float(np.mean(ndcgs)),
            "Coverage": len(all_recommended) / len(catalogue)
        }

    final_results = pd.DataFrame([
        {"Model": "Most Popular - All Ratings",
         **evaluate_popular(popular_all_final, test_truth, final_seen, catalogue)},
        {"Model": "Most Popular - Positive Ratings",
         **evaluate_popular(popular_positive_final, test_truth, final_seen, catalogue)},
        {"Model": f"Item-kNN {best_similarity} k={best_k}",
         **evaluate_knn_final()}
    ])
    assert len(final_results) == 3
    assert not final_results.isna().any().any()
    for metric in ["Precision@10", "Recall@10", "NDCG@10", "Coverage"]:
        assert final_results[metric].between(0, 1).all()
    assert (final_results["Users"] == len(test_truth)).all()
    final_results.to_csv(OUTPUT_DIR / "part1_test_results_raw.csv", index=False)

    protocol_summary = {
        "seed": 42, "rating_threshold": int(POSITIVE_THRESOLD),
        "n_users": int(n_users), "n_movies_rated": int(n_items),
        "n_movies_catalogue": int(len(catalogue)),
        "n_interactions": int(n_interactions), "density": float(density),
        "train_rows": len(train), "validation_rows": len(validation), "test_rows": len(test),
        "train_cut": str(train_cut_date), "validation_cut": str(val_cut_date),
        "eligible_validation_users": len(validation_truth),
        "eligible_test_users": len(test_truth),
        "selected_knn_similarity": best_similarity, "selected_knn_k": best_k,
        "cold_start_test_users": sum(user not in final_seen for user in test_truth),
        "top10pct_user_interaction_share": float(user_share),
        "top10pct_movie_interaction_share": float(movie_share),
        "rating_percentages": {str(k): float(v) for k, v in rating_percentage.items()}
    }
    with (OUTPUT_DIR / "protocol_summary.json").open("w", encoding="utf-8") as f:
        json.dump(protocol_summary, f, indent=2)

    print("\nFINAL TEST RESULTS:\n", final_results.round(4).to_string(index=False))
    print("FINAL TEST CHECKS PASSED!")
    print("Outputs saved to:", OUTPUT_DIR.resolve())
    return final_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MovieLens 1M Part 1 experiments")
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--output-dir", default="part1_outputs")
    args = parser.parse_args()
    main(args.data_dir, args.output_dir)
