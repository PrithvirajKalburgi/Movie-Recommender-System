"""Small independent checks; run: python -m tests.test_metrics"""
import numpy as np
import pandas as pd
from part1 import ranking_metrics_at_k, recommend_popular, evaluate_popular


def test_metrics():
    p, r, n = ranking_metrics_at_k(
        [10,20,30,40,50,60,70,80,90,100], {10,30,110,120}
    )
    assert np.isclose(p, 0.2)
    assert np.isclose(r, 0.5)
    assert np.isclose(n, 0.5856, atol=0.0001)


def test_seen_items():
    history = pd.DataFrame({"user_id": [1, 1, 2], "movie_id": [10, 30, 20]})
    train_seen = history.groupby("user_id")["movie_id"].apply(set).to_dict()
    assert train_seen[1] == {10, 30}
    scores = pd.Series([3,2,1,0,0,0,0,0,0,0,0,0], index=pd.Index(range(10,22),name="movie_id"))
    recs = recommend_popular(1, scores, history, n=10)
    assert 10 not in set(recs.index)
    assert len(recs) == 10
    truth = {1:{11}}
    out = evaluate_popular(scores, truth, train_seen, pd.Index(range(10,22)), k=10)
    assert out["Users"] == 1
    assert 0 <= out["Precision@10"] <= 1


if __name__ == "__main__":
    test_metrics()
    test_seen_items()
    print("UNIT TESTS PASSED")
