import numpy as np
from sklearn.decomposition import TruncatedSVD


class CollaborativeFilteringModel:
    """Matrix factorization recommender using SVD."""

    def __init__(self, n_components: int = 20, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.model = TruncatedSVD(
            n_components=n_components, random_state=random_state
        )
        self.user_factors = None
        self.item_factors = None

    def fit(self, user_item_matrix):
        """Fit the SVD model on the user-item matrix."""
        self.user_factors = self.model.fit_transform(user_item_matrix)
        self.item_factors = self.model.components_.T

    def recommend(self, user_idx: int, top_n: int = 10) -> np.ndarray:
        """Return top N item indices recommended for a given user."""
        scores = self.user_factors[user_idx] @ self.item_factors.T
        top_items = np.argsort(scores)[::-1][:top_n]
        return top_items
