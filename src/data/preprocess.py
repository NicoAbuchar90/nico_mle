import pandas as pd
from scipy.sparse import csr_matrix


def load_data(filepath: str) -> pd.DataFrame:
    """Load raw interaction data (user_id, item_id, rating)."""
    df = pd.read_csv(filepath)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and deduplicate interaction data."""
    df = df.dropna()
    df = df.drop_duplicates(subset=["user_id", "item_id"])
    return df


def build_user_item_matrix(df: pd.DataFrame):
    """Convert interactions dataframe to a sparse user-item matrix."""
    users = df["user_id"].astype("category")
    items = df["item_id"].astype("category")

    matrix = csr_matrix(
        (df["rating"], (users.cat.codes, items.cat.codes))
    )

    user_index = dict(enumerate(users.cat.categories))
    item_index = dict(enumerate(items.cat.categories))

    return matrix, user_index, item_index
