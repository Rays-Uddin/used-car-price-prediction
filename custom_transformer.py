import numpy as np, pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class OrderedTargetEncoder(BaseEstimator, TransformerMixin):
    """
    A custom transformer for ordered target encoding of categorical features.

    This encoder replaces categorical values with their smoothed target means,
    and then maps these smoothed means to an ordered integer sequence.
    It handles unknown categories by imputing with the global mean or raising an error.

    Attributes:
        columns (list): List of column names to encode. If None, all object-type columns are encoded.
        smoothing (float): Smoothing factor for target mean calculation. Higher values reduce the impact of small categories.
        handle_unknown (str): Strategy to handle unknown categories during transform.
                              'mean': Imputes unknown categories with the mean order of known categories.
                              'error': Raises a ValueError if unknown categories are encountered.
        encodings_ (dict): Stores the smoothed target means for each category in each column.
        global_mean_ (float): The overall mean of the target variable.
        mapping_order_ (dict): Stores the ordered integer mapping for each category in each column.
    """

    def __init__(self, columns=None, smoothing=1.0, handle_unknown="mean"):
        """
        Initializes the OrderedTargetEncoder.

        Args:
            columns (list, optional): List of column names to encode. If None, all object-type columns are encoded.
                                      Defaults to None.
            smoothing (float, optional): Smoothing factor for target mean calculation. Defaults to 1.0.
            handle_unknown (str, optional): Strategy to handle unknown categories ('mean' or 'error').
                                            Defaults to "mean".
        """
        self.columns = columns
        self.smoothing = smoothing
        self.handle_unknown = handle_unknown
        self.encodings_ = {}
        self.global_mean_ = None
        self.mapping_order_ = {}

    def fit(self, X, y):
        """
        Fits the encoder to the training data.

        Calculates the global mean of the target and smoothed target means for each category
        in the specified columns. It then creates an ordered integer mapping based on these smoothed means.

        Args:
            X (pd.DataFrame or np.ndarray): The input features (training data).
            y (pd.Series or np.ndarray): The target variable.

        Returns:
            self (OrderedTargetEncoder): The fitted encoder instance.
        """

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.columns or [f'col_{i}' for i in range(X.shape[1])])

        y = pd.Series(y, name="target")


        if self.columns is None:
            self.columns = X.columns

        self.global_mean_ = y.mean()


        for col in self.columns:
            df_temp = pd.concat([X[col], y], axis=1)

            stats = df_temp.groupby(col)["target"].agg(["mean", "count"])

            smoothing_factor = stats["count"] / (stats["count"] + self.smoothing)
            stats["smoothed_mean"] = smoothing_factor * stats["mean"] + (1-smoothing_factor) * self.global_mean_

            stats = stats.sort_values(by = "smoothed_mean")

            ordered_map = {category: i for i, category in enumerate(stats.index)}

            self.encodings_[col] = stats["smoothed_mean"].to_dict()

            self.mapping_order_[col] = ordered_map

        return self

    def transform(self, X):
        """
        Transforms the input features using the fitted encoder.

        Replaces categorical values with their corresponding ordered integer from the mapping.
        Handles unknown categories based on the `handle_unknown` strategy.

        Args:
            X (pd.DataFrame or np.ndarray): The input features to transform.

        Returns:
            pd.DataFrame: The transformed features with ordered integer encoded columns.

        Raises:
            ValueError: If `handle_unknown` is 'error' and unknown categories are encountered.
        """

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.columns)

        X_encoded = X.copy()

        for col in self.columns:

            X_encoded[col] = X_encoded[col].map(self.mapping_order_[col])

            if X_encoded[col].isnull().any() and self.handle_unknown == "mean":

                mean_order = np.mean(list(self.mapping_order_[col].values()))
                X_encoded[col] = X_encoded[col].fillna(mean_order)

            elif X_encoded[col].isnull().any() and self.handle_unknown == "error":
                raise ValueError(f"Unknown categories encountered in column {col} and handle_unknown is set to 'error'.")

        return X_encoded