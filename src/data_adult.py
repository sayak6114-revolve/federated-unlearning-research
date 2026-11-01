import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class AdultDataset(Dataset):
    """PyTorch Dataset wrapper for Adult Census data."""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_and_preprocess_adult():
    """
    Fetches the Adult dataset from OpenML, cleans missing values,
    one-hot encodes categorical attributes, scales numeric attributes,
    and returns processed numpy arrays (X, y) with feature dimensions.
    """
    print("Loading Adult dataset...")
    data = fetch_openml("adult", version=2, as_frame=True, parser="auto")
    df = data.frame.dropna()

    # Target column: income (>50K vs <=50K)
    target_col = "class"
    y = (df[target_col].astype(str).str.contains(">50K")).astype(int).values

    # Feature matrix
    X_df = df.drop(columns=[target_col])

    # Separate numeric and categorical columns
    numeric_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_df.select_dtypes(exclude=[np.number]).columns.tolist()

    # One-hot encode categorical features
    X_categorical = pd.get_dummies(X_df[categorical_cols], drop_first=True)

    # Scale numeric features
    scaler = StandardScaler()
    X_numeric = pd.DataFrame(
        scaler.fit_transform(X_df[numeric_cols]),
        columns=numeric_cols,
        index=X_df.index
    )

    # Combine features
    X_processed = pd.concat([X_numeric, X_categorical], axis=1).astype(np.float32).values
    input_dim = X_processed.shape[1]

    print(f"Dataset preprocessed: {X_processed.shape[0]} samples, {input_dim} features, {y.sum()} positive classes.")
    return X_processed, y, input_dim

def partition_adult_federated(num_clients=10, batch_size=64, test_size=0.2, random_state=42):
    """
    Partitions the preprocessed Adult dataset across `num_clients` in a federated setup.
    Returns:
        client_loaders: dict {client_id: DataLoader}
        client_datasets: dict {client_id: (X_k, y_k)}
        test_loader: DataLoader for the held-out test set
        input_dim: int, number of features
    """
    X, y, input_dim = load_and_preprocess_adult()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    test_dataset = AdultDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Split train data among clients (IID / stratified)
    np.random.seed(random_state)
    indices = np.random.permutation(len(y_train))
    client_splits = np.array_split(indices, num_clients)

    client_loaders = {}
    client_datasets = {}

    for client_id, split_idx in enumerate(client_splits):
        X_k = X_train[split_idx]
        y_k = y_train[split_idx]
        client_datasets[client_id] = (X_k, y_k)
        ds = AdultDataset(X_k, y_k)
        client_loaders[client_id] = DataLoader(ds, batch_size=batch_size, shuffle=True)

    print(f"Partitioned {len(X_train)} training samples across {num_clients} clients (~{len(X_train)//num_clients} per client).")
    return client_loaders, client_datasets, test_loader, input_dim
