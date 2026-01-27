"""
Common utilities and helper functions.
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path


def ensure_directory(path):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def get_data_dir():
    """Get the standard data directory path."""
    return r"C:\Users\NHaq\OneDrive - University College Cork\CTG\data\ctu-chb-intrapartum-cardiotocography-database-1.0.0"


def get_output_dir():
    """Get the standard output directory path."""
    return r"C:\Users\NHaq\OneDrive - University College Cork\CTG\outputs"


def get_models_dir():
    """Get the models directory path."""
    models_dir = os.path.join(get_output_dir(), 'models')
    ensure_directory(models_dir)
    return models_dir


def list_hea_files(data_dir=None):
    """Get all .hea files from data directory."""
    if data_dir is None:
        data_dir = get_data_dir()
    hea_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.hea')])
    return hea_files


def save_csv(df, filename, output_dir=None):
    """Save dataframe to CSV in output directory."""
    if output_dir is None:
        output_dir = get_output_dir()
    ensure_directory(output_dir)
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)
    return filepath


def load_csv(filename, output_dir=None):
    """Load CSV from output directory."""
    if output_dir is None:
        output_dir = get_output_dir()
    filepath = os.path.join(output_dir, filename)
    return pd.read_csv(filepath)


def save_model(model, model_name, models_dir=None):
    """Save trained model to disk."""
    if models_dir is None:
        models_dir = get_models_dir()
    import pickle
    filepath = os.path.join(models_dir, f"{model_name}.pkl")
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    return filepath


def load_model(model_name, models_dir=None):
    """Load trained model from disk."""
    if models_dir is None:
        models_dir = get_models_dir()
    import pickle
    filepath = os.path.join(models_dir, f"{model_name}.pkl")
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model
