"""Placeholder script: checks that Docker and the data mount work."""
from pathlib import Path

import pandas as pd

DATA_DIR = Path('data')

print('Container is running!')
print(f'pandas version: {pd.__version__}')

train_path = DATA_DIR / 'train.csv'
if not train_path.exists():
    raise FileNotFoundError(
        f'{train_path} not found. Put train.csv in src/data/ and mount it '
        'with -v (see README).')

train = pd.read_csv(train_path)
print(f'Loaded train.csv: {train.shape[0]} rows, {train.shape[1]} columns')