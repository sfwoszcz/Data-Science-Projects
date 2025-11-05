from dataclasses import dataclass
import pandas as pd

@dataclass
class Columns:
    timestamp: str = 'timestamp'
    signal: str = 'vibration'

def read_csv(path: str, cols: Columns = Columns()) -> pd.DataFrame:
    df = pd.read_csv(path)
    req = [cols.timestamp, cols.signal]
    for c in req:
        if c not in df.columns:
            raise ValueError(f'Missing column: {c}')
    return df[req].copy()
