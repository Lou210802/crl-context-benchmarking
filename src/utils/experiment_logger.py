import os
from typing import Dict, Any
import pandas as pd


class ExperimentLogger:
    def __init__(self, output_dir: str, experiment_name: str):
        self.output_dir = os.path.abspath(output_dir)
        self.experiment_name = experiment_name
        self.data = []
        os.makedirs(self.output_dir, exist_ok=True)

    def log(self, data_point: Dict[str, Any]):
        self.data.append(data_point)

    def save(self) -> str:
        filepath = os.path.abspath(os.path.join(self.output_dir, f"{self.experiment_name}_results.csv"))
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df = pd.DataFrame(self.data)
        df.attrs['mode'] = self.experiment_name
        df.to_csv(filepath, index=False)
        print(f"Results saved to {filepath}")
        return filepath

    @staticmethod
    def load(filepath: str) -> pd.DataFrame:
        return pd.read_csv(filepath)

