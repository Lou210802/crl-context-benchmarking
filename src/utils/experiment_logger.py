import os
import json
from typing import Dict, Any
import pandas as pd


class ExperimentLogger:
    """
    Handles experiment logging, creating dedicated run directories, saving metrics CSV,
    and recording hyperparameters to config.yaml.
    """

    def __init__(self, output_dir: str, experiment_name: str):
        self.output_dir = os.path.abspath(output_dir)
        self.experiment_name = experiment_name
        self.data = []
        os.makedirs(self.output_dir, exist_ok=True)

    def log(self, data_point: Dict[str, Any]) -> None:
        """
        Logs a single training metric step dictionary.
        """
        self.data.append(data_point)

    def save_config(self, config_dict: Dict[str, Any]) -> str:
        """
        Saves experiment hyperparameters to config.yaml in the run directory.
        """
        def sanitize_native_types(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: sanitize_native_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [sanitize_native_types(v) for v in obj]
            elif hasattr(obj, "item"):
                return obj.item()
            return obj

        sanitized_config = sanitize_native_types(config_dict)
        filepath = os.path.join(self.output_dir, "config.yaml")
        try:
            import yaml
            with open(filepath, "w") as f:
                yaml.dump(sanitized_config, f, default_flow_style=False, sort_keys=False)
        except ImportError:
            with open(filepath, "w") as f:
                json.dump(sanitized_config, f, indent=4)
        print(f"Hyperparameters saved to {filepath}")
        return filepath

    def save(self) -> str:
        """
        Saves logged metrics to CSV file in the run directory.
        """
        filepath = os.path.join(self.output_dir, f"{self.experiment_name}_results.csv")
        df = pd.DataFrame(self.data)
        df.attrs["mode"] = self.experiment_name
        df.to_csv(filepath, index=False)
        print(f"Results saved to {filepath}")
        return filepath

    @staticmethod
    def load(filepath: str) -> pd.DataFrame:
        return pd.read_csv(filepath)

