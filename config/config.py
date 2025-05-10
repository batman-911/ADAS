import yaml
import torch
import os

class Config:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, "r") as f:
            self._config = yaml.safe_load(f)

    def get(self, *keys, default=None):
        cfg = self._config
        for key in keys:
            if key in cfg:
                cfg = cfg[key]
            else:
                return default
        return cfg

    def __getitem__(self, key):
        return self._config[key]

    def __repr__(self):
        return repr(self._config)

    def get_device(self):
        """Return the appropriate device based on config or environment."""
        device = self.get("device", default="auto")

        if device == "auto":
            # Auto-selection logic based on availability
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")  # for Apple M1/M2 chips
            else:
                return torch.device("cpu")
        elif device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif device == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")