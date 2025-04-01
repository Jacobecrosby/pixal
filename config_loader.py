import yaml
import os
from types import SimpleNamespace

class ConfigNamespace(SimpleNamespace):
    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def to_dict(self):
        def convert(obj):
            if isinstance(obj, ConfigNamespace):
                return {k: convert(v) for k, v in obj.__dict__.items()}
            return obj
        return convert(self)

def _dict_to_namespace(d):
    if isinstance(d, dict):
        return ConfigNamespace(**{k: _dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [_dict_to_namespace(i) for i in d]
    return d

def load_config(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        config_data = yaml.safe_load(f)

    config = _dict_to_namespace(config_data)

    print("\n📄 Loaded config:", path)
    print("-------------------------")
    for k in config.__dict__.keys():
        print(f"• {k}")

    return config
