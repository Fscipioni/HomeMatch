import os
import json

def load_config_value(key, config_path="~/config.json"):
    """
    Load a value from a configuration file based on the provided key.
    
    Args:
        key (str): The key whose value needs to be retrieved.
        config_path (str): Path to the configuration file (default: "~/config.json").
    
    Returns:
        str: The value associated with the key in the configuration file.
    
    Raises:
        FileNotFoundError: If the configuration file is not found.
        KeyError: If the key is not found in the configuration file.
    """
    # Expand the user's home directory if needed
    config_path = os.path.expanduser(config_path)
    
    # Open and parse the configuration file
    try:
        with open(config_path, "r") as config_file:
            config = json.load(config_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    
    # Retrieve the value for the specified key
    if key not in config:
        raise KeyError(f"Key '{key}' not found in the configuration file.")
    
    return config[key]
