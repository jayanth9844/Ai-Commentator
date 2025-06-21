import os
import requests
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm

def download_and_load_gpt2(model_size, models_dir):
    # Ensure the requested model size is valid
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    # Configure directory structure and file requirements
    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    # Initialize directory and fetch model files
    os.makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path)

    # Load model configuration and parameters from downloaded files
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, "hparams.json")))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    return settings, params

def download_file(url, destination):
    try:
        # Initialize HTTP request with SSL verification disabled
        response = requests.get(url, stream=True, verify=False)

        # Extract file size from response headers
        file_size = int(response.headers.get("content-length", 0))

        # Skip download if file exists with matching size
        if os.path.exists(destination):
            file_size_local = os.path.getsize(destination)
            if file_size == file_size_local:
                print(f"File already exists and is up-to-date: {destination}")
                return

        # Configure download parameters
        block_size = 1024  # 1 KB chunks

        # Set up progress tracking with filename
        progress_bar_description = url.split("/")[-1]
        with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
            # Stream file contents and write to disk
            with open(destination, "wb") as file:
                for chunk in response.iter_content(block_size):
                    progress_bar.update(len(chunk))
                    file.write(chunk)

    except requests.exceptions.RequestException as e:
        print(f"Error downloading the file: {e}")
        print(f"Please check the URL: {url}")

def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    # Initialize parameter structure with empty layer blocks
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # Extract variables from checkpoint
    for name, _ in tf.train.list_variables(ckpt_path):
        # Load and reshape variable data
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # Parse variable name structure
        variable_name_parts = name.split("/")[1:]  # Remove 'model/' prefix

        # Determine target dictionary for variable storage
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        # Create nested dictionary structure
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        # Store variable array in appropriate location
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params