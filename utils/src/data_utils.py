import os
import numpy as np
from PIL import Image
import json

def read_file(filepath):
    """
    Returns string content from file

    Args:
      path : str
        path to file where data will be stored
    """
    data = ""
    with open(filepath, "r") as f:
        data = f.read()

    return data

def parse_json(json_string, ignore_keys=[]):
    data = json.loads(json_string)
    for ignore_key in ignore_keys:
        data.pop(ignore_key, None)
  
    return data


def read_paths(filepath, prefix=None):
    """
    Reads a list of paths from a file

    Args:
      path : str
        path to file where data will be stored
    """
    path_list = []
    with open(filepath) as f:
        while True:
            path = f.readline().rstrip("\n")
            # If there was nothing to read
            if path == "":
                break

            if prefix:
                path = prefix + path

            path_list.append(path)

    return path_list


def load_depth_with_validity_map(path):
    """
    Loads a depth map and validity map from a 16-bit PNG file

    Args:
      path : str
        path to 16-bit PNG file

    Returns:
      numpy : depth map
      numpy : binary validity map for available depth measurement locations
    """
    # Loads depth map from 16-bit PNG file
    z = np.array(Image.open(path), dtype=np.float32)
    # Assert 16-bit (not 8-bit) depth map
    z = z / 256.0
    z[z <= 0] = 0.0
    v = z.astype(np.float32)
    v[z > 0] = 1.0
    return z, v


def load_depth(path):
    """
    Loads a depth map from a 16-bit PNG file

    Args:
      path : str
        path to 16-bit PNG file

    Returns:
      numpy : depth map
    """
    # Loads depth map from 16-bit PNG file
    z = np.array(Image.open(path), dtype=np.float32)
    # Assert 16-bit (not 8-bit) depth map
    z = z / 256.0
    z[z <= 0] = 0.0
    return z


def save_depth(z, path):
    """
    Saves a depth map to a 16-bit PNG file

    Args:
      z : numpy
        depth map
      path : str
        path to store depth map
    """
    z = np.uint32(z * 256.0)
    z = Image.fromarray(z, mode="I")
    z.save(path)


def load_validity_map(path):
    """
    Loads a validity map from a 16-bit PNG file

    Args:
      path : str
        path to 16-bit PNG file

    Returns:
      numpy : binary validity map for available depth measurement locations
    """
    # Loads depth map from 16-bit PNG file
    v = np.array(Image.open(path), dtype=np.float32)
    assert np.all(np.unique(v) == [0, 256])
    v[v > 0] = 1
    return v


def save_validity_map(v, path):
    """
    Saves a validity map to a 16-bit PNG file

    Args:
      v : numpy
        validity map
      path : str
        path to store validity map
    """
    v[v <= 0] = 0.0
    v[v > 0] = 1.0
    v = np.uint32(v * 256.0)
    v = Image.fromarray(v, mode="I")
    v.save(path)


def write_paths(filepath, paths):
    """
    Stores line delimited paths into file

    Arg(s):
        filepath : str
            path to file to save paths
        paths : list[str]
            paths to write into file
    """

    with open(filepath, "w") as o:
        for idx in range(len(paths)):
            o.write(paths[idx] + "\n")

def get_categories_from_vild_json_file(filepath):
    json_data = read_file(filepath)
    categories_dict = parse_json(json_data, ignore_keys=['annotations', 'info', 'licenses', 'images'])
    categories = [category['name'] for category in categories_dict['categories']]
    return categories