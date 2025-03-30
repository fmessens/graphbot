import os
import sys
from datetime import datetime


def create_timestamped_folder(prefix, base_path="contexts"):
    """create_timestamped_folder

    Creates a new folder with a timestamp in its name at the specified base path.
    The folder name will be a combination of the provided prefix and the current
    timestamp. If the base path does not exist, it will be created.

    Args:
    prefix (str): A string to be used as the prefix of the new folder name.
    base_path (str, optional): The path to the base directory where the new
    folder will be created. Defaults to "contexts".

    Returns:
    str: The full path to the newly created folder."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{prefix}_{timestamp}"
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    folder_path = os.path.join(base_path, folder_name)
    os.makedirs(folder_path)
    return folder_path


def find_python_files(directory):
    """find_python_files

    This function takes a directory as an argument and returns a list of all Python files
    in the directory and its subdirectories.

    Args:
    directory (str or pathlike): The directory to search for Python files.

    Returns:
    list of strings: A list of paths to all Python files in the specified directory
    and its subdirectories."""
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    return python_files
