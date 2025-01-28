import os
import sys
from datetime import datetime

def create_timestamped_folder(prefix, base_path='contexts'):
    """
    Creates a folder with a timestamp suffix.

    :param base_path: The path where the folder will be created.
    :return: The full path of the created folder.
    """
    # Get the current timestamp formatted as YYYYMMDD_HHMMSS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create the folder name
    folder_name = f"{prefix}_{timestamp}"
    
    # Combine the base path with the folder name
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    folder_path = os.path.join(base_path, folder_name)
    
    # Create the folder
    os.makedirs(folder_path)
    
    return folder_path


def find_python_files(directory):
    """
    Check if the directory contains any .py files.

    :param directory: The directory to check for .py files.
    :return: A list of .py files found in the directory.
    """
    # List to store found .py files
    python_files = []

    # Walk through the directory
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    return python_files