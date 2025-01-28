import os
import argparse
from dotenv import load_dotenv

load_dotenv()

from functions.utilities import find_python_files, create_timestamped_folder
from functions.graph_context import get_context_graph
from functions.embedding_retr import write_embeddings, get_Qobj, retrieve_chunks

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Check for .py files in a specified directory.')
    parser.add_argument('--path', type=str, help='The path to the directory to check.')
    parser.add_argument('--prefix', type=str, help='The prefix of the timestamped folder', default='context')
    # Parse the arguments
    args = parser.parse_args()

    # Validate that at least one of the arguments is provided
    if not args.path:
        print("Error: You must provide --path")
        return
    directory = args.path
    if not os.path.isdir(directory):
        print(f"Error: The path '{directory}' is not a valid directory.")
        return
    python_files = find_python_files(directory)
    if python_files:
        context_path = create_timestamped_folder(args.prefix)
        get_context_graph(context_path, directory)
        write_embeddings(context_path)
    else:
        print("No .py files found in the specified directory.")
        return
    
    
    

if __name__ == "__main__":
    main()