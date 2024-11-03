import os
import argparse
from dotenv import load_dotenv

load_dotenv()

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from functions.utilities import find_python_files, create_timestamped_folder
from functions.graph_context import get_context_graph
from functions.embedding_retr import write_embeddings, get_Qobj, retrieve_chunks
from llm_interface import llm


prompt_template = PromptTemplate(
    input_variables=["retrieved_chunks", "user_input"],
    template="You are a helpful assistant. Here are some relevant pieces of information: {retrieved_chunks}\n"
              "Now respond to the user: {user_input}"
)
chat_chain = LLMChain(llm=llm, prompt=prompt_template)

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Check for .py files in a specified directory.')
    parser.add_argument('--path', type=str, help='Optional: The path to the directory to check.')
    parser.add_argument('--context_path', type=str, help='If path is not given you can use a previously created context. Contexts get saved under "context_[ts]" folder')

    # Parse the arguments
    args = parser.parse_args()

    # Validate that at least one of the arguments is provided
    if not args.path and not args.context_path:
        print("Error: You must provide either --path or --context_path.")
        return
    if args.path and args.context_path:
        print("Error: You can only provide one of --path or --context_path, not both.")
        return

    # If context_path is provided, create it if it doesn't exist
    if args.context_path:
        context_path = args.context_path
    else:
        directory = args.path
        if not os.path.isdir(directory):
            print(f"Error: The path '{directory}' is not a valid directory.")
            return
        python_files = find_python_files(directory)
        if python_files:
            context_path = create_timestamped_folder('context')
            get_context_graph(context_path, directory)
            write_embeddings(context_path)
        else:
            print("No .py files found in the specified directory.")
            return
    QObj = get_Qobj(context_path)
    while True:
        # Get user input
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        
        retrieved_chunks = retrieve_chunks(QObj, user_input)
        # Generate response using LangChain and Mistral model
        response = chat_chain.run(
            user_input=user_input,
            retrieved_chunks="\n".join(retrieved_chunks)
        )
        
        # Display response
        print("Bot:", response)
    
    
    

if __name__ == "__main__":
    main()