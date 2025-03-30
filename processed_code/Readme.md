# graphbot
A chatbot on you codebase and more.

## explanations
It generates explanations of all functions in a codebase. This by representing the code as a dependency graph (DAG), and giving context of all used internal functions. This is done in topological order (which is possible in a DAG)

The create_context.py generates the code DAG with all explanations.
```
python create_context.py --path [path]
```
where path is the folder where the codebase resides

## chat
The chat cli does the same as the above and includes a graph RAG system for chatting with your codebase. Here you can fillout a path to a codebase or use a previously create context.
```
python chat.py --path [path] --context_path [local context path]
```

## llm interface

Swap out the llm api (langchain compatible) in llm_interface.py