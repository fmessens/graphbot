import yaml

from networkx.drawing.nx_pydot import from_pydot, write_dot
import pydot
import code2flow
import json
import os
import sqlite3
import ast
import pandas as pd
import networkx as nx
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain_mistralai.chat_models import ChatMistralAI

from llm_interface import llm

with open('settings.yml', 'r') as f:
    settings = yaml.safe_load(f)


def get_context_graph(output_folder, codeloc):
    # convert codebase to dag
    os.makedirs(os.path.join(output_folder, 'tempdata'))
    tempdot = os.path.join(output_folder, 'tempdata/callers.dot')
    code2flow.code2flow(codeloc, 
                        tempdot)
    with open(tempdot, 'r') as f:
        source_G = f.read()
    source_G = source_G.replace('name=','naming=')
    G2 = from_pydot(pydot.graph_from_dot_data(source_G)[0]) # type: ignore
    metadata = {x: get_file_class_func(x, G2) for x in list(G2.nodes)}
    filenames = {k: v[0] for k,v in metadata.items()}
    classnames = {k: v[1] for k,v in metadata.items()}
    function_names = {k: v[2] for k,v in metadata.items()}
    functionstrs = {k: extract_function_with_used_imports(v[0], v[2], v[1])
                    for k,v in metadata.items()}

    nx.set_node_attributes(G2, filenames, 'file_name')
    nx.set_node_attributes(G2, classnames, 'class_name')
    nx.set_node_attributes(G2, function_names, 'function_name')
    nx.set_node_attributes(G2, functionstrs, 'functionstr')
    dag = to_dag(G2)
    # get explain
    reverse_topological_order = list(reversed(list(nx.topological_sort(dag))))

    for node in reverse_topological_order:
        dag.nodes[node]['explain'] = explainfunc(dag.nodes[node]['functionstr'], node, dag)

    for node in reverse_topological_order:
        for k, v in dict(dag.nodes[node]).items():
            if isinstance(v, tuple) or isinstance(v, set):
                dag.nodes[node][k] = ','.join(list(v))
    # write
    data = nx.node_link_data(dag)
    with open(os.path.join(output_folder, settings['graph_loc']), "w") as f:
        json.dump(data, f, indent=2)
    


def to_dag(G2):
    dag = nx.condensation(G2)

    for supernode in dag.nodes:
        # Get the original nodes in this SCC
        scc_nodes = dag.nodes[supernode]['members']

        dag.nodes[supernode]['file_name'] = ','.join(G2.nodes[node]['file_name']
                                                    for node in scc_nodes),
        dag.nodes[supernode]['class_name'] = ','.join(G2.nodes[node]['class_name']
                                                    for node in scc_nodes),
        dag.nodes[supernode]['function_name'] = ','.join(G2.nodes[node]['function_name'] 
                                                        for node in scc_nodes),
        dag.nodes[supernode]['functionstr'] = ' | '.join(G2.nodes[node]['functionstr'] 
                                                        for node in scc_nodes)

    if not nx.is_directed_acyclic_graph(dag):
        raise ValueError("Graph is not a DAG")
    return dag


def extract_function_with_used_imports(file_name, function_name, class_name=None):
    # Read the content of the file
    with open(file_name, "r") as file:
        file_content = file.read()

    # Parse the file content to AST
    tree = ast.parse(file_content)

    # Initialize variables
    function_code = None
    imports = {}

    # First pass: capture import statements and their names
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports[alias.name] = alias.name
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imports[alias.name] = f"{node.module}.{alias.name}"

    # Second pass: find the desired function and its used imports
    for node in ast.walk(tree):
        if class_name and isinstance(node, ast.ClassDef) and node.name == class_name:
            # Capture the class inheritance (bases)
            class_bases = [ast.get_source_segment(file_content, base) for base in node.bases]
            class_inheritance = f"{node.name}({', '.join(class_bases)})" if class_bases else node.name

            for class_node in node.body:
                if isinstance(class_node, ast.FunctionDef) and class_node.name == function_name:
                    function_code = ast.get_source_segment(file_content, class_node)
                    used_imports = set()

                    # Analyze the function body for used imports
                    for stmt in class_node.body:
                        used_imports.update(get_used_names(stmt))

                    # Filter the imports to include only those used in the function
                    used_imports_code = [f"from {imports[name]}" if name in imports else f"import {name}"
                                         for name in used_imports if name in imports]

                    # Return class inheritance, used imports, and the function code
                    return f"class {class_inheritance}:\n\n" + "\n".join(used_imports_code) + "\n\n" + function_code if function_code else None

        elif not class_name and isinstance(node, ast.FunctionDef) and node.name == function_name:
            function_code = ast.get_source_segment(file_content, node)
            used_imports = set()

            # Analyze the function body for used imports
            for stmt in node.body:
                used_imports.update(get_used_names(stmt))

            # Filter the imports to include only those used in the function
            used_imports_code = [f"from {imports[name]}" if name in imports else f"import {name}"
                                 for name in used_imports if name in imports]

            return "\n".join(used_imports_code) + "\n\n" + function_code if function_code else None

    return ''  # Return emtpy if the function is not found


def get_used_names(node):
    """Recursively collect names used in the AST node."""
    used_names = set()
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.Name):
            used_names.add(child.id)
        elif isinstance(child, (ast.Call, ast.Attribute)):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                used_names.add(child.func.id)
            elif isinstance(child, ast.Attribute):
                used_names.add(child.attr)
        used_names.update(get_used_names(child))
    return used_names


def get_file_class_func(node, G2):
    basis_node = G2.nodes[node]
    filestr = basis_node['naming'].split('::')[0]
    syntaxstr = basis_node['naming'].split('::')[1]
    file_name = filestr.replace('"','').replace('\\','/')+'.py'
    if '.' in syntaxstr:
        class_name = syntaxstr.split('.')[0].replace('"','')
        function_name = syntaxstr.split('.')[1].replace('"','')
    else:
        class_name = ''
        function_name = syntaxstr.replace('"','')
    return file_name, class_name, function_name

def get_fun(x):
    return extract_function_with_used_imports(**x.to_dict())

def explainfunc(function_code, node, graph):
    """
    Generates an explanation for the given function_code. If the node has children,
    it includes explanations of the children in the prompt.
    """
    # Step 1: Connect to SQLite database (create it if it doesn't exist)
    conn = sqlite3.connect('explanations_cache.db')
    cursor = conn.cursor()

    # Step 2: Create the cache table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS explanations_cache (
        function_code TEXT PRIMARY KEY,
        explanation TEXT
    )
    ''')

    # Step 3: Check if the explanation for the function code is already cached
    cursor.execute("SELECT explanation FROM explanations_cache WHERE function_code = ?", (function_code,))
    cached_explanation = cursor.fetchone()

    if cached_explanation:
        # Step 4: Return the cached explanation if available
        print('from cache')
        explanation = cached_explanation[0]
    else:
        # Step 5: Include children's explanations if the node has children
        children = list(graph.successors(node))
        children_explanations = []
        if children:
            for c in children:
                # Get explanation for each child node from the dataframe
                child = graph.nodes[c]
                child_explanation = child['explain']
                children_explanations.append(f"Child node {child} explanation: {child_explanation}")

        # Join children's explanations into a single string
        children_explanation_text = "\n".join(children_explanations)

        # Step 6: Prepare the prompt including the children's explanations
        prompt_template = """
        You are an expert Python programmer. \
Your task is to explain the following function in simple terms with a docstring. \
Provide a step-by-step explanation. Please only provide the docstring without any other words.

        Function code:
        {function_code}

        This is the context of used functions

        {children_explanation_text}
        """
        # Initialize the prompt with function code and children explanations
        prompt = PromptTemplate(
            input_variables=["function_code",
                             "children_explanation_text"],
            template=prompt_template
        )

        # Step 8: Create the LLMChain
        llm_chain = LLMChain(prompt=prompt, llm=llm)

        # Step 9: Prepare inputs for the LLMChain
        inputs = {
            "function_code": function_code,
            "children_explanation_text": children_explanation_text
        }

        # Step 10: Run the LLM chain to generate the explanation
        explanation = llm_chain.run(inputs)

        # Step 9: Store the explanation in the cache
        cursor.execute("INSERT INTO explanations_cache (function_code, explanation) VALUES (?, ?)", (function_code, explanation))

    # Step 10: Commit the transaction and close the connection
    conn.commit()
    conn.close()

    # Step 11: Return the explanation
    return explanation


