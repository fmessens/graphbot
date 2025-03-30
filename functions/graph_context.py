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

with open("settings.yml", "r") as f:
    settings = yaml.safe_load(f)


def get_context_graph(output_folder, codeloc):
    """get_context_graph

    This function generates a context graph for a given code location (codeloc) and saves it as a JSON file in the specified output folder. The context graph is a directed graph where nodes represent functions and edges represent function calls. Each node in the graph contains attributes such as file name, class name, function name, and the function's source code. The function also caches explanations of functions using a SQLite database to avoid unnecessary computation.

    Args:
        output_folder (str): The path to the output folder where the context graph JSON file will be saved.
        codeloc (str): The code location for which the context graph is generated.

    Returns:
        None"""
    os.makedirs(os.path.join(output_folder, "tempdata"))
    tempdot = os.path.join(output_folder, "tempdata/callers.dot")
    code2flow.code2flow(codeloc, tempdot)
    with open(tempdot, "r") as f:
        source_G = f.read()
    source_G = source_G.replace("name=", "naming=")
    G2 = from_pydot(pydot.graph_from_dot_data(source_G)[0])
    metadata = {x: get_file_class_func(x, G2) for x in list(G2.nodes)}
    filenames = {k: v[0] for (k, v) in metadata.items()}
    classnames = {k: v[1] for (k, v) in metadata.items()}
    function_names = {k: v[2] for (k, v) in metadata.items()}
    functionstrs = {
        k: extract_function_with_used_imports(v[0], v[2], v[1])
        for (k, v) in metadata.items()
    }
    nx.set_node_attributes(G2, filenames, "file_name")
    nx.set_node_attributes(G2, classnames, "class_name")
    nx.set_node_attributes(G2, function_names, "function_name")
    nx.set_node_attributes(G2, functionstrs, "functionstr")
    dag = to_dag(G2)
    reverse_topological_order = list(reversed(list(nx.topological_sort(dag))))
    for node in reverse_topological_order:
        dag.nodes[node]["explain"] = explainfunc(
            dag.nodes[node]["functionstr"], node, dag
        )
    for node in reverse_topological_order:
        for k, v in dict(dag.nodes[node]).items():
            if isinstance(v, tuple) or isinstance(v, set):
                dag.nodes[node][k] = ",".join(list(v))
    data = nx.node_link_data(dag)
    with open(os.path.join(output_folder, settings["graph_loc"]), "w") as f:
        json.dump(data, f, indent=2)


def to_dag(G2):
    """Converts a graph with possible circular dependencies into a Directed Acyclic Graph (DAG) by condensing strongly connected components (SCCs) into single nodes.

    Args:
        G2 (networkx.DiGraph): A directed graph representing dependencies between nodes.

    Returns:
        networkx.DiGraph: A directed acyclic graph (DAG) with the same nodes as G2, but with condensed SCCs. Each node in the returned DAG contains attributes 'file_name', 'class_name', 'function_name', and 'functionstr' that are the respective aggregated values from the nodes in the SCC it represents.
    """
    dag = nx.condensation(G2)
    for supernode in dag.nodes:
        scc_nodes = dag.nodes[supernode]["members"]
        dag.nodes[supernode]["file_name"] = (
            ",".join((G2.nodes[node]["file_name"] for node in scc_nodes)),
        )
        dag.nodes[supernode]["class_name"] = (
            ",".join((G2.nodes[node]["class_name"] for node in scc_nodes)),
        )
        dag.nodes[supernode]["function_name"] = (
            ",".join((G2.nodes[node]["function_name"] for node in scc_nodes)),
        )
        dag.nodes[supernode]["functionstr"] = " | ".join(
            (G2.nodes[node]["functionstr"] for node in scc_nodes)
        )
    if not nx.is_directed_acyclic_graph(dag):
        raise ValueError("Graph is not a DAG")
    return dag


def extract_function_with_used_imports(file_name, function_name, class_name=None):
    with open(file_name, "r") as file:
        file_content = file.read()
    tree = ast.parse(file_content)
    function_code = None
    imports = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports[alias.name] = alias.name
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imports[alias.name] = f"{node.module}.{alias.name}"
    for node in ast.walk(tree):
        if class_name and isinstance(node, ast.ClassDef) and (node.name == class_name):
            class_bases = [
                ast.get_source_segment(file_content, base) for base in node.bases
            ]
            class_inheritance = (
                f"{node.name}({', '.join(class_bases)})" if class_bases else node.name
            )
            for class_node in node.body:
                if (
                    isinstance(class_node, ast.FunctionDef)
                    and class_node.name == function_name
                ):
                    function_code = ast.get_source_segment(file_content, class_node)
                    used_imports = set()
                    for stmt in class_node.body:
                        used_imports.update(get_used_names(stmt))
                    used_imports_code = [
                        f"from {imports[name]}" if name in imports else f"import {name}"
                        for name in used_imports
                        if name in imports
                    ]
                    return (
                        f"class {class_inheritance}:\n\n"
                        + "\n".join(used_imports_code)
                        + "\n\n"
                        + function_code
                        if function_code
                        else None
                    )
        elif (
            not class_name
            and isinstance(node, ast.FunctionDef)
            and (node.name == function_name)
        ):
            function_code = ast.get_source_segment(file_content, node)
            used_imports = set()
            for stmt in node.body:
                used_imports.update(get_used_names(stmt))
            used_imports_code = [
                f"from {imports[name]}" if name in imports else f"import {name}"
                for name in used_imports
                if name in imports
            ]
            return (
                "\n".join(used_imports_code) + "\n\n" + function_code
                if function_code
                else None
            )
    return ""


def get_used_names(node):
    """get_used_names

    This function takes an abstract syntax tree (AST) node as an argument and returns a set of unique names that are used in the given node and its descendants. It uses depth-first search to traverse the tree and identify the names.

    Args:
    node (ast.AST): The root node of the abstract syntax tree to search for used names.

    Returns:
    set(str): A set of unique names that are used in the given node and its descendants.
    """
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
    """get_file_class_func

    This function extracts the file name, class name, and function name from a given node in a graph database (G2) based on the node's 'naming' attribute. The 'naming' attribute follows a specific format of "file::class.function" or "file::function".

    Args:
    node (dict): A dictionary containing node information from the G2 graph database with 'naming' as one of the keys.
    G2 (networkx.Graph): A graph database containing nodes and edges with node information stored as attributes.

    Returns:
    tuple: A tuple containing the file name, class name, and function name extracted from the 'naming' attribute of the given node.
    """
    basis_node = G2.nodes[node]
    filestr = basis_node["naming"].split("::")[0]
    syntaxstr = basis_node["naming"].split("::")[1]
    file_name = filestr.replace('"', "").replace("\\", "/") + ".py"
    if "." in syntaxstr:
        class_name = syntaxstr.split(".")[0].replace('"', "")
        function_name = syntaxstr.split(".")[1].replace('"', "")
    else:
        class_name = ""
        function_name = syntaxstr.replace('"', "")
    return (file_name, class_name, function_name)


def get_fun(x):
    """get_fun

    This function is used to extract a function or class method with its used imports from a given file.

    Args:
        x [dict]: A dictionary containing the file name, function name, and optionally the class name.

    Returns:
        str: A string containing the function or class method with its used imports."""
    return extract_function_with_used_imports(**x.to_dict())


def explainfunc(function_code, node, graph):
    """explainfunc

    This function explains a given Python function using a language model. It takes as input the function code, a node, and a graph. The node represents the function to be explained, and the graph is used to retrieve explanations of any child nodes (i.e. functions called within the given function). The function caches the explanations in a SQLite database to avoid unnecessary computation.

    Args:
        function_code (str): The code of the function to be explained.
        node (any): The node representing the function in the graph.
        graph (networkx.DiGraph): A directed graph containing nodes representing functions and edges representing function calls.

    Returns:
        str: A simple explanation of the given function, generated by a language model.
    """
    cache_path = settings["llm_cache"]
    conn = sqlite3.connect(cache_path)
    cursor = conn.cursor()
    cursor.execute(
        "\n    CREATE TABLE IF NOT EXISTS explanations_cache (\n        function_code TEXT PRIMARY KEY,\n        explanation TEXT\n    )\n    "
    )
    cursor.execute(
        "SELECT explanation FROM explanations_cache WHERE function_code = ?",
        (function_code,),
    )
    cached_explanation = cursor.fetchone()
    if cached_explanation:
        print("from cache")
        explanation = cached_explanation[0]
    else:
        children = list(graph.successors(node))
        children_explanations = []
        if children:
            for c in children:
                child = graph.nodes[c]
                child_explanation = child["explain"]
                children_explanations.append(
                    f"Child node {child} explanation: {child_explanation}"
                )
        children_explanation_text = "\n".join(children_explanations)
        prompt_template = "\n        You are an expert Python programmer. Your task is to explain the following function in simple terms with a docstring. Provide a step-by-step explanation. Please only provide the docstring without any other words.\n\n        Function code:\n        {function_code}\n\n        This is the context of used functions\n\n        {children_explanation_text}\n        "
        prompt = PromptTemplate(
            input_variables=["function_code", "children_explanation_text"],
            template=prompt_template,
        )
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        inputs = {
            "function_code": function_code,
            "children_explanation_text": children_explanation_text,
        }
        explanation = llm_chain.run(inputs)
        cursor.execute(
            "INSERT INTO explanations_cache (function_code, explanation) VALUES (?, ?)",
            (function_code, explanation),
        )
    conn.commit()
    conn.close()
    return explanation
