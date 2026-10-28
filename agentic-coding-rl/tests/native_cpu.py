"""Execute pinned Miles' pure function bodies without its CUDA import graph.

Only decorators are omitted (not function bodies). This tests reduction/layout
and dispatcher contracts, not Megatron kernels, NCCL, or a real distributed run.
"""
import ast
import copy
import logging

import torch

from launcher import checkout


def native_functions(relative_path: str, names: tuple[str, ...], **bindings):
    path = checkout("miles") / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"))
    definitions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    selected = set()

    def visit(name):
        if name in selected:
            return
        selected.add(name)
        for node in ast.walk(definitions[name]):
            if isinstance(node, ast.Name) and node.id in definitions:
                visit(node.id)

    for name in names:
        visit(name)
    namespace = {"torch": torch, "logger": logging.getLogger("native_cpu"), **bindings}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                namespace[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    for name in selected:
        node = copy.deepcopy(definitions[name])
        node.decorator_list = []
        nodes.append(node)
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace
