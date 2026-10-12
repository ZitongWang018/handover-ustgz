#!/usr/bin/env python
"""Probe: can thinktool(py3.10) import textworld/alfworld from py310deps?"""
import os
import sys
import traceback

T = os.path.expanduser("~/ztwang/selfplay/third_party/py310deps")
SRC = os.path.expanduser("~/ztwang/selfplay/src")
sys.path.insert(0, SRC)
sys.path.insert(0, T)

print("python:", sys.version)
print("path head:", sys.path[:4])

def step(label, fn):
    try:
        r = fn()
        print(f"[OK]   {label} -> {r}")
        return True
    except Exception as e:
        print(f"[FAIL] {label}: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

step("import numpy", lambda: __import__("numpy").__version__)
step("import torch", lambda: __import__("torch").__version__)
step("alfworld_compat.install", lambda: __import__("alfworld_compat").install())
step("fast_downward.load_lib", lambda: __import__("alfworld_compat").self_check())
step("import textworld", lambda: __import__("textworld").__version__)
step("import textworld.gym", lambda: __import__("textworld.gym").__name__ if hasattr(__import__("textworld"), "gym") else str(__import__("textworld.gym")))
step("import gym", lambda: __import__("gym").__version__)
step("import alfworld", lambda: __import__("alfworld").__file__)
step("import alfworld.agents.environment.alfred_tw_env",
     lambda: __import__("alfworld.agents.environment.alfred_tw_env", fromlist=["AlfredTWEnv"]).AlfredTWEnv)
print("done")
