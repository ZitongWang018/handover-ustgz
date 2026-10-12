#!/bin/bash
# Step1: install ALFWorld deps into ~/ztwang/selfplay/third_party/py310deps for thinktool (py3.10)
set -x
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thinktool
PY=$HOME/miniconda3/envs/thinktool/bin/python
T=$HOME/ztwang/selfplay/third_party/py310deps
mkdir -p "$T"

echo "########## 1) core wheels (no-deps) ##########"
"$PY" -m pip install --target "$T" --no-cache-dir --no-deps \
  textworld==1.6.2 fast_downward_textworld==20.6.4 alfworld==0.3.3

echo "########## 2) pure-python / light deps ##########"
"$PY" -m pip install --target "$T" --no-cache-dir \
  "numpy==1.26.4" "tqdm>=4.17.1" cffi networkx more_itertools tatsu==5.8.3 \
  hashids jericho mementos termcolor prompt_toolkit pyyaml

echo "########## 3) gym 0.26.2 (pure python) copied from agentenv-alfworld ##########"
A=$HOME/miniconda3/envs/agentenv-alfworld/lib/python3.9/site-packages
cp -r "$A/gym" "$T/" && echo "gym copied"
cp -r "$A/gym_notices" "$T/" && echo "gym_notices copied"
cp -r "$A"/gym-0.26.2.dist-info "$T/" && echo "gym dist-info copied"
cp -r "$A"/gym_notices-0.1.0.dist-info "$T/" && echo "gym_notices dist-info copied"

echo "########## 4) result listing ##########"
ls "$T"
echo "########## DONE step1 ##########"
