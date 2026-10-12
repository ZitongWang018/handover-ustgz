#!/bin/bash
# Step2: add ai2thor (needed by alfworld.agents.environment.__init__ chain) + progressbar2
set -x
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thinktool
PY=$HOME/miniconda3/envs/thinktool/bin/python
T=$HOME/ztwang/selfplay/third_party/py310deps

"$PY" -m pip install --target "$T" --no-cache-dir --no-deps ai2thor==2.1.0
"$PY" -m pip install --target "$T" --no-cache-dir progressbar2

echo "########## DONE step2 ##########"
