#!/bin/bash

PYTHON_SCRIPT="run.py"

for i in {1..4}
do
    echo "Running: python3 $PYTHON_SCRIPT --dataset $i"
    python3 "$PYTHON_SCRIPT" --dataset $i
done
