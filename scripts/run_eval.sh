#!/bin/bash

python -m src.eval_transfer --mode whole
python -m src.eval_transfer --mode lesion
python -m src.eval_transfer --mode background
python -m src.eval_transfer --mode bbox70
python -m src.eval_transfer --mode low_whole