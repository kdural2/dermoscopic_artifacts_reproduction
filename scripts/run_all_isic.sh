#!/bin/bash

# Run melanoma training with different modes
python -m src.train_melanoma --mode whole --num_folds 3 --epochs 8
python -m src.train_melanoma --mode lesion --num_folds 3 --epochs 8
python -m src.train_melanoma --mode background --num_folds 3 --epochs 8
python -m src.train_melanoma --mode bbox70 --num_folds 3 --epochs 8
python -m src.train_melanoma --mode low_whole --num_folds 3 --epochs 8