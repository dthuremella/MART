#!/bin/bash

# Handle SIGINT (CTRL+C) to kill all child processes
trap "kill 0" SIGINT

python main_eth.py --config "$1" --gpu 1 --dataset eth &
python main_eth.py --config "$1" --gpu 1 --dataset hotel &
python main_eth.py --config "$1" --gpu 1 --dataset univ &
python main_eth.py --config "$1" --gpu 0 --dataset zara1 &
python main_eth.py --config "$1" --gpu 0 --dataset zara2 &
wait
