#!/bin/bash

# Handle SIGINT (CTRL+C) to kill all child processes
trap "kill 0" SIGINT

python main_eth.py --config "$1" --gpu "$2" --dataset eth --tag "$3" &
python main_eth.py --config "$1" --gpu "$2" --dataset hotel --tag "$3" &
python main_eth.py --config "$1" --gpu "$2" --dataset univ --tag "$3" &
python main_eth.py --config "$1" --gpu "$2" --dataset zara1 --tag "$3" &
python main_eth.py --config "$1" --gpu "$2" --dataset zara2 --tag "$3" &
wait
