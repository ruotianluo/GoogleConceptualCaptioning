#!/bin/sh

iteration=$3

if ! [ "$iteration" = "" ]; then
iteration='-'$iteration
fi

mkdir log_$2
cd log_$2
ln -s ../log_$1/event* ./
cp ../log_$1/model$iteration.pth model.pth
cp ../log_$1/infos_$1$iteration.pkl infos_$2.pkl
cd ../
