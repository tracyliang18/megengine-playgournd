#!/bin/bash
I=64
for e in 4 16 64 256 1024
do
  for ((i=1;i<=10;i++))
  do
    out=outs/model_${e}_exp_${i}
    python3 train.py -e 1024 -i 32 -s $out --epoch 20 &
  done
  wait

  for ((i=1;i<=10;i++))
  do
    out=outs/model_${e}_exp_${i}
    python3 plot.py -f $out/log.txt --epoch 20
  done
  wait
done
