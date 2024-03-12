#!/bin/bash

cd tsp
./buildlocal.sh
cd ../
cd build/Release
# djibouti d198 a280 lin318 pcb442 rat783 d1291
for city in djibouti d198 a280 lin318 pcb442 rat783 d1291; do
	./bin/aco_cuda_v2_ref -p $city -i 15 -r 1 >> ll_aco.out 
done
#for city in 8; do
      #  ./bin/aco_cuda_v3_ref -p $city -i 15 -r 1
#done

