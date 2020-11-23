#!/bin/bash 

for value in {0..24}
do
    qsub -v JOB_ID=${value} submit_data_gen_single.pbs
done
