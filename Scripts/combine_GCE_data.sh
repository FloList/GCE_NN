#!/bin/bash 
# Bash script that submits the PBS jobs for combining the data.

declare -a scripts=("combine_data_from_models_for_letter_A.py" "combine_data_from_models_for_letter_O.py" "combine_data_from_models_for_letter_F.py")

for ((i = 0; i < ${#scripts[@]}; i++))
do
    qsub -v script=${scripts[$i]} submit_combine_data_single.pbs
done
