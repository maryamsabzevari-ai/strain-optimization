#!/bin/bash -l                                                         
#SBATCH -p batch                                                                
#SBATCH -t 30:00:00                                                             
#SBATCH -c 16                                                                   
#SBATCH --mem=8G                                                                


module load anaconda3
#mkdir /tmp/testname                                 
#for val in $Arr; do
#    echo $val
#    #srun python sim_lactic_ed.py $val $val
#    #sleep 1
#done
srun python str_recom_acet_gsch.py   $1 $1
#for i in 123 234 345 567 789 980; do
#    srun python sim_lactic_ed.py $i $i
#    printf '%0.02f\n' $i
#done
                                            



