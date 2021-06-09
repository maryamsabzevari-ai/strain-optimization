#!/bin/bash -l                                                                  
#SBATCH -p batch                                                                
#SBATCH -t 50:00:00                                                             
#SBATCH -c 16                                                                   
#SBATCH --mem=8G                                                                


#module load anaconda2
#mkdir /tmp/testname                                                            

declare -a arr2=(0 1 2 3 4 5 6 7 8  )

for i in "${arr2[@]}"
do
     sbatch  run.sh $i  &
done


