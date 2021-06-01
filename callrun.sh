#!/bin/bash -l                                                                  
#SBATCH -p batch                                                                
#SBATCH -t 50:00:00                                                             
#SBATCH -c 16                                                                   
#SBATCH --mem=8G                                                                


#module load anaconda2
#mkdir /tmp/testname                                                            
#set Arr[0]=123
#set Arr[1]=234
#set Arr[2]=345
#set Arr[3]=456
#set Arr[4]=567
#set Arr[5]=789
#set Arr[6]=890
#for val in $Arr; do
#    echo $val
#    #srun python sim_lactic_ed.py $val $val
#    #sleep 1
#done

#for i in 123 234 345 567 789 980; do
#    srun python sim_lactic_ed.py $i $i
#    printf '%0.02f\n' $i
#done

declare -a arr2=(0 1 2 3 4 5 6 7 8  )
#declare -a arr2=(123  234  345 567  789)

#declare -a arr2=(123  234  345 567 789 980 111 222 )
#declare -a arr2=(333 444 )
#declare -a arr2=(123  234  345 567 789 980 111 222 333 444 213 346 984 645  )
#declare -a arr2=(123  234  345 567 789 980 111 222 333 444 213 346 984 645 926 340 82 24 33 20 90 21 22 23 24 25)
for i in "${arr2[@]}"
do
     sbatch  run.sh $i  &
done


