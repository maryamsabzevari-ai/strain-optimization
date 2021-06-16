#!/bin/bash -l                                                         
#SBATCH -p batch                                                                
#SBATCH -t 30:00:00                                                             
#SBATCH -c 16                                                                   
#SBATCH --mem=8G                                                                


module load anaconda3

srun python strain_recomendar.py   $1 $1

                                            



