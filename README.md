# Strain design optimization using Reinforcement Learning
## Overview
- In this project a multi-agent reinforcement learning (MARL) approach is implemented that learns from experiments to tune enzyme levels in the production host so that the product yield is optimized.
- A comprehensive empirical evaluation is presented on the genome-scale kinetic model of
E. coli (k-ecoli457), evaluating the sample complexity, noise tolerance, and stability of the
designs.
- For more information about the method and results check out the manuscript in "bioRxiv link"
## How to use the software
The code can be run for a specific product ("acetate" determined in the main function of strain_recomendar.py) by providing a seed and running run.sh. For the paper, the results are averaged over the seeds provided in callrun.sh. 
You can change the product= "acetate" line in strain_recomendar.py to optimize the yield for other products such as "succinate" or "ethanol" which their corresponding enzyme and target indexes are presented in "param_file.yaml". 
The code can be run for any other enzymes (as controllable factors) and product (as target to be optimized), when you add the correponding indexes to the "param_file.yaml" file.

## Dependencies
- numpy
- scikit-learn

## Citing the paper: 

## License
MIT
## For any doubt or question you can contact:
maryam.sabzevari@aalto.fi
