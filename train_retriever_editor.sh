#!/bin/bash
module purge                     # clear environment modules inherited from submission

python -m editing.train_llama2_mend 
        --save_name "nq_contriever" 
        --dataset "nq"
