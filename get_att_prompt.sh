#!/bin/bash
module purge                     # clear environment modules inherited from submission
python -m pipeline.gen_att_prompt
