#!/usr/bin/env bash

source test.config
source general.config

source $venv_folder

cd $proj_folder

#python $greedy_tag_file $test_file $q_mle_file $e_mle_file $predict_greedy_tag_file
python $hmm_tag_file $test_file $q_mle_file $e_mle_file $predict_greedy_tag_file
