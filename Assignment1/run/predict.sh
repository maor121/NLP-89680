#!/usr/bin/env bash

source test.config
source general.config

source $venv_folder

cd $proj_folder

echo "HMM Greedy"
python $greedy_tag_file $test_file $q_mle_file $e_mle_file $predict_greedy_tag_file

echo "HMM Viterbi"
python $hmm_tag_file $test_file $q_mle_file $e_mle_file $predict_hmm_tag_file

echo "MEMM Greedy"
python $greedy_max_ent_tag_file $test_file $sklearn_model_file $predict_greedy_max_ent_tag_file $features_map_file