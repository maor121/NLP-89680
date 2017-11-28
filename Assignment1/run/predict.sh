#!/usr/bin/env bash

source general.config

source $venv_folder

cd $proj_folder

for file in $test_files; do
    echo "------------------------------------------------------------------------------------------------------"
    file_name=${file#*/}   # remove prefix ending in "/"
    pred_file_start="${output_folder}${file_name}"

    echo "File: ${file_name}"
    echo "HMM Greedy"
    python $greedy_tag_file $file $q_mle_file $e_mle_file "${pred_file_start}.greedy_tag.pred"

    echo "HMM Viterbi"
    python $hmm_tag_file $file $q_mle_file $e_mle_file "${pred_file_start}.hmm_tag.pred"

    echo "MEMM Greedy"
    python $greedy_max_ent_tag_file $file $sklearn_model_file "${pred_file_start}.greedy_max_ent_tag.pred" $features_map_file

    echo "MEMM Viterbi"
    python $memm_tag_file $file $sklearn_model_file "${pred_file_start}.memm_tag.pred" $features_map_file
    echo "------------------------------------------------------------------------------------------------------"
done