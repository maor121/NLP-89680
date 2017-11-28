#!/usr/bin/env bash

source general.config

source $venv_folder

cd $proj_folder

for pair in $eval_files; do
    input_file=${pair#*+}   # remove prefix ending in "+"
    tagged_file=${pair%+*}   # remove suffix starting with "+"

    input_filename=${input_file#*/} # remove prefix ending with /
    pred_file_start="${output_folder}${input_filename}"

    echo "---------------------------------- Greedy Tag ----------------------------------------"
    python $ner_file $tagged_file "${pred_file_start}.greedy_tag.pred" $is_span
    echo ""

    echo "----------------------------------- HMM Tag ------------------------------------------"
    python $ner_file $tagged_file "${pred_file_start}.hmm_tag.pred" $is_span
    echo ""

    echo "---------------------------- Greedy Max Entropy Tag ----------------------------------"
    python $ner_file $tagged_file "${pred_file_start}.greedy_max_ent_tag.pred" $is_span
    echo ""

    echo "----------------------------------- MEMM Tag -----------------------------------------"
    python $ner_file $tagged_file "${pred_file_start}.memm_tag.pred" $is_span
    echo ""
done