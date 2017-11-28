#!/usr/bin/env bash

source ner.config
source general.config

source $venv_folder

cd $proj_folder

for pair in $eval_files; do
    input_file=${pair#*+}   # remove prefix ending in "+"
    tagged_file=${pair%+*}   # remove suffix starting with "+"
    echo "---------------------------------- Greedy Tag ----------------------------------------"
    python $ner_file $tagged_file "${input_file}.greedy_tag.pred" $is_span
    echo ""

    echo "----------------------------------- HMM Tag ------------------------------------------"
    python $ner_file $tagged_file "${input_file}.hmm_tag.pred" $is_span
    echo ""

    echo "---------------------------- Greedy Max Entropy Tag ----------------------------------"
    python $ner_file $tagged_file "${input_file}.greedy_max_ent_tag.pred" $is_span
    echo ""

    echo "----------------------------------- MEMM Tag -----------------------------------------"
    python $ner_file $tagged_file "${input_file}.memm_tag.pred" $is_span
    echo ""
done