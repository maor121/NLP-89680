#!/usr/bin/env bash

#!/usr/bin/env bash

source test.config
source general.config

source $venv_folder

cd $proj_folder

echo "---------------------------------- Greedy Tag ----------------------------------------"
python $ner_file $eval_file $predict_greedy_tag_file $is_span
echo ""

echo "----------------------------------- HMM Tag ------------------------------------------"
python $ner_file $eval_file $predict_hmm_tag_file $is_span
echo ""

echo "---------------------------- Greedy Max Entropy Tag ----------------------------------"
python $ner_file $eval_file $predict_greedy_max_ent_tag_file $is_span
echo ""

echo "----------------------------------- MEMM Tag -----------------------------------------"
python $ner_file $eval_file $predict_memm_tag_file $is_span
echo ""

