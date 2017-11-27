#!/bin/bash

source ner.config
source general.config

source $venv_folder

cd $proj_folder

#HMM
echo "MLETrain.py"
python $mle_train_file $train_file $q_mle_file $e_mle_file

#MEMM
echo "ExtractFeatures.py"
python $extract_features_file $train_file $features_file
echo "ConvertFeatures.py"
python $convert_features_file $features_file $features_vec_file $features_map_file
echo "Train_Solver.py"
python $train_solver_file $features_vec_file $sklearn_model_file
