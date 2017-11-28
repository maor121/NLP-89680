#!/bin/bash

source general.config

source $venv_folder

cd $proj_folder

#HMM
echo "MLETrain.py ${mle_train_file} ${train_file} ${q_mle_file} ${e_mle_file}"
python $mle_train_file $train_file $q_mle_file $e_mle_file
echo ""

#MEMM
echo "ExtractFeatures.py ${extract_features_file} ${train_file} ${features_file}"
python $extract_features_file $train_file $features_file
echo ""

echo "ConvertFeatures.py ${features_file} ${features_vec_file} ${features_map_file}"
python $convert_features_file $features_file $features_vec_file $features_map_file
echo ""

echo "Train_Solver.py ${features_vec_file} ${sklearn_model_file}"
python $train_solver_file $features_vec_file $sklearn_model_file
echo ""