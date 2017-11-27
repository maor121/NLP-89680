#!/home/ofir/Devl/git/Deep-Learning-89687/venv/bin/activate

proj_folder="/home/ofir/Devl/git/NLP-89680/Assignment1"
src_folder="src/"
data_folder="data-ner/"
output_folder="output-ner/"

#HMM
train_file=$data_folder"train"
q_mle_file=$output_folder"q.mle"
e_mle_file=$output_folder"e.mle"

#MEMM
features_file=$output_folder"features_file"
features_vec_file=$output_folder"features_vec_file"
features_map_file=$output_folder"features_map_file"
sklearn_model_file=$output_folder"sklearn_model"

mle_train_file=$src_folder"hmm1/MLETrain.py"
extract_features_file=$src_folder"memm1/ExtractFeature.py"
convert_features_file=$src_folder"memm1/ConvertFeatures.py"
train_solver_file=$src_folder"memm1/TrainSolver.py"

echo python $pyFile $train_file $q_mle_file $e_mle_file

cd $proj_folder

#HMM
python $mle_train_file $train_file $q_mle_file $e_mle_file

#MEMM
python $extract_features_file $train_file $features_file
python $convert_features_file $features_file $features_vec_file $features_map_file
python $train_solver_file $features_vec_file $sklearn_model_file
