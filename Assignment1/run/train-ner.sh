#!/home/ofir/Devl/git/Deep-Learning-89687/venv/bin/activate

proj_folder="/home/ofir/Devl/git/NLP-89680/Assignment1"
src_folder="src/"
data_folder="data-ner/"
output_folder="output-ner/"

train_file=$data_folder"train"
q_mle_file=$output_folder"q.mle"
e_mle_file=$output_folder"e.mle"

pyFile=$src_folder"hmm1/MLETrain.py"

echo python $pyFile $train_file $q_mle_file $e_mle_file

cd $proj_folder
python $pyFile $train_file $q_mle_file $e_mle_file
