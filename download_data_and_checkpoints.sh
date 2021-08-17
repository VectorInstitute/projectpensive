USER_ID=$1

echo Downloading data...
scp $USER_ID@v.vectorinstitute.ai:/checkpoint/gshensvm/BAISO_checkpoints/data/train-balanced-sarcasm-processed.csv civility/recommender/

echo Downloading civility filter checkpoint...
mkdir civility/classifier/results/
scp -r $USER_ID@v.vectorinstitute.ai:/checkpoint/gshensvm/BAISO_checkpoints/model_checkpoints/civility_model civility/classifier/results/final_model

echo Downloading diversity embeddings...
mkdir diversity/datasets
scp $USER_ID@v.vectorinstitute.ai:/checkpoint/gshensvm/BAISO_checkpoints/embeddings/subreddit_embeddings.csv diversity/datasets/
scp $USER_ID@v.vectorinstitute.ai:/checkpoint/gshensvm/BAISO_checkpoints/embeddings/sarcasm-embeddings-processed.pt diversity/datasets/

echo Downloading recommender engine checkpoint...
scp -r $USER_ID@v.vectorinstitute.ai:/checkpoint/gshensvm/BAISO_checkpoints/model_checkpoints/recommender_model civility/recommender/final_model