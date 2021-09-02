echo Downloading data...
cp -r /checkpoint/gshensvm/BAISO_checkpoints/data/train-balanced-sarcasm-processed.csv civility/recommender/

echo Downloading civility filter checkpoint...
mkdir civility/classifier/results/
cp -r /checkpoint/gshensvm/BAISO_checkpoints/model_checkpoints/civility_model civility/classifier/results/final_model

echo Downloading diversity embeddings...
mkdir diversity/datasets
cp -r /checkpoint/gshensvm/BAISO_checkpoints/embeddings/subreddit_embeddings.csv diversity/datasets/
cp -r /checkpoint/gshensvm/BAISO_checkpoints/embeddings/sarcasm-embeddings-processed.pt diversity/datasets/

echo Downloading recommender engine checkpoint...
cp -r /checkpoint/gshensvm/BAISO_checkpoints/model_checkpoints/recommender_model civility/recommender/final_model
