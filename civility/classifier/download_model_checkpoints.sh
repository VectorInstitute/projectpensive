echo Downloading model checkpoints from remote server...

mkdir results
scp mnas@v.vectorinstitute.ai:baiso/baiso/civility/classifier/output.log results/output.log
scp -r mnas@v.vectorinstitute.ai:baiso/baiso/civility/classifier/results/final_model results/final_model