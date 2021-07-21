echo Downloading model checkpoints from remote server...

scp mnas@v.vectorinstitute.ai:baiso/baiso/civility/classifier/output.log
scp -r mnas@v.vectorinstitute.ai:baiso/baiso/civility/classifier/results/final_model results/final_model