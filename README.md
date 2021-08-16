# Welcome to Project Pensive
Project Pensive is open source software that leverages the latest AI research to benefit those offering recommendations.
The goal of Project Pensive is to build better Recommender Engines that incorporate content conscious filters to improve
content quality and user experience.

To get started:
1. Clone this repo: `git clone https://github.com/VectorInstitute/baiso.git`
2. Install code packages: `python setup.py install`
3. Install package dependencies: `pip install -r requirements.txt`


## Data Download
To run any training programs or demos, you will need to download the data. Assuming you have a Kaggle account, go to 
[the dataset](https://www.kaggle.com/sherinclaudia/sarcastic-comments-on-reddit) and download the zip file. Unzip and 
move the csv file to `baiso/civility/recommender/train-balanced-sarcasm.csv`. Once this is done, traverse to 
`baiso/civility/recommender` and run `python preprocess_data.py`. If done correctly, you will have all data you need
to run experiments and the demo.

## Model Checkpoints
If you want to run the demo, you will need model checkpoints for the civility filter and the recommender. You will also
need precomputed embeddings for the diversity filter.

You can either train the filters yourself (see Civility Filter and Recommender Engine sections) or download the 
checkpoints from the server. To acquire the precomputed embeddings for the diversity filter, see the Diversity Filter
section.

## Civility Filter
Code for the civility filter can be found under `baiso/civility/classifier`. To train a model on the vaughan server, 
run `sbatch run.slurm`.  

## Diversity Filter
Written by Sheen

## Recommender Engine
Code for the recommender engine can be found under `baiso/civility/recommender`. To train a recommender on the vaughan 
server, run `sbatch run.slurm`

## Demo
To run the demo, do the following

1. Make sure you have the necessary datasets and model checkpoints
2. Traverse to demo folder: `cd demo`
3. Run streamlit app: `streamlit run main.py`