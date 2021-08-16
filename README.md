# Welcome to Project Pensive
Project Pensive is open source software that leverages the latest AI research to benefit those offering recommendations.
The goal of Project Pensive is to build better Recommender Engines that incorporate content conscious filters to improve
content quality and user experience.

To get started:
1. Clone this repo: `git clone https://github.com/VectorInstitute/baiso.git`
2. Install code packages: `python setup.py install`
3. Install package dependencies: `pip install -r requirements.txt`

## Civility Filter
Code for the civility filter can be found under `baiso/civility/classifier`. To train a model on the vaughan server, 
run `sbatch run.slurm`.  

## Diversity Filter
Code for the diversity filter can be found under `baiso/diversity`. To generate subreddit embeddings, go to `basio/diversity/embeddings` and run the `subreddit2vec.ipynb` file. To generate comment emebddings, go to `basio/diversity/embeddings` and run the `sentencetransformers.ipynb` file.

## Recommender Engine
Code for the recommender engine can be found under `baiso/civility/recommender`. To train a recommender on the vaughan 
server, run `sbatch run.slurm`

## Demo
To run the demo, do the following

1. Make sure you have the necessary datasets and model checkpoints
2. Traverse to demo folder: `cd demo`
3. Run streamlit app: `streamlit run main.py`
