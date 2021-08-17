# Welcome to Project Pensive
Project Pensive is open source software that leverages the latest AI research to benefit those offering recommendations.
The goal of Project Pensive is to build better Recommender Engines that incorporate content conscious filters to improve
content quality and user experience.

To get started:
1. Clone this repo: `git clone https://github.com/VectorInstitute/baiso.git`
2. Install code packages: `python setup.py install`
3. Install package dependencies: `pip install -r requirements.txt`


## External to Vector
### Data Download
To run any training programs or demos, you will need to download the data. Assuming you have a Kaggle account, go to 
[the dataset](https://www.kaggle.com/sherinclaudia/sarcastic-comments-on-reddit) and download the zip file. Unzip and 
move the csv file to `baiso/civility/recommender/train-balanced-sarcasm.csv`. Once this is done, traverse to 
`baiso/civility/recommender` and run `python preprocess_data.py`. If done correctly, you will have all data you need
to run experiments and the demo. Another dataset that needs to be downloaded can be found 
[here](https://www.kaggle.com/timschaum/subreddit-recommender). Unzip this file and move to 
`baiso/diversity/reddit_user_data_count.csv`. Follow the instructions under the Diversity Filter section below to 
correctly preprocess the data.

#### Model Checkpoints
If you want to run the demo, you will need model checkpoints for the civility filter and the recommender. You will also
need precomputed embeddings for the diversity filter.

You will need to train the filters yourself (see Civility Filter and Recommender Engine sections). To acquire the 
precomputed embeddings for the diversity filter, see the Diversity Filter section.

#### Civility Filter
Code for the civility filter can be found under `baiso/civility/classifier`. To train a model, run `python train.py`.  

#### Diversity Filter
Code for the diversity filter can be found under `baiso/diversity`. To generate subreddit embeddings, go to 
`basio/diversity/embeddings` and run the `subreddit2vec.ipynb` file and to generate comment embeddings, run the 
`sentencetransformers.ipynb` file.

#### Recommender Engine
Code for the recommender engine can be found under `baiso/civility/recommender`. To train a recommender, run 
`python train.py`


## Internal to Vector
If you have access to the vaughan server, you can run a shell script to download all necessary data and model 
checkpoints. To do so, run `sh download_data_and_checkpoints your_cluster_id`.

## Demo
To run the demo, do the following

1. Make sure you have the necessary datasets and model checkpoints
2. Traverse to demo folder: `cd demo`
3. Run streamlit app: `streamlit run main.py`
