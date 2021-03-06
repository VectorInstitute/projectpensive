# Welcome to Project Pensive
Project Pensive is open source software that leverages the latest AI research to benefit those offering recommendations.
The goal of Project Pensive is to build better Recommender Engines that incorporate content conscious filters to improve
content quality and user experience.

See also the project <a href="https://docs.google.com/document/d/183vqr8mFDhFE92pTfXzVoYuNUEMev0PG8RbYa9TZvdc/edit">charter</a> and <a href="https://docs.google.com/document/d/10L_s9FlOrmwOYsUlRg2WE5d8SxivJfP55Y8PTinYH4A/edit#heading=h.3kadlhlnhygj">technical agenda</a>

To get started:
1. Clone this repo: `git clone https://github.com/VectorInstitute/projectpensive.git`
2. Install code packages: `python setup.py install`
3. Install package dependencies: `pip install -r requirements.txt`

Notes:
1. This is a one-off demo, the code repository will not be maintained.
2. This code was written with a  Python 3.7 interpreter. It is recommended to use Python 3.7 or newer.


## External to Vector
### Data Download
To run any training programs or demos, you will need to download the data. Assuming you have a Kaggle account, go to
[the dataset](https://www.kaggle.com/sherinclaudia/sarcastic-comments-on-reddit) and download the zip file. Unzip and
move the csv file to `projectpensive/civility/recommender/train-balanced-sarcasm.csv`. Once this is done, traverse to
`projectpensive/civility/recommender` and run `python preprocess_data.py`. If done correctly, you will have all data you need
to run experiments and the demo. Another dataset that needs to be downloaded can be found
[here](https://www.kaggle.com/timschaum/subreddit-recommender). Unzip this file, create a folder under diversity `diversity/datasets` and add the file there as
`projectpensive/diversity/datasets/reddit_user_data_count.csv`. Follow the instructions under the Diversity Filter section below to
correctly preprocess the data.

#### Model Checkpoints
If you want to run the demo, you will need model checkpoints for the civility filter and the recommender. You will also
need precomputed embeddings for the diversity filter.

You will need to train the filters yourself (see Civility Filter and Recommender Engine sections). To acquire the
precomputed embeddings for the diversity filter, see the Diversity Filter section.

#### Civility Filter
Code for the civility filter can be found under `projectpensive/civility/classifier`. To train a model, run `python train.py`.

#### Diversity Filter
Code for the diversity filter can be found under `projectpensive/diversity`. To generate subreddit embeddings, go to
`basio/diversity/embeddings` and run the `subreddit2vec.ipynb` file. Once you have generated the `vectors.tsv` and `metadata.tsv` files and ensured that they are in the `datasets` folder, run `basio/diversity/generate-subreddit-dataframe.py`. To generate comment embeddings, run the `generate-comment-embeddings.py` file.

#### Recommender Engine
Code for the recommender engine can be found under `projectpensive/civility/recommender`. To train a recommender, run
`python train.py`


## Internal to Vector
If you have access to the vaughan server, you can run a shell script to download all necessary data and model
checkpoints. To do so, run `sh download_data_and_checkpoints your_cluster_id`.

## Demo
To run the demo, do the following

1. Make sure you have the necessary datasets and model checkpoints
2. Traverse to demo folder: `cd demo`
3. Run streamlit app: `streamlit run main.py`

## References
1. [Improving Recommendation Diversity](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.8.5232&rep=rep1&type=pdf) (2001, Keith Bradley and Barry Smyth)
2. [Improving Recommendation Lists Through Topic Diversification](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.62.9683&rep=rep1&type=pdf) (2005, Cai-Nicolas Ziegler et al.)
3. [Enhancing Diversity-Accuracy Technique on User-Based Top-N Recommendation Algorithms](https://sci-hub.se/https://ieeexplore.ieee.org/document/6605824/references#references) (2013, Wichian Premchaiswadi et al.)

## Contributing
There is no active support for this repository.

## The Team
This was work done by Michael Laurent Nasello and Sheen Thusoo during their summer 2021 internship at the Vector Institute with help from Ron Bodkin and Shems Saleh(during her time here at Vector).

## License
This repo has a MIT-style licence, as found in [LICENSE](LICENSE) file.
