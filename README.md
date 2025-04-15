# DS-5500
_Building Machine Learning Models to Predict the Outcome of a Pitch in an MLB Game_<br><br/>


## Overview
Problem: How do you find the most optimal way for hitters and pitchers to approach an at-bat? What is the most likely outcome for a given pitch profile and batter? 

Understanding the most likely outcomes for each offering in a pitcher’s repertoire helps a baseball team game plan and develop more in-depth approaches to each at-bat, thus providing a competitive advantage and putting teams in a better position to win games.

#### Working folder link: https://drive.google.com/drive/folders/13-GU_hNipxnFBtZqqZxMusCat3K89r6n?usp=sharing<br><br/>


## Data
We are using MLB's Statcast data, which can be access at Baseball Savant: https://baseballsavant.mlb.com/statcast_search. In this project, we use the python package _pybaseball_ to directly access the Baseball Savant API via python. More information about pybaseball can be found on the package's GitHub page: https://github.com/jldbc/pybaseball. Pybaseball can be installed using pip.

The dataset that we are using includes a row for every pitch during every at-bat in the 2024 MLB season, with a variety of variables describing player, game, pitch, and batted-ball metrics. An example including 100 rows of our dataset can be found in the file [statcast_2024_sample.csv](https://github.com/zybecker/DS-5500/blob/9f6b05c6843e258cb980093d1f9937fcac10660f/statcast_2024_sample.csv). We include instructions on how to download the full dataset in the files included in the repository.

Currently, we have three models: a gradient-boosted model (XGBoost), a linear neural network, and a recurrent neural network. The linear neural network consists of fully-connected linear layers, and the recurrent neural network utilizes GRU cells. These models currently all predict the same batted ball outcome, _estimated weighted on-base average (wOBA) using speed angle_. This variable measures how a hit ball is likely to perform based on exit velocity and speed angle independent of defense. More information about wOBA can be found at the [Statcast glossary](https://www.mlb.com/glossary/statcast/expected-woba).<br><br/>


## Getting Started
### Prerequisites
To run our code, first clone this repository in the typical git way:
1. Clone repo
   ```sh
   git clone https://github.com/zybecker/Predicting-Batted-Ball-Metrics.git
   ```
2. Install required packaged from requirements.txt
    ```sh
    pip install -r requirements.txt
    ```
For more information on cloning git repositories and installing dependencies, see [GitHub's guide to cloning repositories](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository)

### Accessing and Preparing Data
After cloning the repository and installing dependencies, to run the model do the following:
1. Run **Download2024Data.ipynb** to download a csv containing 2024 Statcast data (2024pitches.csv). This file contains the dataset needed to run the models.
2. Run **Pitch Clustering.ipynb** to create a csv of pitch clusters (pitch_clusters.csv). This runs an unsupervised clustering algorithm on pitches.
3. At this point, move on to the code for the models.

### Running Model Code
Once the data is downloaded and ready to be used, run the code for the various models. The two neural network models share a set of code.
#### XGBoost:
1. Run **XGBoost/Updated XGBoost.ipynb** to run train and test the XGBoost model.
#### Neural Networks:
1. First, run **Neural Networks/Data_preprocessing.ipynb** to preprocess the data for the models. This will create two PyTorch tensor files in **Neural Networks/datasets** to be used later, one for inputs and one for targets.
2. Run **Neural Networks/Hyperparameter_tuning.ipynb** to tune the hyperparameters for both models. This will create two JSON files in the **Neural Networks/tuning** folder to be used when configuring the final model training. Also, two csv file with all hyperparameter combinations and their results during tuning is saved.
3. Finally, to train the models with the best hyperparameters and evaluate their performance, run **Neural Networks/Final_model_training.ipynb**. This will also create a tensor file for the testing dataset in **Neural Networks/datasets**.
<br><br/>

