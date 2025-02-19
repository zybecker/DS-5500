# DS-5500
_Building Machine Learning Models to Predict the Outcome of a Pitch in an MLB Game_<br><br/>


## Overview
Problem: How do you find the most optimal way for hitters and pitchers to approach an at-bat? What is the most likely outcome for a given pitch profile and batter? 

Understanding the most likely outcomes for each offering in a pitcher’s repertoire helps a baseball team game plan and develop more in-depth approaches to each at-bat, thus providing a competitive advantage and putting teams in a better position to win games.

#### Presentation 1 link: https://docs.google.com/presentation/d/1fpL3Jh0kxGMI2qUpWTzKJ9C3V_SD8_k7z_SFKnGPTd0/edit?usp=sharing

#### Working folder link: https://drive.google.com/drive/folders/13-GU_hNipxnFBtZqqZxMusCat3K89r6n?usp=sharing<br><br/>


## Data
We are using MLB's Statcast data, which can be access at Baseball Savant: https://baseballsavant.mlb.com/statcast_search. In this project, we use the python package _pybaseball_ to directly access the Baseball Savant API via python. More information about pybaseball can be found on the package's GitHub page: https://github.com/jldbc/pybaseball. Pybaseball can be installed using pip.

The dataset that we are using includes a row for every pitch during every at-bat in the 2024 MLB season, with a variety of variables describing player, game, pitch, and batted-ball metrics. An example including 100 rows of our dataset can be found in the file [statcast_2024_sample.csv](https://github.com/zybecker/DS-5500/blob/9f6b05c6843e258cb980093d1f9937fcac10660f/statcast_2024_sample.csv). We include instructions on how to download the full dataset in the files included in the repository.

Currently, we have two models: a gradient-boosted model (XGBoost) and a recurrent neural network. These two models currently predict the same batted ball outcome, _estimated weighted on-base average (wOBA) using speed angle_. This variable measures how a hit ball is likely to perform based on exit velocity and speed angle independent of defense. More information about wOBA can be found at the [Statcast glossary](https://www.mlb.com/glossary/statcast/expected-woba).<br><br/>


## Getting Started
### Prerequisites
To run our code, first clone this repository in the typical git way:
1. Clone repo
   ```sh
   git clone https://github.com/zybecker/DS-5500.git
   ```
2. Install required packaged from requirements.txt
    ```sh
    pip install -r requirements.txt
    ```
For more information on cloning git repositories and installing dependencies, see [GitHub's guide to cloning repositories](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository)

### Running
After cloning the repository and installing dependencies, to run the model do the following:
1. Run **Download2024Data.ipynb** to download a csv containing 2024 Statcast data (2024pitches.csv). This file contains the dataset needed to run the models.
2. Run **Pitch Clustering.ipynb** to create a csv of pitch clusters (pitch_clusters.csv). This runs an unsupervised clustering algorithm on pitches.
3. Run the files for the two models, **DS5500 XGBoost.ipynb** and **RNN_Model.ipynb**. Currently, data preprocessing is located within these files. Make sure that these two csv files from the above steps are in the same directory as the model notebooks!<br><br/>

