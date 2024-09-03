# ALCER3D

This repository contains the source code of the paper "Adaptive Learning Constraints for Enhanced Retrieval of Complex Indoor 3D Scenarios".

## Scenes and textual Features
### Features Preparation
To prepare features the [prepare_features.py](https://github.com/aliabdari/ALCER3D/blob/main/train_and_evaluation/available_features/prepare_features.py) should be used. For example, to obtain the features of the 3dfront dataset the following steps are needed from the root of the repository:
```
cd ./train_and_evaluation/available_features
python prepare_features.py -dataset 3dfront
```
## Similarity Amounts
To have the similarity amount based on the dataset and the language model used to obtain the metric it can be downloaded directly from [link](https://drive.google.com/file/d/14c4vAY045WWp2u7sAJWMwIti2TQupRON/view?usp=sharing). Then the zip file should be extracted to obtain the scenes_relevances, which should be placed in the root directory ('./ALCER3D/scenes_relevances') of the project.

## Train and Evaluation
To train and evaluate the method the [train_eval.py](https://github.com/aliabdari/ALCER3D/blob/main/train_and_evaluation/train_eval.py) module should be used.
to simply run the code you can use a command like the below sample command for using one thresholds and two margins. 
```
python train_eval.py -custom_margin -dataset 3dfront -status 0 -thresholds 0.55 -margins 0.25,0.45
```
### Arguments

- -dataset: It is used to select the dataset that we want to use. So far it could be '3dfront' or 'museums'
- -custom_margin: by using this argument it will use custom margins and thresholds and without putting this argument the code will be run in the default mode which does not have any threshold and uses one margin
- -status: It declares how many thresholds and margins will be used. This argument could be 0, 1, or 2. When it is 0 we have one threshold and two margins, when it is 1 we have two thresholds and three margins, and when it is 2 we have three thresholds and four margins
```
    status 0 ->  One Threshold, Two Margins
    status 1 ->  Two Thresholds, Three Margins
    status 2 ->  Three Thresholds, Four Margins
```
- -margins: It allows to define of the margins used in the loss function. It accepts a tuple value like: (0.25,0.35). Note that the number of the margins and the thresholds should match (One of these three options: One Threshold, Two Margins, Two Thresholds, Three Margins, and Three Thresholds, Four Margins)
- -thresholds: It allows to define the thresholds. It accepts a tuple value and again should follow the chosen format of the margins as explained in the 'margins' argument. 
- -simmodel: with this switch, we can specify the similarity model that will be used to measure the similarity between descriptions it could be 'distilroberta' or 'MiniLM' or 'gte-large'
- -simmodel2: with this switch, we can specify the second similarity model that will be used to measure the similarity between descriptions it could be 'distilroberta' or 'MiniLM' or 'gte-large' based on the selected model for the first similarity model
- -simmodel3: with this switch, we can specify the third similarity model that will be used to measure the similarity between descriptions it could be 'distilroberta' or 'MiniLM' or 'gte-large' based on the selected model for the second similarity model
