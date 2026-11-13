# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
– This model was developed by Connor Johnson.
– This model was developed in November 2026.
– This model is a Random Forest Classifier from sklearn.
– Hyperparameters are the default values for the model and have not been modified.

## Intended Use
– This model is intended to predict whether an individual's salary exceeds $50,000 based on their demographic data. 
– This model is was developed as part of a Udacity course and is not should not be used for real world use.

## Training Data
– An 80/20 training test split of the census.csv data was used to train the model where 80% was used for training.

## Evaluation Data
– The data used is from the cleaned census.csv file. This is census income data from 1994.
– The remaining 20% of the 80/20 split in the census.csv data was used for the evaluation.
– One hot encoding was used for the categorical features. A label binarizer for the labels.

## Metrics
– The metrics used to evaluate the model performance were Precision, Recall, and F1-score.
– The model performance was evaluated as the following: Precision: 0.8056 | Recall: 0.5512 | F1: 0.6546.

## Ethical Considerations
– The data used is publicly available census data and does not risk exposing any private information. 
– The census data used for this model is from 1994 and should not be used as an indicator of modern trends.

## Caveats and Recommendations
– Current census data should be used if this model is adapted for real world use. 