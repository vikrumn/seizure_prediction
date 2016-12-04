# seizure_prediction
For the Melbourne University AES/MathWorks/NIH Seizure Prediction Kaggle competition.
Using the thousands of EEG signals that were provided, first extracts statistical data (correlations between channels, means, 
standard deviations, skewness, kurtosis) in process_data_stats.py and process_old_test_data.py.  Then performs logistic
regression and trains a random forest classifier on this data, ultimately averaging the class probabilities from both.  
This achieved a 70% accuracy score on the final (private leaderboard) testing data, while the maximum achieved was 80%.
https://www.kaggle.com/c/melbourne-university-seizure-prediction
