# Air Pollution Predication

Air pollution prediction is one of the popular problems of regression. This data set includes hourly air pollutants data from 12 nationally-controlled air-quality monitoring sites from the Beijing Municipal Environmental Monitoring Center. The meteorological data in each air-quality site are matched with the nearest weather station from the China Meteorological Administration. The time period is from March 1st, 2013 to February 28th, 2017. Missing data are denoted as NA.

Based on the dataset, separate the data into training (2013-2016) and testing (2017) sets. This data consists of multiple attributes, Xi(e.g. TEMP, temperature) and air pollution index (PM2.5 concentration only) as output Y. You need to develop the prediction models based on:


Result:
the error of each implementation of simple regression and multiple regression.
                    TEMP       DEMP      PRES      RAIN       WD        WSPM      Multiple
Training error     0.0093      0.0085    0.0089    0.0089     0.0087    0.0086	  153.39
  
Test error         0.0115      0.00998   0.0112    0.01207    0.01483   0.0089    29.127

Error difference   0.0022      0.0015    0.0023    0.0031     0.00601   0.0003    124.26



Authors:

Xiaojian Xie 7821950

YanLam Ng 7775665

Group: 9

Air Pollution Predictor:

Tensorflow version: v1

To run the program simply run the python code either for simple_regression_model.py or mutiple_regression_model.py. 

How to run in terminal:

1.go the the project directory "AirPollutionPredication" 

2.run 'python Code/{filename}'

For simple_regression_model, we use polynomial regression model with degree 3 to predict PM2.5 by TEMP, PRES, DEWP, RAIN, wd, and WSPM.

For mutiple_regression_model, we use linear regression model to predict.


