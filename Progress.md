# Progress
This file should capture everything I have worked on. All my findings, and all my failures :)

## Week 1 - ?

### Working on the Dataset
This dataset has two files:
- The train.csv - has the size of 8921483, 75
- The test.csv - has the size of 7853253, 75

I realized that I need to change the problem statement a tad bit, since I do not have the true values for the test.csv 
file. 
~~Instead of predicting the probability of a device having malware~~, I will be predicting if a device has malware or not 
(Binary Classification) using the train.csv

**I will be using only the train.csv henceforth.**

#### Task 1 - Cleaning the datafile train.csv
These columns have too many missing values to be of any use and thus are dropped from the dataset:
-     PuaMode, Census_ProcessorClass, Census_InternalBatteryType, Census_IsFlightingInternal,
      Census_ThresholdOptIn, Census_IsWIMBootEnabled, SmartScreen, DefaultBrowsersIdentifier

_**This makes the total count of the columns to 75, down from 83**_

I am making two sets of training files:
1. Drops all nan records from the dataframe (this is going to end up having ~6.9M records)
    Based on skewness and missing data
2. Replaces all nan values with the median of that particular column (this is going to end up having ~8.5M)


#### Task 2 - Getting a base classifier accuracy
Things to do :
1. Encode features
2. Do feature engineering
Use the os version numbers to make it into a time column. And subtract the earliest time from the column to make it a numeric column

Using the Larger file of the two training files, the base accuracy achieved was 62% on the neural network

