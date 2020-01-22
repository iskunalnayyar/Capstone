# Progress
This file should capture everything I have worked on. All my finding, and all my failures :)

## Week 1 - ?

### Working on the Dataset
This dataset has two files:
- The train.csv - has the size of 8921483, 75
- The test.csv - has the size of 7853253, 75

I realized that i need to change the problem statement a tad bit, since i do not have the true values for the test.csv 
file. 
~~Instead of predicting the probability of a device having malware~~, I will be predicting if a device has malware or not 
(Binary Classification) using the train.csv

**I will be using only the train.csv henceforth.**

#### Task 1 - Cleaning the datafile train.csv
These columns have too many missing values to be of any use and thus are dropped from the dataset:
-     PuaMode, Census_ProcessorClass, Census_InternalBatteryType, Census_IsFlightingInternal,
      Census_ThresholdOptIn, Census_IsWIMBootEnabled, SmartScreen, DefaultBrowsersIdentifier

_**This makes the total count of the columns to 75, down from 83**_

I am making two set of training files:
1. Drops all nan records from the datafram (this is going to end up having ~4.9M records)
2. Replaces all nan values with the median of that particular column (this is going to end up having ~6.5M)



#### Task 2 - Getting a base classifier accuracy


