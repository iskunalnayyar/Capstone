# Progress
This file should capture everything I have worked on. All my findings, and all my failures :)

## Week 1 - 3

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
    Will be using this dataset for now.


#### Task 2 - Getting a base classifier accuracy
Things to do :
1. _Encode features_
2. Do feature engineering
3. Use the os version numbers to make it into a time column. And subtract the earliest time from the column to make it a numeric column
4. Do literature review
5. Come up with Genetic Algorithm 

Using the Larger file of the two training files, the base accuracy achieved was 62% on the neural network using a 3 layer 
architecture using Keras, viz, Input, dense layer and the output layer with sigmoid activation function.



## Week 3 - ?

### Literature Review
##### Efficient feature selection using one-pass generalized classifier neural network and binary bat algorithm with a novel fitness function - Akshata K. Naik, Venkatanareshbabu Kuppili & Damodar Reddy Edla

Proposes a wrapper approach of feature selection using a bio-inspired algorithm, i.e., Binary Bat algorithm along with 
One-pass Generalized Classifier Neural Network (OGCNN). 
Binary Bat Algorithm is inspired by the ecological behaviour of bats, & to be specific how they hunt for prey. 
OGCNN is based on GCNN which has 5 layers : Input, summation, normalization, & output.
GCNN calculates the smoothing parameter using the mean-based (gradient descent) functions & standard deviation. The paper talks about
how the accuracy parameter of a model fails to capture the case of class imbalance. And hence proposes to use the specificity &
sensitivity along with the accuracy of the model. The equation to minimize looks like :
- ψ = γ(Fs/FT) + β(p log2(p) + q log2(q)) + α(1 − Ac)
- Where Fs is the # of selected features
- Ft is the total # of features in the dataset
- The term p log2(p) + q log2(q) has the minimum value when sensitivity = specificity
- Ac is the accuracy of the model on current selected features
- α, β, γ are the paramters that control the weights of fraction of features selected, entropy of sensitivity, specificity,
 & accuracy respectively. These parameters sum to 1.

The proposed wrapper approach is employed on 5 publicly available datasets (which are binary in nature). The metrics used 
to measure the performance are, accuracy, specificity, sensitivity and f1 score.

##### A new wrapper feature selection approach using neural network - Md.Monirul KabiraMd, Monirul Islamb, and Kazuyuki Murase

Proposes a constructive approach to feature selection (CAFS) which employs a 3 layer feed forward neural network & uses the 
concept of wrapper approach, & sequential search strategy. The proposed CAFS not only emphasizes not only on selecting the most 
relevant features but also determining the appropriate architecture of the neural network. The idea being that the generalization
of a network is greatly impacted by the number of hidden neurons in the neuralnet architecture itself. Thus, the approach will
perform both the selection of features as well as the architecture of the neural network. This approach also utilizes the 
correlation information to guide the feature selection process. I think this is potential problem since, there may be a 
dataset where there's minimal to no correlation. How would the feature selection process work then? The approach looks like :
- Divide the original feature set into two subsets. Most correlating feature subset (S) and the least correlating feature set (D).
- Select a 3 layer feed-forward neuralnet with number of neurons set to 2 & 1 for the input and hidden layers respectively.
- Initially select one feature from S & D else, select according to feature addition criterion.
- Partially train the NN for a specified number of epochs.
- Check the termination criterion (the error of NN on the validation set), exit if satisfied. Else, continue.
- Check if training error decreases by a certain amount after the specified epochs. Train further if yes else continue
- Compute contribution of previously added feature based on Classification Accuracy.
- Check criterion of adding feature, add if necessary else continue.
- Add one hidden neuron to the NN and train again.

The proposed wrapper approach is employed on 5 publicly available datasets similar to ones used by the previous approach. 