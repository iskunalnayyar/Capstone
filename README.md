# Capstone
Feature Selection in a neural network loop using Genetic Algorithm

## Problem Statement

Suppose we have a neural network model and a large set of features to choose from. For the purpose of training, it might be best to choose the most optimal set of features and train the model on that set to achieve the best possible accuracy from the model. An exhaustive selection of features would yield in 2N combinations (where N is the number of features). This process requires lots of computational work and, if the number of features is big, becomes impracticable. Therefore, we need intelligent methods that allow the selection of features in practice.
One such approach is the use of Genetic Algorithm (GA) to do feature selection. The GA is a heuristic optimization method inspired by the procedures of natural evolution. For feature selection the function to optimize is the performance metric of the model on the test dataset.

## Expected outcome & evaluation
The GA will be used to select the features & feed them to the neural network model. The optimizable parameter of the model (could be any of – rmse, mse, accuracy, etc.) needs to be either - increased (in case of accuracy) or decreased (in case of the error or loss). Every generation results in a different population set based on mutation & crossover. At each generation, the optimizable parameter will be saved, and the best candidate of the lot will be iterated upon by the GA. The overall outcome will be two-fold, one will be the plot of every generation versus the chosen performance metric. Second is the output of the GA – the list of best performing features. The experimentation can be conducted by working on the same problem statement but for a different dataset.
## Plan of action
The plan of action is broken down into the following tasks:
•	Literature Review
•	Finding a large enough dataset with a vast set of features, and problem formulation for that dataset.
•	Cleaning the dataset 
•	Building a neural network model for the purpose of prediction/classification
•	Implementing the Genetic Algorithm
•	Improving the Genetic Algorithm if necessary

##### The milestones may be planned as follows:
Milestone 1	Literature Review, Dataset & Problem Statement formulation, Cleaning Dataset
Milestone 2	Building the Neural Network and implementing the Genetic Algorithm
Milestone 3	Iterating on the Genetic Algorithm as necessary, finalizing report and poster.

### Possible hurdles
The two biggest hurdles that I may need to overcome are:
•	Finding the ‘right’ dataset.
•	The Genetic Algorithm getting stuck in a local maximum or local minimum, depending on the parameter.
•	 Providing the best result very early on in the generation cycle. 
•	GA being sensitive to the initial population.
