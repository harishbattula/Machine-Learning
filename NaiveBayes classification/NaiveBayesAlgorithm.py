# Importing essential Libraries for naivebayes algorithm 
import pandas as pd
import numpy as np
from operator import itemgetter


# Creating class for NaiveBayesAlgorithm..
class NaiveBayesAlgorithm():

    def __init__(self, Inputs, targets):
        """
            1.Get the train dataset
            2.It is good if we already splitted x and y labels before creating instance of this class
            3.In this example we already splitted x and y labels
            4.Do the Preprocessing work using Pandas and numpy
            5.Split the data into no of classifications available in dataset
            6.Since we are using titanic.csv from kaggle we only have two predictions i.e; survived or not
            7.1 means survived and 0 means not survived
            8.Then after data cleaning we split the data to two sets i.e; survived set and not survived set
            9.Apply the conditional probability for each feature in cleaned dataset
            10.Record the conditional probabilities of each feature in dataset
            11.Then apply baye's theorem and predict the outcomes of test dataset and compare then with original dataset
        """

        # Get unique classifications that we get from data
        self.classifications = np.unique(targets).tolist()
        
        # Total no of features
        self.n = np.shape(Inputs)[1]
        # Total no of samples
        self.m = np.shape(Inputs)[0]

        # set small number epsilon to get rid of any zeros
        self.epsilon = 0.001

        # Merge Inputs and targets for ease of indexing
        A = np.concatenate([Inputs, targets], axis=1)

        # Turn np array to pandas dataframe
        df = pd.DataFrame(A)

        # Make list of split datasets based on classifications
        self.classifiers = {}
        
        for c in self.classifications:
            x = df.loc[df[self.n] == c]
            self.classifiers[c] = x


        self.allLikelihoods = {}
        self.allPriors = {}

        for c in self.classifications:
            # for each class, calculate the corresponding probabilities
            X = self.classifiers[c]
            M = np.shape(X)[0]
            
            # calculate the class prior probability (out of all examps, what prob is the class?)
            self.allPriors[c] = M/self.m 


            # calculate total amount of counts for each variable in the class
            total = X.loc[:, X.columns[:-1]].to_numpy().sum()
            
        
            # calculate all likelihood terms P(N|c) using multinomial distribution
            likelihoods = {}

            for feature in range(self.n):

                featureOccurrences = X.loc[:, X.columns[feature]].to_numpy().sum()

                
                # return +1 for stability on top and bottom
                likelihood = (featureOccurrences + 1) / (total + self.n)
                likelihoods[feature] = (likelihood)
            
            self.allLikelihoods[c] = likelihoods
            
            print(self.allLikelihoods)
        
    
    def fit(self, x):
        """fits the model to a new example.
           x: a matrix where each column is a feature, and each row is a training example. If 
              multiple training examples, answer returned will be a numpy array of predictions in 
              size ([examples, 1]). IMPORTANT: INPUT MUST NOT BE A RANK ONE ARRAY. IF SUBMITTING
              A SINGLE EXAMPLE, MUST BE IN FORMAT np.array([[1, 2, 3, 4]]).
        """
    

        # run through each calculated conditional probability and multiply it by how much it appears in 
        # new example
        
        # if single example, reshape into 2d array so we can iterate properly
        
        exampleResults = [] 

        logits = {} 
        for ex in x:
            for c in self.classifications:
                runningCondProbs = []
                for feature in range(self.n):
                    currentConditionalProb = round(ex[feature]*self.allLikelihoods[c][feature], 4)
                    currentConditionalProb += self.epsilon
                    runningCondProbs.append(currentConditionalProb)
            
                unnormalizedClassProb = (np.prod(runningCondProbs))*self.allPriors[c]
                    
                logits[c] = unnormalizedClassProb
            
            assignedClass = max(logits.items(), key=itemgetter(1))[0]
            exampleResults.append(assignedClass)
          
        
        # return the exampleResults as a (examples, 1) size matrix
        return ((np.array(exampleResults).reshape((np.shape(x)[0], -1))))

 

# LOAD TRAINING DATA =================================================================================

df = pd.read_csv('train.csv')

# Drop unnecessary columns 
# I assume that these columns won't show any impacy on our predictions
df = df.drop(['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'], axis=1)

# Turn male female into dummies and drop old male/female column 
dummies = pd.get_dummies(df['Sex'])
df = df.drop(['Sex'], axis=1)

# Add dummies to dataframe
df= pd.concat([df, dummies], axis=1)

# Seperate Targets from inputs
y = df['Survived']
df = df.drop(['Survived'], axis=1)

# Fill any NaN values with the mean of the each respected column
df = df.fillna(df.mean())

# convert both to numpy arrays to pass to MultnomialNaiveBayes
A = df.values
y = y.values.reshape((df.shape[0],1))




# LOAD TEST DATA =======================================================================================
df2 = pd.read_csv('test.csv')
passengerNos = df2['PassengerId']

# Drop unnecessary columns like we did before in train data
df2 = df2.drop(['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'], axis=1)

# Turn male female into dummies and drop old male/female column 
dummies = pd.get_dummies(df2['Sex'])
df2 = df2.drop(['Sex'], axis=1)

# Add dummies to dataframe
df2 = pd.concat([df2, dummies], axis=1)
X = df2.to_numpy()


# IMPLEMENT NAIVE BAYES ================================================================================

# Pass Training data into NaiveBayesAlgorithm
MNB = NaiveBayesAlgorithm(A, y)



# Fit new examples (test set)
yhat = MNB.fit(X)
