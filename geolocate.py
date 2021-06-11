###################################
# CS B551 Fall 2020, Assignment #4
#
# Your names and user ids:
# Cody Harris (harrcody)
#
# geolocate.py
# Handles training and testing of either
# a bayes net or decision tree to predict
# the location of a tweet based on the
# words in the tree

import sys
import pickle
from BayesNetClassifier import BayesNetClassifier
from BayesNetPredictor import BayesNetPredictor
from DecisionTreeClassifier import DecisionTreeClassifier
from DecisionTreePredictor import DecisionTreePredictor

def main():
    if len(sys.argv) != 5:
        raise(Exception('Invalid parameters in program call'))
    
    #Train Bayes Net
    #Outputs a pickle file of probabilities
    if sys.argv[1] == 'train' and sys.argv[2] == 'bayes':
        print('Training Bayes Net....')
        bayes = BayesNetClassifier(sys.argv[3], sys.argv[4])
        bayes.fit()
        bayes.top_words()
        bayes.pickle_probs()
    
    #Train Decision Tree
    #Outputs a pickle file of the tree in dictionary form
    if sys.argv[1] == 'train' and sys.argv[2] == 'dtree':
        print('Training Decision Tree....')
        dtree = DecisionTreeClassifier(sys.argv[3], sys.argv[4], 50/32000, 675/32000, 7)
        dtree.fit()
        dtree.pickle_tree()
        
    #Predicting from Trained Models    
    elif sys.argv[1] == 'test':
        model_name = sys.argv[2]
        with open(model_name, 'rb') as handle:
            model = pickle.load(handle)
            
        #Predicting bayes
        if model['type'] == 'bayes':
            bayes_pred = BayesNetPredictor(model, sys.argv[3], sys.argv[4])
            bayes_pred.predict()
        
        #Predicting Decision Tree
        if model['type'] == 'dtree':
            dtree_pred = DecisionTreePredictor(model, sys.argv[3], sys.argv[4])
            dtree_pred.predict()

if __name__ == '__main__':
    main()