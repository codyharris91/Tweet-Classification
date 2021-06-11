import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

class DecisionTreePredictor:
    def __init__(self, model, test_file, out_file):
        self.t_file = test_file
        self.o_file = out_file
        self.model = model['tree']
        self.words = []
        
    # Read in Testing Data    
    def read_data(self, fname):
        text = []
        clean = []
        labels = []
        file = open(fname, 'r');
        for line in file:
            #Source https://stackoverflow.com/questions/12883376/remove-the-first-word-in-a-python-string
            split =line.split(' ', 1)
            text += [split[1]]
            clean += [line]
            labels += [split[0]]
        return text, clean, labels
    
    # Simply returns true or false
    # When a value is 0 or 1
    def num_to_bool(self, val):
        if val == 1:
            return True
        elif val == 0:
            return False
        
    # Predicts each tweet based on traversing the tree
    def predict_tweet(self, tweet, tree, level):
        #if level == 0:
        if type(tree) == dict:
            curr_node = list(tree.keys())[0]
            if curr_node in self.words:
                ind = self.words.index(curr_node)
                if self.num_to_bool(tweet[ind]):
                    val = self.predict_tweet(tweet, tree[curr_node]['y'], level + 1)
                    return val
                else:
                    val = self.predict_tweet(tweet, tree[curr_node]['n'], level + 1)
                    return val
            else:
                val = self.predict_tweet(tweet, tree[curr_node]['n'], level + 1)
                return val
        else:
            return tree
    
    # Calls above methods to form a prediction
    def predict(self):
        predicted = []
        acc = 0
        corpus, full_lines, truth = self.read_data(self.t_file)
        vect = CountVectorizer(binary = True)
        csr_corp = vect.fit_transform(corpus) 
        test_vals = csr_corp.toarray()
        self.words = vect.get_feature_names()
        for i in range(test_vals.shape[0]):
            predicted.append(self.predict_tweet(test_vals[i, :], self.model, 0))
        for i in range(len(predicted)):
            if predicted[i] == truth[i]:
                acc +=1
        print('Decision Tree Accuracy:', acc/len(predicted))
        with open(self.o_file, 'w') as text_file:
            for i,l in enumerate(full_lines):
                text_file.write(predicted[i] + ' ' + l)
