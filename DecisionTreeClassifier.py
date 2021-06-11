####################################
# CS B551 Fall 2020, Assignment #4
#
# Your names and user ids:
# Cody Harris (harrcody)
#
# DecisionTreeClassifier.py
# Builds a Decision Tree

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import time
import pickle

class DecisionTreeClassifier:
    def __init__(self, train_file, out_file, min_df, max_df, max_depth):
        self.train_file = train_file
        self.out_file = out_file
        self.min_df = min_df
        self.max_df = max_df
        self.max_depth = max_depth
        self.cities = []
        self.dtree = {}
    
    # Save tree to a pickle file
    def pickle_tree(self):
        model = {'type': 'dtree', 'tree': self.dtree}
        #source: https://stackoverflow.com/questions/11218477/how-can-i-use-pickle-to-save-a-dict
        with open(self.out_file, 'wb') as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Read in Training Data
    def read_data(self, fname):
        text = []
        labels = []
        file = open(fname, 'r');
        for line in file:
            #Source https://stackoverflow.com/questions/12883376/remove-the-first-word-in-a-python-string
            split =line.split(' ', 1)
            text += [split[1]]
            labels += [split[0]]
        self.cities = list(set(labels))
        return text, labels
    
    # Takes a column of 1s and 0s for a given word
    # and computes the entropy of that word
    def calc_entropy(self, truth, labels):
        labels = np.array(labels)
        total = truth.shape[0]
        nz_ind = truth.getcol(0).nonzero()[0]
        z_ind = [i for i in range(len(labels)) if i not in nz_ind]
        nz_labs = labels[nz_ind]
        z_labs = labels[z_ind]
        nz_tot = len(nz_labs)
        z_tot = len(z_labs)
        nz_entropy = 0
        z_entropy = 0
        nz_unique, nz_counts = np.unique(nz_labs, return_counts=True)
        z_unique, z_counts = np.unique(z_labs, return_counts=True)
        nz_dict = dict(zip(nz_unique, nz_counts))
        z_dict = dict(zip(z_unique, z_counts))
        for city in self.cities:
            if city in nz_dict.keys():
                nz_city_count = nz_dict[city]
                nz_entropy += -(nz_city_count/nz_tot) * np.log2(nz_city_count/nz_tot)
            if city in z_dict.keys():
                z_city_count = z_dict[city]
                z_entropy += -(z_city_count/z_tot) * np.log2(z_city_count/z_tot)
        entropy = (nz_tot/total) * nz_entropy + (z_tot/total) * z_entropy
        return entropy
    
    # Loops through all words and finds the word with the lowest entropy
    def get_split(self, corpus, labels, words):
        best_entropy = 999999999999999
        best_index = -1
        for i, word in enumerate(list(words)):
            curr_entropy = self.calc_entropy(corpus.getcol(i), labels)
            if curr_entropy < best_entropy:
                best_entropy = curr_entropy
                best_index = i
        return best_index
    
    # Finds the location with the highest count in a list
    # Source: https://stackoverflow.com/questions/6252280/find-the-most-frequent-number-in-a-numpy-array
    def highest_count(self, a):
        u, indices = np.unique(a, return_inverse=True)
        return u[np.argmax(np.bincount(indices))]
    # Stop Copy

    # Builds a dictionary based on splitting on words that have the lowest entropy        
    def build_tree(self, corpus, labels, words, depth):
        split_ind = self.get_split(corpus, labels, words)
        nz_ind = corpus.getcol(split_ind).nonzero()[0]
        z_ind = [i for i in range(len(labels)) if i not in nz_ind]
        remove_ind = [i for i in range(len(words)) if i != split_ind]
        corpus = corpus[:, remove_ind]
        split_word = words[split_ind]
        del words[split_ind]
        if len(nz_ind) > 0 and len(z_ind) > 0:
            tree = {split_word: {'y': {}, 'n':{}}}
            for i in ['y', 'n']:
                if i == 'y':
                    if len(set(labels[nz_ind])) != 1 and depth > 0:
                        tree[split_word][i] = self.build_tree(corpus[nz_ind, :], labels[nz_ind], words, depth - 1)
                    else:
                        tree[split_word][i] = self.highest_count(labels[nz_ind])
                elif i == 'n':
                    if len(set(labels[z_ind])) != 1 and depth > 0:
                        tree[split_word][i] = self.build_tree(corpus[z_ind, :], labels[z_ind], words, depth - 1)
                    else:
                        tree[split_word][i] = self.highest_count(labels[z_ind])
        else:
            tree = {}
            tree[split_word] = self.highest_count(labels)
                    
        return tree
    
    # Prints the top three layers of the tree
    def print_tree(self, d, depth, offset = 0):
        if depth >= 0:
            if offset == 0:
                print('Top 3 levels of trained tree')
                to_print = list(d.keys())[0]
                print('+-', "'", to_print, "'", sep = '')
                self.print_tree(d[to_print], depth-1, offset = offset + 1)
            elif offset == 1:
                for i in ['y', 'n']:
                    if type(d[i]) == dict:
                        key = list(d[i].keys())[0]
                        print('| +-', "'", key, "'", sep = '')
                        self.print_tree(d[i][key], depth-1, offset = offset + 1)
                    else:
                        print('| +-', d[i], sep = '')
            elif offset == 2:
                for i in ['y', 'n']:
                    if type(d[i]) == dict:
                        key = list(d[i].keys())[0]
                        print('|   +-', "'", key, "'", sep = '')
                        self.print_tree(d[i][key], depth-1, offset = offset + 1)
                    else:
                        print('|   +-', d[i], sep = '')
            elif offset == 3:
                for i in ['y', 'n']:
                    if type(d[i]) == dict:
                        key = list(d[i].keys())[0]
                        print('|     +-', "'", key, "'", sep = '')
                        self.print_tree(d[i][key], depth-1, offset = offset + 1)
                    else:
                        print('|     +-', d[i], sep = '')            
        else:
            return
    # Calls above functions to fit the Decision Tree Model
    def fit(self):
        corpus, labs = self.read_data(self.train_file)
        vect = CountVectorizer(min_df = self.min_df, max_df = self.max_df, binary = True)
        csr_corp = vect.fit_transform(corpus)   
        self.dtree = self.build_tree(csr_corp, np.array(labs), vect.get_feature_names(), self.max_depth)
        self.print_tree(self.dtree, 3)