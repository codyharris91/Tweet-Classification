###################################
# CS B551 Fall 2020, Assignment #4
#
# Your names and user ids:
# Cody Harris (harrcody)
#
# BayesNetClassifier.py
# Builds probability dictionaries
# That will be used in a bayes net
# Prediction

import time
import pickle

class BayesNetClassifier:
    def __init__(self, train_file, out_file):
        self.train_file = train_file
        self.out_file = out_file
        self.word_count_loc = {}
        self.word_probs = {}
        self.l_probs = {}
        self.word_counts = {}
        self.common_words = {}
        self.cities = []
        self.total_words = 0
        
    # Saves probabilites to a pickle file
    def pickle_probs(self):
        all_probs = {'type': 'bayes', 'location': self.l_probs, 'words': self.word_probs, 'total': self.total_words, 'cities': self.cities}
        #source: https://stackoverflow.com/questions/11218477/how-can-i-use-pickle-to-save-a-dict
        with open(self.out_file, 'wb') as handle:
            pickle.dump(all_probs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Loads training data
    # Some ideas taken from label.py given by Dr. Crandall    
    def read_data(self, fname):
        exemplars = []
        file = open(fname, 'r');
        for line in file:
            data = tuple([w if i == 0 else w.lower() for i, w in enumerate(line.split())])
            exemplars += [data]
        return exemplars
    
    # Source: https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
    # Fastest way to find max value in dict
    def max_val(self, d, c): 
        top = {}
        for i in range(c):
            v=list(d.values())
            k=list(d.keys())
            max_key = k[v.index(max(v))]
            top[max_key] = max(v)
            del d[max_key]
        return top
    # Stop Copy
    
    # Pretty print of top 5 words per location
    def print_values(self, d):
        for key in d.keys():
            print('{:<20}'.format(key.replace('_',' ')), end = '')
            for i, val in enumerate(d[key]):
                print('{:<20}'.format(val), end = '')
            print()
    
    # Fits the Bayes Net model
    def fit(self):
        t0 = time.time()
        tweets = self.read_data(self.train_file)
        total_tweets = len(tweets)
        words_per_city = {}
        for twe in tweets:
            city = twe[0]
            if city in self.l_probs.keys():
                self.l_probs[city] += 1/total_tweets
            else:
                self.l_probs[city] = 1/total_tweets
                words_per_city[city] = 0
            for word in twe[1:]:
                if city in self.word_count_loc.keys():
                    if word in self.word_count_loc[city].keys():                    
                        self.word_count_loc[city][word] += 1
                        words_per_city[city] +=1
                    else:
                        self.word_count_loc[city][word] = 1
                        words_per_city[city] +=1
                else:
                    self.word_count_loc[city] = {}
                    self.word_count_loc[city][word] = 1
                    words_per_city[city] +=1
                    
                if word in self.word_counts.keys():
                    self.word_counts[word] += 1
                else:
                    self.word_counts[word] = 1
                    
        #source: https://stackoverflow.com/questions/17095163/remove-a-dictionary-key-that-has-a-certain-value
        self.common_words = {k:v for k,v in self.word_counts.items() if v >= 5}
        self.cities = list(self.word_count_loc.keys())
        self.total_words = len(self.word_counts.keys())
        #print(self.max_val(self.word_counts,30))
        for city in self.cities:
            self.word_probs[city] = {w: c/words_per_city[city] for w,c in self.word_count_loc[city].items()}
            
    # Find top 5 words by location
    def top_words(self):
        most_pop = {}
        top_five = {}
        for city in self.cities:
            self.word_count_loc[city] = {k: v for k, v in sorted(self.word_count_loc[city].items(), key=lambda x: x[1], reverse = True)}
            most_pop[city] = {k:v/self.common_words[k]  for k,v in self.word_count_loc[city].items() if k in self.common_words.keys()}
            top_five[city] = self.max_val(most_pop[city], 5)
        print()
        print('Top 5 Words Per Location')
        print('-------------------------------------------------------------------------------------------------------------------')
        self.print_values(top_five)