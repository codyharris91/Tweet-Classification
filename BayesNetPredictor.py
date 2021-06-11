import numpy as np

class BayesNetPredictor:
    def __init__(self, model, test_file, out_file):
        self.t_file = test_file
        self.o_file = out_file
        self.loc = model['location']
        self.w_probs = model['words']
        #self.missing_prob = 1 / model['total']
        self.missing_prob = 1 / 400000
        self.cities = model['cities']
    
    # Loads Testing Data
    # Some ideas taken from label.py given by Dr. Crandall    
    def read_data(self, fname):
        exemplars = []
        text = []
        file = open(fname, 'r');
        for line in file:
            data = tuple([w if i == 0 else w.lower() for i, w in enumerate(line.split())])
            text += [line]
            exemplars += [data]
        return exemplars, text
    
    # Source: https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
    # Fastest way to find max value in dict
    def max_prob(self, d):
         v=list(d.values())
         k=list(d.keys())
         return k[v.index(max(v))]
    # Stop Copy
    
    # Predicts Location    
    def predict(self):
        true = []
        predicted = []
        correct = 0
        words, lines = self.read_data(self.t_file)
        for twe in words:
            true.append(twe[0])
            probs = {}
            for city in self.cities:
                probs[city] = np.prod([self.w_probs[city][w] if w in self.w_probs[city].keys() else self.missing_prob for w in twe[1:]]) * self.loc[city]
            predicted.append(self.max_prob(probs))
        for i in range(len(true)):
            if true[i] == predicted[i]:
                correct += 1
        print('Predicted Bayes Net Accuracy: ', correct/len(true))
        with open(self.o_file, 'w') as text_file:
            for i,l in enumerate(lines):
                text_file.write(predicted[i] + ' ' + l)
    
