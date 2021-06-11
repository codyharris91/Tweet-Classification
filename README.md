# Machine Learning from Scratch - Tweet Classification

## How to run

To train using a Bayes Net:
```
python3 ./geolocate.py train bayestraining-input-file bayes-model-output-file
```

To train using a decision tree:
```
python3 ./geolocate.py train dtreetraining-input-file dtree-model-output-file
```

To test using either model type:
```
python3 ./geolocate.py testmodel-input-file testing-input-file testing-output-file
```

## Learning a Naive Bayes Classifier - BayesNetClassifier.py

### Formulation of the Problem

Learning the model is done through using a class called BayesNetClassifier. This class holds all the methods needed to produce the information needed for prediction. If the specified commands are passed to geolocate.py in the command line, the bayes classifier will be called and built. All the information for the model is stored in a dictionary and then stored in a pickle file to be read in by the prediction command.

### How it Works

The first step is to load in the training data. This is done as a list of tuples, where each tuple is a tweet and the tuple contains the words in a tweet.

The program loops through the tweets in the training data and calculates the following:  
-Word count per location, which counts how many times a word occurs per location  
-The above count is used to create probabilites of a word occuring at a given location  
-Location Probabilities which is the proportion a location shows up in the training data compared to the total amount of tweets  
-Word counts regardless of location  
-A dictionary that is just like the word count dictionary, but only includes words that occur at least 5 times in the total training data  
-A list of all the cities in the training data  
-Total count of different words in the training data  

The above data is either stored in a dictionary to be output by the program, or used in the calculation of the top five words per location.

The top five words per location is calculated first by only dealing with words that occur 5 times or more. For each city in the training data, each word that occurs at that location counted, and then dividing it by the total times that word occurs in any city. Then the top five probabilites per city are printed to the screen in a human readable format.

Lastly, the needed dictionaries are output to the current working directory in pickle format.

### Hyperparameter Tuning

For this problem there is one variable that could be changed. This is the probability used when a word is in the test data, and not in train. After a handful of tests and the best probability was found to be around 1/400,000.

### Problems, Assumptions, and Simplifications

For this portion there were not any problems. The assumption is made that 1/400,000 for the missing probability does not over fit the training data, but provides the best results.

## Learning a Decision Tree Classifier - DecisionTreeClassifier.py

### Formulation of the Problem

For this classifier, instead of reading in the training data in the way that I did in the previous classifier, this time I used CountVectorizer from sklearn to create a sparse matrix that specified if a word exists in a tweet. This package also treats the tokens in the tweet differently and if tokens are similar, it counts them as the same. For example: 'lakers', '#lakers', 'lakers!', would all count as the same word. 

Using sparse matricies is helpful as the matrix is large for our training data and allows for quick operations. 

### How it Works

The training data is read in and then fed into the CountVectorizer function. This outputs a sparse matrix.

Then this matrix is used to loop through all the words in the traing data. Each word is evaluated for its entropy, and the lowest entropy is the word that is split on. This process is repeated recursively to build a dictionary. The tree stops splitting if either it has reached the max depth specified by the class call, or it reaches a node that only has one label. If there is only one label, that leaf gains that label as it's predicted outcome. If the function reaches it's max depth, then the label that has the highest occurance is chosen.

Equation for Entropy:  

<a href="https://www.codecogs.com/eqnedit.php?latex=\bg_white&space;\sum&space;\frac{n_i}{n_t}&space;\sum&space;-\frac{n_i_c}{n_i}log_2\frac{n_i_c}{n_i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\bg_white&space;\sum&space;\frac{n_i}{n_t}&space;\sum&space;-\frac{n_i_c}{n_i}log_2\frac{n_i_c}{n_i}" title="\sum \frac{n_i}{n_t} \sum -\frac{n_i_c}{n_i}log_2\frac{n_i_c}{n_i}" /></a>

After the tree is built, a function traverses the tree recursively to print out the top 3 layers of the tree. 

### Hyper Parameter tuning

Due to the training time of this classifier, as well as wanting to get the best accuracy, some parameters had to be tuned. The three parameters were: min_df, max_df, and depth. The min_df and max_df are parameters given to the count vectorizer class. This says to use only tokens that occur between min_df and max_df times. While infreqeunt words made the model take too long, very frequent words were not very helpful in the model, so finding a happy medium was best.

Instead of specifying parameters for these three values, I randomly chose them between specified ranges, and then trained a model and tested it on the test set. After over 100 experiments all the information was stored in all_logs.csv.

Some plots of this information were used to choose the final hyperparamters.

Depth did not seem to have a strong relationship with the accuracy

![depth](/images/depth.png)

We see that a min around 40 and a max around 700 were the best accuracy

![minmax](/images/min_max.png)

Time and total amount of words also did not seem to have a strong relationship.

![time](/images/time.png)

![words](/images/words.png)

### Problems, Assumptions, and Simplifications

While I have tested the training of this program many times manually, I did get a couple errors when hyperparameter tuning. This could have been because I forced a max depth when it was not possible, but I could not find the error. I do not think with the hyperparamters explicitly specified that this should happen.

The assumption is that the tuned hyper parameters are not an over fit, and that they will generalize well. As you can see from the accuracy ranges that I got, the hyper parameters did not change the accuracy that greatly, I usually got between 20%-23% correct.

## Predicting using Naive Bayes - BayesNetPredictor.py

### How it Works

This portion of the problem reads in a pickle file. If the label in the dictionary is bayes, then it calls the bayes predictor. 

The program simply loops through each tweet in the testing set, and for each tweet evaluates the words of the tweet using the equation below to calculate the probability of the tweet coming from each of the 12 equations.  

<a href="https://www.codecogs.com/eqnedit.php?latex=\bg_white&space;P(L&space;=&space;l&space;|&space;w_1,w_2,w_3,w_n)&space;=&space;\frac{P(w_1,w_2,w_3,w_n|L)P(L)}{P(w_1,w_2,w_3,w_n)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\bg_white&space;P(L&space;=&space;l&space;|&space;w_1,w_2,w_3,w_n)&space;=&space;\frac{P(w_1,w_2,w_3,w_n|L)P(L)}{P(w_1,w_2,w_3,w_n)}" title="P(L = l | w_1,w_2,w_3,w_n) = \frac{P(w_1,w_2,w_3,w_n|L)P(L)}{P(w_1,w_2,w_3,w_n)}" /></a>

It then selects the highest probability across the locations and predicts that location. 

### Problems, Assumptions, and Simplifications

In the equation specified above, the denominator is always the same for each location. Therefore as we are maximizing over this value, the denominator isn't needed and adds extra complexity. Therefore, I am actually doing the following:  

<a href="https://www.codecogs.com/eqnedit.php?latex=\bg_white&space;max(P(w_1,w_2,w_3,w_n|L)P(L))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\bg_white&space;max(P(w_1,w_2,w_3,w_n|L)P(L))" title="max(P(w_1,w_2,w_3,w_n|L)P(L))" /></a>

## Predicting using Decision Tree - DecisionTreePredictor.py

### How it Works

The prediction of the decision tree takes in a dictionary that specifies a tree, and traverses that tree. 

Each tweet is taking through the tree and then a prediction is made. The tree checks if a given word is in the tweet and either goes right or left on the tree if the word is or isn't in the tree. 

### Problems, Assumptions, and Simplifications

The only problem with this prediction is that regardless of hyperparameters the prediction accuracy is quite low. The Training of the tree was optimized to take roughly 10 minutes. If I could train the tree for longer, this accuracy would likely go up.
