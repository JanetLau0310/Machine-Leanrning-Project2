# Machine-Leanrning-Project2
Sentiment Analysis via Bag-of-Words and Word2Vec

### About the Data 
You have been supplied with several thousand single-sentence reviews, collected from three domains: imdb.com, amazon.com, and yelp.com. Each review consists of a sentence, and has been assigned a binary label indicating the sentiment (1 for positive and 0 for negative) of that sentence. Your goal is to develop binary classiers that can generate the sentiment-labels for new sentences, automating the assessment process.<br>
The provided data consists of 2400 training examples. There are also 600 testing inputs, for which no y-values are given.

## Part One: Classifying Review Sentiment with Bag-of-Words Features
### Introducion
In this part the basic idea is to construct the Bag-of-Words Features for every review sentence in the training and testing data set. 
This NLP problem sometimes can be very complicated, but Bag-of-Words Model is the most traditional way to build the feature vector of every sentence.<br>
In this project, I will consider two vectorization way in sklearn: __CountVectorizer and TfidfVectorizer.__ Both of their idea is to count the frequency of every *word* in the specific data and then build the feature vector for the sentences. Then I used them separately into the models I generated, and applied the K-Fold Validation in each model.

Here are the preprocessing steps I did in CountVectorizer part.<br>
- Load the data from the data_reviews file. 
- Import the CountVectorizer package, and get the raw vectors of every sentence in train and test data set.
- Filter the words in sentences: such as filter the meaningless symbols ( like comma, etc ), the number, and the stopwords in nltk. Also, I changed them into the lower case.
- After than I can build a dictionary, the key is the vocabulary appear in the filtered data, and the value is its frequency ( >1 )
- Finally the new training and testing set I got have shape:<br>
**x_train.shape = (2400, 1795)**<br>
**x_test.shape = (600,1795)**<br>
_1795 is the dimension of the feature vector._
	
In TfidfVectorizer part:
- Import the TfidfVectorizer package
- Combine the text in training and the test set. Then use TfidfVectorizer to  fit them.
- Separate back into training and test sets. 
**x_train.shape = (2400, 5155)**<br>
**x_test.shape = (600,5155)**<br>
_Some justification for why you made the decisions you did._

It is clear to choose the CountVectorizer and TfidfVectorizer methods to build the feature vector. Both of them consider the frequency in the data set, and especially, tf-idf consider the different weight of different words.




## Part Two: Classifying Review Sentiment with Word Embeddings
### Introduction
We have provided a file containing pre-trained embedding vectors for 400,000 possible vocabulary words from GloVe. Each line of that file consists of a word, followed by a 50-value embedding vector for it. <br>
**Problem in this part is that word2vec only represent the vectors of words, not for the sentences. In order to do the sentiment analysis we need to know the feature vectors of every sentences.**<br>
There are some ways to gain the sentence embedding:<br>
- Count the avg of words' vectors in a sentence.
- Also we can also sum or concatenate the word vectors.
- Use the tf-idf as weight, count the Weighted Arithmetic Mean

Other solutions using supervised learning such as CNN. But doing this basic counting word is easier.After trying to use average / sum / tf-idf weight, I think the avg one seems to perform better than the other.

In this data processing part I do it roughly, just do a basic filter work and do the mathematical work.After filtering part I can miss the weird or meaningless words.

After creating the sentence embedding, we can apply them into the Model work. 

