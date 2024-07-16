# Text Classification and Word Embeddings Project

## Project Overview

This project is divided into two main parts:

1. **Text Classification**:
    - Implemented using a Rule-Based Classifier.
    - Implemented using a Bag of Words Classifier.

2. **Word Embeddings**:
    - Implemented using the Skip-gram model of Word2Vec.

## Part 1: Text Classification

### Rule-Based Classifier

A rule-based classifier uses a set of manually crafted rules to classify text data. This approach relies on domain knowledge and specific patterns identified in the text to make predictions.

### Bag of Words Classifier

The Bag of Words (BoW) model is a simple and commonly used method in natural language processing. It transforms text into a fixed-size vector by counting the frequency of each word in the document, ignoring grammar and word order but keeping multiplicity.

### Steps to Implement Text Classification

1. **Preprocessing**:
    - Tokenization: Splitting the text into individual words.
    - Lowercasing: Converting all text to lowercase to maintain uniformity.
    - Removing stop words: Removing common words that do not contribute much to the meaning.

2. **Rule-Based Classifier**:
    - Define rules based on the domain knowledge.
    - Apply these rules to classify the text.

3. **Bag of Words Classifier**:
    - Create a vocabulary of all unique words in the training dataset.
    - Convert each document into a vector based on word frequency.
    - Train a machine learning model (e.g., Naive Bayes, Logistic Regression) on these vectors.

## Part 2: Word Embeddings using Skip-gram Model

### Word2Vec

Word2Vec is a popular technique to learn word embeddings, which are dense vector representations of words. The meaning of a word is determined by the context in which it occurs, and words with similar meanings have similar representations.

### Architectures of Word2Vec

1. **Continuous Bag of Words (CBOW)**:
    - Predicts the center word from the surrounding context words.

2. **Skip-gram**:
    - Predicts surrounding context words from the center word.
    - Implemented in this project.

### Skip-gram Model

The Skip-gram model is designed to predict the context words for a given center word. It learns word representations by maximizing the probability of the context words given a center word.

### Steps to Implement Skip-gram Model

1. **Preprocessing**:
    - Tokenization: Splitting the text into individual words.
    - Building the vocabulary: Creating a dictionary of all unique words in the dataset.
    - Generating training examples: Creating pairs of center words and context words.

2. **Training the Model**:
    - Define the neural network architecture for the Skip-gram model.
    - Train the model using the generated training examples.
    - Optimize the model parameters to learn the word embeddings.


## Requirements

- Python 3.x
- Libraries: numpy, pandas, scikit-learn, gensim, nltk

## References

- [Word2Vec Explained](https://www.tensorflow.org/tutorials/text/word2vec)
- [Text Classification with Scikit-Learn](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)

## Acknowledgements

- Thanks to the developers of the libraries used in this project.
- Special thanks to the authors of the datasets used for training and evaluation.

---

Feel free to explore the code and experiment with different configurations to improve the models! If you encounter any issues or have suggestions, please open an issue or submit a pull request.

Happy coding!

