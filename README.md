# Fake News Detection

Fake News Detection in Python
 we have used various natural language processing techniques and machine learning algorithms to classify fake news articles using sci-kit libraries from python. 

### Prerequisites

What software we needed to install:

1. Python

2. Python ide=> (Anaconda or Google Colab)

3. Download and install below 3 packages:
   - Sklearn (scikit-learn)
   - numpy
   - scipy
   - pandas
   - matplotlib
   - re
   - nltk
   - seaborn



#### Dataset used
The data source used for this project is LIAR dataset which contains 3 files with .tsv format for test, train and validation. Below is some description about the data files used for this project.
  • id: unique id for a news article
  • title: the title of a news article
  • author: author of the news article
  • text: the text of the article; could be incomplete
  • label: a label that marks the article as potentially unreliable
          o 1: Fake
          o 0: Real
(20800  records in dataset)


#### Data cleaning
# Drop duplicates
display(news_dataset.drop_duplicates())
# replacing the null values with empty string
news_dataset = news_dataset.fillna('')
# merging the author name and news title
news_dataset['content'] = news_dataset['author']+' '+news_dataset['title']
# separating the data & label
X = news_dataset.drop(columns='label', axis=1)
Y = news_dataset['label']

#### FeatureSelection
 we have performed feature extraction and selection methods from sci-kit and nltklearn python libraries. For feature selection, we have used methods like term frequency tf-tdf weighting. for selection method, we have used train_test_split method.
**converting the textual data to numerical data**
vectorizer = TfidfVectorizer()
vectorizer.fit(X)
X = vectorizer.transform(X)

**Splitting the dataset to training & test data**
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify=Y, random_state=3)

#### classifier
 we have build Logistic Regression classifiers for predicting the fake news detection. 
 model = LogisticRegression()
 model.fit(X_train, Y_train)

#### prediction
Our finally best performing classifier **Logistic Regression**. It takes an news article as input from user then model is used for final classification output that is shown to user along with probability of truth.

X_new = X_test[1000]
prediction = model.predict(X_new)
print(prediction)
if (prediction[0]==0):
  print('The news is Real')
else:
  print('The news is Fake')

### Performance
**Logistic Regression Classifier**

model = LogisticRegression()
model.fit(X_train, Y_train)

### Accuracy
As we can see that our models accuracy score on the training data is 0.9859975961538462 and 0.9745192307692307 accuracy score on the training data.
# accuracy score on the training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score of the training data : ', training_data_accuracy)

# accuracy score on the test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score of the test data : ', test_data_accuracy)

## Data set link:
https://www.kaggle.com/competitions/fake-news/data
