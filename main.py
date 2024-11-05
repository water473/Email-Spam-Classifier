import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 
import seaborn as sns
import nltk

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer



nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
df = pd.read_csv('spam_ham_dataset.csv')

print(df.head())
print(df.isnull().sum())

# Pre-processing
df['text'] = df['text'].str.lower()
df['text_punc'] = df['text'].str.replace('[^A-z ]','',regex=True).str.replace(' +',' ',regex=True).str.strip()

print(df['text_punc'].head())

# Tokenizes words (Separates them and puts them into a list)
df['text_punc_tok'] = df['text_punc'].apply(lambda x: word_tokenize(x))

# Gets rid of unnecessary words
stopwords = nltk.corpus.stopwords.words('english')
df['text_punc_tok_sw'] = df['text_punc_tok'].apply(lambda tokens: [word for word in tokens if word not in stopwords])
print(df['text_punc_tok_sw'].head())

# Reduces words to their root form
lemmatizer = WordNetLemmatizer()
df['text_punc_tok_sw_lemma'] = df['text_punc_tok_sw'].apply(lambda tokens: [lemmatizer.lemmatize(word) for word in tokens])
print(df['text_punc_tok_sw_lemma'].head())

# Split into training and testing data
y = np.array(df['label_num'])
df['text_punc_tok_sw_lemma'] = df['text_punc_tok_sw_lemma'].str.join(' ')
X = np.array(df['text_punc_tok_sw_lemma'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize Data
cw = CountVectorizer()
X_train = cw.fit_transform(X_train)
X_test = cw.transform(X_test)

# Train the model
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Accuracy Score
accuracy = accuracy_score(y_pred, y_test)
f1 = f1_score(y_pred, y_test, average="weighted")
print("The accuracy was: " + str(accuracy))
print("F1 Score:", str(f1))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()