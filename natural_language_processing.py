from nltk.corpus import stopwords
from nltk import PorterStemmer

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import confusion_matrix, f1_score

import pandas as pd

df = pd.read_table("data/SMSSpamCollection.txt", header=None)

labels = df[0]

# Convert labels to numbers
le = LabelEncoder()
labels_enc = le.fit_transform(labels)

raw_text = df[1]

# Lowercase the corpus
processed = raw_text.str.lower()

# Remove punctuation, white spaces
processed = processed.str.replace(r'[^\w\d\s]', ' ')
processed = processed.str.replace(r'\s+', ' ')
processed = processed.str.replace(r'^\s+|\s+?$', '')

# Remove stop words
stop_words = stopwords.words('english')
processed = processed.apply(lambda x: ' '.join(
    term for term in x.split() if term not in set(stop_words))
)

# Remove word stems using a Porter stemmer
porter = PorterStemmer()
processed = processed.apply(lambda x: ' '.join(
    porter.stem(term) for term in x.split())
)

# Construct a design matrix using an n-gram model and a tf-idf statistics
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
features = vectorizer.fit_transform(processed)

# Prepare the training and test sets using an 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    features,
    labels_enc,
    test_size=0.2,
    random_state=4,
    stratify=labels_enc
)

# Train SVM with a linear kernel on the training set
clf = svm.LinearSVC(loss='hinge')
clf.fit(X_train, y_train)

# Evaluate the classifier on the test set
y_pred = clf.predict(X_test)

# Compute the F1 score
print('F1 Score - {}'.format(f1_score(y_test, y_pred)))

# Display a confusion matrix
confusion_matrix = pd.DataFrame(
    confusion_matrix(y_test, y_pred),
    index=[['actual', 'actual'], ['spam', 'ham']],
    columns=[['predicted', 'predicted'], ['spam', 'ham']]
)
print(confusion_matrix)

# test against new messages 
def pred(msg):
    msg = vectorizer.transform([msg])
    prediction = svm.predict(msg)
    return prediction[0]