import os
import numpy
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib

NEWLINE = '\n'

TRAVEL = 'Travel'
OTHER = 'Other'

SOURCES = [
    ('data/travel',    TRAVEL),
    ('data/other',    OTHER),
]

SKIP_FILES = {'cmds', '.DS_Store'}

SEED = 0 # for reproducibility

# Class to extend the Sklearn classifier
class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        #params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
    def fit(self,x,y):
        return self.clf.fit(x,y)

    def score(self,x,y):
        return self.clf.fit(x,y)
    
    def feature_importances(self,x,y):
        print(self.clf.fit(x,y).feature_importances_)
    
#Model Parameters    
svc_params = {
    'kernel': 'linear',
    'C': 1
}    

nb_params = {
    'alpha':2
}

#Model Objects
svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)
nb = SklearnHelper(clf=MultinomialNB, seed=SEED, params =nb_params)



def read_files(path):
    #Reads all files in all directories mentioned in SOURCES
    for root, dir_names, file_names in os.walk(path):
        for path in dir_names:
            read_files(os.path.join(root, path))
        for file_name in file_names:
            if file_name not in SKIP_FILES:
                file_path = os.path.join(root, file_name)
                if os.path.isfile(file_path):
                    past_header, lines = False, []
                    f = open(file_path, encoding="latin-1")
                    for line in f:
                        if past_header:
                            lines.append(line)
                        elif line == NEWLINE:
                            past_header = True
                    f.close()
                    content = NEWLINE.join(lines)
                    yield file_path, content

def build_data_frame(path, classification):
    #Returns a data frame of all the files read using read_files()
  data_frame = DataFrame({'text': [], 'class': []})
  for file_name, text in read_files(path):
    data_frame = data_frame.append(
        DataFrame({'text': [text], 'class': [classification]}, index=[file_name]))
  return data_frame

data = DataFrame({'text': [], 'class': []})
for path, classification in SOURCES:
    data = data.append(build_data_frame(path, classification))

data = data.reindex(numpy.random.permutation(data.index))

#Training data
X_train = numpy.asarray(data['text'])
X_test = numpy.asarray(data['class'])
count_vectorizer = CountVectorizer()
counts = count_vectorizer.fit_transform(X_train)
targets = X_test
clf = nb.fit(counts, targets)
clf_svc = svc.fit(counts, targets)

X_train, X_test, y_train, y_test = train_test_split(
    X_train, X_test, test_size=0.4, random_state=0)

X_test = count_vectorizer.transform(X_test)
print(clf.score(X_test, y_test))
print(clf_svc.score(X_test, y_test))

pipeline = Pipeline([
    ('vectorizer',  CountVectorizer()),
    ('classifier',  nb) ])

pipeline.fit(data['text'].values, data['class'].values)
joblib.dump(pipeline, 'travel.pkl')