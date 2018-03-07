import os
import numpy
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib

NEWLINE = '\n'

TRAVEL = 'Travel'
OTHER = 'Other'

SOURCES = [
    ('data/travel',    TRAVEL),
    ('data/other',    OTHER),
]

SKIP_FILES = {'cmds'}

def read_files(path):
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
  data_frame = DataFrame({'text': [], 'class': []})
  for file_name, text in read_files(path):
    data_frame = data_frame.append(
        DataFrame({'text': [text], 'class': [classification]}, index=[file_name]))
  return data_frame

data = DataFrame({'text': [], 'class': []})
for path, classification in SOURCES:
    data = data.append(build_data_frame(path, classification))

data = data.reindex(numpy.random.permutation(data.index))

count_vectorizer = CountVectorizer()
counts = count_vectorizer.fit_transform(numpy.asarray(data['text']))

classifier = MultinomialNB()
targets = numpy.asarray(data['class'])
classifier.fit(counts, targets)

pipeline = Pipeline([
    ('vectorizer',  CountVectorizer()),
    ('classifier',  MultinomialNB()) ])

pipeline.fit(data['text'].values, data['class'].values)
joblib.dump(pipeline, 'travel.pkl')

