Install the requirements file by runnning:
```python
pip install -r requirements.txt
```
To classify emails simply run: 
```python
python predict.py
```
This app requires permission to read emails and modify labels on your gmail.  

(1) It will ask for a gmail login id and password.  
(2) Grant modification permissions.  
(3) It will collect the latest emails and then classify them.  
(4) All travel emails will be labelled on the user's Gmail account.  
  
train.py implements multinomial naive bayes using scikit-learn. It implements binary classification and creates a model called 'travel.pkl'. train_tfidf.py does the same thing but it uses tf-idf vectorisation. However, I found that predictions were more accurate without it. I am only passing one feature (the email body) to the classifier. 
