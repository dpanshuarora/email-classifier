import sys
from auth import *
import urllib.request as urllib2
from googleapiclient.errors import HttpError
from os import path
import io
import imaplib

import csv
import string

import email
import base64
from html.parser import HTMLParser
import html2text

from sklearn.externals import joblib
import numpy as np # linear algebra
import pandas as pd

#Global variables to store message ids and messages
msg_id_list=[]
msg_body_list=[]

def GetMimeMessage(service, user_id, msg_id):
  """Get a Message and use it to create a MIME Message.

  Args:
    service: Authorized Gmail API service instance.
    user_id: User's email address. The special value "me"
    can be used to indicate the authenticated user.
    msg_id: The ID of the Message required.

  Returns:
    A MIME Message, consisting of data from Message.
  """
  try:
    message = service.users().messages().get(userId=user_id, id=msg_id,
                                             format='raw').execute()

    h = html2text.HTML2Text()
    h.ignore_links = True

    msg_str = base64.urlsafe_b64decode(message['raw'].encode('ASCII'))
    mime_msg = email.message_from_bytes(msg_str)
    body = ''

    #Iterate through multipart emails
    if mime_msg.is_multipart():
            for part in mime_msg.walk():
                if part.is_multipart():
                    for subpart in part.get_payload():
                        if subpart.is_multipart():
                            for subsubpart in subpart.get_payload():
                                body = body + str(subsubpart.get_payload(decode=True)) + '\n'
                        else:
                            body = body + str(subpart.get_payload(decode=True)) + '\n'
                else:
                    body = body + str(part.get_payload(decode=True)) + '\n'
    else:
        body = body + str(mime_msg.get_payload(decode=True)) + '\n'

    body = bytes(body,'utf-8').decode('unicode-escape')
    body = h.handle(body)
    #print(body)
    msg_id = str(msg_id)
    global msg_id_list
    global msg_body_list
    msg_id_list.append(msg_id)
    msg_body_list.append(body)

  except HttpError as error:
    print('An error occurred: %s' % error)
    return None

def ListLabels(service, user_id):
  """Get a list all labels in the user's mailbox.

  Args:
    service: Authorized Gmail API service instance.
    user_id: User's email address. The special value "me"
    can be used to indicate the authenticated user.

  Returns:
    A list all Labels in the user's mailbox.
  """
  try:
    response = service.users().labels().list(userId=user_id).execute()
    labels = response['labels']
    for label in labels:
      print('Label id: %s - Label name: %s' % (label['id'], label['name']))
    return labels
  except errors.HttpError as error:
    print('An error occurred: %s' % error)
    return None

def StoreEmails(msg_body_list):
    """Store all collected emails in separate files in the directory Data/emails.
    """
    count = 0
    file_path = path.relpath("data/emails")
    for msg_body in msg_body_list:
        with open(file_path + "/" + str(count) + ".txt", "a", encoding = 'utf-8') as fp:
            fp.write(msg_body)
        count = count+1


def UpdateLabels(service, predictions_df):
    """Updates labels of travel related emails on Gmail
    """
    print("\n\nScanning all emails and updating labels")
    for row in predictions_df.itertuples():
        if(row[1]!='Other'):
            message_id = row[2]
            message = service.users().messages().modify(userId='me', id=message_id, body=
            {
            "addLabelIds": ["Label_3"] #Label_3 is the Label ID for Travel label in Gmail
            }
            ).execute()
            #print(row)
    print("\n\nLabels for travel related emails have been updated.")    

def main():

    """Creates a Gmail API service object 
    """
    credentials = get_credentials()
    http = credentials.authorize(httplib2.Http())
    service = discovery.build('gmail', 'v1', http=http)

    """#Uncomment this to output a list of label names of the user's Gmail account.
    ListLabels(service,'me')
    """
   
    messages = service.users().messages().list(userId='me', labelIds = "INBOX").execute().get('messages', [])
    #INBOX is the label ID for inbox. This can be changed.
    print("Collecting all emails." + "\n" + "This may take a while...")

    print ("Total messages in inbox: ", str(len(messages)))

    for message in messages:
        GetMimeMessage(service,'me',message['id'])
    
    clf = joblib.load('travel.pkl') #Load previously trained model for travel emails.
    global msg_id_list
    global msg_body_list

    #Uncomment this to store all collected emails to Data/emails.
    StoreEmails(msg_body_list)

    nparray_msg_body = np.array(msg_body_list)
    predictions = clf.predict(nparray_msg_body)
    predictions_df = pd.DataFrame({'Predictions': predictions, 'id': msg_id_list})

    UpdateLabels(service, predictions_df)        


if __name__ == '__main__':
    main()