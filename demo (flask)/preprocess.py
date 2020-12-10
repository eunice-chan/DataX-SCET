import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

one_hot_columns =  ['Gender', 'College major?', 'How outdoorsy are you?',
                    'When is your preferred time to hang out with friends?',
                    'What is your preferred way of spending time with friends?',
                    'How often do you like to spend time with your friends?',
                    'How many people do you like to spend time with at once?',
                    'What is your top love language?', 'Introvert or extrovert?']

def preprocess_users(data):
  
  columns = list(data.columns)
  
  cat_indices = []
  for col in one_hot_columns:
    cat_indices += [columns.index(col)]

  data_pipeline = ColumnTransformer([
      ('categorical', OneHotEncoder(), cat_indices)
  ], remainder='passthrough')

  return data_pipeline.fit_transform(data)


def preprocess_user_pair(args, user_data, User1ID, User2ID):

  user1 = {
          'Age':int(args.get('age1') if args.get('age1') else "0"),
          'Gender':args.get('gender1', ""),
          'College major?':args.get('major1', ""),
          'How outdoorsy are you?':args.get('outdoor1', ""),
          'When is your preferred time to hang out with friends?':args.get('time1', ""),
          'What is your preferred way of spending time with friends?':args.get('hanging1', ""),
          'How often do you like to spend time with your friends?':args.get('freq1', ""),
          'How many people do you like to spend time with at once?':args.get('size1', ""),
          'What is your top love language?':args.get('lovelang1', ""),
          'Introvert or extrovert?':args.get('vert1', ""),
          'UserID':User1ID
          }
  
  
  user2 = {
          'Age':int(args.get('age2') if args.get('age2') else "0"),
          'Gender':args.get('gender2', ""),
          'College major?':args.get('major2', ""),
          'How outdoorsy are you?':args.get('outdoor2', ""),
          'When is your preferred time to hang out with friends?':args.get('time2', ""),
          'What is your preferred way of spending time with friends?':args.get('hanging2', ""),
          'How often do you like to spend time with your friends?':args.get('freq2', ""),
          'How many people do you like to spend time with at once?':args.get('size2', ""),
          'What is your top love language?':args.get('lovelang2', ""),
          'Introvert or extrovert?':args.get('vert2', ""),
          'UserID':User2ID
          }
  
  all_users = user_data.append(user1, ignore_index=True)
  all_users = all_users.append(user2, ignore_index=True) 
  
  all_users[one_hot_columns].fillna('', inplace=True)
  all_users[["UserID", "Age"]].fillna(0, inplace=True)
  
  all_users = preprocess_users(all_users)
  
  return all_users[User1ID].todense(), all_users[User2ID].todense(), all_users