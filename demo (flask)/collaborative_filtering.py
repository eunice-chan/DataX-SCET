import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

def similarity(x, y):
  return cosine_similarity(x.reshape(1,-1), y.reshape(1,-1))

def similarity_to_user(user, all_users):
  similarity_to_user_array = []
  for curr_user in all_users:
    similarity_to_user_array.extend(similarity(user, curr_user)[0])
  return similarity_to_user_array

def suggest_activity(N, user, all_users, activities_df, activity_score_name):
  # Score users by how similar they are to the given user.
  user_similarity_array = similarity_to_user(user, all_users)

  user_similarity_df = pd.DataFrame({"User Similarity": user_similarity_array, "UserID": np.array(all_users[:,-1].todense().reshape(1,-1).astype(np.uint64)).flatten()})
  df = user_similarity_df.merge(activities_df, on='UserID').drop('UserID', axis=1)

  # Multiply the similarity of the users with the activity score. 
  df['Activity Match'] = [a*b for a, b in zip(df['User Similarity'], df[activity_score_name])]

  # From the top N most similar rows, find the most common features (assuming features are independent). These belong to the "ideal" activity.
  # Based on the features of the "ideal" activity, score activities based on how similar the features are to the top features.
  # Return the ordered list of activities ranked by similarity to the "ideal" activity.
  # >>>> CURRENT: Return top from sorted list of activities.
  df = df.sort_values('Activity Match', ascending=False)
  top_N = df.iloc[:N]
  sorted_activities = top_N['Activity Name'].value_counts(sort=True, ascending=False)

  return sorted_activities.index