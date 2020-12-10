import pandas as pd
import numpy as np

def match_prefs(pref1, pref2):
  pref1 = list(pref1)
  pref2 = list(pref2)
  ranked_pref = {}

  for i in pref1:
    if i in pref2:
      ranked_pref[i] = (pref1.index(i) + 1) * (pref2.index(i) + 1)

  sorted_activities = sorted(ranked_pref, key=ranked_pref.__getitem__)

  return sorted_activities

activity_to_features = {
  'HIKING':[1, 1, 1, 0, 0, 1, 1, 0],
  'JOURNALING':[0, 0, 1, 1, 1, 0, 0, 0],
  'READING NONFICTION':[0, 0, 1, 1, 0, 0, 0, 0],
  'DRAWING':[0, 0, 1, 0, 1, 0, 0, 0],
  'HANGING OUT WITH FRIENDS. (PRE-COVID)':[1, 1, 1, 0, 0, 1, 1, 1],
  'SOCIAL DISTANCE EXERCISE (WALK, RUN, ETC.)':[1, 1, 1, 0, 0, 1, 1, 0],
  'NETFLIX PARTY':[0, 0, 0, 0, 1, 1, 0, 0],
  'VIDEO CHAT HANGOUT':[0, 0, 1, 0, 0, 1, 1, 0],
  'WINE TASTING/COCKTAIL SHAKE UP':[0, 0, 1, 0, 1, 1, 1, 1],
  'TRIVIA CONTESTS':[0, 0, 0, 0, 1, 1, 1, 0],
  'VIRTUAL ESCAPE ROOM':[0, 0, 1, 0, 1, 1, 1, 0],
  'ARTS AND CRAFTS':[0, 0, 1, 0, 1, 1, 0, 0]
}

suggestions_from_features = {
  'PICNIC':[1, 0, 1, 0, 1, 1, 1, 1],
  'GROUP GAMES (AMONG US, CODE NAMES, ETC.)':[0, 0, 0, 0, 1, 1, 1, 0],
  'GRABBING FOOD OR DRINKS TOGETHER':[1, 0, 0, 0, 0, 0, 1, 1],
  'STUDY TOGETHER':[0, 0, 0, 1, 0, 1, 0, 0],
  'VIDEO GAMES':[0, 0, 0, 0, 1, 1, 1, 0],
  'COOKING/BAKING CLASS':[0, 0, 1, 0, 1, 1, 1, 1],
  'PAINTING SOCIAL':[1, 0, 0, 0, 1, 1, 1, 0],
  'BOOK CLUB':[0, 0, 0, 1, 1, 1, 1, 0],
  'KARAOKE':[0, 0, 0, 0, 1, 1, 1, 0],
  'COOKING/BAKING COMPETITION':[0, 0, 1, 0, 1, 1, 1, 1],
  'WORKOUT SESSION':[1, 1, 1, 0, 0, 1, 0, 0],
  'SELF-CARE SHEET MASK + TEA SESSION':[0, 0, 1, 0, 1, 0, 1, 1],
  'ONLINE SHOPPING SESSION':[0, 0, 0, 0, 1, 1, 0, 0]
}

def distance(x, y):
  return np.sqrt(np.sum((np.array(x) - np.array(y))**2))

def activity_similarity(activity):
  similarity = {}
  
  for suggestion in suggestions_from_features.keys():
    similarity[suggestion.title()] = distance(activity, suggestions_from_features[suggestion])
  
  ranked = sorted(similarity, key=similarity.__getitem__)
  return [(r, similarity[r]) for r in ranked]

def rank_new_activities(activities):
  """
  List of Series.
  """
  all = {}
  for ranking in activities:
    for activity, rank in ranking:
      if activity in all:
        all[activity] = max(all[activity], rank)
      else:
        all[activity] = rank
  ranked = sorted(all, key=all.__getitem__)
  return [(r, all[r]) for r in ranked]

def generalize(activities):
  return rank_new_activities([activity_similarity(activity_to_features[activity.strip().upper()]) for activity in activities])
