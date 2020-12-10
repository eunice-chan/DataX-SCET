#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from flask import Flask, request, render_template, jsonify

import pandas as pd
import numpy as np
import json

import preprocess
import collaborative_filtering
import matching

# Support for gomix's 'front-end' and 'back-end' UI.
app = Flask(__name__, static_folder='public', template_folder='views')

# Features from CSV
deleted_features = ['Favorite movie genre?', 
                    'What qualities do you look for in a friend?', 
                    'Social distance run']
user_features = ['Age', 'Gender', 'College major?', 'How outdoorsy are you?',
                'When is your preferred time to hang out with friends?',
                'What is your preferred way of spending time with friends?',
                'How often do you like to spend time with your friends?',
                'How many people do you like to spend time with at once?',
                'What is your top love language?', 'Introvert or extrovert?']
activites_features = ['Hiking',
                      'Journaling',
                      'Reading nonfiction',
                      'Drawing',
                      'Hanging out with friends. (pre-COVID)',
                      'Social distance exercise (walk, run, etc.)', 
                      'Netflix party',
                      'Video chat hangout', 
                      'Wine tasting/Cocktail shake up',
                      'Trivia contests', 
                      'Virtual escape room', 
                      'Arts and crafts']
user_id = ['UserID']

# The databases (Pandas Dataframe). Currently stored in-memory because it's very small and time constraints.
user_data = pd.read_csv('https://cdn.glitch.com/5703bdb4-9be1-469e-8438-35964f70e9d6%2Fall_user_survey.csv?v=1607630616070', index_col=0)
all_activities = pd.read_csv('https://cdn.glitch.com/5703bdb4-9be1-469e-8438-35964f70e9d6%2Fall_activities.csv?v=1607598296969', index_col=0)


### WEB-APP STUFF ###
@app.after_request
def apply_kr_hello(response):
    """Adds some headers to all responses."""
  
    # Made by SCET Team. 
    if 'MADE_BY' in os.environ:
        response.headers["X-Was-Here"] = os.environ.get('MADE_BY')
    
    # Powered by Flask. 
    response.headers["X-Powered-By"] = os.environ.get('POWERED_BY')
    return response


@app.route('/')
def homepage():
    """Displays the homepage."""
    return render_template('index.html')
  
  
@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    """
    Simple API endpoint.
    """
    
    N = 10
    
    
    if request.method == 'GET':
    
      return render_template('results.html', N=N, u1=["Come back when you've filled out the survey!"], u2=["Come back when you've filled out the survey!"], match=["Come back when you've filled out the survey!"])
    
    if request.method == 'POST':
        
      vals = request.values.to_dict()
      
      # Hardcoded.
      uid = max(user_data["UserID"].astype(np.uint64))
      
      user1, user2, all_users = preprocess.preprocess_user_pair(vals, user_data, uid+1, uid+2)

      user1_ranked_pref = collaborative_filtering.suggest_activity(N, user1, all_users, all_activities, 'User Enjoyment Level')
      user2_ranked_pref = collaborative_filtering.suggest_activity(N, user2, all_users, all_activities, 'User Enjoyment Level')

      match = matching.match_prefs(user1_ranked_pref, user2_ranked_pref)
      
      

      return render_template('results.html', N=N, u1=matching.generalize(user1_ranked_pref), u2=matching.generalize(user2_ranked_pref), match=matching.generalize(match))

@app.template_filter()
def round(value):
    value = float(value)
    return "{:.5f}".format(value)
  
if __name__ == '__main__':
    app.run()