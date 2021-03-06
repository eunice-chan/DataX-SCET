# Project Overview

## Problem
  It has always been difficult to maintain and create new interpersonal relationships. However, now due to quarantine, the typical channels of hanging out with your friends, having study groups, and bumping into one another are closed off due to social distancing. So, more than ever, college students find themselves isolated and as a result, their mental health suffers. Social media, despite its name, does not provide a very good job at encouraging people to keep connected or establish new connections on their platform. 
  
 ## Solution
  Our solution creates a social media platform that utilizes data signals and behavioral economics to nudge users to interact with and deepen the bonds with their community. Our nudges will include activities that people should do with a specific person that they are friends wth or are in a campus organization with. To create our model, we tried two different methods including supervised learning, and collaborative filtering.  
  
 # Repo Overview
| Filename                            | Description |
| :---------------------------------: | :---------: | 
| ipynb/Supervised_Models.ipynb       | Initial formulation as a supervised learning task using a Support Vector Multi-Output Regression and a Decision Tree Multi-Output Regression  | 
| ipynb/Collaborative_Filtering.ipynb | Second solution attempt using collaborative filtering. |
| ipynb/Future_Direction.ipynb        | Prototype of how it would interact with live events (a future direction). |
| flask (demo)/*                      | Contains code for demo (more information below). |

## Our Approaches
### Supervised Learning
  In the supervised learning model, it first loads in the Google forms data and converts it into a dataframe. Then another dataframe labeled “individual_acitivity” is created that cleans the elements of the dataframe and only contains all of the activity ranking responses. A “relationships” datatframe is created to pair each user in the dataframe with all of the other users, to then make and “activities_df” that includes each user pair for each activity, and both of the users rankings  for that specific activity. Also, so we don’t have to load in the google forms data each time, we use base_path that maps the data from the google form into the dataframes. Before the data gets preprocessed, we create a “full_relationship_data” dataframe by merging the relationships and individual_data, and activities frames all together. This dataframe then gets OneHotEncoder to convert the categorical data into indices. For testing purposes, the data gets split into train, test, and validation data. Once the data was ready, multiple linear regression was performed using both SVR and decision tree models. Both however performed very poorly, so this is when we pivoted to collaborative filtering.   
### Collaborative Filtering
  In the collaborative filtering model, the data is imported and preprocessed the same way as it was in the supervised learning model. To create the collaborative filtering algorithm, three  methods are created including the similarity, similarity_to_user, and suggest_acitivty. The similarity function uses the built in sklearn, cosine_similarity function to compute the cosine similarity between two imputed matrices. Similarity_to_user inputs a specific user and all other users in an array, then computes the cosine similarity between the user and every other user using the similarity function, which is returned as an array. The suggest_acitivity function first scores all other users by how similar they are to the given user using the similariy_to_user method. Then it multiplies the similarity of the users with the activity score (inputted). Then from the top N (inputted value) most similar rows, t finds the most common features, assuming features are independent, which belong to the ideal activity. Based on the features of the ideal activity, it scores activities based on how similar the features are to the top features. This function then returns an ordered list of activities ranked by its similarity to the ideal activity. Before this gets applied to all users, random pairs are created and the responses in the dataframes are cleaned. After this, the suggest_acitivity function is applied to each pair. This is the current model we are using to generate nudges.   
 ### Next Steps 
  For the future of the model, we create another file labeled “Future_Direction” that imports datasets that ideally will contain different events around a city. This will give us a wider range of potential activity suggestions to nudge people, while still using the collaborative filtering algorithm.  

# About the Data
CSV file of response data not included to protect the privacy of the participants. The `.ipynb` files include the head/tail of the data to show representative data. 

Survey used to collect the data can be found [here](https://docs.google.com/forms/d/e/1FAIpQLSc0vSS6KeY69VW-CB-nO3ZT569Zq73CD1di9bMIdFD40qp70g/viewform). 

You can view the suggestions given to the participants in our small-scale deployment [here](https://github.com/eunice-chan/DataX-SCET/blob/main/Suggestions.md).

# How to Run
We provide two ways to interface with our solution: (1) ipynb with the algorithm and (2) a web demo.

# .ipynb
To demo the ipynbs, please generate synthetic data or collect actual data in the form shown in the `.ipynb`s. Once that is done, simply run through all the cells.

# Demo
A web demo can be found [here](https://modern-productive-cheque.glitch.me).  
The associated code is uploaded to the repository, but can also be found [here](https://glitch.com/edit/#!/modern-productive-cheque).
To run the web demo, fill in the relevant information and press the button at the bottom to generate suggestions for the pair.
