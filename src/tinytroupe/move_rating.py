#TODO: Add a docstring to this file and a main method.
import json
import sys
import os
sys.path.insert(0, '..')

import pandas as pd

from tinytroupe.agent import TinyPerson
from tinytroupe.factory import TinyPersonFactory
from tinytroupe.extraction import ResultsExtractor
from tqdm import tqdm
import random

TinyPerson.all_agents

def extract_agent_answer(user):
    extractor = ResultsExtractor()

    choices =[]
    extraction_objective="Find the movie rating score. Extract the movie rating score number and justification for the rate."
    situation=""

    res = extractor.extract_results_from_agent(user,                                    
                                            extraction_objective=extraction_objective,
                                            situation=situation,
                                            fields=["movie_rating_score", "justification"],
                                            fields_hints={"movie_rating_score": "Must be an float, not a string."},
                                            verbose=True
                                        )

    choices.append(res)
    return res

def build_eval_request(historical_rates, new_movie, genre, favorite_movies, disliked_movies):
    eval_request_msg =\
f"""
    Your Task is to give a rate for the movie {new_movie} that you have watched. 
    DO NOT OVER THINK, THINK ONLY ONCE, AFTER GIVING YOUR RATE, YOU ARE DONE.
            **Movie Watched:**  
            - **Title**: {new_movie}  
            - **Genre**: {genre}  
            Provide a movie rating score and a justification to the movie that you watched and do not do nothing more.
            ---
            **Possibles movie rating score are 0.5 to 5 rounded by .5:**  
            Your favorite_movies: {favorite_movies}
            Your disliked movies: {disliked_movies}
            Here is your historical movie ratings (ranging from 0.5 to 5 in 0.5 increments):
            {historical_rates}
"""
    
    return eval_request_msg


def load_or_create_agent(user_id):
    """
    Loads an existing agent if it exists, otherwise creates a new one from the user profile dataset.
    
    Args:
        user_id (str): The user ID to load or create
        factory (TinyPersonFactory): The factory to create a new agent
        
    Returns:
        TinyPerson: The loaded or created agent
    """
    # Read the user profile dataset
    user_profiles_df = pd.read_csv('user_full_profiles.tsv', sep='\t')
    
    # Find the user in the dataset
    user_data = user_profiles_df[user_profiles_df['userId'] == user_id]
        
    # Extract user profile information
    user_row = user_data.iloc[0]
    
    # Build profile summary from user data
    favorite_movies = user_row.get('favorite_movies', '')
    disliked_movies = user_row.get('disliked_movies', '')

    agent_path = os.path.join(os.path.dirname(__file__), f'./agents/{user_id}.agent.json')
    
    # Check if agent specification already exists
    if os.path.exists(agent_path):
        print(f"Loading agent for user {user_id}")
        return TinyPerson.load_specification(json.load(open(agent_path))), favorite_movies, disliked_movies
    
    profile_summary = f"""
    "openness":"{user_row.get('openness', 'medium')}",
    "conscientiousness":"{user_row.get('conscientiousness', 'medium')}",
    "extraversion":"{user_row.get('extraversion', 'medium')}",
    "agreeableness":"{user_row.get('agreeableness', 'medium')}",
    "neuroticism":"{user_row.get('neuroticism', 'medium')}",
    "sensation_seeking":"{user_row.get('sensation_seeking', 'medium')}",
    "empathy":"{user_row.get('empathy', 'medium')}",
    "cognitive_style":"{user_row.get('cognitive_style', 'mixed')}",
    "need_for_closure":"{user_row.get('need_for_closure', 'medium')}",
    "preferred_genres":"{user_row.get('preferred_genres', '')}",
    "favorite_movies":"{favorite_movies}",
    "disliked_movies":"{disliked_movies}",
    "observations":"{user_row.get('observations', '')}"
    """
    
    # Create user spec with the profile summary
    custom_user_spec = f""" 
        **User Profile Summary:**  
        {profile_summary}
    """
    # Generate the person using the factory
    factory = TinyPersonFactory(custom_user_spec)
    user = factory.generate_person(custom_user_spec)
    
    # Save the specification
    user.save_specification(agent_path)
    
    print(f"Created and saved new agent for user {user_id}")
    return user, favorite_movies, disliked_movies
        
# Read dataframe
val_df = pd.read_csv("validation_set.csv")
users_id = random.sample(range(100, 601), 22)

for user_id in users_id:
    user_df = val_df[val_df["userId"] == user_id]
    if user_df.shape[0] > 40:
        print(f"Skipping user {user_id} due to larger data.")
        continue
    
    print(f"Running for the user: {user_id}")
    past_ratings_df = pd.read_csv("train_gpt_with_user.csv", sep="\t")
    historical_rates = past_ratings_df[past_ratings_df["userId"] == user_id]["prompt"].values[0].split(":")[1].strip()

    # Load a Agent:
    user, favorite_movies, disliked_movies = load_or_create_agent(user_id)


    true_label = []

    for _, row in tqdm(user_df.iterrows(), total=user_df.shape[0], desc="Processing movies"):
        new_movie = row["title"]
        genre = row["genres"]
        true_label = row["rating"]
        situation = "you have watched the movie {new_movie} and you are going to rate the movie by choosing a numerical value between 0.5 to 5 always rounded by .5"
        user.change_context(situation)
        eval_request_msg = build_eval_request(historical_rates, new_movie, genre, favorite_movies, disliked_movies)
        user.listen_and_act(eval_request_msg, n=2)

    # Extract the predicted label
    results = extract_agent_answer(user)
    results_df = pd.DataFrame(results)
    results_df['userId'] = user_id
    try:
        results_df['movieId'] = user_df['movieId'].values
        results_df['true_labels'] = user_df['rating'].values
    except ValueError:
        print("Error: The number of rows in results_df and user_df do not match. Please check the data.")

    # Rename the columns
    results_df.rename(columns={"movie_rating_score": "predicted_label"}, inplace=True)

    # Export the dataframe to a CSV file
    results_df.to_csv(f"user_{user_id}_evaluation.csv", index=False)