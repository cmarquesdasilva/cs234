"""Script to orchestrate the agents workflow integration with TinyTroupe."""
import argparse
from src.prompts import analyze_movie_attributes, predict_movie_rating_based_user_past_rate, predict_movie_rating_based_on_user_profile_summary
import pandas as pd
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Run the agents workflow.")
    parser.add_argument(
        "--task",
        type=str,
        default="predict_movie_rate",
        help="The task to run.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="user_profile",
        help="The task to run.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="src/data/validation_set.csv",
        help="The agent to run.",
    )

    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df = df[df["userId"] == 22]

    print("Size of the input dataframe: ", df.shape)
    print(f"Running prompt: {args.task}")

    if args.task == "analyze_movie_attributes":

        results = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            movie = row["title"]
            new_df = analyze_movie_attributes(movie)
            new_df["movieId"] = row["movieId"]
            results.append(new_df)

        results_df = pd.concat(results)
        results_df.to_csv("src/movies_attr.csv", index=False)

    elif args.task == "predict_movie_rate":
        if args.prompt == "past_experice":
            users_profile = pd.read_csv("src/data/train_gpt_with_user.csv", sep="\t")
        else:
            users_profile = pd.read_csv("src/data/user_full_profiles.tsv", sep="\t")
        
        results = []
        
        for _, row in tqdm(df.iterrows(), total=len(df)):
            movie = row["title"]
            genre = row["genres"]
            user_id = row["userId"]
            if args.prompt == "past_experice":
                user_profile = users_profile[users_profile["userId"] == user_id]["prompt"].values[0]
                json_output = predict_movie_rating_based_user_past_rate(user_profile, movie, genre)
            else:
                user_profile = users_profile[users_profile["userId"] == user_id].iloc[0]

                print(user_profile['openness'] == "high")
                context = f"""
                This user has a **{user_profile['openness']}** level of openness, meaning they {("enjoy exploring new and creative storytelling" if user_profile['openness'] == "high" else "prefer familiar and structured narratives")}.
                They have **{user_profile['conscientiousness']}** conscientiousness, which means they {("appreciate well-structured, thought-provoking films" if user_profile['conscientiousness'] == "high" else "are more relaxed about plot structure and prefer entertainment over depth")}.
                Their **{user_profile['extraversion']}** extraversion suggests they {("enjoy social, energetic movies" if user_profile['extraversion'] == "high" else "prefer introspective, slow-paced, or character-driven films")}.
                They have **{user_profile['agreeableness']}** agreeableness, meaning they {("gravitate toward emotionally engaging and warm movies" if user_profile['agreeableness'] == "high" else "do not necessarily seek out emotional or feel-good movies")}.
                They have **{user_profile['neuroticism']}** neuroticism, which means they {("may prefer emotionally intense films" if user_profile['neuroticism'] == "high" else "tend to enjoy light-hearted or balanced emotional narratives")}.
                Their **{user_profile['sensation_seeking']}** level of sensation-seeking indicates they {("love high-adrenaline action and thrillers" if user_profile['sensation_seeking'] == "high" else "enjoy slower-paced, thoughtful storytelling")}.
                They have a **{user_profile['cognitive_style']}** cognitive style, so they {("appreciate non-linear, complex storytelling" if user_profile['cognitive_style'] == "non-linear" else "prefer structured, easy-to-follow narratives")}.
                They have a **{user_profile['need_for_closure']}** need for closure, meaning they {("prefer movies with clear, satisfying endings" if user_profile['need_for_closure'] == "high" else "enjoy open-ended or ambiguous stories")}.
                
                Their favorite genres include: **{user_profile['preferred_genres']}**.
                They typically enjoy movies like: **{user_profile['favorite_movies']}**.
                They tend to dislike movies like: **{user_profile['disliked_movies']}**.

                Observations: {user_profile['observations']}
                """

                json_output = predict_movie_rating_based_on_user_profile_summary(context, movie, genre)
            
            print("Predicted Rate: ", json_output["rating"])
            print("True Rate: ", row["rating"])

            new_df = pd.DataFrame([json_output])
            new_df["userId"] = user_id
            new_df["true_label"] = row["rating"]
            new_df.rename(columns={"rating": "predicted_label"}, inplace=True)
            results.append(new_df)

        results_df = pd.concat(results)
        results_df.to_csv("user_rates.csv", index=False)
    else:
        raise ValueError(f"Invalid task: {args.task}")

if __name__ == "__main__":
    main()