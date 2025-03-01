import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

# Load environment variables
endpoint_url = os.getenv("ENDPOINT_URL")
azure_key = os.getenv("AZURE_OPENAI_KEY")
deployment = os.getenv("DEPLOYMENT_NAME")
api_verions = os.getenv("API_VERSION")

if not all([endpoint_url, azure_key, deployment]):
    raise ValueError("Missing required environment variables: check ENDPOINT_URL, AZURE_OPENAI_KEY, and DEPLOYMENT_NAME")

# Initialize Azure LLM
azure_llm = AzureChatOpenAI(
    azure_endpoint=endpoint_url,
    api_key=azure_key,
    api_version=api_verions,
    deployment_name=deployment
)

# Agent memory
agent_memory = {}

def round_to_half_step(value: float) -> float:
    """Rounds the rating to the nearest 0.0 or 0.5 step."""
    return round(value * 2) / 2


def scan_user_psychological_trait(user_data: str) -> str:
    """Summarizes user preferences into a JSON-structured response."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an AI assistant analyzing a user's movie rating history. "
            "Your task is to generate a structured JSON summary of their movie preferences based on psychological traits, "
            "cognitive styles, and emotional tendencies.\n\n"
            "Analyze the user's ratings and extract insights related to:\n"
            "- **Big Five Personality Traits** (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism) based on their preferred genres and storytelling complexity.\n"
            "- **Sensation Seeking** (Do they prefer high-action, thrilling movies, or slower, thought-provoking narratives?)\n"
            "- **Empathy and Emotional Intelligence** (Do they enjoy emotionally deep films, or do they prefer plot-driven stories?)\n"
            "- **Cognitive Style** (Do they prefer structured, linear narratives, or non-linear, experimental films?)\n"
            "- **Need for Closure vs. Need for Exploration** (Do they prefer movies with clear resolutions, or ambiguous, open-ended stories?)\n\n"
            "Your response must be a structured JSON object with the following keys:\n"
            "{{\n"
            '  "personality_traits": {{\n'
            '    "openness": "low/medium/high",\n'
            '    "conscientiousness": "low/medium/high",\n'
            '    "extraversion": "low/medium/high",\n'
            '    "agreeableness": "low/medium/high",\n'
            '    "neuroticism": "low/medium/high"\n'
            "  }},\n"
            '  "sensation_seeking": "low/medium/high",\n'
            '  "empathy": "low/medium/high",\n'
            '  "cognitive_style": "structured/non-linear/mixed",\n'
            '  "need_for_closure": "low/medium/high",\n'
            '  "preferred_genres": ["list of genres"],\n'
            '  "favorite_movies": ["list of movies with high ratings"],\n'
            '  "disliked_movies": ["list of movies with low ratings"],\n'
            '  "observations": "Additional insights based on their preferences."\n'
            "}}\n\n"
            "Use the user's ratings to determine these attributes as accurately as possible."
        )),
        ("human", "Below is the user's movie rating history:\n\n{user_data}\n\nGenerate a structured JSON summary that encompasses user psychological factors."),
    ])
    chain = prompt | azure_llm | JsonOutputParser(key="summary")
    try:
        output = chain.invoke({"user_data": user_data})  # Returns a dictionary
        return output  # Directly returns the extracted "summary" key
    except Exception as e:
        print("Error summarizing user data:", e)
        return "Unable to parse user profile"


def predict_movie_rating(profile_summary: str, new_movie: str, genre: str) -> dict:
    """Predicts a numeric rating (0, 1, or 2) and provides a short justification."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            """You are role-playing as a regular moviegoer who has just watched a movie. 
            
            **Context:**
            - You have specific movie preferences based on your personality and past experiences.
            - You are **not a professional critic**, but you are sharing your personal thoughts.

            ---
            
            **User Profile Summary:**  
            {profile_summary}
            
            **Movie Watched:**  
            - **Title**: {new_movie}  
            - **Genre**: {genre}  

            **Your Task:**  
            - **Describe your experience watching this movie** in a few sentences.
            - **Did you enjoy it? Why or why not?**  
            - **How well does it match your personality traits and movie preferences?**  
            - **Was it exciting enough for you? Or too slow?**  
            - **Did you feel emotionally connected to the story?**  
            - **Did you find the movie predictable or engaging based on your cognitive style?**  
            - **Would you recommend it to someone with similar tastes?**  
            
            ---
            
            **Final Rating (0.5 to 5):**  
            - **0.5** → Strongly disliked it, found it almost unwatchable.  
            - **1** → Did not enjoy it, but it had minor redeeming qualities.  
            - **1.5** → Mostly disliked it, but a few aspects were decent.  
            - **2** → It was below average, not engaging, and not something I would watch again.  
            - **2.5** → Just okay, neither great nor terrible.  
            - **3** → It was decent, had some strong points but also some flaws.  
            - **3.5** → Liked it, but it wasn’t outstanding.  
            - **4** → Enjoyed it a lot and would likely recommend it.  
            - **4.5** → Loved it, nearly perfect with only minor issues.  
            - **5** → Absolutely loved it! One of the best movies I’ve seen.  

            **Response Format:**  
            Your answer **must** be a **valid JSON object** containing two keys: `"rating"` and `"justification"`.  
            Do **not** include any additional text outside of the JSON.  
            
            **Example valid JSON response:**  
            ```json
            {{ "rating": 5, "justification": "I absolutely loved this movie! As someone who enjoys Sci-Fi and deep storytelling, 'Blade Runner 2049' was a perfect fit for me. The slow-burn narrative and visually stunning world-building fully immersed me. The emotional themes resonated with my high empathy and openness to complex ideas." }}
            ```
            """
        )),
        ("human", (
            "Based on your personality traits and movie preferences, predict a numeric rating (0.5 to 5) and justify your decision in **strict JSON format**."
        ))
    ])

    output_parser = JsonOutputParser(key='rating')  # Extract full JSON response
    chain = prompt | azure_llm | output_parser

    response = chain.invoke({"profile_summary": profile_summary, "new_movie": new_movie, "genre": genre})  # Returns JSON dict
    try:
        # Ensure the response is in valid JSON format
        rating = response.get("rating")
        justification = response.get("justification", "No justification provided.")
        
        return {"rating": rating, "justification": justification}

    except Exception as e:
        print("Error parsing response:", e)
        return {"rating": "Error", "justification": "Unable to generate a valid rating."}