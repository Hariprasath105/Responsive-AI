import os
from openai import OpenAI
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def generate_story(prompt):
    """Generates a short story based on a prompt using the OpenAI API."""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  
            messages=[
                {"role": "system", "content": "You are a creative storyteller."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,  
            temperature=0.8  
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Example usage
if __name__ == "__main__":
    prompt = "Once upon a time, in a small village nestled between the mountains, a young girl discovered a magical key."
    
    story = generate_story(prompt)
    
    if story:
        print(story)
