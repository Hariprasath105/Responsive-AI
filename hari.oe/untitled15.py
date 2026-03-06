import random

# 1. Fixed the function name (no spaces)
def generate_image(prompt):
    # Responsible AI: Content Moderation
    # Fixed the list syntax and the "if any" logic
    restricted_keywords = ["harmful", "illegal", "bias"]
    
    if any(keyword in prompt.lower() for keyword in restricted_keywords):
        return "ERROR: Generated content violated safety policy."
    
    return f"Image generated: A beautiful digital painting of {prompt}"

# 2. User Input
user_prompt = "A serene sunset over a mountain"
print(generate_image(user_prompt))

# 3. Risky Input Example
risky_prompt = "A harmful image of..."
print(generate_image(risky_prompt))