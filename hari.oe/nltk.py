import math
import numpy as np
from collections import Counter
import re
import nltk
from nltk import ngrams
from nltk.tokenize import word_tokenize

nltk.download('punkt')

def calculate_perplexity(text, n=2):
    """Calculates the perplexity of text using a simple n-gram model."""
    try:
        text = re.sub(r'[^\w\s]', '', text.lower())
    
        tokens = word_tokenize(text)
        
        if not tokens:
            return 0.0
        
        n_grams = list(ngrams(tokens, n))
        
        if not n_grams:
            return 0.0
        
        freq_dist = Counter(n_grams)
        
        N = len(n_grams)
        
        log_prob_sum = 0.0
        for ng in n_grams:
            count = freq_dist[ng]
            prob = count / N
            if prob > 0:
                log_prob_sum += math.log(prob)
        
        avg_log_prob = log_prob_sum / N
        
        perplexity = math.exp(-avg_log_prob)
        
        return perplexity
    
    except Exception as e:
        print(f"Error calculating perplexity: {e}")
        return None

if __name__ == "__main__":
    sample_text = "This is a sample text for testing perplexity calculation."
    
    result = calculate_perplexity(sample_text, n=2)
    
    if result is not None:
        print(f"Perplexity: {result:.2f}")
    else:

        print("Failed to calculate perplexity.")
