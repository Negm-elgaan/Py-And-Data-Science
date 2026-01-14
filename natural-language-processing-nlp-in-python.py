# Import nltk
import nltk
# Download the punkt_tab package 
nltk.download("punkt_tab")

text = """
The stock market saw a significant dip today. Experts believe the downturn may continue.
However, many investors are optimistic about future growth.
"""

# Tokenize the text into sentences
sentences = nltk.sent_tokenize(text)
print(sentences)