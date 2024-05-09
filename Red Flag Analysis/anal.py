import pandas as pd
import spacy
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Load SpaCy English model with word embeddings
try:
    nlp = spacy.load('en_core_web_md')
except OSError:
    print('Downloading language model for the spaCy POS tagger\n'
        "(don't worry, this will only happen once)")
    from spacy.cli import download
    download('en_core_web_md')
    nlp = spacy.load('en_core_web_md')

# Function to compute sentence embedding
def get_sentence_embedding(sentence):
    doc = nlp(sentence)
    if len(doc) == 0:  # Check if the doc is empty
        return np.zeros((nlp.meta["vectors"]["width"],))  # Return zero vector of proper length
    embeddings = [token.vector for token in doc if token.has_vector]
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros((nlp.meta["vectors"]["width"],))  # Return zero vector if no token vectors found


#Function to calculate semantic scores for a given sentence
def get_semantic_score(sentence, reference_embeddings):
    sentence_embedding = get_sentence_embedding(sentence)
    scores = [cosine_similarity(sentence_embedding, ref_emb) for ref_emb in reference_embeddings]
    return max(scores)  #return the highest similarity score

# Function to calculate cosine similarity
def cosine_similarity(vec1, vec2):
    if np.all(vec1 == 0) or np.all(vec2 == 0):
        return 0.0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Define reference phrases and compute their embeddings
reference_phrases = [
    "never opened",
    "still sealed",
    "must sell fast",
    "contact me soon",
    "offer won't last long","cash only", "urgent sale", "need gone today", "no questions asked",
    "quick transaction", "serious buyers only", "first come first serve",
    "no returns", "price is firm", "selling cheap", "act fast",
    "limited time offer", "confidential sale", "direct deal", "available immediately","need to sell quickly", "looking for a quick sale", "urgent sale needed",
    "unopened", "still in original packaging", "brand new condition",
    "unbroken seal", "factory sealed", "not opened"
]
reference_embeddings = [get_sentence_embedding(phrase) for phrase in reference_phrases]
varied_references = [phrase.split() for phrase in reference_phrases]

# Load data from CSV
data = pd.read_csv('processed_data_craigslist_v2_deduplicated.csv', header=None, low_memory=False)

# Define score dictionaries
max_bleu = {'score': 0, 'paragraph': '', 'index': 0}
max_semantic = {'score': 0, 'paragraph': '', 'index': 0}
max_composite = {'score': 0, 'paragraph': '', 'index': 0}

smoother = SmoothingFunction()


# Process every 1000th entry
for index, row in data.iloc[::1000].iterrows():
    # Convert the paragraph to string to ensure compatibility with text processing methods
    input_paragraph = str(row[6])
    sentences = input_paragraph.strip().split('. ')
    bleu_scores = {}
    semantic_scores = {}
    final_scores = {}
    for sentence in sentences:
        candidate = sentence.split()
        # BLEU score
        try:
            bleu_score = sentence_bleu(varied_references, candidate, weights=(1.0, 0.0, 0.0, 0.0), smoothing_function=smoother.method1)
        except Exception as e:
            print(f"Error computing BLEU score for sentence: {sentence}, error: {e}")
            bleu_score = 0
        bleu_scores[sentence] = bleu_score
        # Semantic score
        semantic_score = get_semantic_score(sentence, reference_embeddings)
        semantic_scores[sentence] = semantic_score
        # Composite score
        adjusted_semantic_score = 1.0 if semantic_scores[sentence] > 0.7 else semantic_scores[sentence] / 0.7
        final_score = 0.6 * bleu_scores[sentence] + 0.4 * adjusted_semantic_score
        final_scores[sentence] = final_score
    # Check for max scores
    average_bleu = np.mean(list(bleu_scores.values()))
    average_semantic = np.mean(list(semantic_scores.values()))
    average_composite = np.mean(list(final_scores.values()))
    if average_bleu > max_bleu['score']:
        max_bleu.update({'score': average_bleu, 'paragraph': input_paragraph, 'index': index})
    if average_semantic > max_semantic['score']:
        max_semantic.update({'score': average_semantic, 'paragraph': input_paragraph, 'index': index})
    if average_composite > max_composite['score']:
        max_composite.update({'score': average_composite, 'paragraph': input_paragraph, 'index': index})

# Print the results
print("Highest BLEU Score:")
print(f"Index: {max_bleu['index']}, Score: {max_bleu['score']}, Paragraph: {max_bleu['paragraph']}")
print("\nHighest Semantic Score:")
print(f"Index: {max_semantic['index']}, Score: {max_semantic['score']}, Paragraph: {max_semantic['paragraph']}")
print("\nHighest Composite Score:")
print(f"Index: {max_composite['index']}, Score: {max_composite['score']}, Paragraph: {max_composite['paragraph']}")
