import pandas as pd
import spacy
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from heapq import heappush, heappop

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
    if len(doc) == 0:
        return np.zeros((nlp.meta["vectors"]["width"],))
    embeddings = [token.vector for token in doc if token.has_vector]
    return np.mean(embeddings, axis=0) if embeddings else np.zeros((nlp.meta["vectors"]["width"],))

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
reference_phrases2 = [
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

reference_phrases = [
    "must sell immediately", "need to sell ASAP", "urgent sale", "available now",
    "quick sale needed", "act now", "limited time only", "discreet transaction",
    "confidential deal", "no questions asked", "private sale", "discreet only",
    "no receipt", "not registered", "no return policy", "as is", "no warranty"
]
reference_embeddings = [get_sentence_embedding(phrase) for phrase in reference_phrases]
varied_references = [phrase.split() for phrase in reference_phrases]

# Load data from CSV
data = pd.read_csv('processed_data_craigslist_v2_deduplicated.csv', header=None, low_memory=False)

# Heap to store top 5 scores
top_bleu = []
top_semantic = []
top_composite = []
smoother = SmoothingFunction()
# Process every 1000th entry (to save time)
for index, row in data.iloc[::1000].iterrows():
    input_paragraph = str(row[6])
    sentences = input_paragraph.strip().split('. ')
    bleu_scores = {}
    semantic_scores = {}
    final_scores = {}
    for sentence in sentences:
        candidate = sentence.split()
        bleu_score = sentence_bleu(varied_references, candidate, weights=(1.0, 0.0, 0.0, 0.0), smoothing_function=smoother.method1)
        semantic_score = get_semantic_score(sentence, reference_embeddings)
        bleu_scores[sentence] = bleu_score
        semantic_scores[sentence] = semantic_score
        adjusted_semantic_score = 1.0 if semantic_scores[sentence] > 0.7 else semantic_scores[sentence] / 0.7
        final_score = 0.6 * bleu_score + 0.4 * adjusted_semantic_score
        final_scores[sentence] = final_score

    # Calculate overall post scores and manage top 5 lists
    overall_bleu = np.mean(list(bleu_scores.values()))
    overall_semantic = np.mean(list(semantic_scores.values()))
    overall_composite = np.mean(list(final_scores.values()))
    heappush(top_bleu, (overall_bleu, index, input_paragraph))
    heappush(top_semantic, (overall_semantic, index, input_paragraph))
    heappush(top_composite, (overall_composite, index, input_paragraph))
    if len(top_bleu) > 5: heappop(top_bleu)
    if len(top_semantic) > 5: heappop(top_semantic)
    if len(top_composite) > 5: heappop(top_composite)

# Print top 5 results
print("Top 5 BLEU Scores:")
for score, idx, para in sorted(top_bleu, reverse=True):
    print(f"Index: {idx}, Score: {score}, Paragraph: {para}\n")

print("Top 5 Semantic Scores:")
for score, idx, para in sorted(top_semantic, reverse=True):
    print(f"Index: {idx}, Score: {score}, Paragraph: {para}\n")

print("Top 5 Composite Scores:")
for score, idx, para in sorted(top_composite, reverse=True):
    print(f"Index: {idx}, Score: {score}, Paragraph: {para}\n")
