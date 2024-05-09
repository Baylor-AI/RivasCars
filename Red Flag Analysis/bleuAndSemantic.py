import spacy
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

#Load SpaCy English model with word embeddings
try:
    nlp = spacy.load('en_core_web_md')
except OSError:
    print('Downloading language model for the spaCy POS tagger\n'
        "(don't worry, this will only happen once)")
    from spacy.cli import download
    download('en_core_web_md')
    nlp = spacy.load('en_core_web_md')

#Function to compute sentence embedding
def get_sentence_embedding(sentence):
    doc = nlp(sentence)
    if len(doc) == 0:  #Check if the doc is empty
        return np.zeros((nlp.meta["vectors"]["width"],))  #Return zero vector of proper length
    embeddings = [token.vector for token in doc if token.has_vector]
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros((nlp.meta["vectors"]["width"],))  #Return zero vector if no token vectors found


#Function to calculate cosine similarity
def cosine_similarity(vec1, vec2):
    if np.all(vec1 == 0) or np.all(vec2 == 0):
        return 0.0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

#Define reference phrases and compute their embeddings
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


#Function to calculate semantic scores for a given sentence
def get_semantic_score(sentence, reference_embeddings):
    sentence_embedding = get_sentence_embedding(sentence)
    scores = [cosine_similarity(sentence_embedding, ref_emb) for ref_emb in reference_embeddings]
    return max(scores)  #return the highest similarity score

#Calculate BLEU and Semantic scores
weights_bleu = (1.0, 0.0, 0.0, 0.0)  #top performing weights from previous results
smoother = SmoothingFunction().method1

#Example input
input_paragraph = """
I have a totally not stolen sealed camera that I need gone today. No questions asked about the sale. I guarantee you we can have the sale done with quickly.
"""
#Process input
sentences = input_paragraph.strip().split('. ')
bleu_scores = {}
semantic_scores = {}
for sentence in sentences:
    candidate = sentence.split()
    #BLEU score
    bleu_score = sentence_bleu(varied_references, candidate, weights=weights_bleu, smoothing_function=smoother)
    bleu_scores[sentence] = bleu_score
    #Semantic score
    semantic_score = get_semantic_score(sentence, reference_embeddings)
    semantic_scores[sentence] = semantic_score

#Weighted average of BLEU and Semantic scores (0.5 each)
#Adjusted composite score calculation with new thresholds
final_scores = {}
threshold = 0.7  #Example threshold for high semantic similarity
for sentence in sentences:
    adjusted_semantic_score = 1.0 if semantic_scores[sentence] > threshold else semantic_scores[sentence] / threshold
    final_score = 0.6 * bleu_scores[sentence] + 0.4 * adjusted_semantic_score
    final_scores[sentence] = final_score


final_scores
#Print the results
print("Results for each sentence:")
for sentence in sentences:
    print("\nSentence:", sentence)
    print("BLEU Score:", bleu_scores[sentence])
    print("Semantic Score:", semantic_scores[sentence])
    print("Composite Score (Weighted Average):", final_scores[sentence])
