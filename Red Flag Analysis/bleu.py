import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

#Hardcoded input paragraph
input_paragraph = """
I have a totally not stolen sealed camera that I need gone today. No questions asked about the sale. I guarantee you we can have the sale done with quickly.
"""

#Hardcoded "red flag" phrases
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

#Manually splitting the paragraph into sentences
sentences = input_paragraph.strip().split('. ')
sentences = [s.strip() for s in sentences if s]

#Prepare references for BLEU (each phrase as a separate list of words)
references = [phrase.split() for phrase in reference_phrases]

#Smoothing function for BLEU calculation
smoother = SmoothingFunction().method7

#Adjusted weights for BLEU calculation to include up to 4-grams
weights = (1, 0, 0, 0)  #unigram weighted

#Calculate BLEU score for each sentence against the references
bleu_scores = {}
for sentence in sentences:
    candidate = sentence.split()
    score = sentence_bleu(references, candidate, weights=weights, smoothing_function=smoother)
    bleu_scores[sentence] = score

#Print BLEU scores
for sentence, score in bleu_scores.items():
    print(f"\"{sentence}\": {score:.3f}")
