try:
    import spacy
except:
    exit()
from collections import Counter
nlp = spacy.load("en_core_web_sm")

def extract_aspects(text, topn=5):
    doc = nlp(text)
    # noun chunks + nouns
    aspects = [chunk.text.lower() for chunk in doc.noun_chunks]
    aspects += [token.text.lower() for token in doc if token.pos_ == 'NOUN']
    c = Counter(aspects)
    return [a for a,_ in c.most_common(topn)]

# Example:
# print(extract_aspects("The battery life is great but the shipping was slow. The screen quality is fantastic."))
