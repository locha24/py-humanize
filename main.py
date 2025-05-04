import random
from textblob import TextBlob
from nltk.corpus import wordnet
import nltk
import re

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

def add_human_touch(text, error_rate=0.02, shuffle_sentences=False):
    words = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(words)
    
    for i in range(len(words)):
        if random.random() < error_rate and len(words[i]) > 3:
            pos = random.randint(0, len(words[i]) - 2)
            words[i] = words[i][:pos] + words[i][pos + 1] + words[i][pos] + words[i][pos + 2:]
    
    modified_text = ' '.join(words)
    if random.random() < error_rate * 2:
        modified_text = modified_text.replace(',', '', 1)
    if random.random() < error_rate * 2:
        modified_text = modified_text.replace('.', '', 1)
    
    if shuffle_sentences:
        sentences = nltk.sent_tokenize(modified_text)
        random.shuffle(sentences)
        modified_text = ' '.join(sentences)
    
    return modified_text

def paraphrase_text(text, intensity=0.3):
    blob = TextBlob(text)
    words = blob.words
    
    for i in range(len(words)):
        if random.random() < intensity:
            synsets = wordnet.synsets(words[i])
            if synsets:
                synonyms = set()
                for syn in synsets:
                    for lemma in syn.lemmas():
                        if lemma.name() != words[i]:
                            synonyms.add(lemma.name().replace('_', ' '))
                if synonyms:
                    words[i] = random.choice(list(synonyms))
    
    return ' '.join(words)

def humanize_ai_text(text, error_rate=0.03, shuffle_sentences=False, paraphrase_intensity=0.4):

    text = paraphrase_text(text, intensity=paraphrase_intensity)
    text = add_human_touch(text, error_rate=error_rate, shuffle_sentences=shuffle_sentences)
    return text

if __name__ == "__main__":
    ai_generated_text = input("Enter AI-generated text: ")
    
    humanized_text = humanize_ai_text(
        ai_generated_text,
        error_rate=0.05,
        shuffle_sentences=True,
        paraphrase_intensity=0.5
    )
    
    print("AI-generated text:\n", ai_generated_text)
    print("\nHumanized text:\n", humanized_text)