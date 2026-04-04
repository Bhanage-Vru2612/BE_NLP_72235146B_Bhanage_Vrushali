import nltk
import random
from collections import defaultdict, Counter
from nltk.util import ngrams
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

class NGramModel:
    def __init__(self, n=3):
        self.n = n
        self.context_counts = defaultdict(Counter)
        self.vocab = set()
        self.stop_words = set(stopwords.words('english'))

    def train(self, text):
        sentences = sent_tokenize(text.lower())
        tokenized_sents = [word_tokenize(sent) for sent in sentences]
        filtered_sents = [[w for w in sent if w.isalpha() and len(w) > 2 and w not in self.stop_words] 
                          for sent in tokenized_sents]
        
        for sent in filtered_sents:
            if len(sent) >= self.n - 1:
                # Pad with start/end tokens
                padded_sent = ['<s>'] * (self.n - 1) + sent + ['</s>']
                n_grams = ngrams(padded_sent, self.n)
                for gram in n_grams:
                    context = ' '.join(gram[:-1])
                    word = gram[-1]
                    self.context_counts[context][word] += 1
                    if word != '<s>' and word != '</s>':
                        self.vocab.add(word)
        
        self.vocab = sorted(list(self.vocab))

    def generate(self, seed_words=None, length=20):
        if seed_words is None or len(seed_words) == 0:
            context = ' '.join(['<s>'] * (self.n - 1))
        else:
            context = ' '.join(['<s>'] * max(0, self.n - 1 - len(seed_words)) + seed_words)
        
        generated = seed_words[:] if seed_words else []
        
        for _ in range(length):
            if context not in self.context_counts:
                context = ' '.join(['<s>'] * (self.n - 1))
            
            next_words = self.context_counts[context]
            if not next_words:
                break
            
            # Laplace (add-one) smoothing
            vocab_size = len(self.vocab) + 2  # +2 for <s>, </s>
            total_count = sum(next_words.values())
            probs = {w: (next_words[w] + 1) / (total_count + vocab_size) 
                     for w in next_words}
            
            next_word = random.choices(list(probs.keys()), weights=list(probs.values()))[0]
            generated.append(next_word)
            
            context = ' '.join(generated[-(self.n - 1):])
            if next_word == '</s>':
                break
        
        # Clean output
        output = ' '.join(generated)
        output = output.replace(' <s>', '').replace('</s>', '').strip()
        return ' '.join(output.split()[:length])  # Trim to requested length