import spacy
import numpy as np
from bilm import Batcher, TokenBatcher
nlp = spacy.load('en')
class generate_samples:
    '''
    Generate samples of training data or testing data for data analysis
    '''
    def __init__(self, data, labels, word_to_idx, max_sent_len=30, is_training=True):
        '''
        Args:
        data: numpy
        labels: numpy
        '''
        self.data = data
        self.labels = labels
        self.word_to_idx = word_to_idx
        self.is_training = is_training
        self.max_sent_len = max_sent_len
        self.index = 0
        
    def sent_split(self, sent):
        '''
        Split a sentence into tokens
        '''
        words = []
        sent = nlp(sent.strip())
        for w in sent:
            words.append(w.text.lower())
        return words
        
    def generate_random_samples(self, sents, labels, batch_size=64):
        '''
        Select a batch_size of sentences
        Transform each sentence into a sequence of idx
        '''
        indice = np.random.choice(len(sents), batch_size)
        sents = sents[indice]
        labels = labels[indice]
        #sent_vecs, sent_lens = self.create_sent_idx(sents)
        sent_vecs, sent_lens = self.create_sent_idx(sents)
        return sent_vecs, labels, sent_lens
        #return self.create_sent_idx(sents), labels, sent_lens
    
    
    def create_sent_idx(self, sents):
        '''
        Map sents into idx
        '''
        sents_lens = list(map(self.sent2idx, sents))
        sents_idx, sents_lens = zip(*sents_lens)
        return sents_idx, sents_lens
        
        
    def sent2idx(self, sent):
        '''Map a sentence into a sequence of idx'''
        sent_idx = []
        words = self.sent_split(str(sent))
        lens = len(words)
        ##Cut long sentences
        if lens > self.max_sent_len:
            words = words[:self.max_sent_len]
            lens = self.max_sent_len
        for w in words:
            idx = self.word_to_idx.get(w)
            idx = idx if idx else self.word_to_idx['<unk>']
            sent_idx.append(idx)
        ###Pad short sentences
        for i in np.arange(lens, self.max_sent_len):
            idx = self.word_to_idx['<pad>']
            sent_idx.append(idx)
        return sent_idx, lens
    
    def generate(self, batch_size=64):
        if self.is_training:
            sent_vecs, sent_labels, lengths = self.generate_random_samples(self.data, 
                                                               self.labels,
                                                              batch_size)
        else:
            start = self.index
            end = start + batch_size
            if end > len(self.data):
                print('Out of sample size')
                self.index = 0
            sents = self.data[start:end]
            sent_labels = self.labels[start:end]
            sent_vecs, lengths = self.create_sent_idx(sents)
            self.index = end
        return sent_vecs, sent_labels, lengths
    
class generate_token_samples:
    '''
    Generate samples of training data or testing data for data analysis
    '''
    def __init__(self, data, labels, vocab_file, max_sent_len=20, is_training=True):
        '''
        Args:
        data: numpy
        labels: numpy
        '''
        self.data = data
        self.labels = labels
        self.batcher = TokenBatcher(vocab_file)
        self.is_training = is_training
        self.max_sent_len = max_sent_len
        self.index = 0
        
    def sent_split(self, sent):
        '''
        Split a sentence into tokens
        '''
        words = []
        sent = nlp(sent.strip())
        for w in sent:
            words.append(w.text.lower())
        return words
        
    def generate_random_samples(self, sents, labels, batch_size=64):
        '''
        Select a batch_size of sentences
        Transform each sentence into a sequence of idx
        '''
        indice = np.random.choice(len(sents), batch_size)
        sents = sents[indice]
        labels = labels[indice]
        #sent_vecs, sent_lens = self.create_sent_idx(sents)
        sent_vecs, sent_lens = self.create_sent_idx(sents)
        return sent_vecs, labels, sent_lens
        #return self.create_sent_idx(sents), labels, sent_lens
    
    
    def create_sent_idx(self, sents):
        '''
        Map sents into char embeddings
        '''
        sents_lens = list(map(self.sent2idx, sents))
        sents_words, sents_lens = zip(*sents_lens)
        sents_idx = self.batcher.batch_sentences(list(sents_words))
        return sents_idx, sents_lens
        
        
    def sent2idx(self, sent):
        '''Map a sentence into a sequence of idx'''
        sent_idx = []
        words = self.sent_split(str(sent))
        lens = len(words)
        ##Cut long sentences
        if lens > self.max_sent_len:
            words = words[:self.max_sent_len]
            lens = self.max_sent_len
        ###Pad short sentences
        for i in np.arange(lens, self.max_sent_len):
            words.append('<unk>')
        return words, lens
    
    def generate(self, batch_size=64):
        if self.is_training:
            sent_vecs, sent_labels, lengths = self.generate_random_samples(self.data, 
                                                               self.labels,
                                                              batch_size)
        else:
            start = self.index
            end = start + batch_size
            if end > len(self.data):
                print('Out of sample size')
                self.index = 0
            sents = self.data[start:end]
            sent_labels = self.labels[start:end]
            sent_vecs, lengths = self.create_sent_idx(sents)
            self.index = end
        return sent_vecs, sent_labels, lengths    
    
class generate_char_samples:
    '''
    Generate samples of training data or testing data for data analysis
    '''
    def __init__(self, data, labels, vocab_file, max_sent_len=20, is_training=True):
        '''
        Args:
        data: numpy
        labels: numpy
        '''
        self.data = data
        self.labels = labels
        self.batcher = Batcher(vocab_file, 50)
        self.is_training = is_training
        self.max_sent_len = max_sent_len
        self.index = 0
        
    def sent_split(self, sent):
        '''
        Split a sentence into tokens
        '''
        words = []
        sent = nlp(sent.strip())
        for w in sent:
            words.append(w.text.lower())
        return words
        
    def generate_random_samples(self, sents, labels, batch_size=64):
        '''
        Select a batch_size of sentences
        Transform each sentence into a sequence of idx
        '''
        indice = np.random.choice(len(sents), batch_size)
        sents = sents[indice]
        labels = labels[indice]
        #sent_vecs, sent_lens = self.create_sent_idx(sents)
        sent_vecs, sent_lens = self.create_sent_idx(sents)
        return sent_vecs, labels, sent_lens
        #return self.create_sent_idx(sents), labels, sent_lens
    
    
    def create_sent_idx(self, sents):
        '''
        Map sents into char embeddings
        '''
        sents_lens = list(map(self.sent2idx, sents))
        sents_words, sents_lens = zip(*sents_lens)
        sents_idx = self.batcher.batch_sentences(list(sents_words))
        return sents_idx, sents_lens
        
        
    def sent2idx(self, sent):
        '''Map a sentence into a sequence of idx'''
        sent_idx = []
        words = self.sent_split(str(sent))
        lens = len(words)
        ##Cut long sentences
        if lens > self.max_sent_len:
            words = words[:self.max_sent_len]
            lens = self.max_sent_len
        ###Pad short sentences
        #for i in np.arange(lens, self.max_sent_len):
            #words.append('<UNK>')
        return words, lens
    
    def generate(self, batch_size=64):
        if self.is_training:
            sent_vecs, sent_labels, lengths = self.generate_random_samples(self.data, 
                                                               self.labels,
                                                              batch_size)
        else:
            start = self.index
            end = start + batch_size
            if end > len(self.data):
                print('Out of sample size')
                self.index = 0
            sents = self.data[start:end]
            sent_labels = self.labels[start:end]
            sent_vecs, lengths = self.create_sent_idx(sents)
            self.index = end
        return sent_vecs, sent_labels, lengths