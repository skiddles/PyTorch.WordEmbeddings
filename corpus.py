import os
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class Corpus:

    rootdir = ''
    Documents_Docs = []  # A list of Documents, which contain a list of sentences, which contain a list of words
    carry_docs = []
    filenames = []
    tokenized = False

    def __init__(self):
        self.rootdir = ''

    def set_root(self, root_dir):
        self.rootdir = root_dir

    def load_corpus(self):
        for file_ix, file in enumerate(os.listdir(self.rootdir)):
            if file_ix > 0:
                pass
            self.filenames.append(file)
            f = open('/'.join([self.rootdir, file]), 'r', encoding='latin-1')
            raw = f.read()
            f.close()
            raw = re.sub('\n+', ' ', raw)
            self.Documents_Docs.append(raw)

    def return_documents(self, style='JOINED'):
        if style == 'JOINED':
            for i in range(len(self.Documents_Docs)):
                tokens = self.Documents_Docs[i]
                self.carry_docs.extend(tokens)
            return self.carry_docs
        else:
            assert(1==2), "This has not been tested yet"
            return self.Documents_Docs

    def cleanse(self, remove_pattern=r'([._-]{2,})',
                tokenize=True,
                lowercase=True,
                strip_punctuation=True,
                keep_only_alpha=True,
                stem=True,
                lemma=False,
                stop='ENGLISH'):

        # First apply remove_pattern
        if remove_pattern:
            for i in range(len(self.Documents_Docs)):
                self.Documents_Docs[i] = re.sub(remove_pattern, '', self.Documents_Docs[i])

        if tokenize:
            print("Tokenizing the text!")
            for i in range(len(self.Documents_Docs)):
                tokens = word_tokenize(self.Documents_Docs[i])
                self.carry_docs.append(tokens)
            self.Documents_Docs = self.carry_docs
            self.carry_docs = []

        if lowercase:
            print("Converting to lowercase!")
            for i in range(len(self.Documents_Docs)):
                tokens = self.Documents_Docs[i]
                l_tokens = [w.lower() for w in tokens]
                self.carry_docs.append(l_tokens)
            self.Documents_Docs = self.carry_docs
            self.carry_docs = []

        if strip_punctuation:
            print("Removing punctuation!")
            table = str.maketrans('', '', string.punctuation)
            for i in range(len(self.Documents_Docs)):
                tokens = self.Documents_Docs[i]
                s_tokens = [w.translate(table) for w in tokens]
                self.carry_docs.append(s_tokens)
            self.Documents_Docs = self.carry_docs
            self.carry_docs = []

        if keep_only_alpha:
            print("Removing non-alpha tokens!")
            for i in range(len(self.Documents_Docs)):
                tokens = self.Documents_Docs[i]
                a_tokens = [word for word in tokens if word.isalpha()]
                self.carry_docs.append(a_tokens)
            self.Documents_Docs = self.carry_docs
            self.carry_docs = []

        if stop:
            stop_words = ''
            print("Applying stopwords!")
            if stop == 'ENGLISH':
                stop_words = set(stopwords.words('english'))
            else:
                stop_words = set(stop)

            for i in range(len(self.Documents_Docs)):
                tokens = self.Documents_Docs[i]
                s_tokens = [w for w in tokens if not w in stop_words]
                self.carry_docs.append(s_tokens)
            self.Documents_Docs = self.carry_docs
            self.carry_docs = []

        if stem:
            pass

        if lemma:
            pass
