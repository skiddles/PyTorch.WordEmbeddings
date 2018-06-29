from torch.utils.data import Dataset
import numpy as np
import pickle
import os
from CustomCorpus import corpus as cc
import argparse


class TrigramsDataset(Dataset):
    def __init__(self, params):
        self.params = params
        self.context_ids = None
        self.target_ids = None
        self.sentences = ''
        self.vocab = None
        self.word_to_ix = {}
        assert (params.clean_dir is not None), "Must include a clean corpus parameter."
        self.sentences = load_corpus(params.clean_dir)
        self.prepare_trigrams()

    def __len__(self):
        return len(self.context_ids)

    def __getitem__(self, idx):
        context = self.context_ids.iloc(idx)
        target = self.target_ids.iloc(idx)
        sample = {'context': context, 'target': target}
        return sample

    def prepare_trigrams(self, clip=False):
        # build a list of tuples.  Each tuple is ([ word_i-2, word_i-1 ], target word)
        trigrams = [([self.sentences[i], self.sentences[i + 1]], self.sentences[i + 2])
                    for i in range(len(self.sentences) - 2)]

        self.vocab = set(self.sentences)
        self.word_to_ix = {word: i for i, word in enumerate(self.vocab)}

        context_idxs = np.zeros([len(trigrams), 2], dtype=np.long)
        target_idxs = np.zeros([len(trigrams), 1], dtype=np.long)

        i = 0
        for context, target in trigrams:
            # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
            # into integer indices and wrap them in variables)
            ids = [self.word_to_ix[w] for w in context]
            context_idxs[i, 0] = ids[0]
            context_idxs[i, 1] = ids[1]
            target_idxs[i, 0] = [self.word_to_ix[target]][0]
            i += 1

        if not clip:
            self.context_ids = context_idxs
            self.target_ids = target_idxs
        else:
            self.context_ids = context_idxs[0:clip]
            self.target_ids = target_idxs[0:clip]

def load_corpus(in_clean, in_root=None, in_reload=False):
    # This tests for a clean file.  If it does not exist, it asserts that the directory
    # is at least accessible to save a clean file in.
    if not os.path.exists(in_clean):
        # in_reload = True
        assert (os.path.exists(os.path.dirname(in_clean))), "%s does not exist. Please check for the" \
                                                            " file and reload from the root directory" \
                                                            " using --reload and --root." % in_clean

        print("Could not load corpus from %s. Creating corpus." % in_clean)
        beigebooks = cc.Corpus()
        beigebooks.set_root(in_root)
        beigebooks.load_corpus()
        beigebooks.cleanse()
        sentences = beigebooks.return_documents(style='JOINED')
        pickle.dump(sentences, open(in_clean, 'wb'))
        print("Corpus saved to %s" % in_clean)

        return sentences
    # This tests for a clean file and will use the file if the reload flag is not set
    # to True, meaning reload from the corpus
    elif os.path.exists(in_clean) and not in_reload:
        print("Loading corpus from %s" % in_clean)
        sentences = pickle.load(open(in_clean, 'rb'))
        return sentences

    elif in_reload:
        print("in_root: ", repr(in_root))
        print(type(in_root))
        assert (os.path.exists(in_root)), 'the root directory is not provided or invalid. A' \
                                          ' valid root must be provided if reloading data from the corpus.' \
                                          ' Root passed is \n%s' % in_root
        return False


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', dest='root_dir', default=None,
                        help='the root directory of the corpus')
    parser.add_argument('--clean', dest='clean_dir',  required=True,
                        help='the directory to save the cleansed text to or load from')
    parser.add_argument('--reload', dest='reload', action='store_true')
    return parser.parse_args()


""" Call this main function directly from the command line to recreate the corpus file. You
    do not need to call it every time only when there is new data in the corpus or the 
    corpus.pkl file does not exists.  The corpus.pkl file is the cleansed corpus. 
"""
if __name__ == '__main__':
    params = get_args()

    corpus = load_corpus(in_root=params.root_dir, in_clean=params.clean_dir)
