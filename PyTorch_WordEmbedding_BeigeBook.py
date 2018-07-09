import TrigramDataset.trigrams_dataset as tgd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import time
import argparse
import os


class NGramLanguageModeler(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super().__init__()
        print("Vocab contains %d words." % vocab_size)
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((len(inputs), -1))
        out1 = F.relu(self.linear1(embeds))
        out2 = self.linear2(out1)
        log_probs = F.log_softmax(out2, dim=1)
        return log_probs


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean', dest='clean_dir',  required=True, default=r'./TrigramDataset/trigrams_data/Corpus.pkl',
                        help='the directory to load the cleansed corpus from')
    parser.add_argument('--clip', dest='clip', required=False, default=None,
                        help='Limit the number of records for debugging purposes')
    parser.add_argument('--save', dest='save_dir', required=True, default=r"./data/final_model.md",
                        help='the directory to save and load the in-progress model from')
    parser.add_argument('--batch-size', dest='MINI_BATCH', type=int, default=1000, metavar='N',
                        help='input batch size for training (default: 1000)')
    parser.add_argument('--epochs', dest='EPOCHS', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--num-processes', type=int, default=2, metavar='N',
                        help='how many training processes to use (default: 2)')
    parser.add_argument('--embedding-dims', dest='EMBEDDING_DIM', type=int, default=300, metavar='N',
                        help='Need help (default: 300)')
    parser.add_argument('--context-size', dest='CONTEXT_SIZE', type=int, default=2, metavar='N',
                        help='Need help (default: 2)')
    return parser.parse_args()


def save_checkpoint(state, filename):
    torch.save(state, filename)


def train(model, optimizer):
    optimizer.zero_grad()
    # log_probs = model(torch.Tensor(text_data.context_ids[minibatchids]).long())
    for data in dataloader:
        log_probs = model(torch.LongTensor(data['context']))
        y = torch.autograd.Variable(
                torch.squeeze(torch.LongTensor(data['target'])))

        loss = loss_function(log_probs, y)
        loss.backward()
        optimizer.step()
    return loss


if __name__ == '__main__':

    loss_function = nn.NLLLoss()

    parms = get_args()
    torch.manual_seed(1)
    text_data = tgd.TrigramsDataset(parms)
    dataloader = DataLoader(text_data,
                            batch_size=parms.MINI_BATCH,
                            shuffle=False,
                            num_workers=2)

    vocab_length = len(text_data.get_vocab())
    model = NGramLanguageModeler(vocab_length, parms.EMBEDDING_DIM, parms.CONTEXT_SIZE)
    # model.share_memory()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    if not os.path.exists(parms.save_dir):
        if not os.path.exists(os.path.split(parms.save_dir)[0]):
            os.mkdir(os.path.split(parms.save_dir)[0])
        assert (os.path.exists(os.path.split(parms.save_dir)[0])), "It appears that the save_dir could not be created."
        parms.start_epoch = 0
        # weights_init(model, new=True)
    else:
        checkpoint = torch.load(parms.save_dir)
        parms.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(parms.save_dir, checkpoint['epoch']))
        f2 = open('./data/loaded_weights.txt', 'w')
        for x in model.parameters():
            f2.write(str(x.data))
        f2.close()


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    losses = []

    if parms.start_epoch>0:
        print("start_epoch is %d" % parms.start_epoch)
        EPOCH_RANGE = range(parms.start_epoch, parms.EPOCHS)
    else:
        print("start_epoch is 0")
        EPOCH_RANGE = range(0, parms.EPOCHS)

    for epoch in EPOCH_RANGE:
        current_start = 0
        total_loss = torch.Tensor([0])
        s_elapsed_time = 0
        t = time.process_time()

        loss = train(model, optimizer)
        total_loss += loss.item()
        elapsed_time = time.process_time() - t

        print("Epoch: %d Elapsed Time %0.2f  Loss: %0.6f" % (epoch, np.log10(elapsed_time/float(parms.clip)), loss.item()))
        losses.append(total_loss)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, parms.save_dir)

        f = open('./data/prev_weights.txt', 'w')
        for x in model.parameters():
            f.write(str(x.data))
        f.close()




