import TrigramDataset.trigrams_dataset as tgd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import time
import argparse
import copy
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
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean', dest='clean_dir',  required=True,
                        help='the directory to load the cleansed corpus from')
    parser.add_argument('--clip', dest='clip', required=False,
                        help='Limit the number of records for debugging purposes')
    parser.add_argument('--save', dest='save_dir', required=True,
                        help='the directory to save and load the in-progress model from')
    parser.add_argument('--batch-size', dest='MINI_BATCH', type=int, default=1000, metavar='N',
                        help='input batch size for training (default: 64)')
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


if __name__ == '__main__':
    parms = get_args()

    torch.manual_seed(1)

    text_data = tgd.TrigramsDataset(parms)
    if parms.clip:
        text_data.prepare_trigrams(parms.clip)
    else:
        text_data.prepare_trigrams()

    dataloader = DataLoader(text_data,
                            batch_size=parms.MINI_BATCH,
                            shuffle=True,
                            num_workers=3)

    model = NGramLanguageModeler(len(text_data.vocab), parms.EMBEDDING_DIM, parms.CONTEXT_SIZE)
    model.share_memory()
    # Define an optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    if not os.path.exists(parms.save_dir):
        if not os.path.exists(os.path.split(parms.save_dir)[0]):
            os.mkdir(os.path.split(parms.save_dir)[0])
        assert (os.path.exists(os.path.split(parms.save_dir)[0])), "It appears that the save_dir could not be created."
        parms.start_epoch = 0
    else:
        checkpoint = torch.load(parms.save_dir)
        parms.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(parms.save_dir, checkpoint['epoch']))

        model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    losses = []
    loss_function = nn.NLLLoss()

    if parms.start_epoch>0:
        print("start_epoch is %d" % parms.start_epoch)
        EPOCH_RANGE = range(parms.start_epoch, parms.EPOCHS)
    else:
        print("start_epoch is 0")
        EPOCH_RANGE = range(0, parms.EPOCHS)

    for epoch in EPOCH_RANGE:

        final_model_wts = copy.deepcopy(model.state_dict())
        current_start = 0
        total_loss = torch.Tensor([0])
        s_elapsed_time = 0

        keep_going = True
        while keep_going:
            if current_start + parms.MINI_BATCH < len(text_data.target_ids):
                minibatchids = slice(current_start, current_start + parms.MINI_BATCH -1)
            else:
                minibatchids = slice(current_start, len(text_data.target_ids))
                keep_going = False

            t = time.process_time()
            # Step 2. Recall that torch *accumulates* gradients. Before passing in a
            # new instance, you need to zero out the gradients from the old
            # instance
            model.zero_grad()

            # Step 3. Run the forward pass, getting log probabilities over next
            # words
            log_probs = model(torch.Tensor(text_data.context_ids[minibatchids]).long())

            # Step 4. Compute your loss function. (Again, Torch wants the target
            # word wrapped in a variable)
            loss = loss_function(log_probs,
                                 torch.autograd.Variable(
                                     torch.squeeze(
                                        torch.Tensor(text_data.target_ids[minibatchids]).long()
                                        )
                                     )
                                 ) * parms.MINI_BATCH

            # Step 5. Do the backward pass and update the gradient
            loss.backward()

            optimizer.step()

            # Get the Python number from a 1-element Tensor by calling tensor.item()
            total_loss += loss.item()
            elapsed_time = time.process_time() - t
            s_elapsed_time += elapsed_time
            t_avg_time = elapsed_time/parms.MINI_BATCH
            s_avg_time = s_elapsed_time/(current_start + parms.MINI_BATCH)
            print('Epoch: %d  Batch Size: %s  Processed: %d  Batch Avg Time: %0.2f  EP Avg Time: %0.2f  '
                  'Loss: %0.2f  TotalLoss: %0.1f' % (epoch, str.rjust(str(parms.MINI_BATCH), 6, ' '),
                                                     current_start+parms.MINI_BATCH,
                                                     np.log10(t_avg_time), np.log10(s_avg_time),
                                                     loss.item(), total_loss.item()))
            if keep_going:
                current_start = current_start + parms.MINI_BATCH
        losses.append(total_loss)
        # Check out this site https://discuss.pytorch.org/t/saving-and-loading-a-model-in-pytorch/2610/3
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, parms.save_dir)

    # # load best model weights
    # final_model_wts = copy.deepcopy(model.state_dict())
    # model.load_state_dict(final_model_wts)

# The loss decreased every iteration over the training data!
# print(len(losses))
# print(losses[-1].item())



