import sys
import torch
from torch.autograd import Variable
import numpy as np

cuda = torch.cuda.is_available()


def predict_labels(Q, X):
    Q.eval()

    latent_y = Q(X)[0]
    pred_labels = torch.argmax(latent_y, dim=1)
    return pred_labels

def get_categorial(label, n_classes=10):
    latent_y = np.eye(n_classes)[label].astype('float32')
    latent_y = torch.from_numpy(latent_y)
    return Variable(latent_y)

def sample_categorical(batch_size, n_classes=10, label=None):
    '''
     Sample from a categorical distribution
     of size batch_size and # of classes n_classes
     return: torch.autograd.Variable with the sample
    '''
    cat = np.random.randint(0, 10, batch_size)
    cat = np.eye(n_classes)[cat].astype('float32')
    cat = torch.from_numpy(cat)
    return Variable(cat)

def classification_accuracy(Q, data_loader):
    correct = 0
    N = len(data_loader.dataset)

    for batch_idx, (X, target) in enumerate(data_loader):

        X.resize_(data_loader.batch_size, Q.input_size)
        X, target = Variable(X), Variable(target)
        if cuda:
            X, target = X.cuda(), target.cuda()

        # encoding phase
        pred = predict_labels(Q, X)
        correct += pred.eq(target.data).cpu().sum()

    return 100. * correct / N

def zero_grad_all(*models):
    [m.zero_grad() for m in models]

def train_all(*models):
    [m.train() for m in models]

def eval_all(*models):
    [m.eval() for m in models]

def report_loss(epoch, all_losses, descriptions):
    '''
    Print loss.
    '''
    base_loss_report = '\nEpoch-{}; '.format(epoch)

    for loss, desc in zip(all_losses, descriptions):
        base_loss_report += '{}: {:.4}; '.format(desc, loss.item())

    print(base_loss_report)

def report_progress(percent, barLen=20):
    sys.stdout.write("\rcurrent epoch:: ")
    progress = ""
    for i in range(barLen):
        if i < int(barLen * percent):
            progress += "="
        else:
            progress += " "
    sys.stdout.write("[ %s ] %.2f%%" % (progress, percent * 100))
    sys.stdout.flush()
