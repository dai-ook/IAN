import torch
import logging
import time
def loadLogger(dataset, name):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt="[ %(asctime)s ] %(message)s",
                                  datefmt="%a %b %d %H:%M:%S %Y")
    sHandler = logging.StreamHandler()
    sHandler.setFormatter(formatter)
    logger.addHandler(sHandler)
    fHandler = logging.FileHandler('./result' + '/{0}-{1}-{2}-log.txt'.format(name, dataset, time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())), mode='w')
    fHandler.setLevel(logging.DEBUG)
    fHandler.setFormatter(formatter)
    logger.addHandler(fHandler)

    return logger
# Precision
def get_recall(pre, truth, j=None, index8= None):
    """
    :param pre: (B,K) TOP-K indics predicted by the model
    :param truth: (B,1) the truth value of test samples
    :return: recall(Float), the recall score
    """
    truths = truth.expand_as(pre)
    hits = (pre == truths).nonzero()
    hits8 = hits.transpose(0, 1)
    hits_and8 = []
    t = hits8[0]
    if j == 19:
        # h = hits.cpu().numpy()
        # np.savetxt('test222_layer1.csv',h)
        for i in index8:
            if i in hits8[0]:
                hits_and8.append(i)
    if len(hits) == 0:
        return 0
    n_hits = (pre == truths).nonzero().size(0)
    recall = n_hits / truths.size(0)
    return recall

# MRR
def get_mrr(pre, truth):
    """
    :param pre: (B,K) TOP-K indics predicted by the model
    :param truth: (B, 1) real label
    :return: MRR(Float), the mrr score
    """
    targets = truth.view(-1, 1).expand_as(pre)
    # ranks of the targets, if it appears in your indices
    hits = (targets == pre).nonzero()
    if len(hits) == 0:
        return 0
    ranks = hits[:, -1] + 1
    ranks = ranks.float()
    r_ranks = torch.reciprocal(ranks)  # reciprocal ranks
    mrr = torch.sum(r_ranks).data / targets.size(0)
    return mrr


    """
    :param pre: (B,K) TOP-K indics predicted by the model
    :param truth: (B,1) the truth value of test samples
    :return: recall(Float), the recall score
    """
    truths = truth.expand_as(pre)
    hits = (pre == truths).nonzero()

    hits8 = hits.transpose(0, 1)
    hits_and8 = []
    t = hits8[0]
    if j == 19:
        # h = hits.cpu().numpy()
        # np.savetxt('test222_layer1.csv',h)
        for i in index8:
            if i in hits8[0]:
                hits_and8.append(i)
    if len(hits) == 0:
        return 0

    top_20 = (pre == truths).nonzero()
    n_hits = (pre == truths).nonzero().size(0)

    top_20[top_20.nonzero()]

    recall = n_hits / truths.size(0)
    return diversity_all