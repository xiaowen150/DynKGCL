import argparse
import random
from time import time
from data_loader import load_data
from config import default_configs
from train import train
import tensorflow as tf

seed_value = 555
# np.random.seed(seed_value)
random.seed(seed_value)
tf.set_random_seed(seed_value)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True,
                    choices=list(default_configs.keys()),
                    help='choose dataset to run with default parameters')

args = parser.parse_args()

config = default_configs[args.dataset]
for k, v in config.items():
    if hasattr(args, k) and getattr(args, k) is not None:
        continue
    setattr(args, k, v)


def load_contrastive_pairs(dataset, n_user, n_item):
    """
    Load contrastive pairs from a file.
    Each line in the file should have the format: user_id, pos_item_id, neg_item_id, label
    """
    contrastive_pairs = []
    contrastive_file = '../data/' + dataset + '/contrastive_pairs.txt'

    with open(contrastive_file, 'r') as f:
        for line in f:
            user, pos_item, neg_item, label = line.strip().split('\t')
            user = int(user)
            pos_item = int(pos_item)
            neg_item = int(neg_item)
            label = float(label)

            if user < n_user and pos_item < n_item and neg_item < n_item:
                contrastive_pairs.append((user, pos_item, neg_item, label))
            else:
                print(f"Invalid index found: user {user}, pos_item {pos_item}, neg_item {neg_item}")

    print(f'Loaded {len(contrastive_pairs)} valid contrastive pairs.')
    return contrastive_pairs


show_loss = False
show_time = False
show_topk = True

t = time()

data = load_data(args)
n_user, n_item = data[0], data[1]

contrastive_pairs = load_contrastive_pairs(args.dataset, n_user, n_item)

train(args, data, contrastive_pairs, show_loss, show_topk)

if show_time:
    print('time used: %d s' % (time() - t))
