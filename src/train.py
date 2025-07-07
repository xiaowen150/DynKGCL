import os

import tensorflow as tf
import numpy as np
from model import DynKGCL
import time
import psutil
def train(args, data, contrastive_pairs, show_loss, show_topk):
    n_user, n_item, n_entity, n_relation = data[0], data[1], data[2], data[3]
    train_data, eval_data, test_data = data[4], data[5], data[6]
    adj_entity, adj_relation = data[7], data[8]

    model = DynKGCL(args, n_user, n_entity, n_relation, adj_entity, adj_relation)

    # top-K evaluation settings
    user_list, train_record, test_record, item_set, k_list = topk_settings(show_topk, train_data, test_data, n_item)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        best_eval_auc = 0
        best_eval_f1 = 0
        patience =2
        no_improve_epochs = 0
        epoch_times = []

        for step in range(args.n_epochs):
            epoch_start_time = time.time()  

            np.random.shuffle(train_data)
            start = 0

            while start + args.batch_size <= train_data.shape[0]:
                feed_dict = get_feed_dict(model, train_data, start, start + args.batch_size)
                _, loss = model.train(sess, feed_dict)
                start += args.batch_size
                if show_loss:
                    print("CTR Loss at step {}: {}".format(start, loss))


            contrastive_loss = contrastive_learning(sess, model, contrastive_pairs, args.batch_size)
            print("Contrastive Loss at epoch {}: {:.4f}".format(step, contrastive_loss))

 
            train_auc, train_f1 = ctr_eval(sess, model, train_data, args.batch_size)
            eval_auc, eval_f1 = ctr_eval(sess, model, eval_data, args.batch_size)
            test_auc, test_f1 = ctr_eval(sess, model, test_data, args.batch_size)

            print('epoch %d    train auc: %.4f  f1: %.4f    eval auc: %.4f  f1: %.4f    test auc: %.4f  f1: %.4f'
                  % (step, train_auc, train_f1, eval_auc, eval_f1, test_auc, test_f1))
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time  
            epoch_times.append(epoch_duration)
            print("Epoch {} duration: {:.2f} seconds".format(step, epoch_duration))  
            # Check if early stopping should be triggered
            if eval_auc > best_eval_auc:
                best_eval_auc = eval_auc
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            # top-K evaluation only if we're on the last epoch before early stopping
            if no_improve_epochs == patience - 1:
                precision, recall, hr, ndcg = topk_eval(
                    sess, model, user_list, train_record, test_record, item_set, k_list, args.batch_size)

                print('Precision: ', end='')
                for i in precision:
                    print('%.4f\t' % i, end='')
                print()

                print('Recall: ', end='')
                for i in recall:
                    print('%.4f\t' % i, end='')
                print()

                print('HR: ', end='')
                for i in hr:
                    print('%.4f\t' % i, end='')
                print()

                print('NDCG: ', end='')
                for i in ndcg:
                    print('%.4f\t' % i, end='')
                print('\n')

            # Early stopping check
            if no_improve_epochs >= patience:
                print(f"Early stopping triggered after {no_improve_epochs} epochs without improvement.")
                break

        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        total_time = sum(epoch_times)
        print(f"Average epoch time: {avg_epoch_time:.2f} seconds")
        print(f"Total training time: {total_time:.2f} seconds")


        process = psutil.Process(os.getpid())
        mem_used = process.memory_info().rss / 1024 / 1024
        print(f"Peak memory usage: {mem_used:.2f} MB")


def contrastive_learning(sess, model, contrastive_pairs, batch_size):
    """
    Perform contrastive learning using the positive and negative pairs.
    """
    start = 0
    losses = []
    while start + batch_size <= len(contrastive_pairs):
        feed_dict = get_contrastive_feed_dict(model, contrastive_pairs, start, start + batch_size)
        _, loss = sess.run([model.contrastive_optimizer, model.contrastive_loss], feed_dict)
        losses.append(loss)
        start += batch_size

    # Padding last batch if necessary
    if start < len(contrastive_pairs):
        feed_dict = get_contrastive_feed_dict(model, contrastive_pairs, start, len(contrastive_pairs))
        _, loss = sess.run([model.contrastive_optimizer, model.contrastive_loss], feed_dict)
        losses.append(loss)

    return np.mean(losses)


def get_contrastive_feed_dict(model, contrastive_pairs, start, end):
    users = [pair[0] for pair in contrastive_pairs[start:end]]
    items_1 = [pair[1] for pair in contrastive_pairs[start:end]]
    items_2 = [pair[2] for pair in contrastive_pairs[start:end]]
    labels = [pair[3] for pair in contrastive_pairs[start:end]]

    feed_dict = {model.user_indices: users,
                 model.pos_item_indices: items_1,
                 model.neg_item_indices: items_2,
                 model.contrastive_labels: labels}
    return feed_dict


def topk_settings(show_topk, train_data, test_data, n_item):
    if show_topk:
        user_num = 100
        k_list = [1, 2, 5, 10, 20, 50, 100]
        # k_list = [10, 20]
        train_record = get_user_record(train_data, True)
        test_record = get_user_record(test_data, False)
        user_list = list(set(train_record.keys()) & set(test_record.keys()))
        if len(user_list) > user_num:
            user_list = np.random.choice(user_list, size=user_num, replace=False)
        item_set = set(list(range(n_item)))
        return user_list, train_record, test_record, item_set, k_list
    else:
        return [None] * 5


def get_feed_dict(model, data, start, end, contrastive=False, contrastive_pairs=None):
    if not contrastive:
        feed_dict = {
            model.user_indices: data[start:end, 0],
            model.item_indices: data[start:end, 1],
            model.labels: data[start:end, 2],
            model.pos_item_indices: np.zeros((end - start,), dtype=np.int64),
            model.neg_item_indices: np.zeros((end - start,), dtype=np.int64),
            model.contrastive_labels: np.zeros((end - start,), dtype=np.float32)
        }
    else:
        users = [pair[0] for pair in contrastive_pairs[start:end]]
        items_1 = [pair[1] for pair in contrastive_pairs[start:end]]
        items_2 = [pair[2] for pair in contrastive_pairs[start:end]]
        labels = [pair[3] for pair in contrastive_pairs[start:end]]

        feed_dict = {
            model.user_indices: users,
            model.pos_item_indices: items_1,
            model.neg_item_indices: items_2,
            model.contrastive_labels: labels
        }

    return feed_dict


def ctr_eval(sess, model, data, batch_size):
    start = 0
    auc_list = []
    f1_list = []
    while start + batch_size <= data.shape[0]:
        auc, f1 = model.eval(sess, get_feed_dict(model, data, start, start + batch_size))
        auc_list.append(auc)
        f1_list.append(f1)
        start += batch_size
    return float(np.mean(auc_list)), float(np.mean(f1_list))


def topk_eval(sess, model, user_list, train_record, test_record, item_set, k_list, batch_size):
    precision_list = {k: [] for k in k_list}
    recall_list = {k: [] for k in k_list}
    hr_list = {k: [] for k in k_list}
    ndcg_list = {k: [] for k in k_list}

    for user in user_list:
        test_item_list = list(item_set - train_record[user])
        item_score_map = dict()
        start = 0
        while start + batch_size <= len(test_item_list):
            items, scores = model.get_scores(sess, {model.user_indices: [user] * batch_size,
                                                    model.item_indices: test_item_list[start:start + batch_size]})
            for item, score in zip(items, scores):
                item_score_map[item] = score
            start += batch_size

        # padding the last incomplete minibatch if exists
        if start < len(test_item_list):
            items, scores = model.get_scores(
                sess, {model.user_indices: [user] * batch_size,
                       model.item_indices: test_item_list[start:] + [test_item_list[-1]] * (
                               batch_size - len(test_item_list) + start)})
            for item, score in zip(items, scores):
                item_score_map[item] = score

        item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
        item_sorted = [i[0] for i in item_score_pair_sorted]

        for k in k_list:
            hit_num = len(set(item_sorted[:k]) & test_record[user])
            precision_list[k].append(hit_num / k)
            recall_list[k].append(hit_num / len(test_record[user]))

            # Calculate Hit Rate (HR)
            hr = 1.0 if len(set(item_sorted[:k]) & test_record[user]) > 0 else 0.0
            hr_list[k].append(hr)

            # Calculate NDCG
            dcg = 0.0
            for rank, item in enumerate(item_sorted[:k]):
                if item in test_record[user]:
                    dcg += 1.0 / np.log2(rank + 2)
            idcg = sum((1.0 / np.log2(i + 2) for i in range(min(len(test_record[user]), k))))
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_list[k].append(ndcg)

    precision = [np.mean(precision_list[k]) for k in k_list]
    recall = [np.mean(recall_list[k]) for k in k_list]
    hr = [np.mean(hr_list[k]) for k in k_list]
    ndcg = [np.mean(ndcg_list[k]) for k in k_list]

    return precision, recall, hr, ndcg


def get_user_record(data, is_train):
    user_history_dict = dict()
    for interaction in data:
        user = interaction[0]
        item = interaction[1]
        label = interaction[2]
        if is_train or label == 1:
            if user not in user_history_dict:
                user_history_dict[user] = set()
            user_history_dict[user].add(item)
    return user_history_dict
