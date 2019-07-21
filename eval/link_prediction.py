from __future__ import division, print_function
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn import linear_model
from collections import defaultdict
import random

np.random.seed(123)
operatorTypes = ["HAD"]


def write_to_csv(test_results, output_name, model_name, dataset, time_steps, mod='val'):
    """Output result scores to a csv file for result logging"""
    with open(output_name, 'a+') as f:
        for op in test_results:
            print("{} results ({})".format(model_name, mod), test_results[op])
            _, best_auc = test_results[op]
            f.write("{},{},{},{},{},{},{}\n".format(dataset, time_steps, model_name, op, mod, "AUC", best_auc))


def get_link_score(fu, fv, operator):
    """Given a pair of embeddings, compute link feature based on operator (such as Hadammad product, etc.)"""
    fu = np.array(fu)
    fv = np.array(fv)
    if operator == "HAD":
        return np.multiply(fu, fv)
    else:
        raise NotImplementedError


def get_link_feats(links, source_embeddings, target_embeddings, operator):
    """Compute link features for a list of pairs"""
    features = []
    for l in links:
        a, b = l[0], l[1]
        f = get_link_score(source_embeddings[a], target_embeddings[b], operator)
        features.append(f)
    return features


def get_random_split(train_pos, train_neg, val_pos, val_neg, test_pos, test_neg):
    """ Randomly split a given set of train, val and test examples"""
    all_data_pos = []
    all_data_neg = []

    all_data_pos.extend(train_pos)
    all_data_neg.extend(train_neg)
    all_data_pos.extend(test_pos)
    all_data_neg.extend(test_neg)

    # re-define train_pos, train_neg, test_pos, test_neg.
    random.shuffle(all_data_pos)
    random.shuffle(all_data_neg)

    train_pos = all_data_pos[:int(0.2 * len(all_data_pos))]
    train_neg = all_data_neg[:int(0.2 * len(all_data_neg))]

    test_pos = all_data_pos[int(0.2 * len(all_data_pos)):]
    test_neg = all_data_neg[int(0.2 * len(all_data_neg)):]
    print("# train :", len(train_pos) + len(train_neg), "# val :", len(val_pos) + len(val_neg),
          "#test :", len(test_pos) + len(test_neg))
    return train_pos, train_neg, val_pos, val_neg, test_pos, test_neg


def evaluate_classifier(train_pos, train_neg, val_pos, val_neg, test_pos, test_neg, source_embeds, target_embeds):
    """Downstream logistic regression classifier to evaluate link prediction"""
    test_results = defaultdict(lambda: [])
    val_results = defaultdict(lambda: [])

    test_auc = get_roc_score_t(test_pos, test_neg, source_embeds, target_embeds)
    val_auc = get_roc_score_t(val_pos, val_neg, source_embeds, target_embeds)

    # Compute AUC based on sigmoid(u^T v) without classifier training.
    test_results['SIGMOID'].extend([test_auc, test_auc])
    val_results['SIGMOID'].extend([val_auc, val_auc])

    test_pred_true = defaultdict(lambda: [])
    val_pred_true = defaultdict(lambda: [])

    for operator in operatorTypes:
        train_pos_feats = np.array(get_link_feats(train_pos, source_embeds, target_embeds, operator))
        train_neg_feats = np.array(get_link_feats(train_neg, source_embeds, target_embeds, operator))
        val_pos_feats = np.array(get_link_feats(val_pos, source_embeds, target_embeds, operator))
        val_neg_feats = np.array(get_link_feats(val_neg, source_embeds, target_embeds, operator))
        test_pos_feats = np.array(get_link_feats(test_pos, source_embeds, target_embeds, operator))
        test_neg_feats = np.array(get_link_feats(test_neg, source_embeds, target_embeds, operator))

        train_pos_labels = np.array([1] * len(train_pos_feats))
        train_neg_labels = np.array([-1] * len(train_neg_feats))
        val_pos_labels = np.array([1] * len(val_pos_feats))
        val_neg_labels = np.array([-1] * len(val_neg_feats))

        test_pos_labels = np.array([1] * len(test_pos_feats))
        test_neg_labels = np.array([-1] * len(test_neg_feats))
        train_data = np.vstack((train_pos_feats, train_neg_feats))
        train_labels = np.append(train_pos_labels, train_neg_labels)

        val_data = np.vstack((val_pos_feats, val_neg_feats))
        val_labels = np.append(val_pos_labels, val_neg_labels)

        test_data = np.vstack((test_pos_feats, test_neg_feats))
        test_labels = np.append(test_pos_labels, test_neg_labels)

        logistic = linear_model.LogisticRegression()
        logistic.fit(train_data, train_labels)
        test_predict = logistic.predict_proba(test_data)[:, 1]
        val_predict = logistic.predict_proba(val_data)[:, 1]

        test_roc_score = roc_auc_score(test_labels, test_predict)
        val_roc_score = roc_auc_score(val_labels, val_predict)

        val_results[operator].extend([val_roc_score, val_roc_score])
        test_results[operator].extend([test_roc_score, test_roc_score])

        val_pred_true[operator].extend(zip(val_predict, val_labels))
        test_pred_true[operator].extend(zip(test_predict, test_labels))

    return val_results, test_results, val_pred_true, test_pred_true


def get_roc_score_t(edges_pos, edges_neg, source_emb, target_emb):
    """Given test examples, edges_pos: +ve edges, edges_neg: -ve edges, return ROC scores for a given snapshot"""
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(source_emb, target_emb.T)
    pred = []
    pos = []
    for e in edges_pos:
        pred.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(1.0)

    pred_neg = []
    neg = []
    for e in edges_neg:
        pred_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(0.0)

    pred_all = np.hstack([pred, pred_neg])
    labels_all = np.hstack([np.ones(len(pred)), np.zeros(len(pred_neg))])
    roc_score = roc_auc_score(labels_all, pred_all)
    return roc_score
