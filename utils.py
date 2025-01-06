import sys
import os
import torch
import random
import math
import csv
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, average_precision_score, recall_score, \
    precision_score, precision_recall_curve
from sklearn.metrics import auc as PRAUC
from numpy import argmax
import copy
import scipy.sparse as sp
import numpy as np
from scipy import sparse
import dgl
from scipy.stats import multivariate_normal
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd
import pickle
import stat_rnn






def get_metrics(target_edges, org_adj, reconstructed_adj):
    reconstructed_adj =  sparse.csr_matrix(torch.sigmoid(reconstructed_adj).detach().numpy())
    org_adj = sparse.csr_matrix(org_adj)
    prediction = []
    true_label = []
    counter = 0
    for edge in target_edges:
        prediction.append(reconstructed_adj[edge[0], edge[1]])
        prediction.append(reconstructed_adj[edge[1], edge[0]])
        true_label.append(org_adj[edge[0], edge[1]])
        true_label.append(org_adj[edge[1], edge[0]])

    pred = np.array(prediction)
    
    
    precision, recall, thresholds = precision_recall_curve(true_label, pred)
    filter = recall >= 0.8  # or any other recall level you deem necessary
    best_threshold = thresholds[np.argmax(precision[filter])] if any(filter) else 0.5
    Threshold = best_threshold
    pr_auc = PRAUC(recall, precision)

    # fscore = (2 * precision * recall) / (precision + recall)
    # ix = argmax(fscore)
    # Threshold = thresholds[ix]
    # Threshold = 0.5
    # thresholds = np.append(thresholds, 1)
    # acc = [accuracy_score(true_label, prediction >= t) for t in thresholds]
    
    pred[pred > Threshold] = 1.0
    pred[pred < Threshold] = 0.0
    pred = pred.astype(int)


    precision = precision_score(y_pred=pred, y_true=true_label)
    recall = recall_score(y_pred=pred, y_true=true_label)
    auc = roc_auc_score(y_score=prediction, y_true=true_label)
    acc = accuracy_score(y_pred=pred, y_true=true_label, normalize=True)
    ap = average_precision_score(y_score=prediction, y_true=true_label)

    hr_ind = np.argpartition(np.array(prediction), -1*len(pred)//5)[-1*len(pred)//5:] # dividing by 5 to get top 20%
    HR = precision_score(y_pred=np.array(pred)[hr_ind], y_true=np.array(true_label)[hr_ind])
    
    
    return auc, acc, ap, precision, recall, HR, np.max(thresholds)



# def roc_auc_single(prediction, true_label):
#     pred = np.array(prediction)
#     pred[pred > .5] = 1
#     pred[pred < .5] = 0
#     pred = pred.astype(int)
#     # pred = prob_to_one_hot(pred)

#     precision = precision_score(y_pred=pred, y_true=true_label)
#     recall = recall_score(y_pred=pred, y_true=true_label)
#     auc = roc_auc_score(y_score=prediction, y_true=true_label)
#     acc = accuracy_score(y_pred=pred, y_true=true_label, normalize=True)
#     ap = average_precision_score(y_score=prediction, y_true=true_label)
#     hr_ind = np.argpartition(np.array(prediction), -1*len(pred)//5)[-1*len(pred)//5:] # dividing by 5 to get top 20%
#     HR = precision_score(y_pred=np.array(pred)[hr_ind], y_true=np.array(true_label)[hr_ind])
#     pred = np.array(prediction)
    
#     return auc, acc, ap, precision, recall, HR



def roc_auc_single(predictions_list, true_labels_list):
    # Lists to store metrics for each adjacency matrix
    all_aucs = []
    all_accs = []
    all_aps = []
    all_precisions = []
    all_recalls = []
    all_HRs = []
    
    # Process each adjacency matrix's predictions and true labels
    for pred_single, true_single in zip(predictions_list, true_labels_list):
        # Convert to numpy array
        pred = np.array(pred_single)
        
        # Threshold predictions
        pred_binary = pred.copy()
        pred_binary[pred_binary > .5] = 1
        pred_binary[pred_binary < .5] = 0
        pred_binary = pred_binary.astype(int)
        
        # Calculate metrics
        precision = precision_score(y_pred=pred_binary, y_true=true_single)
        recall = recall_score(y_pred=pred_binary, y_true=true_single)
        auc = roc_auc_score(y_score=pred_single, y_true=true_single)
        acc = accuracy_score(y_pred=pred_binary, y_true=true_single, normalize=True)
        ap = average_precision_score(y_score=pred_single, y_true=true_single)
        
        # Calculate HR for top 20%
        hr_ind = np.argpartition(np.array(pred_single), -1*len(pred_binary)//5)[-1*len(pred_binary)//5:]
        HR = precision_score(y_pred=np.array(pred_binary)[hr_ind], 
                           y_true=np.array(true_single)[hr_ind])
        
        # Store metrics
        all_aucs.append(auc)
        all_accs.append(acc)
        all_aps.append(ap)
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_HRs.append(HR)
    
    return all_aucs, all_accs, all_aps, all_precisions, all_recalls, all_HRs




def roc_auc_estimator_labels(re_labels, labels, org_labels):
    prediction = []
    true_label = []

    for i in range(len(labels)):
        prediction.append(re_labels[i].detach().numpy())
        true_label.append(labels[i].detach().numpy())
    prediction = np.array(prediction)
    true_label = np.array(true_label)
    num_classes = true_label.shape[1]  # Number of classes
    # pred = prediction
    # pred =
    # pred[pred > .5] = 1.0
    # pred[pred < .5] = 0.0
    # pred = pred.astype(int)
    pred = prob_to_one_hot(prediction)

    precision = precision_score(y_pred=pred, y_true=true_label, average="weighted")
    recall = recall_score(y_pred=pred, y_true=true_label, average="weighted")

    roc_auc_scores = []
    seen_classes = 0

    for i in range(num_classes):
        # Calculate ROC-AUC for each class
        y_true = torch.from_numpy(true_label[:, i])
        y_pred = torch.from_numpy(prediction[:, i])
        y_true = torch.cat([y_true, torch.tensor([0])])
        y_pred = torch.cat([y_pred, torch.tensor([0])])
        if len(y_true.nonzero()) > 0:
            seen_classes += 1
            roc_auc = roc_auc_score(y_true, y_pred)
            roc_auc_scores.append(roc_auc)

    average_roc_auc = sum(roc_auc_scores) / seen_classes


    acc = accuracy_score(y_pred=pred, y_true=true_label)
    ap = average_precision_score(y_score=prediction, y_true=true_label)

    f1_score_macro = f1_score(true_label, pred, average ="macro")
    return average_roc_auc, acc, ap, precision, recall, f1_score_macro

def prob_to_one_hot(y_pred):
    ret = np.zeros(y_pred.shape)
    indices = np.argmax(y_pred, axis=1)
    for i in range(y_pred.shape[0]):
        ret[i][indices[i]] = 1
    return ret

# std_z_recog, m_z_recog, z_recog, re_adj_recog, re_feat_recog, re_recog_labels = run_network(features, org_adj_list, labels, inductive_model, targets, sampling_method,
#                                                             is_prior=False)

def run_network(feats, adj_list, labels, model, targets, sampling_method, is_prior):
    graph_dgl = []

    pre_self_loop_train_adj = []

    train_matrix = []

    for adj in adj_list:

        sparse_adj = sparse.csr_matrix(adj)

        pre_self_loop_train_adj.append(sparse_adj)

        tr_matrix = sparse_adj + sp.eye(adj.shape[0])

        train_matrix.append(tr_matrix.todense())

        src, dst = tr_matrix.nonzero()

        graph_dgl.append(dgl.graph((src, dst), num_nodes=adj.shape[0]))
    std_z, m_z, z, re_adj, reconstructed_feat, reconstructed_labels = model(graph_dgl, feats, labels, targets, sampling_method, is_prior, train=False)
    return std_z, m_z, z, re_adj, reconstructed_feat, reconstructed_labels


def get_pdf(mean_p, std_p, mean_q, std_q, z, targets):

    pdf_all_z_p = 0
    pdf_all_z_q = 0
    for i in targets:
        # TORCH
        cov_p = np.diag(std_p.detach().numpy()[i] ** 2)
        dist_p = torch.distributions.multivariate_normal.MultivariateNormal(mean_p[i], torch.tensor(cov_p))
        pdf_all_z_p += dist_p.log_prob(z[i]).detach().numpy()

        cov_q = np.diag(std_q.detach().numpy()[i] ** 2)
        dist_q = torch.distributions.multivariate_normal.MultivariateNormal(mean_q[i], torch.tensor(cov_q))
        pdf_all_z_q += dist_q.log_prob(z[i]).detach().numpy()
    return pdf_all_z_p, pdf_all_z_q

def weight_labels(labels):
    n_samples = labels.shape[0]
    labels_ind = torch.argmax(torch.from_numpy(labels), dim=1)
    class_counts = torch.bincount(labels_ind)
    class_weights = []
    num_classes = labels.shape[1]
    for i in range(0,num_classes):
        class_weights.append(n_samples/(class_counts[i]*num_classes))
    return torch.tensor(class_weights)
    # labels = torch.argmax(torch.from_numpy(labels), dim=1)
    # # labels = torch.from_numpy(labels)
    # class_counts = torch.bincount(labels)
    #
    # # Calculate the total number of samples
    # total_samples = len(labels)
    #
    # # Calculate class frequencies (class_counts / total_samples)
    # class_frequencies = class_counts.float() / total_samples
    #
    # # Calculate inverse class frequencies to use as class weights
    # class_weights = 1.0 / class_frequencies
    # class_weights /= class_weights.sum()


def weight_edges(labels):
    # labels = torch.from_numpy(labels)
    n_samples = labels.shape[0]*labels.shape[1]
    # labels_ind = torch.argmax(torch.from_numpy(labels), dim=1)
    class_counts = torch.tensor([(labels.shape[0] ** 2 - torch.sum(labels)),torch.sum(labels) ])
    class_weights = []
    num_classes = 2
    for i in range(0,num_classes):
        class_weights.append(n_samples/(class_counts[i]*num_classes))
    return torch.tensor(class_weights)

def test(test_edges, org_adj, run_network, features, labels, inductive_model, targets, sampling_method):


    adj_list_copies = []

    for adj in org_adj:
        adj_copy = copy.deepcopy(adj)
        
        for i, j in test_edges:
            adj_copy[i][j] = 0
            
        adj_list_copies.append(adj_copy)

    std_z_prior, m_z_prior, z_prior, re_adj_prior, re_feat_prior, re_prior_labels = run_network(features,
                                                                                                adj_list_copies,
                                                                                                labels,
                                                                                                inductive_model,
                                                                                                targets,
                                                                                                sampling_method,
                                                                                                is_prior=True)
    re_adj_prior_sig = torch.sigmoid(re_adj_prior)
    re_label_prior_sig = torch.sigmoid(re_prior_labels)
    pred_single_link = [[] for _ in range(len(re_adj_prior))]
    true_single_link = [[] for _ in range(len(re_adj_prior))]
    pred_single_label = []
    true_single_label = []
    for i, j in test_edges:
        for adj_idx in range(len(re_adj_prior)):
            pred_single_link[adj_idx].append(re_adj_prior_sig[adj_idx][i][j].detach().numpy())
            true_single_link[adj_idx].append(org_adj[adj_idx][i][j])
        pred_single_label.append(re_label_prior_sig[i])
        true_single_label.append(labels[i])
    auc, val_acc, val_ap, precision, recall, HR = roc_auc_single(pred_single_link, true_single_link)
    auc_l, acc_l, ap_l, precision_l, recall_l, F1_score = roc_auc_estimator_labels(pred_single_label, true_single_label,
                                                                                   labels)
    return auc, val_acc, val_ap, precision, recall, HR, auc_l, acc_l, ap_l, precision_l, recall_l, F1_score


def reduce_node_features(x, y, random_seed, n_components=5):
    np.random.seed(random_seed)
    model = ExtraTreesClassifier()
    model.fit(x, y)
    feat_importances = pd.Series(model.feature_importances_)
    important_feats = np.array(feat_importances.nlargest(n_components).index)
    x_reduced = x[:, important_feats]
    return x_reduced, important_feats



def descrizer(graph, threshold=.5):
    """

    :param graph: numpy array
    :return: discretize numpy array using the threshold
    """
    graph[graph >= 0.5] = 1
    graph[graph < 0.5] = 0
    return graph


def Hemogenizer(adj_matrix):
    """

    :param adj_matrix: given the numpy tesnsor, homegenize it into matix
    :return:
    """
    return adj_matrix.sum(0)


def generator(model, computation_graph, in_features,  num_sam = 10):

    """use the sample and generate  attiributed graph"""



    generate_graph = []
    for sample_i in range(num_sam):
        std_z, m_z, z, reconstructed_adj_logit, reconstructed_x, reconstructed_labels = model(computation_graph, in_features)
        reconstructed_adjacency = torch.sigmoid(reconstructed_adj_logit)
        reconstructed_x_prob = torch.sigmoid(reconstructed_x)
        reconstructed_labels_prob = torch.sigmoid(reconstructed_labels)
        graph =reconstructed_adjacency.detach().numpy()
        graph = descrizer(graph)
        graph = Hemogenizer(graph)
        generate_graph.append([graph, reconstructed_x_prob.detach().numpy()])
    return generate_graph

def SaveSamples(model, computation_graph, in_features, ref_graph,ref_feature, dir,  num_sam = 10):
    generate_graph = generator(model, computation_graph, in_features,  num_sam = 10)
    refrence_graph = []

    refrence_graph.append([Hemogenizer(ref_graph.detach().numpy()), ref_feature.detach().numpy()])


    if not os.path.exists(dir):
        os.makedirs(dir)

    # np.save(dir + setting+'_generatedGraphs_.npy', generate_graph, allow_pickle=True)
    # np.save(dir + setting+'refGraphs.npy', refrence_graph, allow_pickle=True)
    with open(dir + 'generatedGraphs.npy', 'wb') as file:
        pickle.dump(generate_graph, file)

    with open(dir + 'refGraphs.npy', 'wb') as file:
        pickle.dump(refrence_graph, file)

    stat_rnn.mmd_eval([stat_rnn.to_nx(G[0]) for G in generate_graph], [stat_rnn.to_nx(G[0]) for G in refrence_graph], True)

