#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 19:13:06 2021

@author: pnaddaf
"""

import sys
import os
import argparse
import time

import numpy as np
from scipy import sparse
import scipy.sparse as sp  # alias for sparse
import pickle
import random
import torch
import torch.nn.functional as F
import pyhocon
import dgl
import random

from dgl.nn.pytorch import GraphConv as GraphConv

from dataCenter import *
from utils import *
from models import *
import timeit
import csv
from bayes_opt import BayesianOptimization
from loss import *
from motif_count import *

# New import for TensorBoard logging
from torch.utils.tensorboard import SummaryWriter


def train_model(data_center, features, args, device):
    dataset = args.dataSet
    decoder = args.decoder_type
    encoder = args.encoder_type
    num_of_relations = args.num_of_relations  # different type of relation
    num_of_comunities = args.num_of_comunities  # number of communities
    batch_norm = args.batch_norm
    DropOut_rate = args.DropOut_rate
    encoder_layers = [int(x) for x in args.encoder_layers.split()]
    epoch_number = args.epoch_number
    subgraph_size = args.num_node
    lr = args.lr
    is_prior = args.is_prior
    targets = args.targets
    sampling_method = args.sampling_method
    ds = args.dataSet
    loss_type = args.loss_type

    # ----------------------------------
    # Create a separate SummaryWriter for each metric
    # ----------------------------------
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    metrics = [
        "loss",
        "edge_loss",
        "feat_loss",
        "node_classification_loss",
        "z_kl_loss",
        "motif_loss",
        "accuracy",
        "val_edge_loss",
        "val_feat_loss",
        "val_node_classification_loss",
        "val_total_loss",
        "val_motif_loss"
    ]
    writers = {}
    for metric in metrics:
        # e.g. runs/Cora_dgl_motif_True_tuning_True_loss_20250404-083611
        run_name = f"{args.dataSet}_motif_{args.motif_loss}_tuning_{args.tuning}_{metric}_{timestamp}"
        writers[metric] = SummaryWriter(log_dir=os.path.join("runs", run_name))
    # Global step tracker to persist epoch count between different phases
    global_step_tracker = {"step": 0}

    # ------------------------------
    # Data loading and preprocessing
    # ------------------------------
    original_adj_full = torch.FloatTensor(getattr(data_center, ds + '_adj_lists')).to(device)
    node_label_full = torch.FloatTensor(getattr(data_center, ds + '_labels')).to(device)

    val_indx = getattr(data_center, ds + '_val_edge_idx')
    train_indx = getattr(data_center, ds + '_train_edge_idx')

    # Shuffling the data and selecting a subset
    if subgraph_size == -1:
        subgraph_size = original_adj_full.shape[0]
    elemnt = min(original_adj_full.shape[0], subgraph_size)
    indexes = list(range(original_adj_full.shape[0]))
    np.random.shuffle(indexes)
    indexes = indexes[:elemnt]
    original_adj = original_adj_full[indexes, :]
    original_adj = original_adj[:, indexes]

    node_label = [np.array(node_label_full[i], dtype=np.float16) for i in indexes]
    features = features[indexes]
    number_of_classes = len(node_label_full[0])

    # Check for Encoder
    if encoder == "Multi_GCN":
        encoder_model = multi_layer_GCN(num_of_comunities, latent_dim=num_of_comunities, layers=encoder_layers)
    elif encoder == "Multi_GAT":
        encoder_model = multi_layer_GAT(num_of_comunities, latent_dim=num_of_comunities, layers=encoder_layers)
    elif encoder == "Multi_GIN":
        encoder_model = multi_layer_GIN(num_of_comunities, latent_dim=num_of_comunities, layers=encoder_layers)
    elif encoder == "Multi_SAGE":
        encoder_model = multi_layer_SAGE(num_of_comunities, latent_dim=num_of_comunities, layers=encoder_layers)
    else:
        raise Exception("Sorry, this Encoder is not implemented; check the input args")

    # Check for Decoder
    if decoder == "ML_SBM":
        decoder_model = MultiLatetnt_SBM_decoder(num_of_relations, num_of_comunities, num_of_comunities, batch_norm, DropOut_rate=0.3)
    else:
        raise Exception("Sorry, this Decoder is not implemented; check the input args")

    feature_encoder_model = feature_encoder(features.view(-1, features.shape[1]), num_of_comunities)
    feature_decoder = feature_decoder_nn(features.shape[1], num_of_comunities)
    class_decoder = MulticlassClassifier(number_of_classes, num_of_comunities)

    trainId = getattr(data_center, ds + '_train')
    testId = getattr(data_center, ds + '_test')
    validId = getattr(data_center, ds + '_val')

    adj_train = original_adj.cpu().detach().numpy()[trainId, :][:, trainId]
    adj_val = original_adj.cpu().detach().numpy()[validId, :][:, validId]

    feat_np = features.cpu().data.numpy()
    feat_train = feat_np[trainId, :]
    feat_val = feat_np[validId, :]

    labels_np = np.array(node_label, dtype=np.float16)
    labels_train = labels_np[trainId]
    labels_val = labels_np[validId]

    print('Finish splitting dataset to train and test.')

    adj_train = sp.csr_matrix(adj_train)
    adj_val = sp.csr_matrix(adj_val)

    graph_dgl = dgl.from_scipy(adj_train)
    graph_dgl.add_edges(graph_dgl.nodes(), graph_dgl.nodes())  # add self-loops
    num_nodes = graph_dgl.number_of_dst_nodes()
    adj_train = torch.tensor(adj_train.todense())
    adj_train = adj_train + sp.eye(adj_train.shape[0]).todense()

    graph_dgl_val = dgl.from_scipy(adj_val)
    graph_dgl_val.add_edges(graph_dgl_val.nodes(), graph_dgl_val.nodes())
    num_nodes_val = graph_dgl.number_of_dst_nodes()
    adj_val = torch.tensor(adj_val.todense())
    adj_val = adj_val + sp.eye(adj_val.shape[0]).todense()

    if isinstance(feat_train, np.ndarray):
        feat_train = torch.tensor(feat_train, dtype=torch.float32)
        feat_val = torch.tensor(feat_val, dtype=torch.float32)

    model = VGAE_FrameWork(num_of_comunities,
                           encoder=encoder_model,
                           decoder=decoder_model,
                           feature_decoder=feature_decoder,
                           feature_encoder=feature_encoder_model,
                           classifier=class_decoder)
    optimizer = torch.optim.Adam(model.parameters(), lr)

    pos_wight = torch.true_divide((adj_train.shape[0] ** 2 - torch.sum(adj_train)), torch.sum(adj_train))
    pos_wight_val = torch.true_divide((adj_val.shape[0] ** 2 - torch.sum(adj_val)), torch.sum(adj_val))
    norm = torch.true_divide(adj_train.shape[0] * adj_train.shape[0],
                             ((adj_train.shape[0] * adj_train.shape[0] - torch.sum(adj_train)) * 2))
    norm_val = torch.true_divide(adj_val.shape[0] * adj_val.shape[0],
                                 ((adj_val.shape[0] * adj_val.shape[0] - torch.sum(adj_val)) * 2))
    pos_weight_feat = torch.true_divide((feat_train.shape[0] * feat_train.shape[1] - torch.sum(feat_train)),
                                        torch.sum(feat_train))
    norm_feat = torch.true_divide((feat_train.shape[0] * feat_train.shape[1]),
                                  (2 * (feat_train.shape[0] * feat_train.shape[1] - torch.sum(feat_train))))
    pos_weight_feat_val = torch.true_divide((feat_val.shape[0] * feat_val.shape[1] - torch.sum(feat_val)),
                                            torch.sum(feat_val))
    norm_feat_val = torch.true_divide((feat_val.shape[0] * feat_val.shape[1]),
                                      (2 * (feat_val.shape[0] * feat_val.shape[1] - torch.sum(feat_val))))

    # --------------------------
    # Prepare Motif_Count if needed
    # --------------------------
    if args.motif_loss is True:
        CM = Motif_Count(args)
        CM.setup_function()
        reconstructed_x_slice, reconstructed_labels_m = CM.process_reconstructed_data(
            None, [adj_train],
            feat_train[:, np.array(data_center.important_feats_id)],
            np.array(data_center.important_feats_id),
            torch.tensor(labels_train)
        )
        observed = CM.iteration_function(reconstructed_x_slice, reconstructed_labels_m, mode="ground-truth")
    else:
        CM = None
        observed = None

    print(observed)

    if args.motif_loss is True:
        reconstructed_x_slice, reconstructed_labels_m = CM.process_reconstructed_data(
            None, [adj_val],
            feat_val[:, np.array(data_center.important_feats_id)],
            np.array(data_center.important_feats_id),
            torch.tensor(labels_val)
        )
        observed_val = CM.iteration_function(reconstructed_x_slice, reconstructed_labels_m, mode="ground-truth")
    else:
        observed_val = None

    # Initialize lambda values
    lambda_1 = 1
    lambda_2 = 1
    lambda_3 = 1
    lambda_4 = 1

    # If tuning is True, perform Bayesian optimization
    if args.tuning == "True":
        pbounds = {
            'lambda_1': (0.0, 1.0),
            'lambda_2': (0.0, 1.0),
            'lambda_3': (0.0, 1.0),
            'lambda_4': (0.0, 1.0)
        }
        optimizer_function = make_optimizer_wrapper(
            labels_train, labels_val, dataset, epoch_number, model, graph_dgl, graph_dgl_val, feat_train,
            feat_val, targets, sampling_method, is_prior, loss_type, adj_train, adj_val, norm_feat,
            pos_weight_feat, norm_feat_val, pos_weight_feat_val, num_nodes, num_nodes_val, pos_wight, norm,
            pos_wight_val, norm_val, optimizer, val_indx, trainId, args, observed, CM, data_center, observed_val,
            writers, global_step_tracker  # <-- pass the writers dict
        )
        optimizer_hp = BayesianOptimization(
            f=optimizer_function,
            pbounds=pbounds,
            random_state=42
        )
        optimizer_hp.maximize(
            init_points=1,
            n_iter=1
        )
        print(optimizer_hp.max)

        best_params = optimizer_hp.max['params']
        lambda_1 = best_params['lambda_1']
        lambda_2 = best_params['lambda_2']
        lambda_3 = best_params['lambda_3']
        lambda_4 = best_params['lambda_4']

        with open('./new_weights.csv', 'a', newline="\n") as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow([args.dataSet, lambda_1, lambda_2, lambda_3, lambda_4])

    # For tuning==False, load weights from file
    if args.tuning == "False":
        weights_list = []
        with open('new_weights.csv', 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                processed_row = []
                for item in row:
                    try:
                        processed_row.append(float(item))
                    except ValueError:
                        processed_row.append(item)
                weights_list.append(processed_row)

        for row in weights_list:
            if row[0] in args.dataSet:
                lambda_1 = float(row[1])
                lambda_2 = float(row[2])
                lambda_3 = float(row[3])
                try:
                    lambda_4 = float(row[4])
                except IndexError:
                    lambda_4 = None

        print("weights:", lambda_1, lambda_2, lambda_3, lambda_4)

    # ------------------------
    # Main training loop
    # ------------------------
    for epoch in range(epoch_number):
        model.train()
        # Forward
        std_z, m_z, z, reconstructed_adj, reconstructed_feat, re_labels = model(
            graph_dgl, feat_train, labels_train,
            targets, sampling_method, is_prior, train=True
        )

        reconstructed_adjacency = torch.sigmoid(reconstructed_adj)
        reconstructed_x_prob = torch.sigmoid(reconstructed_feat)
        reconstructed_labels_prob = torch.sigmoid(re_labels)

        if args.devide_rec_adj:
            reconstructed_adjacency = [
                (adj * (1 / args.num_nodes)) for adj in reconstructed_adjacency
            ]

        if args.motif_loss is True:
            reconstructed_x_slice, reconstructed_labels_m = CM.process_reconstructed_data(
                None, [reconstructed_adjacency],
                reconstructed_x_prob[:, np.array(data_center.important_feats_id)],
                np.array(data_center.important_feats_id),
                torch.tensor(reconstructed_labels_prob)
            )
            predicted = CM.iteration_function(reconstructed_x_slice, reconstructed_labels_m, mode="ground-truth")
        else:
            predicted = None

        z_kl, reconstruction_loss, posterior_cost_edges, posterior_cost_features, posterior_cost_classes, acc, val_recons_loss, loss_adj, loss_feat, motif_loss_val = optimizer_VAE(
            lambda_1, lambda_2, lambda_3, lambda_4, labels_train,
            re_labels, loss_type, reconstructed_adj, reconstructed_feat,
            adj_train, feat_train, norm_feat, pos_weight_feat,
            std_z, m_z, num_nodes, pos_wight, norm, val_indx, trainId, args, observed, predicted
        )
        loss = reconstruction_loss + z_kl

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # --------------------------------------
        # Log each metric to its own directory
        # --------------------------------------
        step = global_step_tracker["step"]
        writers["loss"].add_scalar("loss", loss.item(), step)
        writers["edge_loss"].add_scalar("edge_loss", reconstruction_loss.item(), step)
        writers["feat_loss"].add_scalar("feat_loss", posterior_cost_features.item(), step)
        writers["node_classification_loss"].add_scalar("node_classification_loss", posterior_cost_classes.item(), step)
        writers["z_kl_loss"].add_scalar("z_kl_loss", z_kl.item(), step)
        writers["motif_loss"].add_scalar("motif_loss", motif_loss_val, step)
        writers["accuracy"].add_scalar("accuracy", acc, step)

        global_step_tracker["step"] += 1

        print(f"Epoch: {epoch + 1:03d} | Loss: {loss.item():.5f} "
              f"| edge_loss: {reconstruction_loss.item():.5f} "
              f"| feat_loss: {posterior_cost_features.item():.5f} "
              f"| node_classification_loss: {posterior_cost_classes.item():.5f} "
              f"| z_kl_loss: {z_kl.item():.5f} "
              f"| Accuracy: {acc:.3f} "
              f"| motif loss: {motif_loss_val:.5f}")

    model.eval()

    # Close all writers at the end
    for w in writers.values():
        w.close()

    return model, z


def optimize_weights(lambda_1, lambda_2, lambda_3, lambda_4,
                     labels_train, labels_val, dataset, epoch_number, model, graph_dgl, graph_dgl_val, feat_train,
                     feat_val, targets, sampling_method, is_prior, loss_type, adj_train_org, adj_val_org, norm_feat,
                     pos_weight_feat, norm_feat_val, pos_weight_feat_val, num_nodes, num_nodes_val, pos_wight, norm,
                     pos_wight_val, norm_val, optimizer, val_indx, trainId, args, observed, CM, data_center, observed_val,
                     writers, global_step_tracker):
    # Training loop inside weight optimization
    for epoch in range(epoch_number):
        model.train()
        std_z, m_z, z, reconstructed_adj, reconstructed_feat, re_labels = model(
            graph_dgl, feat_train, labels_train,
            targets, sampling_method, is_prior, train=True
        )

        reconstructed_adjacency = torch.sigmoid(reconstructed_adj)
        reconstructed_x_prob = torch.sigmoid(reconstructed_feat)
        reconstructed_labels_prob = torch.sigmoid(re_labels)

        if args.devide_rec_adj:
            reconstructed_adjacency = [
                (adj * (1 / args.num_nodes)) for adj in reconstructed_adjacency
            ]

        if args.motif_loss is True:
            reconstructed_x_slice, reconstructed_labels_m = CM.process_reconstructed_data(
                None, [reconstructed_adjacency],
                reconstructed_x_prob[:, np.array(data_center.important_feats_id)],
                np.array(data_center.important_feats_id),
                torch.tensor(reconstructed_labels_prob)
            )
            predicted = CM.iteration_function(reconstructed_x_slice, reconstructed_labels_m, mode="ground-truth")
        else:
            predicted = None

        z_kl, reconstruction_loss, posterior_cost_edges, posterior_cost_features, posterior_cost_classes, acc, val_recons_loss, loss_adj, loss_feat, motif_loss_val = optimizer_VAE(
            lambda_1, lambda_2, lambda_3, lambda_4, labels_train,
            re_labels, loss_type, reconstructed_adj, reconstructed_feat,
            adj_train_org, feat_train, norm_feat, pos_weight_feat,
            std_z, m_z, num_nodes, pos_wight, norm, val_indx, trainId, args, observed, predicted
        )
        loss = reconstruction_loss + z_kl

        model.eval()
        model.train()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log training metrics for weight optimization
        step = global_step_tracker["step"]
        writers["loss"].add_scalar("loss", loss.item(), step)
        writers["edge_loss"].add_scalar("edge_loss", reconstruction_loss.item(), step)
        writers["feat_loss"].add_scalar("feat_loss", posterior_cost_features.item(), step)
        writers["node_classification_loss"].add_scalar("node_classification_loss", posterior_cost_classes.item(), step)
        writers["z_kl_loss"].add_scalar("z_kl_loss", z_kl.item(), step)
        writers["motif_loss"].add_scalar("motif_loss", motif_loss_val, step)
        writers["accuracy"].add_scalar("accuracy", acc, step)

        global_step_tracker["step"] += 1

        print(f"OptEpoch: {epoch + 1:03d} | Loss: {loss.item():.5f} "
              f"| edge_loss: {reconstruction_loss.item():.5f} "
              f"| feat_loss: {posterior_cost_features.item():.5f} "
              f"| node_classification_loss: {posterior_cost_classes.item():.5f} "
              f"| z_kl_loss: {z_kl.item():.5f} "
              f"| Accuracy: {acc:.3f} "
              f"| motif loss: {motif_loss_val:.5f}")

    model.eval()
    with torch.no_grad():
        std_z_val, m_z_val, z_val, reconstructed_adj_val, reconstructed_feat_val, re_labels_val = model(
            graph_dgl_val, feat_val, labels_val,
            targets, sampling_method, is_prior, train=True
        )

        reconstructed_adjacency_val = torch.sigmoid(reconstructed_adj_val)
        reconstructed_x_prob_val = torch.sigmoid(reconstructed_feat_val)
        reconstructed_labels_prob_val = torch.sigmoid(re_labels_val)

        if args.devide_rec_adj:
            reconstructed_adjacency_val = [
                (adj * (1 / args.num_nodes)) for adj in reconstructed_adjacency_val
            ]

        if args.motif_loss is True:
            reconstructed_x_slice_val, reconstructed_labels_m_val = CM.process_reconstructed_data(
                None,
                [reconstructed_adjacency_val],
                reconstructed_x_prob_val[:, np.array(data_center.important_feats_id)],
                np.array(data_center.important_feats_id),
                torch.tensor(reconstructed_labels_prob_val)
            )
            predicted_val = CM.iteration_function(reconstructed_x_slice_val, reconstructed_labels_m_val, mode="ground-truth")

            zero_indices = [i for i, t in enumerate(observed_val) if torch.any(t == 0)]
            filtered_observed = [g for i, g in enumerate(observed_val) if i not in zero_indices]
            filtered_predicted_val = [p for i, p in enumerate(predicted_val) if i not in zero_indices]
            normalized_observed = [torch.ones_like(t) for t in filtered_observed]
            normalized_predicted_val = [torch.abs(torch.log(p / g)) for p, g in zip(filtered_predicted_val, filtered_observed)]
            motif_loss_v = (torch.sum(torch.stack(normalized_predicted_val)) / len(normalized_predicted_val))
            motif_loss_v = motif_loss_v.cpu()
        else:
            motif_loss_v = 0

    w_l = weight_labels(labels_val)
    posterior_cost_edges = norm * F.binary_cross_entropy_with_logits(reconstructed_adj_val, adj_val_org, pos_weight=pos_wight_val)
    posterior_cost_features = norm_feat * F.binary_cross_entropy_with_logits(reconstructed_feat_val, feat_val, pos_weight=pos_weight_feat)
    posterior_cost_classes = F.cross_entropy(re_labels_val, (torch.tensor(labels_val).to(torch.float64)), weight=w_l)

    cost = posterior_cost_edges + posterior_cost_features + posterior_cost_classes
    if args.motif_loss is True:
        cost += motif_loss_v

    # Log validation metrics
    step = global_step_tracker["step"]
    writers["val_edge_loss"].add_scalar("val_edge_loss", posterior_cost_edges.item(), step)
    writers["val_feat_loss"].add_scalar("val_feat_loss", posterior_cost_features.item(), step)
    writers["val_node_classification_loss"].add_scalar("val_node_classification_loss", posterior_cost_classes.item(), step)
    writers["val_total_loss"].add_scalar("val_total_loss", cost.item(), step)
    if args.motif_loss is True:
        if isinstance(motif_loss_v, torch.Tensor):
            writers["val_motif_loss"].add_scalar("val_motif_loss", motif_loss_v.item(), step)
        else:
            writers["val_motif_loss"].add_scalar("val_motif_loss", motif_loss_v, step)

    global_step_tracker["step"] += 1

    # The return value for BayesianOptimization is negative cost
    return -1 * cost.item()


def make_optimizer_wrapper(labels_train, labels_val, dataset, epoch_number, model, graph_dgl, graph_dgl_val, feat_train,
                           feat_val, targets, sampling_method, is_prior, loss_type, adj_train_org, adj_val_org, norm_feat,
                           pos_weight_feat, norm_feat_val, pos_weight_feat_val, num_nodes, num_nodes_val, pos_wight, norm,
                           pos_wight_val, norm_val, optimizer, val_indx, trainId, args, observed, CM, data_center, observed_val,
                           writers, global_step_tracker):
    def optimize_weights_wrapper(lambda_1, lambda_2, lambda_3, lambda_4):
        return optimize_weights(lambda_1, lambda_2, lambda_3, lambda_4,
                                labels_train, labels_val, dataset, epoch_number, model, graph_dgl, graph_dgl_val, feat_train,
                                feat_val, targets, sampling_method, is_prior, loss_type, adj_train_org, adj_val_org, norm_feat,
                                pos_weight_feat, norm_feat_val, pos_weight_feat_val, num_nodes, num_nodes_val, pos_wight, norm,
                                pos_wight_val, norm_val, optimizer, val_indx, trainId, args, observed, CM, data_center, observed_val,
                                writers, global_step_tracker)
    return optimize_weights_wrapper
