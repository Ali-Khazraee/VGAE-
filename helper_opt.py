#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 19:13:06 2021

@author: pnaddaf
"""

import sys
import os
import argparse

import numpy as np
from scipy.sparse import lil_matrix
import pickle
import random
import torch
import torch.nn.functional as F
import pyhocon
import dgl
import random

from scipy import sparse
from dgl.nn.pytorch import GraphConv as GraphConv

from dataCenter import *
from utils import *
from models import *
import timeit
import csv
from bayes_opt import BayesianOptimization
from loss import *
from motif_count import *

# Import TensorBoard
from torch.utils.tensorboard import SummaryWriter


# %% KDD model
def train_model(data_center, features, args, device):
    dataset = args.dataSet
    decoder = args.decoder_type
    encoder = args.encoder_type
    num_of_relations = args.num_of_relations  # diffrent type of relation
    num_of_comunities = args.num_of_comunities  # number of comunities
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
    
    # Create a unique log directory based on experiment parameters
    log_dir = f"runs/{ds}_motif_{args.motif_obj}_tuning_{args.tuning}"
    tb_writer = SummaryWriter(log_dir=log_dir)
    
    # Log hyperparameters to TensorBoard
    tb_writer.add_text('Hyperparameters/dataset', ds)
    tb_writer.add_text('Hyperparameters/encoder', encoder)
    tb_writer.add_text('Hyperparameters/decoder', decoder)
    tb_writer.add_text('Hyperparameters/num_of_relations', str(num_of_relations))
    tb_writer.add_text('Hyperparameters/num_of_comunities', str(num_of_comunities))
    tb_writer.add_text('Hyperparameters/batch_norm', str(batch_norm))
    tb_writer.add_text('Hyperparameters/DropOut_rate', str(DropOut_rate))
    tb_writer.add_text('Hyperparameters/encoder_layers', args.encoder_layers)
    tb_writer.add_text('Hyperparameters/epoch_number', str(epoch_number))
    tb_writer.add_text('Hyperparameters/learning_rate', str(lr))
    tb_writer.add_text('Hyperparameters/motif_obj', str(args.motif_obj))
    tb_writer.add_text('Hyperparameters/is_prior', str(is_prior))
    tb_writer.add_text('Hyperparameters/targets', str(targets))
    tb_writer.add_text('Hyperparameters/sampling_method', str(sampling_method))
    tb_writer.add_text('Hyperparameters/loss_type', str(loss_type))

    original_adj_full= torch.FloatTensor(getattr(data_center, ds+'_adj_lists')).to(device)
    node_label_full= torch.FloatTensor(getattr(data_center, ds+'_labels')).to(device)

    val_indx = getattr(data_center, ds + '_val_edge_idx')
    train_indx = getattr(data_center, ds + '_train_edge_idx')

    # shuffling the data, and selecting a subset of it
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

    # Check for Encoder and redirect to appropriate function
    if encoder == "Multi_GCN":
        encoder_model = multi_layer_GCN(num_of_comunities , latent_dim=num_of_comunities, layers=encoder_layers)
        # encoder_model = multi_layer_GCN(in_feature=features.shape[1], latent_dim=num_of_comunities, layers=encoder_layers)

    elif encoder == "Multi_GAT":
        encoder_model = multi_layer_GAT(num_of_comunities , latent_dim=num_of_comunities, layers=encoder_layers)


    elif encoder == "Multi_GIN":
        encoder_model = multi_layer_GIN(num_of_comunities, latent_dim=num_of_comunities, layers=encoder_layers)

    elif encoder == "Multi_SAGE":
        encoder_model = multi_layer_SAGE(num_of_comunities, latent_dim=num_of_comunities, layers=encoder_layers)

    else:
        raise Exception("Sorry, this Encoder is not Impemented; check the input args")

    # Check for Decoder and redirect to appropriate function

    if decoder == "ML_SBM":
        decoder_model = MultiLatetnt_SBM_decoder(num_of_relations, num_of_comunities, num_of_comunities, batch_norm, DropOut_rate=0.3)

    else:
        raise Exception("Sorry, this Decoder is not Impemented; check the input args")

    feature_encoder_model = feature_encoder(features.view(-1, features.shape[1]), num_of_comunities)
    # feature_encoder_model = MulticlassClassifier(num_of_comunities, features.shape[1])
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

    print('Finish spliting dataset to train and test. ')

    adj_train = sp.csr_matrix(adj_train)
    adj_val = sp.csr_matrix(adj_val)

    graph_dgl = dgl.from_scipy(adj_train)
    graph_dgl.add_edges(graph_dgl.nodes(), graph_dgl.nodes())  # the library does not add self-loops
    num_nodes = graph_dgl.number_of_dst_nodes()
    adj_train = torch.tensor(adj_train.todense())  # use sparse man
    adj_train = adj_train + sp.eye(adj_train.shape[0]).todense()

    graph_dgl_val = dgl.from_scipy(adj_val)
    graph_dgl_val.add_edges(graph_dgl_val.nodes(), graph_dgl_val.nodes())  # the library does not add self-loops
    num_nodes_val = graph_dgl.number_of_dst_nodes()
    adj_val = torch.tensor(adj_val.todense())  # use sparse man
    adj_val = adj_val + sp.eye(adj_val.shape[0]).todense()

    if (type(feat_train) == np.ndarray):
        feat_train = torch.tensor(feat_train, dtype=torch.float32)
        feat_val = torch.tensor(feat_val, dtype=torch.float32)

    model = VGAE_FrameWork(num_of_comunities,
                            encoder = encoder_model,
                            decoder = decoder_model,
                            feature_decoder = feature_decoder,
                            feature_encoder = feature_encoder_model,
                            classifier=class_decoder)
    optimizer = torch.optim.Adam(model.parameters(), lr)

    pos_wight = torch.true_divide((adj_train.shape[0] ** 2 - torch.sum(adj_train)), torch.sum(
        adj_train))  # addrressing imbalance data problem: ratio between positve to negative instance
    pos_wight_val = torch.true_divide((adj_val.shape[0] ** 2 - torch.sum(adj_val)), torch.sum(
        adj_val))
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

    if args.motif_obj == True:
        CM = Motif_Count(args)
        CM.setup_function()
        reconstructed_x_slice, reconstructed_labels_m = CM.process_reconstructed_data(None, 
        [adj_train], feat_train[:,np.array(data_center.important_feats_id)], np.array(data_center.important_feats_id), torch.tensor(labels_train)
    )
        observed = CM.iteration_function(reconstructed_x_slice , reconstructed_labels_m, mode = "ground-truth")
    else:
        CM = None
        observed = None

    print(observed)

    if args.motif_obj == True:
        reconstructed_x_slice, reconstructed_labels_m = CM.process_reconstructed_data(None, 
        [adj_val], feat_val[:,np.array(data_center.important_feats_id)], np.array(data_center.important_feats_id), torch.tensor(labels_val)
    )
        observed_val = CM.iteration_function(reconstructed_x_slice , reconstructed_labels_m, mode = "ground-truth")
    else:
        CM = None
        observed_val = None

    lambda_1 = 1
    lambda_2 = 1
    lambda_3 = 1
    lambda_4 = 1

    # Define optimizer wrapper function before it's used
    def make_optimizer_wrapper(labels_train, labels_val, dataset, epoch_number, model, graph_dgl, graph_dgl_val, feat_train,
                    feat_val, targets, sampling_method, is_prior, loss_type, adj_train_org, adj_val_org, norm_feat,
                    pos_weight_feat, norm_feat_val, pos_weight_feat_val, num_nodes, num_nodes_val, pos_wight, norm,
                    pos_wight_val, norm_val, optimizer, val_indx, trainId, args, observed, CM, data_center, observed_val, tuning_writer=None):
        def optimize_weights_wrapper(lambda_1, lambda_2, lambda_3, lambda_4):
            return optimize_weights(lambda_1, lambda_2, lambda_3, lambda_4, labels_train, labels_val, dataset, epoch_number, model, graph_dgl, graph_dgl_val, feat_train,
                    feat_val, targets, sampling_method, is_prior, loss_type, adj_train_org, adj_val_org, norm_feat,
                    pos_weight_feat, norm_feat_val, pos_weight_feat_val, num_nodes, num_nodes_val, pos_wight, norm,
                    pos_wight_val, norm_val, optimizer, val_indx, trainId, args, observed, CM, data_center, observed_val, tuning_writer)
        return optimize_weights_wrapper
        
    #to find weights
    if args.tuning == "True":
        pbounds = {
            'lambda_1': (0.0, 1.0),
            'lambda_2': (0.0, 1.0),
            'lambda_3': (0.0, 1.0),
            'lambda_4': (0.0, 1.0)
        }
        
        # Create a TensorBoard writer for tuning
        tuning_tb_writer = SummaryWriter(log_dir=f"{log_dir}/tuning")
        
        optimizer_function = make_optimizer_wrapper(labels_train, labels_val, dataset, epoch_number, model, graph_dgl, graph_dgl_val, feat_train,
                    feat_val, targets, sampling_method, is_prior, loss_type, adj_train, adj_val, norm_feat,
                    pos_weight_feat, norm_feat_val, pos_weight_feat_val, num_nodes, num_nodes_val, pos_wight, norm,
                    pos_wight_val, norm_val, optimizer, val_indx, trainId, args, observed, CM, data_center, observed_val, tuning_tb_writer)
        
        optimizer_hp = BayesianOptimization(
            f=optimizer_function,
            pbounds=pbounds,
            random_state=42,
            verbose=2  # Add this to see detailed logs
        )
        
        # Add error handling for the optimization process
        try:
            optimizer_hp.maximize(
                init_points=1,
                n_iter=1
            )
            print(optimizer_hp.max)

            # Extract and print the best values for weights
            best_params = optimizer_hp.max['params']
            lambda_1 = best_params['lambda_1']
            lambda_2 = best_params['lambda_2']
            lambda_3 = best_params['lambda_3']
            lambda_4 = best_params['lambda_4']
            
            # Log best hyperparameters to TensorBoard
            tb_writer.add_hparams(
                {
                    'lambda_1': lambda_1,
                    'lambda_2': lambda_2,
                    'lambda_3': lambda_3,
                    'lambda_4': lambda_4,
                },
                {'hparam/best_objective': optimizer_hp.max['target']}
            )
        except Exception as e:
            print(f"Error during optimization: {e}")
            print("Using default lambda values instead.")
            lambda_1 = 1.0
            lambda_2 = 1.0
            lambda_3 = 1.0
            lambda_4 = 1.0

        with open('./new_weights.csv', 'a', newline="\n") as f:
            writer = csv.writer(f)
            writer.writerow(
                [args.dataSet, lambda_1, lambda_2, lambda_3, lambda_4])

    # to read weights
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
        
        # Log the loaded weights to TensorBoard
        tb_writer.add_hparams(
            {
                'lambda_1': lambda_1,
                'lambda_2': lambda_2,
                'lambda_3': lambda_3,
                'lambda_4': lambda_4 if lambda_4 is not None else 0.0,
            },
            {'hparam/loaded_weights': 1.0}
        )

    for epoch in range(epoch_number):
        model.train()
        # forward propagation by using all train nodes
        std_z, m_z, z, reconstructed_adj, reconstructed_feat, re_labels = model(graph_dgl, feat_train, labels_train,
                                                                                targets, sampling_method,
                                                                                is_prior, train=True)

        reconstructed_adjacency = torch.sigmoid(reconstructed_adj)
        reconstructed_x_prob = torch.sigmoid(reconstructed_feat)
        reconstructed_labels_prob = torch.sigmoid(re_labels)

        if args.devide_rec_adj:
            reconstructed_adjacency = [
                (adj * (1 / args.num_nodes)) for adj in reconstructed_adjacency
            ]

        if args.motif_obj == True:
            reconstructed_x_slice, reconstructed_labels_m = CM.process_reconstructed_data(None, 
            [reconstructed_adjacency], reconstructed_x_prob[:,np.array(data_center.important_feats_id)], np.array(data_center.important_feats_id), torch.tensor(reconstructed_labels_prob)
        )
            predicted = CM.iteration_function(reconstructed_x_slice , reconstructed_labels_m, mode = "ground-truth")
        else:
            predicted = None

        z_kl, reconstruction_loss, posterior_cost_edges, posterior_cost_features, posterior_cost_classes, acc, val_recons_loss, loss_adj, loss_feat, motif_loss = optimizer_VAE(lambda_1, lambda_2,
                                                                                                lambda_3, lambda_4, labels_train,
                                                                                                re_labels, loss_type,
                                                                                                reconstructed_adj,
                                                                                                reconstructed_feat,
                                                                                                adj_train,
                                                                                                feat_train, norm_feat,
                                                                                                pos_weight_feat,
                                                                                                std_z, m_z, num_nodes,
                                                                                                pos_wight, norm, val_indx, train_indx, args, observed, predicted)

        loss = reconstruction_loss + z_kl

        # backward propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print some metrics
        print("Epoch: {:03d} | Loss: {:05f} | edge_loss: {:05f} |feat_loss: {:05f} |node_classification_loss: {:05f} | z_kl_loss: {:05f} | Accuracy: {:03f}| motif loss: {:05f}".format(
            epoch + 1, loss.item(), reconstruction_loss.item(), posterior_cost_features.item(), posterior_cost_classes.item(), z_kl.item(), acc, motif_loss))
        
        # Log metrics to TensorBoard for non-tuning training
        if args.tuning == "False":
            tb_writer.add_scalar('Training/Loss', loss.item(), epoch)
            tb_writer.add_scalar('Training/ReconstructionLoss', reconstruction_loss.item(), epoch)
            tb_writer.add_scalar('Training/EdgeLoss', posterior_cost_edges.item(), epoch)
            tb_writer.add_scalar('Training/FeatureLoss', posterior_cost_features.item(), epoch)
            tb_writer.add_scalar('Training/ClassificationLoss', posterior_cost_classes.item(), epoch)
            tb_writer.add_scalar('Training/KLDivergence', z_kl.item(), epoch)
            tb_writer.add_scalar('Training/Accuracy', acc, epoch)
            tb_writer.add_scalar('Training/MotifLoss', motif_loss, epoch)
            
            # Evaluate on validation set
            model.eval()
            with torch.no_grad():
                std_z_val, m_z_val, z_val, reconstructed_adj_val, reconstructed_feat_val, re_labels_val = model(
                    graph_dgl_val, feat_val, labels_val, targets, sampling_method, is_prior, train=False
                )
                
                # Calculate validation metrics
                w_l = weight_labels(labels_val)
                val_edge_loss = norm_val * F.binary_cross_entropy_with_logits(
                    reconstructed_adj_val, adj_val, pos_weight=pos_wight_val
                )
                val_feat_loss = norm_feat_val * F.binary_cross_entropy_with_logits(
                    reconstructed_feat_val, feat_val, pos_weight=pos_weight_feat_val
                )
                val_class_loss = F.cross_entropy(
                    re_labels_val, torch.tensor(labels_val).to(torch.float64), weight=w_l
                )
                
                # Calculate validation accuracy
                predicted_labels = torch.max(re_labels_val, 1)[1]
                true_labels = torch.max(torch.tensor(labels_val).to(torch.float64), 1)[1]
                val_acc = (predicted_labels == true_labels).sum().item() / len(true_labels)
                
                # Log validation metrics
                tb_writer.add_scalar('Validation/EdgeLoss', val_edge_loss.item(), epoch)
                tb_writer.add_scalar('Validation/FeatureLoss', val_feat_loss.item(), epoch)
                tb_writer.add_scalar('Validation/ClassificationLoss', val_class_loss.item(), epoch)
                tb_writer.add_scalar('Validation/Accuracy', val_acc, epoch)
                
                # Calculate validation motif loss if applicable
                if args.motif_obj == True:
                    reconstructed_adjacency_val = torch.sigmoid(reconstructed_adj_val)
                    reconstructed_x_prob_val = torch.sigmoid(reconstructed_feat_val)
                    reconstructed_labels_prob_val = torch.sigmoid(re_labels_val)
                    
                    if args.devide_rec_adj:
                        reconstructed_adjacency_val = [
                            (adj * (1 / args.num_nodes)) for adj in reconstructed_adjacency_val
                        ]
                    
                    reconstructed_x_slice_val, reconstructed_labels_m_val = CM.process_reconstructed_data(
                        None, 
                        [reconstructed_adjacency_val],
                        reconstructed_x_prob_val[:, np.array(data_center.important_feats_id)],
                        np.array(data_center.important_feats_id),
                        torch.tensor(reconstructed_labels_prob_val)
                    )
                    
                    predicted_val = CM.iteration_function(
                        reconstructed_x_slice_val, reconstructed_labels_m_val, mode="ground-truth"
                    )
                    
                    # Filter out zero indices
                    zero_indices = [i for i, t in enumerate(observed_val) if torch.any(t == 0)]
                    filtered_observed = [g for i, g in enumerate(observed_val) if i not in zero_indices]
                    filtered_predicted_val = [p for i, p in enumerate(predicted_val) if i not in zero_indices]
                    
                    # Calculate normalized motif loss
                    if filtered_observed and filtered_predicted_val:
                        normalized_predicted_val = [torch.abs((torch.log(p / g))) 
                                                    for p, g in zip(filtered_predicted_val, filtered_observed)]
                        val_motif_loss = (torch.sum(torch.stack(normalized_predicted_val))/len(normalized_predicted_val))
                        tb_writer.add_scalar('Validation/MotifLoss', val_motif_loss.item(), epoch)
            
            model.train()

    # Close the TensorBoard writer
    try:
        tb_writer.close()
    except Exception as e:
        print(f"Warning: Could not close TensorBoard writer: {e}")
        
    if args.tuning == "True":
        try:
            if 'tuning_tb_writer' in locals():
                tuning_tb_writer.close()
        except Exception as e:
            print(f"Warning: Could not close tuning TensorBoard writer: {e}")
        
    model.eval()

    return model, z 


def optimize_weights(lambda_1, lambda_2, lambda_3, lambda_4, labels_train, labels_val, dataset, epoch_number, model, graph_dgl, graph_dgl_val, feat_train,
                feat_val, targets, sampling_method, is_prior, loss_type, adj_train_org, adj_val_org, norm_feat,
                pos_weight_feat, norm_feat_val, pos_weight_feat_val, num_nodes, num_nodes_val, pos_wight, norm,
                pos_wight_val, norm_val, optimizer, val_indx, trainId, args, observed, CM, data_center, observed_val, tuning_writer=None):
    
    # Log hyperparameters for this optimization run
    if tuning_writer:
        tuning_writer.add_hparams(
            {
                'lambda_1': lambda_1,
                'lambda_2': lambda_2,
                'lambda_3': lambda_3,
                'lambda_4': lambda_4,
            },
            {}  # Metrics will be added during training
        )
    
    # Lists to track metrics across epochs for this optimization run
    losses = []
    edge_losses = []
    feat_losses = []
    class_losses = []
    kl_losses = []
    accuracies = []
    motif_losses = []
    
    for epoch in range(epoch_number):
        model.train()
        # forward propagation by using all nodes
        std_z, m_z, z, reconstructed_adj, reconstructed_feat, re_labels = model(graph_dgl, feat_train, labels_train,
                                                                                targets, sampling_method,
                                                                                is_prior, train=True)

        reconstructed_adjacency = torch.sigmoid(reconstructed_adj)
        reconstructed_x_prob = torch.sigmoid(reconstructed_feat)
        reconstructed_labels_prob = torch.sigmoid(re_labels)

        if args.devide_rec_adj:
            reconstructed_adjacency = [
                (adj * (1 / args.num_nodes)) for adj in reconstructed_adjacency
            ]

        if args.motif_obj == True:
            reconstructed_x_slice, reconstructed_labels_m = CM.process_reconstructed_data(None, 
            [reconstructed_adjacency], reconstructed_x_prob[:,np.array(data_center.important_feats_id)], np.array(data_center.important_feats_id), torch.tensor(reconstructed_labels_prob)
        )
            predicted = CM.iteration_function(reconstructed_x_slice , reconstructed_labels_m, mode = "ground-truth")
        else:
            predicted = None

        # compute loss and accuracy
        z_kl, reconstruction_loss, posterior_cost_edges, posterior_cost_features, posterior_cost_classes, acc, val_recons_loss, loss_adj, loss_feat, motif_loss = optimizer_VAE(
            lambda_1, lambda_2,
            lambda_3, lambda_4, labels_train,
            re_labels, loss_type,
            reconstructed_adj,
            reconstructed_feat,
            adj_train_org,
            feat_train, norm_feat,
            pos_weight_feat,
            std_z, m_z, num_nodes,
            pos_wight, norm, val_indx, trainId, args, observed, predicted)
        
        loss = reconstruction_loss + z_kl

        # Track metrics
        losses.append(loss.item())
        edge_losses.append(posterior_cost_edges.item())
        feat_losses.append(posterior_cost_features.item())
        class_losses.append(posterior_cost_classes.item())
        kl_losses.append(z_kl.item())
        accuracies.append(acc)
        motif_losses.append(motif_loss)

        # Log the current epoch metrics if writer is provided
        if tuning_writer:
            run_name = f"lambda1_{lambda_1:.3f}_lambda2_{lambda_2:.3f}_lambda3_{lambda_3:.3f}_lambda4_{lambda_4:.3f}"
            tuning_writer.add_scalar(f'Tuning/{run_name}/Loss', loss.item(), epoch)
            tuning_writer.add_scalar(f'Tuning/{run_name}/ReconstructionLoss', reconstruction_loss.item(), epoch)
            tuning_writer.add_scalar(f'Tuning/{run_name}/EdgeLoss', posterior_cost_edges.item(), epoch)
            tuning_writer.add_scalar(f'Tuning/{run_name}/FeatureLoss', posterior_cost_features.item(), epoch)
            tuning_writer.add_scalar(f'Tuning/{run_name}/ClassificationLoss', posterior_cost_classes.item(), epoch)
            tuning_writer.add_scalar(f'Tuning/{run_name}/KLDivergence', z_kl.item(), epoch)
            tuning_writer.add_scalar(f'Tuning/{run_name}/Accuracy', acc, epoch)
            tuning_writer.add_scalar(f'Tuning/{run_name}/MotifLoss', motif_loss, epoch)

        model.train()
        # backward propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print some metrics
        print("Epoch: {:03d} | Loss: {:05f} | edge_loss: {:05f} |feat_loss: {:05f} |node_classification_loss: {:05f} | z_kl_loss: {:05f} | Accuracy: {:03f}| motif loss: {:05f}".format(
            epoch + 1, loss.item(), reconstruction_loss.item(), posterior_cost_features.item(), posterior_cost_classes.item(), z_kl.item(), acc, motif_loss))
    
    # Log aggregate metrics for this optimization run
    if tuning_writer:
        avg_metrics = {
            'tuning/avg_loss': sum(losses) / len(losses),
            'tuning/avg_edge_loss': sum(edge_losses) / len(edge_losses),
            'tuning/avg_feat_loss': sum(feat_losses) / len(feat_losses),
            'tuning/avg_class_loss': sum(class_losses) / len(class_losses),
            'tuning/avg_kl_loss': sum(kl_losses) / len(kl_losses),
            'tuning/avg_accuracy': sum(accuracies) / len(accuracies),
            'tuning/avg_motif_loss': sum(motif_losses) / len(motif_losses)
        }
        tuning_writer.add_hparams(
            {
                'lambda_1': lambda_1,
                'lambda_2': lambda_2,
                'lambda_3': lambda_3,
                'lambda_4': lambda_4,
            },
            avg_metrics
        )
    
    model.eval()
    with torch.no_grad():
        std_z_val, m_z_val, z_val, reconstructed_adj_val, reconstructed_feat_val, re_labels_val = model(graph_dgl_val,
                                                                                                        feat_val,
                                                                                                        labels_val,
                                                                                                        targets,
                                                                                                        sampling_method,
                                                                                                        is_prior,
                                                                                                        train=True)

        reconstructed_adjacency_val = torch.sigmoid(reconstructed_adj_val)
        reconstructed_x_prob_val = torch.sigmoid(reconstructed_feat_val)
        reconstructed_labels_prob_val = torch.sigmoid(re_labels_val)
        
        # Apply the same adjustment as in training if needed
        if args.devide_rec_adj:
            reconstructed_adjacency_val = [
                (adj * (1 / args.num_nodes)) for adj in reconstructed_adjacency_val
            ]
        
        # Compute predicted motif counts for validation
        if args.motif_obj == True:
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
            
            normalized_predicted_val = [torch.abs((torch.log(p / g))) for p, g in zip(filtered_predicted_val, filtered_observed)]
            
            motif_loss_val = (((torch.sum(torch.stack(normalized_predicted_val))/len((normalized_predicted_val)))))
            motif_loss_val = motif_loss_val.cpu()

        else:
            motif_loss_val = 0
        
    # Calculate validation metrics
    w_l = weight_labels(labels_val)
    posterior_cost_edges = norm_val * F.binary_cross_entropy_with_logits(reconstructed_adj_val, adj_val_org,
                                                                     pos_weight=pos_wight_val)
    posterior_cost_features = norm_feat_val * F.binary_cross_entropy_with_logits(reconstructed_feat_val, feat_val,
                                                                             pos_weight=pos_weight_feat_val)
    posterior_cost_classes = F.cross_entropy(re_labels_val, (torch.tensor(labels_val).to(torch.float64)), weight=w_l)

    # Calculate validation accuracy
    predicted_labels = torch.max(re_labels_val, 1)[1]
    true_labels = torch.max(torch.tensor(labels_val).to(torch.float64), 1)[1]
    val_acc = (predicted_labels == true_labels).sum().item() / len(true_labels)
    
    # Log validation metrics
    if tuning_writer:
        run_name = f"lambda1_{lambda_1:.3f}_lambda2_{lambda_2:.3f}_lambda3_{lambda_3:.3f}_lambda4_{lambda_4:.3f}"
        tuning_writer.add_scalar(f'Validation/{run_name}/EdgeLoss', posterior_cost_edges.item(), 0)
        tuning_writer.add_scalar(f'Validation/{run_name}/FeatureLoss', posterior_cost_features.item(), 0)
        tuning_writer.add_scalar(f'Validation/{run_name}/ClassificationLoss', posterior_cost_classes.item(), 0)
        tuning_writer.add_scalar(f'Validation/{run_name}/Accuracy', val_acc, 0)
        tuning_writer.add_scalar(f'Validation/{run_name}/MotifLoss', motif_loss_val, 0)

    # Calculate total validation cost
    cost = posterior_cost_edges + posterior_cost_features + posterior_cost_classes

    if args.motif_obj == True:
        cost += motif_loss_val
        
    # Ensure we return a valid numerical value for the Bayesian optimization
    # Convert to a standard Python float to avoid any TensorFlow/PyTorch tensor issues
    return float(-1.0 * cost.item())