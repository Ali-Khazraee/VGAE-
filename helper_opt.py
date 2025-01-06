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


    original_adj_full= torch.FloatTensor(getattr(data_center, ds+'_adj_lists')).to(device)
    node_label_full= torch.FloatTensor(getattr(data_center, ds+'_labels')).to(device)

    val_indx = getattr(data_center, ds + '_val_edge_idx')
    train_indx = getattr(data_center, ds + '_train_edge_idx')

    # shuffling the data, and selecting a subset of it
    if subgraph_size == -1:
        subgraph_size = original_adj_full.shape[0]
    elemnt = min(original_adj_full.shape[0], subgraph_size)
    indexes = list(range(original_adj_full.shape[0]))
    # np.random.shuffle(indexes)
    # indexes = indexes[:elemnt]



    original_adj = original_adj_full[indexes, :]
    original_adj = original_adj[:, indexes]


    node_label = [np.array(node_label_full[i], dtype=np.float16) for i in indexes]

    features = features[indexes]
    number_of_classes = len(node_label_full[0])



    trainId = getattr(data_center, ds + '_train')
    testId = getattr(data_center, ds + '_test')
    validId = getattr(data_center, ds + '_val')
    #
    # adj_train = getattr(data_center, ds + '_adj_train')
    # adj_val = getattr(data_center, ds + '_adj_val')
    #
    # feat_np = features.cpu().data.numpy()
    # feat_train = feat_np
    # feat_val = feat_np
    #
    #
    # labels_np = np.array(node_label, dtype=np.float16)
    # labels_train = labels_np
    # labels_val = labels_np




    adj_train = original_adj.cpu().detach().numpy()[trainId, :][:, trainId]
    adj_val = original_adj.cpu().detach().numpy()[validId, :][:, validId]

    feat_np = features.cpu().data.numpy()
    feat_train = feat_np[trainId, :]
    feat_val = feat_np[validId, :]

    labels_np = np.array(node_label, dtype=np.float16)
    labels_train = labels_np[trainId]
    labels_val = labels_np[validId]



# ########################################

    edge_labels1 = getattr(data_center, ds + '_edge_labels')

    edge_labels_row_shuffled1 = edge_labels1[indexes, :]

    edge_labels_shuffled = edge_labels_row_shuffled1[:, indexes]

    # Process training data

    edge_labels_train = edge_labels_shuffled

    edge_relType = sp.csr_matrix(np.multiply(edge_labels_train, original_adj))

    rel_type = np.unique(edge_labels1[edge_labels1 != 0])

    # Initialize training matrices

    org_adj = []

    for rel_num in rel_type:

        tm_mtrix = sp.csr_matrix(edge_relType.shape)

        tm_mtrix[edge_relType == int(rel_num)] = 1

        org_adj.append(tm_mtrix.todense())
    
    org_adj = [torch.tensor(matrix) for matrix in org_adj]

########################################



############################################################

    # Get and process edge labels for training

    edge_labels = getattr(data_center, ds + '_edge_labels')

    edge_labels_row_shuffled = edge_labels[indexes, :]

    edge_labels_shuffled = edge_labels_row_shuffled[:, indexes]

    # Process training data

    edge_labels_train = edge_labels_shuffled[trainId, :][:, trainId]

    edge_relType_train = sp.csr_matrix(np.multiply(edge_labels_train, adj_train))

    rel_type = np.unique(edge_labels[edge_labels != 0])

    # Initialize training matrices

    tra_matrix = []

    for rel_num in rel_type:

        tm_mtrix = sp.csr_matrix(edge_relType_train.shape)

        tm_mtrix[edge_relType_train == int(rel_num)] = 1

        tra_matrix.append(tm_mtrix)

    # Process validation data

    edge_labels_shuffled_val = edge_labels_shuffled[validId, :][:, validId]

    edge_relType_val = sp.csr_matrix(np.multiply(edge_labels_shuffled_val, adj_val))

    # Initialize validation matrices

    val_matrix = []

    for rel_num in rel_type:

        val_mtrix = sp.csr_matrix(edge_relType_val.shape)

        val_mtrix[edge_relType_val == int(rel_num)] = 1

        val_matrix.append(val_mtrix)

    # Process training graphs and matrices

    graph_dgl = []

    pre_self_loop_train_adj = []

    train_matrix = []

    for adj in tra_matrix:

        # Keep as sparse until necessary

        sparse_adj = adj

        pre_self_loop_train_adj.append(sparse_adj.todense())

        

        # Add self-loops

        tr_matrix = sparse_adj + sp.eye(adj.shape[0])

        train_matrix.append(tr_matrix.todense())

        

        # Create DGL graph

        src, dst = tr_matrix.nonzero()

        graph_dgl.append(dgl.graph((src, dst), num_nodes=adj.shape[0]))

    # Convert training matrices to torch tensors

    train_matrix = [torch.tensor(mtrix) for mtrix in train_matrix]

    adj_train =  train_matrix

    # Process validation graphs and matrices

    graph_dgl_val = []

    pre_self_loop_val_adj = []

    validation_matrix = []

    for adj in val_matrix:

        # Keep as sparse until necessary

        sparse_adj_val = adj

        pre_self_loop_val_adj.append(sparse_adj_val.todense())

        

        # Add self-loops

        vl_matrix = sparse_adj_val + sp.eye(adj.shape[0])

        validation_matrix.append(vl_matrix.todense())

        

        # Create DGL graph

        src, dst = vl_matrix.nonzero()

        graph_dgl_val.append(dgl.graph((src, dst), num_nodes=adj.shape[0]))

    # Convert validation matrices to torch tensors

    validation_matrix = [torch.tensor(mtrix) for mtrix in validation_matrix]

############################################################


    print('Finish spliting dataset to train and test. ')



    if (type(feat_train) == np.ndarray):
        feat_train = torch.tensor(feat_train, dtype=torch.float32)
        feat_val = torch.tensor(feat_val, dtype=torch.float32)


    # Check for Encoder and redirect to appropriate function
    if encoder == "Multi_GCN":
        encoder_model = multi_layer_GCN(num_of_comunities , latent_dim=num_of_comunities, layers=encoder_layers)
        # encoder_model = multi_layer_GCN(in_feature=features.shape[1], latent_dim=num_of_comunities, layers=encoder_layers)

    elif encoder == "Multi_GAT":
        encoder_model = multi_layer_GAT(num_of_comunities , latent_dim=num_of_comunities, layers=encoder_layers)


    elif encoder == "RGCN_Encoder":

        encoder_model = RGCN_Encoder(
        in_feature=num_of_comunities,
        num_relation=len(graph_dgl),  
        latent_dim=num_of_comunities,
        layers=encoder_layers,
        DropOut_rate=0.3
        )


    elif encoder == "Multi_GIN":
        encoder_model = multi_layer_GIN(num_of_comunities, latent_dim=num_of_comunities, layers=encoder_layers)

    elif encoder == "Multi_SAGE":
        encoder_model = multi_layer_SAGE(num_of_comunities, latent_dim=num_of_comunities, layers=encoder_layers)

    else:
        raise Exception("Sorry, this Encoder is not Impemented; check the input args")

    # Check for Decoder and redirect to appropriate function

    if decoder == "ML_SBM":
        decoder_model = MultiLatetnt_SBM_decoder(num_of_relations, num_of_comunities, num_of_comunities, batch_norm, DropOut_rate=0.3)



    elif decoder == "MultiRelational_SBM_decoder":

        decoder_model = MultiRelational_SBM_decoder(
        number_of_rel=len(graph_dgl), 
        Lambda_dim=num_of_comunities,
        in_dim=num_of_comunities,
        normalize=batch_norm,
        DropOut_rate=0.3
)

    else:
        raise Exception("Sorry, this Decoder is not Impemented; check the input args")

    feature_encoder_model = feature_encoder(features.view(-1, features.shape[1]), num_of_comunities)
    # feature_encoder_model = MulticlassClassifier(num_of_comunities, features.shape[1])
    feature_decoder = feature_decoder_nn(features.shape[1], num_of_comunities)
    class_decoder = MulticlassClassifier(number_of_classes, num_of_comunities)





    model = VGAE_FrameWork(num_of_comunities,
                            encoder = encoder_model,
                            decoder = decoder_model,
                            feature_decoder = feature_decoder,
                            feature_encoder = feature_encoder_model,
                            classifier=class_decoder)
    optimizer = torch.optim.Adam(model.parameters(), lr)

    # adj_train = torch.tensor(adj_train)
    # adj_val = torch.tensor(adj_val)
    # pos_wight = torch.true_divide((adj_train.shape[0] ** 2 - torch.sum(adj_train)), torch.sum(
    #     adj_train))  # addrressing imbalance data problem: ratio between positve to negative instance
    # pos_wight_val = torch.true_divide((adj_val.shape[0] ** 2 - torch.sum(adj_val)), torch.sum(
    #     adj_val))
    # norm = torch.true_divide(adj_train.shape[0] * adj_train.shape[0],
    #                          ((adj_train.shape[0] * adj_train.shape[0] - torch.sum(adj_train)) * 2))
    # norm_val = torch.true_divide(adj_val.shape[0] * adj_val.shape[0],
    #                          ((adj_val.shape[0] * adj_val.shape[0] - torch.sum(adj_val)) * 2))
    # pos_weight_feat = torch.true_divide((feat_train.shape[0] * feat_train.shape[1] - torch.sum(feat_train)),
    #                                     torch.sum(feat_train))

    # norm_feat = torch.true_divide((feat_train.shape[0] * feat_train.shape[1]),
    #                               (2 * (feat_train.shape[0] * feat_train.shape[1] - torch.sum(feat_train))))

    # pos_weight_feat_val = torch.true_divide((feat_val.shape[0] * feat_val.shape[1] - torch.sum(feat_val)),
    #                                         torch.sum(feat_val))
    # norm_feat_val = torch.true_divide((feat_val.shape[0] * feat_val.shape[1]),
    #                                   (2 * (feat_val.shape[0] * feat_val.shape[1] - torch.sum(feat_val))))

################################################################
    num_nodes , _ = original_adj.shape
    num_nodes_val , _ = adj_val.shape
    pos_weights_train = []
    pos_weights_val = []
    norms_train = []
    norms_val = []

    for adj_mat in train_matrix:        
        pos_weight = torch.true_divide((adj_mat.shape[0] ** 2 - torch.sum(adj_mat)), torch.sum(adj_mat))
        pos_weights_train.append(pos_weight)
        
        norm = torch.true_divide(adj_mat.shape[0] * adj_mat.shape[0],
                            ((adj_mat.shape[0] * adj_mat.shape[0] - torch.sum(adj_mat)) * 2))
        norms_train.append(norm)

    for adj_mat in validation_matrix:
        pos_weight = torch.true_divide((adj_mat.shape[0] ** 2 - torch.sum(adj_mat)), torch.sum(adj_mat))
        pos_weights_val.append(pos_weight)
        
        norm = torch.true_divide(adj_mat.shape[0] * adj_mat.shape[0],
                            ((adj_mat.shape[0] * adj_mat.shape[0] - torch.sum(adj_mat)) * 2))
        norms_val.append(norm)

    pos_weight_feat = torch.true_divide((feat_train.shape[0] * feat_train.shape[1] - torch.sum(feat_train)),
                                    torch.sum(feat_train))
    norm_feat = torch.true_divide((feat_train.shape[0] * feat_train.shape[1]),
                                (2 * (feat_train.shape[0] * feat_train.shape[1] - torch.sum(feat_train))))

    pos_weight_feat_val = torch.true_divide((feat_val.shape[0] * feat_val.shape[1] - torch.sum(feat_val)),
                                        torch.sum(feat_val))
    norm_feat_val = torch.true_divide((feat_val.shape[0] * feat_val.shape[1]),
                                    (2 * (feat_val.shape[0] * feat_val.shape[1] - torch.sum(feat_val))))
    
################################################################


    


    mapping_detail = getattr(data_center, ds +'_mapping_details')

    # Calculate reverse mappings once
    movie_rev = []
    director_rev = []
    actor_rev = []
    for node_type, (start, end) in mapping_detail['node_type_to_index_map'].items():
        if node_type == 'movie':
            for i in range(start, end):
                try:
                    movie_rev.append(list(trainId).index(i))
                except ValueError:
                    pass
        elif node_type == 'director':
            for i in range(start, end):
                try:
                    director_rev.append(list(trainId).index(i))
                except ValueError:
                    pass
        elif node_type == 'actor':
            for i in range(start, end):
                try:
                    actor_rev.append(list(trainId).index(i))
                except ValueError:
                    pass

    # Add reverse mapping to mapping_detail
    mapping_detail['reverse_mapping'] = {
        'movie': movie_rev,
        'director': director_rev,
        'actor': actor_rev
    }




    if args.motif_obj == True:
        CM = Motif_Count(args)
        CM.setup_function()
        reconstructed_x_slice, reconstructed_labels_m = CM.process_reconstructed_data(mapping_detail, 
        adj_train, feat_train[:,np.array([ 23, 244,  59,  69, 222])], np.array([ 23, 244,  59,  69, 222]), torch.tensor(labels_train)
    )
        ground_truth = CM.iteration_function(reconstructed_x_slice , reconstructed_labels_m, mode = "ground-truth")


    else:
        CM = None
        ground_truth = None


   

    lambda_1 = 1
    lambda_2 = 1
    lambda_3 = 1
    lambda_4 = 1

    #to find weights
    if args.tuning == "True":
        pbounds = {
            'lambda_1': (0.0, 1.0),
            'lambda_2': (0.0, 1.0),
            'lambda_3': (0.0, 1.0),
            'lambda_4': (0.0, 1.0)
        }
        optimizer_function = make_optimizer_wrapper(labels_train, labels_val, dataset, epoch_number, model, graph_dgl, graph_dgl_val, feat_train,
                    feat_val, targets, sampling_method, is_prior, loss_type, adj_train, validation_matrix, norm_feat,
                    pos_weight_feat, norm_feat_val, pos_weight_feat_val, num_nodes, num_nodes_val, pos_weights_train, norms_train,
                    pos_weights_val, norms_val, optimizer, val_indx, trainId, args, ground_truth, CM, data_center, mapping_detail)
        optimizer_hp = BayesianOptimization(

 
 
            f=optimizer_function,
            pbounds=pbounds,
            random_state=42
        )
        optimizer_hp.maximize(
            init_points=20,
            n_iter=200
       )
        print(optimizer_hp.max)

        #Extract and print the best values for weight1 and weight2
        best_params = optimizer_hp.max['params']
        lambda_1= best_params['lambda_1']
        lambda_2= best_params['lambda_2']
        lambda_3 = best_params['lambda_3']
        lambda_4 = best_params['lambda_4']

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

        print("weights:", lambda_1, lambda_1, lambda_3, lambda_4)






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

            reconstructed_x_slice, reconstructed_labels_m = CM.process_reconstructed_data(mapping_detail, 
            reconstructed_adjacency, reconstructed_x_prob[:,np.array([ 23, 244,  59,  69, 222])], np.array([ 23, 244,  59,  69, 222]), torch.tensor(reconstructed_labels_prob)
        )
            predicted = CM.iteration_function(reconstructed_x_slice , reconstructed_labels_m, mode = "ground-truth")


        else:
            predicted = None






        z_kl, reconstruction_loss,posterior_cost_edges ,posterior_cost_features , posterior_cost_classes, acc, val_recons_loss, loss_adj, loss_feat, motif_loss = optimizer_VAE(lambda_1, lambda_2,
                                                                                                lambda_3,lambda_4, labels_train,
                                                                                                re_labels, loss_type,
                                                                                                reconstructed_adj,
                                                                                                reconstructed_feat,
                                                                                                adj_train,
                                                                                                feat_train, norm_feat,
                                                                                                pos_weight_feat,
                                                                                                std_z, m_z, num_nodes,
                                                                                                pos_weights_train, norms_train, val_indx, train_indx, args, ground_truth, predicted)

        loss = reconstruction_loss + z_kl

        # backward propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # print some metrics
        print("Epoch: {:03d} | Loss: {:05f} | edge_loss: {:05f} |feat_loss: {:05f} |node_classification_loss: {:05f} | z_kl_loss: {:05f} | Accuracy: {:03f}".format(
            epoch + 1, loss.item(), reconstruction_loss.item(),posterior_cost_edges.item() ,posterior_cost_features.item() , posterior_cost_classes.item(), z_kl.item(), acc))
    model.eval()



    return model, z 


def optimize_weights(lambda_1, lambda_2, lambda_3,lambda_4, labels_train, labels_val, dataset, epoch_number, model, graph_dgl, graph_dgl_val, feat_train,
                feat_val, targets, sampling_method, is_prior, loss_type, adj_train_org, adj_val_org, norm_feat,
                pos_weight_feat, norm_feat_val, pos_weight_feat_val, num_nodes, num_nodes_val, pos_weight_train, norms_train,
                pos_weight_val, norms_val, optimizer, val_indx, trainId, args, ground_truth, CM, data_center, mapping_detail):
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

            reconstructed_x_slice, reconstructed_labels_m = CM.process_reconstructed_data(mapping_detail, 
            reconstructed_adjacency, reconstructed_x_prob[:,np.array([ 23, 244,  59,  69, 222])], np.array([ 23, 244,  59,  69, 222]), torch.tensor(reconstructed_labels_prob)
        )
            predicted = CM.iteration_function(reconstructed_x_slice , reconstructed_labels_m, mode = "ground-truth")


        else:
            predicted = None

        # compute loss and accuracy
        z_kl, reconstruction_loss, posterior_cost_edges, posterior_cost_features, posterior_cost_classes, acc, val_recons_loss, loss_adj, loss_feat, motif_loss = optimizer_VAE(
            lambda_1, lambda_2,
            lambda_3,lambda_4, labels_train,
            re_labels, loss_type,
            reconstructed_adj,
            reconstructed_feat,
            adj_train_org,
            feat_train, norm_feat,
            pos_weight_feat,
            std_z, m_z, num_nodes,
            pos_weight_train, norms_train, val_indx, trainId, args, ground_truth, predicted)
        loss = reconstruction_loss + z_kl

        # reconstructed_adj = torch.sigmoid(reconstructed_adj).detach().numpy()

        model.eval()

        model.train()
        # backward propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print some metrics
        print(
            "Epoch: {:03d} | Loss: {:05f} | Reconstruction_loss: {:05f} | z_kl_loss: {:05f} | Accuracy: {:03f}".format(
                epoch + 1, loss.item(), reconstruction_loss.item(), z_kl.item(), acc))
    model.eval()
    with torch.no_grad():
        std_z_val, m_z_val, z_val, reconstructed_adj_val, reconstructed_feat_val, re_labels_val = model(graph_dgl_val,
                                                                                                        feat_val,
                                                                                                        labels_val,
                                                                                                        targets,
                                                                                                        sampling_method,
                                                                                                        is_prior,
                                                                                                        train=True)

    w_l = weight_labels(labels_val)
    if isinstance(adj_val_org, (list, tuple)):
        adj_val_tensors = [torch.tensor(adj) if not isinstance(adj, torch.Tensor) else adj 
                        for adj in adj_val_org]
        adj_val_org = torch.stack(adj_val_tensors)
    elif not isinstance(adj_val_org, torch.Tensor):
        adj_val_org = torch.tensor(adj_val_org)

    if len(adj_val_org.shape) == 2:
        adj_val_org = adj_val_org.expand(2, -1, -1)

    adj_val_org = adj_val_org.to(reconstructed_adj_val.dtype).to(reconstructed_adj_val.device)

    norms_val_tensor = torch.tensor(norms_val).view(-1, 1, 1)       
    pos_weights_val_tensor = torch.tensor(pos_weight_val).view(-1, 1, 1)

    posterior_cost_edges = (norms_val_tensor * F.binary_cross_entropy_with_logits(
        reconstructed_adj_val, 
        adj_val_org,
        pos_weight=pos_weights_val_tensor,
        reduction='none'
    )).mean()
    
    posterior_cost_features = norm_feat_val * F.binary_cross_entropy_with_logits(reconstructed_feat_val, feat_val,
                                                                             pos_weight=pos_weight_feat_val)
    posterior_cost_classes = F.cross_entropy(re_labels_val, (torch.tensor(labels_val).to(torch.float64)), weight=w_l)

    cost = posterior_cost_edges + posterior_cost_features + posterior_cost_classes

    return -1*cost.item()


def make_optimizer_wrapper(labels_train, labels_val, dataset, epoch_number, model, graph_dgl, graph_dgl_val, feat_train,
                feat_val, targets, sampling_method, is_prior, loss_type, adj_train_org, adj_val_org, norm_feat,
                pos_weight_feat, norm_feat_val, pos_weight_feat_val, num_nodes, num_nodes_val, pos_wight, norm,
                pos_wight_val, norm_val, optimizer, val_indx, trainId, args, ground_truth, CM, data_center, mapping_detail):
    def optimize_weights_wrapper(lambda_1, lambda_2, lambda_3, lambda_4):
        return optimize_weights(lambda_1, lambda_2, lambda_3, lambda_4, labels_train, labels_val, dataset, epoch_number, model, graph_dgl, graph_dgl_val, feat_train,
                feat_val, targets, sampling_method, is_prior, loss_type, adj_train_org, adj_val_org, norm_feat,
                pos_weight_feat, norm_feat_val, pos_weight_feat_val, num_nodes, num_nodes_val, pos_wight, norm,
                pos_wight_val, norm_val, optimizer, val_indx, trainId, args, ground_truth, CM, data_center, mapping_detail)
    return optimize_weights_wrapper