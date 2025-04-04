import concurrent.futures
from datetime import datetime
from functools import partial
import numpy as np
import networkx as nx
import os
import pickle as pkl
import subprocess
import time
import sys
import rnn_mmd as mmd
import pickle
from scipy.linalg import eigvalsh
import csv
PRINT_TIME = False


def degree_worker(G):
    return np.array(nx.degree_histogram(G))


def add_tensor(x, y):
    support_size = max(len(x), len(y))
    if len(x) < len(y):
        x = np.hstack((x, [0.0] * (support_size - len(x))))
    elif len(y) < len(x):
        y = np.hstack((y, [0.0] * (support_size - len(y))))
    return x + y


def degree_stats(graph_ref_list, graph_pred_list, is_parallel=False):
    ''' Compute the distance between the degree distributions of two unordered sets of graphs.
    Args:
      graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    '''
    sample_ref = []
    sample_pred = []
    # in case an empty graph is generated
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_ref_list):
                sample_ref.append(deg_hist)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_pred_list_remove_empty):
                sample_pred.append(deg_hist)

    else:
        for i in range(len(graph_ref_list)):
            degree_temp = np.array(nx.degree_histogram(graph_ref_list[i]))
            sample_ref.append(degree_temp)
        for i in range(len(graph_pred_list_remove_empty)):
            degree_temp = np.array(nx.degree_histogram(graph_pred_list_remove_empty[i]))
            sample_pred.append(degree_temp)
    # print(len(sample_ref), len(sample_pred))
    # mmd_dist = mmd.compute_mmd(sample_ref, sample_pred, kernel=mmd.gaussian_emd)
    mmd_dist = mmd.compute_mmd(sample_ref, sample_pred, kernel=mmd.gaussian_tv)
    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing degree mmd: ', elapsed)
    return mmd_dist


def Diam_stats(graph_list):
    graph_list = [G for G in graph_list if not G.number_of_nodes() == 0]
    graph_list = np.array([nx.diameter(G) for G in graph_list])
    print("Average Diam:", str(np.average(graph_list)), "Var:", str(np.var(graph_list)), "Max Diam:",
          str(np.max(graph_list)), "Min Diam:", str(np.min(graph_list)))


def MMD_diam(graph_ref_list, graph_pred_list, is_parallel=False):
    ''' Compute the distance between the degree distributions of two unordered sets of graphs.
    Args:
      graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    '''
    sample_ref = []
    sample_pred = []
    # in case an empty graph is generated
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    for i in range(len(graph_ref_list)):
        try:
            degree_temp = np.array([nx.diameter(graph_ref_list[i])])
            sample_ref.append(degree_temp)
        except:
            print("An exception occurred; disconnected graph in ref set")
    for i in range(len(graph_pred_list_remove_empty)):
        try:
            degree_temp = np.array([nx.diameter(graph_pred_list_remove_empty[i])])
            sample_pred.append(degree_temp)
        except:
            print("An exception occurred; disconnected graph in gen set")
    mmd_dist = mmd.compute_mmd(sample_ref, sample_pred, kernel=mmd.gaussian_tv, is_hist=False)
    return mmd_dist


def clustering_worker(param):
    G, bins = param
    clustering_coeffs_list = list(nx.clustering(G).values())
    hist, _ = np.histogram(
        clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
    return hist


def clustering_stats(graph_ref_list, graph_pred_list, bins=100, is_parallel=False):
    sample_ref = []
    sample_pred = []
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for clustering_hist in executor.map(clustering_worker,
                                                [(G, bins) for G in graph_ref_list]):
                sample_ref.append(clustering_hist)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for clustering_hist in executor.map(clustering_worker,
                                                [(G, bins) for G in graph_pred_list_remove_empty]):
                sample_pred.append(clustering_hist)
        # check non-zero elements in hist
        # total = 0
        # for i in range(len(sample_pred)):
        #    nz = np.nonzero(sample_pred[i])[0].shape[0]
        #    total += nz
        # print(total)
    else:
        for i in range(len(graph_ref_list)):
            clustering_coeffs_list = list(nx.clustering(graph_ref_list[i]).values())
            hist, _ = np.histogram(
                clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
            sample_ref.append(hist)

        for i in range(len(graph_pred_list_remove_empty)):
            clustering_coeffs_list = list(nx.clustering(graph_pred_list_remove_empty[i]).values())
            hist, _ = np.histogram(
                clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
            sample_pred.append(hist)
    #
    # mmd_dist = mmd.compute_mmd(sample_ref, sample_pred, kernel=mmd.gaussian_emd,
    #                            sigma=1.0 / 10, distance_scaling=bins)
    mmd_dist = mmd.compute_mmd(sample_ref, sample_pred, kernel=mmd.gaussian_tv,
                               sigma=1.0 / 10)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing clustering mmd: ', elapsed)
    return mmd_dist


# maps motif/orbit name string to its corresponding list of indices from orca output
motif_to_indices = {
    '3path': [1, 2],
    '4cycle': [8],
}
COUNT_START_STR = 'orbit counts: \n'


def edge_list_reindexed(G):
    idx = 0
    id2idx = dict()
    for u in G.nodes():
        id2idx[str(u)] = idx
        idx += 1

    edges = []
    for (u, v) in G.edges():
        edges.append((id2idx[str(u)], id2idx[str(v)]))
    return edges


def orca(graph):
    tmp_fname = 'eval/orca/tmp.txt'
    f = open(tmp_fname, 'w')
    f.write(str(graph.number_of_nodes()) + ' ' + str(graph.number_of_edges()) + '\n')
    for (u, v) in edge_list_reindexed(graph):
        f.write(str(u) + ' ' + str(v) + '\n')
    f.close()

    output = subprocess.check_output(['./eval/orca/orca', 'node', '4', 'eval/orca/tmp.txt', 'std'])
    output = output.decode('utf8').strip()

    idx = output.find(COUNT_START_STR) + len(COUNT_START_STR)
    output = output[idx:]
    node_orbit_counts = np.array([list(map(int, node_cnts.strip().split(' ')))
                                  for node_cnts in output.strip('\n').split('\n')])

    try:
        os.remove(tmp_fname)
    except OSError:
        pass

    return node_orbit_counts


def motif_stats(graph_ref_list, graph_pred_list, motif_type='4cycle', observed_match=None, bins=100):
    # graph motif counts (int for each graph)
    # normalized by graph size
    total_counts_ref = []
    total_counts_pred = []

    num_matches_ref = []
    num_matches_pred = []

    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]
    indices = motif_to_indices[motif_type]
    for G in graph_ref_list:
        orbit_counts = orca(G)
        motif_counts = np.sum(orbit_counts[:, indices], axis=1)

        if observed_match is not None:
            match_cnt = 0
            for elem in motif_counts:
                if elem == observed_match:
                    match_cnt += 1
            num_matches_ref.append(match_cnt / G.number_of_nodes())

        # hist, _ = np.histogram(
        #        motif_counts, bins=bins, density=False)
        motif_temp = np.sum(motif_counts) / G.number_of_nodes()
        total_counts_ref.append(motif_temp)

    for G in graph_pred_list_remove_empty:
        orbit_counts = orca(G)
        motif_counts = np.sum(orbit_counts[:, indices], axis=1)

        if observed_match is not None:
            match_cnt = 0
            for elem in motif_counts:
                if elem == observed_match:
                    match_cnt += 1
            num_matches_pred.append(match_cnt / G.number_of_nodes())

        motif_temp = np.sum(motif_counts) / G.number_of_nodes()
        total_counts_pred.append(motif_temp)

    mmd_dist = mmd.compute_mmd(total_counts_ref, total_counts_pred, kernel=mmd.gaussian,
                               is_hist=False)
    # print('-------------------------')
    # print(np.sum(total_counts_ref) / len(total_counts_ref))
    # print('...')
    # print(np.sum(total_counts_pred) / len(total_counts_pred))
    # print('-------------------------')
    return mmd_dist


# this functione is used to calculate some of the famous graph properties
def MMD_triangles(graph_ref_list, graph_pred_list):
    """

    :param list_of_adj: list of nx arrays
    :return:
    """
    total_counts_pred = []
    for graph in graph_pred_list:
        total_counts_pred.append([np.sum(list(nx.triangles(graph).values())) / graph.number_of_nodes()])

    total_counts_ref = []
    for graph in graph_ref_list:
        total_counts_ref.append([np.sum(list(nx.triangles(graph).values())) / graph.number_of_nodes()])

    total_counts_pred = np.array(total_counts_pred)
    total_counts_ref = np.array(total_counts_ref)
    mmd_dist = mmd.compute_mmd(total_counts_ref, total_counts_pred, kernel=mmd.gaussian_tv,
                               is_hist=False, sigma=30.0)
    # print("averrage number of tri in ref/ test: ", str(np.average(total_counts_pred)), str(np.average(total_counts_ref)))
    return mmd_dist


def sparsity_stats_all(graph_ref_list, graph_pred_list):
    def sparsity(G):
        return (G.number_of_nodes() ** 2 - len(G.edges)) / G.number_of_nodes() ** 2

    def edge_num(G):
        return len(G.edges)

    total_counts_ref = []
    total_counts_pred = []

    edge_num_ref = []
    edge_num_pre = []
    for G in graph_ref_list:
        sp = sparsity(G)
        total_counts_ref.append([sp])
        edge_num_ref.append(edge_num(G))

    for G in graph_pred_list:
        sp = sparsity(G)
        total_counts_pred.append([sp])
        edge_num_pre.append(edge_num(G))

    total_counts_ref = np.array(total_counts_ref)
    total_counts_pred = np.array(total_counts_pred)
    mmd_dist = mmd.compute_mmd(total_counts_ref, total_counts_pred, kernel=mmd.gaussian_tv,
                               is_hist=False, sigma=30.0)

    # print('-------------------------')
    # print(np.sum(total_counts_ref, axis=0) / len(total_counts_ref))
    # print('...')
    # print(np.sum(total_counts_pred, axis=0) / len(total_counts_pred))
    # print('-------------------------')
    # print("average edge # in test set:")
    # print(np.average(edge_num_ref))
    # print("average edge # in generated set:")
    # print(np.average(edge_num_pre))
    # print('-------------------------')

    return mmd_dist, np.average(edge_num_ref), np.average(edge_num_pre)


def orbit_stats_all(graph_ref_list, graph_pred_list):
    total_counts_ref = []
    total_counts_pred = []

    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    for G in graph_ref_list:
        try:
            orbit_counts = orca(G)
        except:
            print("Unexpected error:", sys.exc_info()[0])
            continue
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
        total_counts_ref.append(orbit_counts_graph)

    for G in graph_pred_list:
        try:
            orbit_counts = orca(G)
        except:
            continue
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
        total_counts_pred.append(orbit_counts_graph)

    total_counts_ref = np.array(total_counts_ref)
    total_counts_pred = np.array(total_counts_pred)
    mmd_dist = mmd.compute_mmd(total_counts_ref, total_counts_pred, kernel=mmd.gaussian_tv,
                               is_hist=False, sigma=30.0)
    # mmd_dist = mmd.compute_mmd(total_counts_ref, total_counts_pred, kernel=mmd.gaussian,
    #                            is_hist=False, sigma=30.0)
    # print('-------------------------')
    # print(np.sum(total_counts_ref, axis=0) / len(total_counts_ref))
    # print('...')
    # print(np.sum(total_counts_pred, axis=0) / len(total_counts_pred))
    # print('-------------------------')
    return mmd_dist

import sys
import traceback
import networkx as nx

def mmd_eval(generated_graph_list, original_graph_list, diam=False):
    try:
        print("Starting mmd_eval with:")
        print(f"  - {len(generated_graph_list)} generated graphs")
        print(f"  - {len(original_graph_list)} original graphs")
        
        # Check graphs before filtering
        for i, G in enumerate(generated_graph_list):
            print(f"Generated graph {i}: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}, selfloops={len(list(nx.selfloop_edges(G)))}")
        
        # Filter empty graphs
        generated_graph_list = [G for G in generated_graph_list if not G.number_of_nodes() == 0]
        print(f"After filtering empty graphs: {len(generated_graph_list)} generated graphs remain")
        
        # Remove self-loops with verification
        for i, G in enumerate(generated_graph_list):
            self_loops = list(nx.selfloop_edges(G))
            print(f"Removing {len(self_loops)} self-loops from generated graph {i}")
            G.remove_edges_from(self_loops)
            
        for i, G in enumerate(original_graph_list):
            self_loops = list(nx.selfloop_edges(G))
            print(f"Removing {len(self_loops)} self-loops from original graph {i}")
            G.remove_edges_from(self_loops)
        
        # Additional filtering with verification
        tmp_generated_graph_list = []
        for i, G in enumerate(generated_graph_list):
            if G.number_of_nodes() > 0:
                tmp_generated_graph_list.append(G)
            else:
                print(f"Removing empty generated graph {i}")
        
        generated_graph_list = tmp_generated_graph_list
        print(f"Final graph count: {len(generated_graph_list)} generated, {len(original_graph_list)} original")
        
        # Check for potential issues in graphs
        for i, G in enumerate(generated_graph_list):
            zero_degree_nodes = [n for n, d in G.degree() if d == 0]
            if zero_degree_nodes:
                print(f"WARNING: Generated graph {i} has {len(zero_degree_nodes)} nodes with zero degree")
            if not nx.is_connected(G):
                print(f"WARNING: Generated graph {i} is not connected")
                
        for i, G in enumerate(original_graph_list):
            zero_degree_nodes = [n for n, d in G.degree() if d == 0]
            if zero_degree_nodes:
                print(f"WARNING: Original graph {i} has {len(zero_degree_nodes)} nodes with zero degree")
            if not nx.is_connected(G):
                print(f"WARNING: Original graph {i} is not connected")
        
        # Start metric calculations with detailed error handling
        print("\nCalculating metrics:")
        
        # Degree stats
        print("Calculating degree stats...")
        try:
            mmd_degree = degree_stats(original_graph_list, generated_graph_list)
            print(f"  - Degree MMD: {mmd_degree}")
        except Exception as e:
            print(f"ERROR in degree_stats: {str(e)}")
            traceback.print_exc()
            mmd_degree = "ERROR"
        
        # Orbit stats with detailed sample tracking
        print("Calculating orbit stats...")
        try:
            # Wrap the orbit_stats_all call to inspect samples before MMD computation
            # This requires a modified version of orbit_stats_all that returns samples
            # If you can't modify orbit_stats_all, use the next approach below
            
            def wrapped_orbit_stats_all(orig_graphs, gen_graphs):
                """Wrapper to intercept and debug orbit samples"""
                # Get orbit counts for both sets of graphs (implementation depends on your code)
                orig_counts = get_graph_orbits(orig_graphs)
                gen_counts = get_graph_orbits(gen_graphs)
                
                print(f"Original graphs orbit sample count: {len(orig_counts)}")
                print(f"Generated graphs orbit sample count: {len(gen_counts)}")
                
                if len(orig_counts) == 0:
                    print("WARNING: Original graphs produced ZERO orbit samples!")
                    print("Graph properties of original graphs:")
                    for i, G in enumerate(orig_graphs):
                        print(f"  Graph {i}: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}, connected={nx.is_connected(G)}")
                
                if len(gen_counts) == 0:
                    print("WARNING: Generated graphs produced ZERO orbit samples!")
                    print("Graph properties of generated graphs:")
                    for i, G in enumerate(gen_graphs):
                        print(f"  Graph {i}: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}, connected={nx.is_connected(G)}")
                
                # If either set is empty, we know we'll get division by zero
                if len(orig_counts) == 0 or len(gen_counts) == 0:
                    raise ValueError(f"Empty orbit samples detected: orig={len(orig_counts)}, gen={len(gen_counts)}")
                
                # Continue with normal orbit_stats_all calculation
                return orbit_stats_all(orig_graphs, gen_graphs)
            
            # Try to use the wrapper function if compatible with your code
            try:
                mmd_4orbits = wrapped_orbit_stats_all(original_graph_list, generated_graph_list)
            except NameError:
                # If get_graph_orbits isn't defined, fall back to inspecting MMD directly
                print("Wrapper not compatible, attempting direct MMD inspection...")
                
                # Modify your rnn_mmd.py file's compute_mmd function to check for zero length
                # If you can't modify that file, patch it at runtime
                original_compute_mmd = sys.modules.get('rnn_mmd', None)
                if original_compute_mmd and hasattr(original_compute_mmd, 'compute_mmd'):
                    original_mmd_fn = original_compute_mmd.compute_mmd
                    
                    def patched_compute_mmd(samples1, samples2, kernel, *args, **kwargs):
                        print(f"MMD samples: len(samples1)={len(samples1)}, len(samples2)={len(samples2)}")
                        if len(samples1) == 0:
                            print("WARNING: First sample set (original graphs) is EMPTY!")
                        if len(samples2) == 0:
                            print("WARNING: Second sample set (generated graphs) is EMPTY!")
                        
                        if len(samples1) == 0 or len(samples2) == 0:
                            return -1  # Avoid division by zero
                        return original_mmd_fn(samples1, samples2, kernel, *args, **kwargs)
                    
                    # Apply the patch
                    original_compute_mmd.compute_mmd = patched_compute_mmd
                
                # Try the regular call with our patch applied
                mmd_4orbits = orbit_stats_all(original_graph_list, generated_graph_list)
            
            print(f"  - Orbit MMD: {mmd_4orbits}")
        except ZeroDivisionError as e:
            print(f"ZERO DIVISION ERROR in orbit_stats_all: {str(e)}")
            print(f"Location: {traceback.format_exc()}")
            
            # Try to inspect mmd.py and orbit_stats functions directly
            print("\nAttempting to diagnose orbit calculation issue:")
            try:
                import inspect
                
                # Try to get source of the relevant functions
                if 'orbit_stats_all' in globals():
                    print("orbit_stats_all function source:")
                    print(inspect.getsource(orbit_stats_all))
                
                if 'mmd' in sys.modules:
                    mmd_module = sys.modules['mmd']
                    if hasattr(mmd_module, 'compute_mmd'):
                        print("compute_mmd function source:")
                        print(inspect.getsource(mmd_module.compute_mmd))
            except Exception as inspect_error:
                print(f"Could not inspect source: {inspect_error}")
            
            mmd_4orbits = -1
        except Exception as e:
            print(f"ERROR in orbit_stats_all: {str(e)}")
            traceback.print_exc()
            mmd_4orbits = -1
        
        # Clustering stats
        print("Calculating clustering stats...")
        try:
            mmd_clustering = clustering_stats(original_graph_list, generated_graph_list)
            print(f"  - Clustering MMD: {mmd_clustering}")
        except Exception as e:
            print(f"ERROR in clustering_stats: {str(e)}")
            traceback.print_exc()
            mmd_clustering = "ERROR"
        
        # Spectral stats
        print("Calculating spectral stats...")
        try:
            mmd_spectral = spectral_stats(original_graph_list, generated_graph_list)
            print(f"  - Spectral MMD: {mmd_spectral}")
        except Exception as e:
            print(f"ERROR in spectral_stats: {str(e)}")
            traceback.print_exc()
            mmd_spectral = "ERROR"
        
        # Diameter stats
        if diam:
            print("Calculating diameter stats...")
            try:
                mmd_diam = MMD_diam(original_graph_list, generated_graph_list)
                print(f"  - Diameter MMD: {mmd_diam}")
            except Exception as e:
                print(f"ERROR in MMD_diam: {str(e)}")
                traceback.print_exc()
                mmd_diam = "ERROR"
        else:
            mmd_diam = "_"
            print("Skipping diameter calculation")

        print('\nFinal results:')
        print('degree', mmd_degree, 'clustering', mmd_clustering, 'orbits', mmd_4orbits, 
              "Spec:", mmd_spectral, "diameter:", mmd_diam)
              
        return (' degree: ' + str(mmd_degree) + ' clustering: ' + str(mmd_clustering) + ' orbits: ' + str(
            mmd_4orbits) + " Spec: " + str(mmd_spectral) + " diameter: " + str(mmd_diam))
            
    except Exception as e:
        print(f"CRITICAL ERROR in mmd_eval: {str(e)}")
        traceback.print_exc()
        return f"Error: {str(e)}"

def load_graphs(graph_pkl):
    import pickle5 as cp
    graphs = []
    with open(graph_pkl, 'rb') as f:
        while True:
            try:
                g = cp.load(f)
            except:
                break
            graphs.append(g)

    return graphs


# if os.path.exists(fname):
#     with open(fname, 'rb') as fid:
#
#             roidb = pickle.loads(fid)
#             print(roidb)
#             print("roidb")


# load a list of graphs
def load_graph_list(fname, remove_self=True, limited_to=1000):
    if fname[-3:] == "pkl":
        glist = load_graphs(fname)
    else:
        with open(fname, "rb") as f:
            glist = np.load(f, allow_pickle=True)
    # np.save(fname+'Lobster_adj.npy', glist, allow_pickle=True)
    graph_list = []
    for G in glist[:limited_to]:
        if type(G) == np.ndarray:
            graph = nx.from_numpy_matrix(G)
        elif type(G) == nx.classes.graph.Graph:
            graph = G
        else:
            graph = nx.Graph()
            if len(G[0]) > 0:
                graph.add_nodes_from(G[0])
                graph.add_edges_from(G[1])
            else:
                continue

        if remove_self:
            graph.remove_edges_from(nx.selfloop_edges(graph))
        graph.remove_nodes_from(list(nx.isolates(graph)))
        Gcc = sorted(nx.connected_components(graph), key=len, reverse=True)
        graph = graph.subgraph(Gcc[0])
        graph = nx.Graph(graph)
        graph_list.append(graph)
    return graph_list


def spectral_worker(G):
    # eigs = nx.laplacian_spectrum(G)
    eigs = eigvalsh(nx.normalized_laplacian_matrix(G).todense())
    spectral_pmf, _ = np.histogram(eigs, bins=200, range=(-1e-5, 2), density=False)
    spectral_pmf = spectral_pmf / spectral_pmf.sum()
    # from scipy import stats
    # kernel = stats.gaussian_kde(eigs)
    # positions = np.arange(0.0, 2.0, 0.1)
    # spectral_density = kernel(positions)

    # import pdb; pdb.set_trace()
    return spectral_pmf


def spectral_stats(graph_ref_list, graph_pred_list, is_parallel=False):
    ''' Compute the distance between the degree distributions of two unordered sets of graphs.
      Args:
        graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
      '''
    sample_ref = []
    sample_pred = []
    # in case an empty graph is generated
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(spectral_worker, graph_ref_list):
                sample_ref.append(spectral_density)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(spectral_worker, graph_pred_list_remove_empty):
                sample_pred.append(spectral_density)

        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #   for spectral_density in executor.map(spectral_worker, graph_ref_list):
        #     sample_ref.append(spectral_density)
        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #   for spectral_density in executor.map(spectral_worker, graph_pred_list_remove_empty):
        #     sample_pred.append(spectral_density)
    else:
        for i in range(len(graph_ref_list)):
            spectral_temp = spectral_worker(graph_ref_list[i])
            sample_ref.append(spectral_temp)
        for i in range(len(graph_pred_list_remove_empty)):
            spectral_temp = spectral_worker(graph_pred_list_remove_empty[i])
            sample_pred.append(spectral_temp)
    # print(len(sample_ref), len(sample_pred))

    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd)
    # mmd.compute_mmd(sample_ref, sample_pred, kernel=mmd.gaussian_emd,
    #                 sigma=1.0 / 10, distance_scaling=bins)
    mmd_dist = mmd.compute_mmd(sample_ref, sample_pred, kernel=mmd.gaussian_tv)
    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing degree mmd: ', elapsed)
    return mmd_dist


def evl_all_in_dir(dir, refrence_file, generated_file):
    # load all th sub dir
    import glob
    sub_dirs = glob.glob(dir + '*', recursive=True)
    print(sub_dirs)
    report = []
    for subdir in sub_dirs:
        try:
            refrence_graphs = load_graph_list(subdir + "/" + refrence_file)
            generated_graphs = load_graph_list(subdir + "/" + generated_file)
            generated_graphs = generated_graphs[:len(refrence_graphs)]

            Stats = mmd_eval(generated_graphs, refrence_graphs, True)
            report.append([subdir, Stats])
        except:
            report.append([subdir, "Error"])
    # statistics_based_MMD = [ [ row] for row in report]
    import csv
    # save the perturbed graph comparion with the ground truth test set in terms of Statistics-based MMD
    with open(dir + '_Stat_based_MMD.csv', 'w') as f:

        # using csv.writer method from CSV package
        write = csv.writer(f)

        write.writerows(report)
    # save the csv file in the dir

def to_nx(G):
    graph = nx.from_numpy_array(G)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    graph.remove_nodes_from(list(nx.isolates(graph)))
    Gcc = sorted(nx.connected_components(graph), key=len, reverse=True)
    graph = graph.subgraph(Gcc[0])
    graph = nx.Graph(graph)
    return graph

def load_attributedGraph_list(fname,num_graph=None):

    with open(fname, 'rb') as file:
        glist = pickle.load(file)
    if num_graph==None:
        num_graph = len(glist)
    graph_list =[]
    for G,X in glist[:num_graph]:
        try:
            graph = to_nx(G)
            graph_list.append(graph)
        except Exception as e:
            print("cpould not read a graph")
            print(e)
    return graph_list


if __name__ == '__main__':

    dir = "/local-scratch/kiarash/LLGF_ruleLearner/sfu-graphlearning/GeneratedSamples/"

    ref_file_name = "/refGraphs.npy"
    gen_graphs_file_name = "/generatedGraphs.npy"

    pattern = ""

    import glob

    sub_dirs = glob.glob(dir + '*', recursive=True)
    print(sub_dirs)
    report = []
    for path in sub_dirs:
        try:
            if pattern in path:
                # plot_the_graphs = False

                gen_f = path+gen_graphs_file_name
                ref_f = path+ref_file_name

                # ===============================================

                refrence_graphs = load_attributedGraph_list(ref_f)
                generated_graphs = load_attributedGraph_list(gen_f, 11)

                # Visualize = False
                # import plotter


                # if (Visualize):
                #     import plotter
                #
                #     for i, G in enumerate(generated_graphs[:20]):
                #         plotter.plotG(G, "generated", file_name= "graph.png")

                result = mmd_eval(generated_graphs, refrence_graphs, True)
                report.append([path+result])
                with open(dir  + "stats_report.csv", "w") as f:
                    writer = csv.writer(f)
                    writer.writerows(report)

                print("=============================================================================")
        except  Exception as e:
                print(e)
