# import pickle as pkl
# import numpy as np
# from scipy.sparse import lil_matrix
# import scipy.sparse as sp

# # Load edge matrices
# with open("IMDB/edges.pkl", 'rb') as f:
#     matrices = pkl.load(f)

# # Print basic information about the matrices
# print("Shape of the matrix:", matrices[0].shape)
# print("Number of non-zero elements:", matrices[0].getnnz())

# # Define node type boundaries based on the matrix structure
# total_nodes = matrices[0].shape[0]
# in_1 = matrices[0].indices.min()  # Start of directors
# in_2 = matrices[0].indices.max() + 1  # Start of actors
# in_3 = total_nodes  # End of actors

# # Create node type labels
# node_types = (
#     [0] * in_1 +          # Movies
#     [1] * (in_2 - in_1) + # Directors
#     [2] * (in_3 - in_2)   # Actors
# )
# node_types = np.array(node_types)

# print("\nNode type distribution:")
# print(f"Movies (Type 0): {sum(np.array(node_types) == 0)} nodes (0 to {in_1-1})")
# print(f"Directors (Type 1): {sum(np.array(node_types) == 1)} nodes ({in_1} to {in_2-1})")
# print(f"Actors (Type 2): {sum(np.array(node_types) == 2)} nodes ({in_2} to {in_3-1})")

# def analyze_matrix(matrix):
#     # Get non-zero indices
#     rows, cols = matrix.nonzero()
    
#     # Get unique source and target node types
#     source_types = np.unique(node_types[rows])
#     target_types = np.unique(node_types[cols])
    
#     type_names = {0: "Movie", 1: "Director", 2: "Actor"}
    
#     print(f"\nMatrix Analysis:")
#     print(f"Number of edges: {matrix.sum()}")
#     print(f"Source node types: {[type_names[t] for t in source_types]}")
#     print(f"Target node types: {[type_names[t] for t in target_types]}")
    
#     # Count connections between types
#     for s_type in source_types:
#         for t_type in target_types:
#             s_mask = node_types[rows] == s_type
#             t_mask = node_types[cols] == t_type
#             count = len(rows[s_mask & t_mask])
#             if count > 0:
#                 print(f"{type_names[s_type]} -> {type_names[t_type]}: {count} connections")

# # Analyze the matrix
# print("\nAnalyzing the matrix structure:")
# analyze_matrix(matrices[3])


import pickle as pkl
import numpy as np
from scipy.sparse import lil_matrix
import scipy.sparse as sp

# Load edge matrices
with open("IMDB/edges.pkl", 'rb') as f:
    matrices = pkl.load(f)

# Get all 4 matrices
matrix0, matrix1, matrix2, matrix3 = matrices

def analyze_source_nodes(matrix, matrix_name):
    # Get rows that have any non-zero elements (these are source nodes)
    source_rows = np.where(matrix.sum(axis=1) > 0)[0]
    
    print(f"\n{matrix_name}:")
    print(f"Number of source nodes: {len(source_rows)}")
    print(f"Source node index range: {source_rows.min()} to {source_rows.max()}")
    
    # Print first few source indices
    print(f"First 10 source indices: {source_rows[:10]}")
    
    # Classify source nodes
    movies = source_rows[source_rows < 4661]
    directors = source_rows[(source_rows >= 4661) & (source_rows < 6931)]
    actors = source_rows[source_rows >= 6931]
    
    print("\nSource node types:")
    if len(movies) > 0:
        print(f"Movies: {len(movies)} nodes (first few: {movies[:5]})")
    if len(directors) > 0:
        print(f"Directors: {len(directors)} nodes (first few: {directors[:5]})")
    if len(actors) > 0:
        print(f"Actors: {len(actors)} nodes (first few: {actors[:5]})")

# Analyze each matrix
print("Node type boundaries:")
print("Movies: 0 to 4660")
print("Directors: 4661 to 6930")
print("Actors: 6931 to 12771")

analyze_source_nodes(matrix0, "Matrix 0")
analyze_source_nodes(matrix1, "Matrix 1")
analyze_source_nodes(matrix2, "Matrix 2")
analyze_source_nodes(matrix3, "Matrix 3")

# Additional verification
print("\nVerifying edge counts:")
print(f"Matrix 0 edges: {matrix0.sum()}")
print(f"Matrix 1 edges: {matrix1.sum()}")
print(f"Matrix 2 edges: {matrix2.sum()}")
print(f"Matrix 3 edges: {matrix3.sum()}")