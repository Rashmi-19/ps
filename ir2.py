#Page Rank using Eigen vector
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def pagerank(adj_matrix, alpha=0.85, tol=1e-6, max_iter=100):

    n = adj_matrix.shape[0]

    # Step 1: Construct hyperlink matrix H
    H = np.zeros((n, n))
    for i in range(n):
        out_links = np.sum(adj_matrix[i])
        if out_links > 0:
            H[i] = adj_matrix[i] / out_links

    # Step 2: Fix dangling nodes
    for i in range(n):
        if np.sum(H[i]) == 0:
            H[i] = np.ones(n) / n

    # Step 3: Google matrix
    G = alpha * H + (1 - alpha) * (np.ones((n, n)) / n)

    # Step 4: Power iteration
    rank = np.ones(n) / n
    for _ in range(max_iter):
        new_rank = rank @ G
        if np.linalg.norm(new_rank - rank, 1) < tol:
            break
        rank = new_rank

    return rank

def pagerank_eigen(adj_matrix, alpha=0.85):
    n = adj_matrix.shape[0]

    # Step 1: Construct hyperlink matrix H
    H = np.zeros((n, n))
    for i in range(n):
        out_links = np.sum(adj_matrix[i])
        if out_links > 0:
            H[i] = adj_matrix[i] / out_links

    # Step 2: Fix dangling nodes
    for i in range(n):
        if np.sum(H[i]) == 0:
            H[i] = np.ones(n) / n

    # Step 3: Google matrix
    G = alpha * H + (1 - alpha) * (np.ones((n, n)) / n)

    # Step 4: Eigen decomposition
    eigvals, eigvecs = np.linalg.eig(G.T)

    # Find the index of eigenvalue closest to 1
    idx = np.argmin(np.abs(eigvals - 1))

    # Corresponding eigenvector
    principal_eigvec = np.real(eigvecs[:, idx])

    # Normalize to sum to 1
    pagerank = principal_eigvec / np.sum(principal_eigvec)

    return pagerank


# ====== Generate Graph ======
G = nx.DiGraph()
edges = [
    (1, 2), (2, 5), (3, 1), (3, 2), (3, 4), (3, 5),
    (4, 5), (5, 4)
]
G.add_edges_from(edges)

# Draw graph
plt.figure(figsize=(6, 6))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_size=1000, node_color="skyblue", font_size=12, arrows=True)
plt.title("Directed Graph")
plt.show()

# ====== Create adjacency matrix ======
adj_matrix = nx.to_numpy_array(G, nodelist=sorted(G.nodes()), dtype=int)
print("Adjacency Matrix:\n", adj_matrix)

# ====== Run PageRank (Eigenvalue Method) ======
pagerank_scores = pagerank_eigen(adj_matrix)
print("PageRank Scores:", pagerank_scores)
print("Sum of scores (should be 1):", np.sum(pagerank_scores))
