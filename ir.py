# MinHash using permutation

import random
import numpy as np

# Example documents
docs = [
    "the cat sat on the mat",
    "the dog sat on the mat",
    "the cat chased the dog"
]

# Step 1: Shingling (k=2 or 3)
def get_shingles(doc, k=2):
    words = doc.split()
    return {" ".join(words[i:i+k]) for i in range(len(words)-k+1)}

k = 2
shingle_set = set()
doc_shingles = []

for d in docs:
    sh = get_shingles(d, k)
    doc_shingles.append(sh)
    shingle_set |= sh

shingles = list(shingle_set)

# Step 2: Build binary shingle–document matrix
matrix = []
for sh in shingles:
    row = [1 if sh in doc_shingles[j] else 0 for j in range(len(docs))]
    matrix.append(row)

sd_matrix = np.array(matrix)

print("Shingle–Document Matrix:")
print(sd_matrix)

# Step 3: MinHash Implementation
num_shingles, num_docs = sd_matrix.shape
num_hashes = 5  # number of permutations
signature = np.full((num_hashes, num_docs), np.inf)

rows = list(range(num_shingles))
permutations = [random.sample(rows, len(rows)) for _ in range(num_hashes)]

for i, perm in enumerate(permutations):
    for col in range(num_docs):
        for row in perm:
            if sd_matrix[row, col] == 1:
                signature[i, col] = row
                break

print("\nSignature Matrix:")
print(signature)

# Step 4: Similarity from signatures
def minhash_sim(col1, col2):
    return np.mean(signature[:, col1] == signature[:, col2])

print("\nSimilarity between Doc1 & Doc2:", minhash_sim(0, 1))
print("Similarity between Doc1 & Doc3:", minhash_sim(0, 2))
print("Similarity between Doc2 & Doc3:", minhash_sim(1, 2))
--------------------------------------------------------------
#MinHash using hash function
import random
import numpy as np

# Example documents
docs = [
    "the cat sat on the mat",
    "the dog sat on the mat",
    "the cat chased the dog"
]

# Step 1: Shingling (k=2 or 3)
def get_shingles(doc, k=2):
    words = doc.split()
    return {" ".join(words[i:i+k]) for i in range(len(words)-k+1)}

k = 2
shingle_set = set()
doc_shingles = []

for d in docs:
    sh = get_shingles(d, k)
    doc_shingles.append(sh)
    shingle_set |= sh

shingles = list(shingle_set)
num_shingles = len(shingles)

# Step 2: Build binary shingle–document matrix
matrix = []
for sh in shingles:
    row = [1 if sh in doc_shingles[j] else 0 for j in range(len(docs))]
    matrix.append(row)

sd_matrix = np.array(matrix)

print("Shingle–Document Matrix:")
print(sd_matrix)

# Step 3: MinHash using hash functions
num_shingles, num_docs = sd_matrix.shape
num_hashes = 5  # number of hash functions
signature = np.full((num_hashes, num_docs), np.inf)

# Create random hash functions of form: h(x) = (a*x + b) % p
p = 2 * num_shingles  # prime > num_shingles
hash_funcs = [(random.randint(1, p-1), random.randint(0, p-1)) for _ in range(num_hashes)]

for i, (a, b) in enumerate(hash_funcs):
    for row in range(num_shingles):
        hash_val = (a * row + b) % p
        for col in range(num_docs):
            if sd_matrix[row, col] == 1:
                if hash_val < signature[i, col]:
                    signature[i, col] = hash_val

print("\nSignature Matrix:")
print(signature.astype(int))

# Step 4: Similarity from signatures
def minhash_sim(col1, col2):
    return np.mean(signature[:, col1] == signature[:, col2])

print("\nSimilarity between Doc1 & Doc2:", minhash_sim(0, 1))
print("Similarity between Doc1 & Doc3:", minhash_sim(0, 2))
print("Similarity between Doc2 & Doc3:", minhash_sim(1, 2))
--------------------------------------------------------------
# Cosine similarity And Euclidean Case

import numpy as np
from numpy.linalg import norm

# Example documents
docs = [
    "this is a cat",
    "this is a dog",
    "cats and dogs are animals",
    "the dog chased the cat"
]

# Step 1: Vocabulary
words = list(set(" ".join(docs).split()))

# Step 2: Term Frequency (TF)
tf = []
for d in docs:
    row = [d.split().count(w) for w in words]
    tf.append(row)
tf = np.array(tf)

# Step 3: Inverse Document Frequency (IDF)
N = len(docs)
idf = np.log(N / (np.count_nonzero(tf, axis=0)))

# Step 4: TF-IDF
tfidf = tf * idf
print("TF-IDF Matrix:\n", tfidf)

# Step 5: Pairwise Cosine Similarity and Euclidean Distance
num_docs = len(docs)

print("\nCosine Similarities:")
for i in range(num_docs):
    for j in range(i+1, num_docs):
        cosine_sim = np.dot(tfidf[i], tfidf[j]) / (norm(tfidf[i]) * norm(tfidf[j]))
        print(f"Doc{i+1} vs Doc{j+1}: {cosine_sim:.4f}")

print("\nEuclidean Distances:")
for i in range(num_docs):
    for j in range(i+1, num_docs):
        euclidean_dist = norm(tfidf[i] - tfidf[j])
        print(f"Doc{i+1} vs Doc{j+1}: {euclidean_dist:.4f}")
--------------------------------------------------------------------------------------
#Page Rank using Power iteration
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


# ====== Generate Graph ======
# Example: Directed graph
G = nx.DiGraph()

# Add edges (you can modify as needed)
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

# ====== Run PageRank ======
pagerank_scores = pagerank(adj_matrix)
print("PageRank Scores:", pagerank_scores)
print("Sum of scores (should be 1):", np.sum(pagerank_scores))
----------------------------------------------------------------------
#Page Rank using Eigen vector
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

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
--------------------------------------------------------------------
import numpy as np

docs = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "the cat played with the dog",
    "dogs and cats are friends"
]


def tokenize(doc):
    return doc.lower().split()

# Build vocabulary
vocab = sorted(set(word for doc in docs for word in tokenize(doc)))
word_index = {w: i for i, w in enumerate(vocab)}

A = np.zeros((len(vocab), len(docs)), dtype=float)

for j, doc in enumerate(docs):
    for word in tokenize(doc):
        A[word_index[word], j] += 1

print("Term–Document Matrix (A):")
print(A)


# A = U Σ V^T
U, s, Vt = np.linalg.svd(A, full_matrices=False)

# Convert singular values into diagonal Σ
Sigma = np.diag(s)

print("\nU (terms -> concepts):\n", U)
print("\nΣ (singular values):\n", Sigma)
print("\nV^T (docs -> concepts):\n", Vt)


k = 2  # latent dimension
U_k = U[:, :k]
Sigma_k = Sigma[:k, :k]
Vt_k = Vt[:k, :]

# Reduced doc vectors in LSI space
doc_vectors = np.dot(Sigma_k, Vt_k).T  # shape: (n_docs, k)
print("\nReduced Document Representations (LSI space):\n", doc_vectors)


query = "cat and dog play together"
q_vec = np.zeros((len(vocab), 1))

for word in tokenize(query):
    if word in word_index:
        q_vec[word_index[word], 0] += 1

# Project query into LSI space: q' = (q^T U_k) Σ_k^-1
q_lsi = np.dot(np.dot(q_vec.T, U_k), np.linalg.inv(Sigma_k))
print("\nQuery Representation (LSI space):\n", q_lsi)

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("\nSimilarity of query with each document:")
for i, doc_vec in enumerate(doc_vectors):
    sim = cosine_sim(q_lsi.flatten(), doc_vec)
    print(f"Doc{i+1}: {sim:.3f}")
 
