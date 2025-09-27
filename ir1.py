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
 
