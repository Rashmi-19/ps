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
