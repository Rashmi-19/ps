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
