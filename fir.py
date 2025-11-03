# rocchio_only.py
# Pure Rocchio Classification (from scratch, no other models)

# ========================================
# 1. HARD-CODED TRAINING DATA
# ========================================
training_data = [
    ("love movie great", 1),
    ("best film ever", 1),
    ("awesome acting", 1),
    ("funny exciting", 1),
    ("recommend watch", 1),
    ("waste time", 0),
    ("boring slow", 0),
    ("hate movie", 0),
    ("bad plot", 0),
    ("not worth", 0)
]

test_reviews = [
    "love best movie",
    "boring waste time",
    "funny and great",
    "bad not recommend"
]

# ========================================
# 2. BUILD VOCABULARY
# ========================================
vocab = set()
for text, _ in training_data:
    for word in text.split():
        vocab.add(word)
vocab = sorted(list(vocab))
V = len(vocab)
print(f"Vocabulary: {vocab}\n")

# ========================================
# 3. BINARY VECTOR (1 if word present)
# ========================================
def to_vector(text):
    words = text.split()
    return [1 if word in words else 0 for word in vocab]

# ========================================
# 4. ROCCHIO: Build class prototypes
# ========================================
alpha, beta = 1.0, 0.5
pos_docs = [to_vector(t) for t, l in training_data if l == 1]
neg_docs = [to_vector(t) for t, l in training_data if l == 0]

N_pos, N_neg = len(pos_docs), len(neg_docs)

# Average positive vector
pos_avg = [0.0] * V
for vec in pos_docs:
    for i in range(V):
        pos_avg[i] += vec[i]
pos_avg = [alpha * (x / N_pos) for x in pos_avg]

# Average negative vector
neg_avg = [0.0] * V
for vec in neg_docs:
    for i in range(V):
        neg_avg[i] += vec[i]
neg_avg = [beta * (x / N_neg) for x in neg_avg]

# Final prototypes
proto_pos = [pos_avg[i] - neg_avg[i] for i in range(V)]
proto_neg = [neg_avg[i] - pos_avg[i] for i in range(V)]

# ========================================
# 5. COSINE SIMILARITY
# ========================================
def dot(a, b):
    return sum(a[i] * b[i] for i in range(len(a)))

def norm(v):
    return (sum(x*x for x in v)) ** 0.5

def cosine(a, b):
    na, nb = norm(a), norm(b)
    return 0.0 if na == 0 or nb == 0 else dot(a, b) / (na * nb)

# ========================================
# 6. CLASSIFY
# ========================================
def rocchio_classify(text):
    vec = to_vector(text)
    return 1 if cosine(vec, proto_pos) > cosine(vec, proto_neg) else 0

# ========================================
# 7. RUN
# ========================================
print("=== ROCCHIO CLASSIFICATION ===")
for r in test_reviews:
    pred = rocchio_classify(r)
    print(f'"{r}" → {"POSITIVE" if pred == 1 else "NEGATIVE"}')

#Bernoulli
training_data = [
    (('Chinese Beijing Chinese'),1),
    (('Chinese Chinese Shanghai'),1),
    (('Chinese Macau'),1),
    (('Tokyo Japan Chinese'),0)
]

test_reviews = [
    'Chinese Chinese Chinese Tokyo Japan'
]


vocab=set()
for review,_ in training_data:
  for word in review.split():
    vocab.add(word)
vocab=list(vocab)
print('Vocabulary:',vocab)

def to_vector(text):
  words=text.split(' ')
  return [1 if word in words else 0 for word in vocab]

pos_docs = [to_vector(t) for t, l in training_data if l == 1]
neg_docs = [to_vector(t) for t,l in training_data if l==0]
N_pos, N_neg = len(pos_docs), len(neg_docs)
N_total = N_pos + N_neg

prior_pos = N_pos / N_total
prior_neg = N_neg / N_total
#P( voc|1)
pos_con={}
for i in range(len(vocab)):
  c=0
  for j in pos_docs:
    if j[i] ==1:
      c+=1
  pos_con[vocab[i]]=(c+1)/(N_pos+2)

neg_con={}
for i in range(len(vocab)):
  c=0
  for j in neg_docs:
    if j[i] ==1:
      c+=1
  neg_con[vocab[i]]=(c+1)/(N_neg+2)

query=to_vector(test_reviews[0])
for i,j in zip(query,vocab):
  positive_class=prior_pos
  negative_class=prior_neg
  
  if i==1:
      positive_class*=pos_con[j]
      negative_class*=neg_con[j]
  else:
      positive_class*=(1-pos_con[j])
      negative_class*=(1-neg_con[j])
print( f'{test_reviews} belong to 1' if positive_class>negative_class else f'{test_reviews} belongs to 0')

#Multinomial NB
training_data = [
    (('Chinese Beijing Chinese'),1),
    (('Chinese Chinese Shanghai'),1),
    (('Chinese Macau'),1),
    (('Tokyo Japan Chinese'),0)
]

test_reviews = [
    'Chinese Chinese Chinese Tokyo Japan'
]


vocab=set()
for review,_ in training_data:
  for word in review.split():
    vocab.add(word)
vocab=list(vocab)
print('Vocabulary:',vocab)

pos_words=[]
neg_words=[]
total_doc=len(training_data)
pos_count=0
neg_count=0

for words, label in training_data:
    word=words.split()
    if label==1:
      pos_words.extend(word)
      pos_count+=1
    else:
      neg_words.extend(word)
      neg_count+=1

prior_pos=pos_count/total_doc
prior_neg=neg_count/total_doc

print(f"Positive docs: {pos_count}, Negative docs: {neg_count}")
print(f"P(Positive) = {prior_pos:.3f}, P(Negative) = {prior_neg:.3f}\n")

total_pos_words = len(pos_words)
total_neg_words = len(neg_words)

alpha = 1
vocab_size = len(vocab)

#P(words| c)
liklihood={}
for uni_word in vocab:
  c=0
  for word in pos_words:
    if word==uni_word:
      c+=1
  liklihood[uni_word]=(c+alpha)/(total_pos_words+vocab_size)

#P(words| not c)
liklihood_neg={}
for uni_word in vocab:
  c=0
  for word in neg_words:
    if word==uni_word:
      c+=1
  liklihood_neg[uni_word]=(c+alpha)/(total_neg_words+vocab_size)
print(liklihood_neg)

# for class 1
ans={'1':[],'0':[]}
for words in test_reviews:
    res=prior_pos
    for word in words.split():
      res*=liklihood[word]
    ans['1'].append(res)

for words in test_reviews:
    res=prior_neg
    for word in words.split():
      res*=liklihood_neg[word]
    ans['0'].append(res)

for i,j in zip(ans['0'],ans['1']):
  for k in range(len(ans['0'])):
    if ans['0'][k]> ans['1'][k]:
      print(test_reviews[k],' belongs to', '1')
    else:
      print(test_reviews[k],' belongs to', '1')

data = [
    [1.0, 1.0], [1.5, 2.0],
    [2.0, 1.5], [2.5, 2.5],
    [3.0, 3.0], [3.5, 3.5],
    [8.0, 8.0], [8.5, 9.0],
    [9.0, 8.5], [9.5, 9.0]
]

K = 2
max_iters = 100
tolerance = 0.001

def distance(c1,c2):
  return ((c2[0]-c1[0])**2+(c2[1]-c2[1])**2)**0.5

centroids=[data[i] for i in range(K)]
print('Initial Centroids:',centroids)

for i in range(max_iters):
  clusters=[[] for i in range(K)]
  # assign points closest to the cluster
  for points in data:
    distances=[ distance(points,c) for c in centroids]
    index=distances.index(min(distances))
    clusters[index].append(points)

  old_centroids=[c[:] for c in centroids]

  for j in range(K):
    if clusters[j]:
      #print(j,'if')
      x=sum(k[0] for k in clusters[j])
      y=sum(k[1] for k in clusters[j])
      centroids[j]= [x/len(clusters[j]),y/len(clusters[j])]
    else:
      #print(j,'else')
      centroids[j]=old_centroids[j]

  moved= sum(distance(centroids[k],old_centroids[k]) for k in range(K))
  print(f'Iteration: {i+1} centroids:{centroids} moved:{moved}')
  if moved<=tolerance:
    print(f"Converged after {i+1} iterations.")
    break
for i in range(K):
  print('Cluster ',i)
  for j in clusters[i]:
    print(j)

# knn_classification_scratch.py
# k-Nearest Neighbors Classifier (from scratch, no libraries)

# ========================================
# 1. HARD-CODED TRAINING DATA
# Each sample: [feature1, feature2], label (0 or 1)
# Example: movie ratings → predict if user will like it
# ========================================
training_data = [
    ([4.5, 5.0], 1),  # high action, high comedy → likes
    ([4.0, 4.8], 1),
    ([3.8, 4.5], 1),
    ([2.0, 1.5], 0),  # low action, low comedy → dislikes
    ([1.8, 2.0], 0),
    ([2.2, 1.7], 0),
    ([4.7, 3.0], 1),
    ([1.5, 4.0], 0),
    ([4.2, 4.0], 1),
    ([2.5, 2.0], 0)
]

# ========================================
# 2. TEST SAMPLES
# ========================================
test_samples = [
    [4.3, 4.7],  # should be 1 (like)
    [2.0, 1.8],  # should be 0 (dislike)
    [3.0, 3.5],
    [4.8, 2.5]
]

# ========================================
# 3. EUCLIDEAN DISTANCE (from scratch)
# ========================================
def distance(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) ** 0.5

# ========================================
# 4. k-NN CLASSIFIER FUNCTION
# k = number of neighbors
# ========================================
def knn_classify(sample, k=3):
    # Step 1: Compute distance to all training points
    distances = []
    for point, label in training_data:
        dist = distance(sample, point)
        distances.append((dist, label))

    # Step 2: Sort by distance
    distances.sort(key=lambda x: x[0])

    # Step 3: Take top k neighbors
    neighbors = distances[:k]

    # Step 4: Majority vote
    vote_pos = sum(1 for _, label in neighbors if label == 1)
    vote_neg = k - vote_pos

    return 1 if vote_pos > vote_neg else 0

# ========================================
# 5. RUN PREDICTIONS
# ========================================
k_value = 3
print(f"=== k-NN CLASSIFICATION (k={k_value}) ===")
print("Format: [action, comedy] → prediction\n")

for i, sample in enumerate(test_samples):
    pred = knn_classify(sample, k=k_value)
    label = "LIKE" if pred == 1 else "DISLIKE"
    print(f"Sample {i+1}: {sample} → {label}")

import numpy as np

# ===========================
# ⿡ Cosine similarity
# ===========================
def cosine_similarity(v1, v2):
    mask = (~np.isnan(v1)) & (~np.isnan(v2))
    if np.sum(mask) == 0:
        return 0
    v1, v2 = v1[mask], v2[mask]
    num = np.dot(v1, v2)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    return num / denom if denom != 0 else 0


# ===========================
# ⿢ Prediction function (with k)
# ===========================
def predict_rating(matrix, target_user, target_item, method='user', k=None):
    """
    Predict rating for a user-item pair using CF with cosine similarity.
    matrix: 2D numpy array (users × items)
    target_user, target_item: indices (0-based)
    method: 'user' or 'item'
    k: use top-k most similar neighbors (if None, use all)
    """
    matrix = np.array(matrix, dtype=float)
    sims, ratings = [], []

    if method == 'user':
        # Compare with all other users
        for u in range(matrix.shape[0]):
            if u == target_user or np.isnan(matrix[u, target_item]):
                continue
            sim = cosine_similarity(matrix[target_user, :], matrix[u, :])
            if sim > 0:
                sims.append((sim, matrix[u, target_item]))

    elif method == 'item':
        # Compare with all other items
        for i in range(matrix.shape[1]):
            if i == target_item or np.isnan(matrix[target_user, i]):
                continue
            sim = cosine_similarity(matrix[:, target_item], matrix[:, i])
            if sim > 0:
                sims.append((sim, matrix[target_user, i]))

    # Sort by similarity (descending)
    sims.sort(reverse=True, key=lambda x: x[0])

    # Keep only top-k if specified
    if k is not None:
        sims = sims[:k]

    if len(sims) == 0:
        return np.nan

    sims, ratings = zip(*sims)
    return np.dot(sims, ratings) / np.sum(sims)


ratings = np.array([
    [5, 4, np.nan, 2, 1, 4],
    [1, 2, 2, np.nan, np.nan, 4],
    [3, np.nan, 2, 4, 2, 1],
    [2, 5, 4, np.nan, 2, 3],
    [np.nan, 4, 3, 5, 2, 1],
    [1, 2, 1, 2, np.nan, 3]
], dtype=float)

# Predict rating for user 2 (u3) on item 2 (i3)
pred_user_k3 = predict_rating(ratings, 5, 4, method='user', k=3)
pred_user_all = predict_rating(ratings, 2, 2, method='user', k=None)

print(f"Top-3 User-based CF prediction for u3 on i3: {pred_user_k3:.2f}")
print(f"All-neighbor User-based CF prediction for u3 on i3: {pred_user_all:.2f}")

import numpy as np

# ==============================
# Step 1: Example Documents
# ==============================
docs = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "the cat played with the dog",
    "dogs and cats are friends"
]

# ==============================
# Step 2: Preprocess & Vocabulary
# ==============================
def tokenize(doc):
    """Convert document to lowercase and split into tokens."""
    return doc.lower().split()

# Build vocabulary (unique words across all documents)
vocab = sorted(set(word for doc in docs for word in tokenize(doc)))
word_index = {w: i for i, w in enumerate(vocab)}

# ==============================
# Step 3: Build Term–Document Matrix
# ==============================
A = np.zeros((len(vocab), len(docs)), dtype=float)

for j, doc in enumerate(docs):
    for word in tokenize(doc):
        A[word_index[word], j] += 1

print("Term–Document Matrix (A):")
print(A)

# ==============================
# Step 4: Singular Value Decomposition (SVD)
# ==============================
# A = U Σ V^T
U, s, Vt = np.linalg.svd(A, full_matrices=False)

# Convert singular values into diagonal matrix Σ
Sigma = np.diag(s)

# ==============================
# Step 5: Display Results
# ==============================
print("\nU (term-to-concept matrix):")
print(U)

print("\nΣ (singular values as diagonal matrix):")
print(Sigma)

print("\nV^T (document-to-concept matrix):")
print(Vt)
