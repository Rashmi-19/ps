# BIM

import math

# ---------- Preprocessing ----------
def preprocess_bim(docs):
    vocab_set = set()
    processed = []
    
    for doc in docs:
        tokens = set(doc.lower().split())  # unique tokens per doc
        processed.append(tokens)
        vocab_set.update(tokens)
    
    vocab = sorted(vocab_set)
    
    # Binary matrix
    binary_matrix = []
    for doc_tokens in processed:
        row = [1 if term in doc_tokens else 0 for term in vocab]
        binary_matrix.append(row)
    
    return vocab, binary_matrix, len(binary_matrix)


# ---------- Phase I ----------
def phase1_estimate(query_terms, vocab, binary_matrix, N_d):
    estimates = {}
    
    for term in query_terms:
        if term not in vocab:
            continue
        
        term_idx = vocab.index(term)
        d_k = sum(1 for doc in binary_matrix if doc[term_idx] == 1)
        
        p_k = 0.5
        q_k_simple = d_k / N_d if N_d > 0 else 0
        q_k = (d_k + 0.5) / (N_d + 1)
        
        estimates[term] = {
            'd_k': d_k,
            'p_k': p_k,
            'q_k_simple': q_k_simple,
            'q_k': q_k
        }
    return estimates


# ---------- Phase II ----------
def phase2_estimate(query_terms, relevant_docs, vocab, binary_matrix, N_d):
    estimates = {}
    N_r = len(relevant_docs)
    
    for term in query_terms:
        if term not in vocab:
            continue
        
        term_idx = vocab.index(term)
        
        r_k = sum(1 for doc_id in relevant_docs if binary_matrix[doc_id][term_idx] == 1)
        d_k = sum(1 for doc in binary_matrix if doc[term_idx] == 1)
        
        p_k_simple = r_k / N_r if N_r > 0 else 0.5
        p_k = (r_k + 0.5) / (N_r + 1)
        
        q_k_simple = (d_k - r_k) / (N_d - N_r) if (N_d - N_r) > 0 else 0.5
        q_k = (d_k - r_k + 0.5) / (N_d - N_r + 1)
        
        estimates[term] = {
            'r_k': r_k,
            'd_k': d_k,
            'N_r': N_r,
            'p_k_simple': p_k_simple,
            'p_k': p_k,
            'q_k_simple': q_k_simple,
            'q_k': q_k
        }
    return estimates


# ---------- RSV ----------
def calculate_rsv(doc_id, query_terms, estimates, vocab, binary_matrix):
    rsv = 0
    for term in query_terms:
        if term not in estimates:
            continue
        
        term_idx = vocab.index(term)
        p_k = estimates[term]['p_k']
        q_k = estimates[term]['q_k']
        
        if binary_matrix[doc_id][term_idx] == 1:
            if p_k > 0 and q_k > 0:
                rsv += math.log(p_k / q_k)
        else:
            if p_k < 1 and q_k < 1:
                rsv += math.log((1 - p_k) / (1 - q_k))
    return rsv


# ---------- Search ----------
def search_phase1(query, vocab, binary_matrix, N_d, top_k=5):
    query_terms = query.lower().split()
    estimates = phase1_estimate(query_terms, vocab, binary_matrix, N_d)
    
    doc_scores = []
    for doc_id in range(N_d):
        rsv = calculate_rsv(doc_id, query_terms, estimates, vocab, binary_matrix)
        doc_scores.append((doc_id, rsv))
    
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    return doc_scores[:top_k]


def search_phase2(query, relevant_docs, vocab, binary_matrix, N_d, top_k=5):
    query_terms = query.lower().split()
    estimates = phase2_estimate(query_terms, relevant_docs, vocab, binary_matrix, N_d)
    
    doc_scores = []
    for doc_id in range(N_d):
        rsv = calculate_rsv(doc_id, query_terms, estimates, vocab, binary_matrix)
        doc_scores.append((doc_id, rsv))
    
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    return doc_scores[:top_k]


# ---------- Printing ----------
def print_estimates(query_terms, vocab, binary_matrix, N_d, relevant_docs=None):
    if relevant_docs is None:
        print("=== PHASE I ESTIMATES ===")
        estimates = phase1_estimate(query_terms, vocab, binary_matrix, N_d)
        for term, est in estimates.items():
            print(f"{term}: d_k={est['d_k']}, p_k={est['p_k']:.3f}, q_k={est['q_k']:.3f}")
    else:
        print("=== PHASE II ESTIMATES ===")
        estimates = phase2_estimate(query_terms, relevant_docs, vocab, binary_matrix, N_d)
        for term, est in estimates.items():
            print(f"{term}: r_k={est['r_k']}, d_k={est['d_k']}, N_r={est['N_r']}")
            print(f"      p_k={est['p_k']:.3f}, q_k={est['q_k']:.3f}")


# ---------- Usage Example ----------
if __name__ == "__main__":
    docs = [
        "information retrieval system",
        "database search query", 
        "information system database",
        "web search engine",
        "query processing system"
    ]
    
    vocab, binary_matrix, N_d = preprocess_bim(docs)
    
    query = "information system"
    query_terms = query.split()
    
    print("=== PHASE I (No Relevance Info) ===")
    print_estimates(query_terms, vocab, binary_matrix, N_d)
    results1 = search_phase1(query, vocab, binary_matrix, N_d)
    print(f"Phase I Results: {results1}")
    
    print("\n=== PHASE II (With Relevance Feedback) ===")
    relevant_docs = [0, 2]  # Assume docs 0 and 2 are relevant
    print_estimates(query_terms, vocab, binary_matrix, N_d, relevant_docs)
    results2 = search_phase2(query, relevant_docs, vocab, binary_matrix, N_d)
    print(f"Phase II Results: {results2}")


# ---------- Quick memory formulas ----------
def bim_formulas():
    print("Phase I (No relevance info):")
    print("  p_k = 0.5")
    print("  q_k = (d_k + 0.5) / (N_d + 1)")
    
    print("\nPhase II (With relevance info):")
    print("  p_k = (r_k + 0.5) / (N_r + 1)")
    print("  q_k = (d_k - r_k + 0.5) / (N_d - N_r + 1)")
    
    print("\nRSV calculation:")
    print("  If term in doc: RSV += log(p_k / q_k)")
    print("  If term not in doc: RSV += log((1-p_k) / (1-q_k))")
