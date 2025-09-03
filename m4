# -*- coding: utf-8 -*-
"""
Boolean Term-Document Model (Function-based)
"""

# ===============================
# BUILD TERM-DOCUMENT MATRIX
# ===============================

def build_matrix(documents):
    """Build term-document matrix and vocabulary"""
    all_terms = set()
    processed_docs = []

    for doc in documents:
        tokens = doc.lower().split()
        processed_docs.append(tokens)
        all_terms.update(tokens)

    vocab = sorted(all_terms)

    # Build matrix: rows = terms, cols = documents
    term_doc_matrix = []
    for term in vocab:
        row = []
        for doc_tokens in processed_docs:
            row.append(1 if term in doc_tokens else 0)
        term_doc_matrix.append(row)

    return vocab, term_doc_matrix


# ===============================
# VECTOR HELPERS
# ===============================

def get_term_vector(term, vocab, term_doc_matrix, documents):
    """Get document vector for a term"""
    term = term.lower()
    if term not in vocab:
        return [0] * len(documents)

    term_idx = vocab.index(term)
    return term_doc_matrix[term_idx]


def boolean_and(term1, term2, vocab, term_doc_matrix, documents):
    """Boolean AND"""
    vec1 = get_term_vector(term1, vocab, term_doc_matrix, documents)
    vec2 = get_term_vector(term2, vocab, term_doc_matrix, documents)
    return [a & b for a, b in zip(vec1, vec2)]


def boolean_or(term1, term2, vocab, term_doc_matrix, documents):
    """Boolean OR"""
    vec1 = get_term_vector(term1, vocab, term_doc_matrix, documents)
    vec2 = get_term_vector(term2, vocab, term_doc_matrix, documents)
    return [a | b for a, b in zip(vec1, vec2)]


def boolean_not(term, vocab, term_doc_matrix, documents):
    """Boolean NOT"""
    vec = get_term_vector(term, vocab, term_doc_matrix, documents)
    return [1 - x for x in vec]


# ===============================
# SEARCH
# ===============================

def search(query, vocab, term_doc_matrix, documents):
    """Search with boolean operators"""
    query = query.lower().strip()

    # Single term
    if ' ' not in query:
        result_vector = get_term_vector(query, vocab, term_doc_matrix, documents)

    # AND
    elif ' and ' in query:
        terms = [t.strip() for t in query.split(' and ')]
        result_vector = get_term_vector(terms[0], vocab, term_doc_matrix, documents)
        for term in terms[1:]:
            term_vec = get_term_vector(term, vocab, term_doc_matrix, documents)
            result_vector = [a & b for a, b in zip(result_vector, term_vec)]

    # OR
    elif ' or ' in query:
        terms = [t.strip() for t in query.split(' or ')]
        result_vector = get_term_vector(terms[0], vocab, term_doc_matrix, documents)
        for term in terms[1:]:
            term_vec = get_term_vector(term, vocab, term_doc_matrix, documents)
            result_vector = [a | b for a, b in zip(result_vector, term_vec)]

    # NOT
    elif ' not ' in query:
        parts = query.split(' not ')
        pos_term = parts[0].strip()
        neg_term = parts[1].strip()

        pos_vec = get_term_vector(pos_term, vocab, term_doc_matrix, documents)
        neg_vec = get_term_vector(neg_term, vocab, term_doc_matrix, documents)
        neg_vec = [1 - x for x in neg_vec]
        result_vector = [a & b for a, b in zip(pos_vec, neg_vec)]

    else:
        result_vector = [0] * len(documents)

    # Return document IDs where result is 1
    return [i for i, val in enumerate(result_vector) if val == 1]


# ===============================
# PRINT MATRIX
# ===============================

def print_matrix(vocab, term_doc_matrix, documents):
    """Print term-document matrix"""
    print("Term-Document Matrix:")
    print("Terms\\Docs", end="")
    for i in range(len(documents)):
        print(f"\tD{i}", end="")
    print()

    for i, term in enumerate(vocab):
        print(f"{term:<10}", end="")
        for val in term_doc_matrix[i]:
            print(f"\t{val}", end="")
        print()


# ===============================
# USAGE EXAMPLE
# ===============================
if __name__ == "__main__":
    docs = [
        "information retrieval system",
        "database search query",
        "information system database",
        "web search engine",
        "query processing system"
    ]

    vocab, term_doc_matrix = build_matrix(docs)

    print_matrix(vocab, term_doc_matrix, docs)

    print("\nSearch Results:")
    print("'information':", search("information", vocab, term_doc_matrix, docs))
    print("'information and system':", search("information and system", vocab, term_doc_matrix, docs))
    print("'search or query':", search("search or query", vocab, term_doc_matrix, docs))
    print("'system not database':", search("system not database", vocab, term_doc_matrix, docs))
