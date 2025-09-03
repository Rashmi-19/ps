# -*- coding: utf-8 -*-
"""
Boolean Retrieval with Inverted Index (Function-based)
"""

# ---------- Build Index ----------
def build_inverted_index(docs):
    index = {}
    for i, doc in enumerate(docs):
        for term in set(doc.lower().split()):
            if term not in index:
                index[term] = []
            index[term].append(i)
    return index


# ---------- Basic Operations ----------
def get(index, term):
    """Get posting list"""
    return index.get(term.lower(), [])


def AND(list1, list2):
    """Intersect two lists"""
    return [x for x in list1 if x in list2]


def OR(list1, list2):
    """Union of two lists"""
    return sorted(set(list1 + list2))


def NOT(posting_list, docs):
    """Complement of posting list"""
    all_docs = list(range(len(docs)))
    return [x for x in all_docs if x not in posting_list]


# ---------- Optimization ----------
def optimize_terms(index, terms, operation='and'):
    term_lengths = [(term, len(get(index, term))) for term in terms]
    
    if operation == 'and':
        # For AND: shortest lists first
        return [term for term, _ in sorted(term_lengths, key=lambda x: x[1])]
    else:
        # For OR: longest lists first
        return [term for term, _ in sorted(term_lengths, key=lambda x: x[1], reverse=True)]


# ---------- Search ----------
def search(query, index, docs):
    q = query.lower()
    
    if ' and ' in q:
        terms = [t.strip() for t in q.split(' and ')]
        terms = optimize_terms(index, terms, 'and')
        result = get(index, terms[0])
        for term in terms[1:]:
            result = AND(result, get(index, term))
            if not result:  # Early stop
                break
        return result
    
    elif ' or ' in q:
        terms = [t.strip() for t in q.split(' or ')]
        terms = optimize_terms(index, terms, 'or')
        result = get(index, terms[0])
        for term in terms[1:]:
            result = OR(result, get(index, term))
        return result
    
    elif ' not ' in q:
        pos, neg = q.split(' not ')
        pos_list = get(index, pos.strip())
        neg_list = get(index, neg.strip())
        return AND(pos_list, NOT(neg_list, docs))
    
    else:
        return get(index, q)


# ---------- Usage Demo ----------
if __name__ == "__main__":
    docs = ["cat dog bird", "dog bird", "cat mouse", "bird eagle", "mouse cat"]
    index = build_inverted_index(docs)
    print("Index:", index)
    
    print("\nQuery: 'cat and bird and dog'")
    terms = ['cat', 'bird', 'dog']
    print("Posting list sizes:")
    for term in terms:
        print(f"  {term}: {len(get(index, term))} docs")
    
    optimized = optimize_terms(index, terms, 'and')
    print(f"Optimized order: {optimized}")
    print(f"Result: {search('cat and bird and dog', index, docs)}")
    
    print("\nOR optimization:")
    or_optimized = optimize_terms(index, terms, 'or')
    print(f"OR order: {or_optimized}")
    print(f"Result: {search('cat or bird or dog', index, docs)}")
