# Vector Search
This hands-on tutorial walks you through the process of implementing different search algorithms for retrieval contents.

## Exercise-1: 
Retrieve the top-k most similar images based on the embedding vectors.

## Goal:
- Implement a simple knn search algorithm to retrieve the top-k most similar images based on the embedding vectors.

## Exercise-2: Maximal Marginal Relevance (MMR)
Marginal relevance means that a document is both relevant to the query and contains minimal similarity to previously selected documents. [ref](https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf). MMR is particularly useful in scenarios where presenting diverse results is important, such as in search engines, recommendation systems, or document summarization tasks.

## Goal:
Implement a simple MMR algorithm for retrieving images? So that the top 3 images do not include duplicate images.

## Exercise-3: Fast and scalable algorithms
### Spatial partitioning algorithms: Example: KDTree
KDTree is a classic algorithm for KNN search. It is a binary tree in which each node represents a hyperplane that partitions the space into two half-spaces. There's an in-memory version of KDTree in scipy.spatial.KDTree. [Example usage](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.query.html)  

## Goal:
Accelerate the image retrieval by a KDTree

### Approximate nearest neighbor search: Example: faiss
faiss is a library for efficient similarity search and clustering of dense vectors. It contains algorithms that search in sets of vectors of any size, up to trillions of vectors. It is developed by Facebook AI Research (FAIR).

## Goal:
Use faiss for vector search.

Further reading: different types of database indices. [Link](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes#summary-of-methods)

