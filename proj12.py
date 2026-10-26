# ============================================================
# PROJECT 12: MATRIX EIGENVALUES AND GOOGLE'S PAGERANK
# PART A: WikiVote Network
# PART B: A-B-C-D Example
# ============================================================

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.linalg import eig


# ============================================================
# PART A - WIKIVOTE NETWORK
# ============================================================
FILE_NAME = "wiki-Vote (1).mat"
N_SMALL = 500
ALPHA = 0.15

print("=" * 60)
print("PART A - WIKIVOTE PAGERANK")
print("=" * 60)


# -------------------- TASK 1 --------------------

data = scipy.io.loadmat(FILE_NAME)
AdjMatrix = csr_matrix(data["Problem"]["A"][0, 0])

m, n = AdjMatrix.shape
total = m * n
nonzero = AdjMatrix.nnz
sparsity = (nonzero / total) * 100

print("\nTASK 1: ADJACENCY MATRIX")
print("Dimensions:", m, "x", n)
print("Total elements:", total)
print("Non-zero elements:", nonzero)
print("Non-zero percentage:", round(sparsity, 4), "%")


# -------------------- TASK 2 --------------------

print("\nTASK 2: MATRIX DIMENSIONS")
print("Rows:", m)
print("Columns:", n)


# -------------------- TASK 3 --------------------

print("\nTASK 3: FIRST 500 NODES")

A500 = AdjMatrix[:N_SMALL, :N_SMALL].toarray().astype(float)

print("Small matrix size:", A500.shape)

G = nx.from_numpy_array(
    A500,
    create_using=nx.DiGraph
)

np.random.seed(42)

pos = {
    i: np.random.rand(2)
    for i in range(N_SMALL)
}

plt.figure(figsize=(8, 6))
nx.draw(
    G,
    pos=pos,
    node_size=8,
    arrows=False,
    with_labels=False
)
plt.title("WikiVote Network - First 500 Nodes")
plt.tight_layout()
plt.savefig("task3_network.png")
plt.show()


# -------------------- TASK 4 --------------------

print("\nTASK 4: GOOGLE MATRIX")

N = N_SMALL

# Number of outgoing links
outgoing = A500.sum(axis=1)

# Transition matrix
S = np.zeros((N, N))

for i in range(N):
    if outgoing[i] > 0:
        S[i, :] = A500[i, :] / outgoing[i]
    else:
        S[i, :] = 1 / N

# Teleportation matrix
E = np.ones((N, N)) / N

# Google matrix
GoogleMatrix = (
    (1 - ALPHA) * S
    + ALPHA * E
)

print("Alpha:", ALPHA)
print("Google Matrix created.")


# -------------------- TASK 5 --------------------

print("\nTASK 5: POWER ITERATION")

rank = np.ones(N) / N
errors = []

for i in range(30):
    new_rank = rank @ GoogleMatrix
    errors.append(
        np.linalg.norm(new_rank - rank, 1)
    )
    rank = new_rank

print("PageRank sum:", rank.sum())
print("Difference after 30 iterations:", errors[-1])

plt.figure(figsize=(7, 5))
plt.plot(range(1, 31), errors, marker="o")
plt.xlabel("Iteration")
plt.ylabel("Difference")
plt.title("PageRank Convergence")
plt.grid(True)
plt.tight_layout()
plt.savefig("task5_convergence.png")
plt.show()


# -------------------- TASK 6 --------------------

print("\nTASK 6: EIGENVALUES")

# PageRank is the left eigenvector
# Therefore calculate eigenvectors of G transpose

values, vectors = eig(GoogleMatrix.T)

index = np.argmin(np.abs(values - 1))

eigenvalue = values[index].real
pagerank = vectors[:, index].real

if pagerank.sum() < 0:
    pagerank = -pagerank

pagerank = np.abs(pagerank)
pagerank = pagerank / pagerank.sum()

print("Eigenvalue closest to 1:", eigenvalue)
print("PageRank sum:", pagerank.sum())


# -------------------- TASK 7 --------------------

print("\nTASK 7: HIGHEST PAGERANK")

max_node = np.argmax(pagerank)

print("Highest PageRank node:", max_node)
print("Highest PageRank:", pagerank[max_node])

top10 = np.argsort(pagerank)[::-1][:10]

print("\nTop 10 nodes:")

for position, node in enumerate(top10, 1):
    print(
        position,
        "Node:", node,
        "PageRank:", round(pagerank[node], 8)
    )


# -------------------- TASK 8 --------------------

print("\nTASK 8: INCOMING LINKS")

incoming = np.asarray(
    AdjMatrix[:N, :N].sum(axis=0)
).ravel()

most_links = np.argmax(incoming)

print("Node with highest PageRank:", max_node)
print("Node with most incoming links:", most_links)
print("Incoming links:", int(incoming[most_links]))

plt.figure(figsize=(7, 5))
plt.scatter(incoming, pagerank, s=10)
plt.xlabel("Incoming Links")
plt.ylabel("PageRank")
plt.title("PageRank vs Incoming Links")
plt.grid(True)
plt.tight_layout()
plt.savefig("task8_pagerank_vs_links.png")
plt.show()


# ============================================================
# PART B - A-B-C-D MINI PROJECT
# ============================================================

print("\n" + "=" * 60)
print("PART B - A-B-C-D PAGERANK")
print("=" * 60)


# -------------------- STEP 1 --------------------

pages = ["A", "B", "C", "D"]

links = {
    "A": ["B", "C"],
    "B": ["C"],
    "C": ["A"],
    "D": ["C"]
}

n = len(pages)


# -------------------- STEP 2 --------------------

A = np.zeros((n, n))

# Rows = destination
# Columns = source

for source, destinations in links.items():

    source_index = pages.index(source)

    for destination in destinations:

        destination_index = pages.index(destination)

        A[destination_index, source_index] = 1

print("\n1. ADJACENCY MATRIX")
print(A)


# -------------------- STEP 3 --------------------

M = np.zeros((n, n))

outgoing = A.sum(axis=0)

for j in range(n):

    if outgoing[j] > 0:
        M[:, j] = A[:, j] / outgoing[j]
    else:
        M[:, j] = 1 / n

print("\n2. TRANSITION MATRIX")
print(M)

print("Column sums:", M.sum(axis=0))


# -------------------- STEP 4 --------------------

d = 0.85

G = (
    d * M
    + (1 - d) * np.ones((n, n)) / n
)

print("\n3. GOOGLE MATRIX")
print(G)


# -------------------- STEP 5 --------------------

values, vectors = np.linalg.eig(G)

index = np.argmin(np.abs(values - 1))

rank_eigen = vectors[:, index].real

if rank_eigen.sum() < 0:
    rank_eigen = -rank_eigen

rank_eigen = rank_eigen / rank_eigen.sum()

print("\n4. EIGENVALUE")
print(values[index].real)

print("\n5. EIGENVECTOR PAGERANK")

for page, score in zip(pages, rank_eigen):
    print(page, ":", round(score, 4))


# -------------------- STEP 6 --------------------

rank = np.ones(n) / n

for iteration in range(1000):

    new_rank = (
        d * M @ rank
        + (1 - d) / n
    )

    if np.linalg.norm(new_rank - rank, 1) < 1e-12:
        rank = new_rank
        break

    rank = new_rank


print("\n6. ITERATIVE PAGERANK")

for page, score in zip(pages, rank):
    print(page, ":", round(score, 4))

print("Iterations:", iteration + 1)


# -------------------- STEP 7 --------------------

print("\n7. COMPARISON")

for page, e, it in zip(
        pages,
        rank_eigen,
        rank):

    print(
        page,
        "Eigenvector:",
        round(e, 4),
        "Iterative:",
        round(it, 4)
    )

print(
    "\nBoth methods equal:",
    np.allclose(rank_eigen, rank)
)


# -------------------- STEP 8 --------------------

order = np.argsort(rank)[::-1]

print("\n8. FINAL RANKING")

for position, i in enumerate(order, 1):
    print(
        position,
        "Page", pages[i],
        "PageRank =", round(rank[i], 4)
    )


# -------------------- STEP 9 --------------------

plt.figure(figsize=(7, 5))

plt.bar(pages, rank)

plt.xlabel("Webpage")
plt.ylabel("PageRank")
plt.title("A-B-C-D PageRank")

plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig("pagerank_scores.png")

plt.show()


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 60)
print("PROJECT 12 COMPLETED SUCCESSFULLY")
print("=" * 60)