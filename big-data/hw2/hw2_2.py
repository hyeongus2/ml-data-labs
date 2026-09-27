import numpy as np 

def norm(v):
    """
    INPUT: vector v
    OUTPUT: L2 norm of v
    """
    return np.sqrt(np.sum(v ** 2))

def matrix_mult(M):
    """
    INPUT: matrix M
    OUTPUT: M^T M and M M^T
    """
    MTM = np.dot(M.T, M)
    MMT = np.dot(M, M.T)
    return MTM, MMT

def power_iteration(M, delta: float = 1e-6, max_iter=10000):
    """
    INPUT: matrix M and threshold delta, max_iter
    OUTPUT: dominant eigenvector of M
    
    DESCRIPTION: Returns the dominant eigenvector of M using power iteration.
    Does not use np.linalg.norm, instead normalizes manually.
    """
    # Step 1: Initialize a random vector `b_k`.
    np.random.seed(42)  # For reproducibility
    b_k = np.random.rand(M.shape[1])
    if norm(b_k) == 0:
        b_k[0] = 1.0
    b_k = b_k / norm(b_k)

    # Step 2: Iterate until convergence or max_iter is reached by updating `b_k` and normalizing.
    for _ in range(max_iter):
        # b_k1 = M * b_k
        b_k1 = np.dot(M, b_k)
        # Normalize b_k1
        if norm(b_k1) == 0:
            b_k1 = np.zeros_like(b_k1)
        else:
            b_k1 = b_k1 / norm(b_k1)
        # Check for convergence
        if norm(b_k1 - b_k) < delta:
            break
        b_k = b_k1

    # Step 3: Return the dominant eigenvector.
    return b_k

def deflate_matrix(M, eigvec):
    """
    INPUT: matrix M, and dominant eigenvector eigvec
    OUTPUT: deflated matrix M after removing the contribution of eigvec
    
    DESCRIPTION: Deflates the matrix M to find subsequent eigenvectors.
    """
    # Deflate the matrix to remove the contribution of the eigenvector.
    # Step 1: Compute the corresponding eigenvalue. (Rayleigh quotient)
    eigenvalue = np.dot(eigvec, np.dot(M, eigvec))
    # Step 2: Update the matrix.
    v_col = eigvec.reshape(-1, 1) # shape (n, 1)
    v_row = eigvec.reshape(1, -1) # shape (1, n)
    M_deflated = M - eigenvalue * np.dot(v_col, v_row) # shape (n, n)
    return M_deflated

def compute_eigenvalues(M):
    """
    INPUT: matrix M
    OUTPUT: list of eigenvalues using power iteration for each eigenvector
    
    DESCRIPTION: Uses power iteration to find the dominant eigenvalue of M,
    deflates the matrix and finds subsequent eigenvalues.
    """
    # Use power iteration to compute multiple eigenvalues.
    # Deflate the matrix for each iteration to find subsequent eigenvalues.
    n = M.shape[0]
    eigenvalues = []
    M_current = M.copy()

    for _ in range(n):
        eigvec = power_iteration(M_current)
        eigenvalue = np.dot(eigvec, np.dot(M_current, eigvec))
        eigenvalues.append(eigenvalue)
        M_current = deflate_matrix(M_current, eigvec)

    return eigenvalues

def svd_manual(M):
    """
    INPUT: matrix M
    OUTPUT: matrices U, Sigma, V using SVD
    
    DESCRIPTION: Computes SVD by calculating eigenvalues and eigenvectors for M^T M and M M^T.
    """
    # Step 1: Compute M^T M and M M^T.
    MTM, _ = matrix_mult(M)
    
    n = MTM.shape[0]
    M_current = MTM.copy()
    eigenvalues = []
    eigenvectors = []

    # Step 2: Compute the eigenvectors using power iteration.
    for _ in range(n):
        eigvec = power_iteration(M_current)
        eigenvalue = np.dot(eigvec, np.dot(M_current, eigvec))
        eigenvalues.append(eigenvalue)
        eigenvectors.append(eigvec)
        M_current = deflate_matrix(M_current, eigvec)

    eigenpairs = sorted(zip(eigenvalues, eigenvectors), key=lambda x: x[0], reverse=True)
    eigenvalues, eigenvectors = zip(*eigenpairs)
    
    # Step 3: Compute singular values and construct U, Sigma, and V.
    V = np.zeros((n, n))
    for i in range(n):
        V[:, i] = eigenvectors[i]

    singular_values = np.sqrt(np.maximum(eigenvalues, 0))  # Ensure non-negative
    Sigma = np.diag(singular_values)

    inverse_singular_values = [1/s if s > 1e-9 else 0 for s in singular_values]
    Sigma_inv = np.diag(inverse_singular_values)
    U = np.dot(M, np.dot(V, Sigma_inv))
    return U, Sigma, V

def matrix_approximation(U, Sigma, V, k):
    """
    INPUT: matrices U, Sigma, V, and integer k
    OUTPUT: matrix approximation with only the top k singular values
    
    DESCRIPTION: Computes the matrix approximation using the top k singular values.
    """
    # Use the top-k singular values to reconstruct the matrix.
    U_k = U[:, :k]
    Sigma_k = Sigma[:k, :k]
    V_k = V[:, :k]

    # M_k = U_k * Sigma_k * V_k^T
    return np.dot(U_k, np.dot(Sigma_k, V_k.T))

def energy_retained(Sigma, k):
    """
    INPUT: list of singular values, integer k
    OUTPUT: percentage of energy retained by the k-dimensional approximation
    
    DESCRIPTION: Computes the percentage of energy retained by the top k singular values.
    The energy is the sum of the squares of the singular values.
    """
    # Compute total energy and retained energy, return the percentage.
    singular_values_squared = np.diag(Sigma) ** 2

    total_energy = np.sum(singular_values_squared)
    retained_energy = np.sum(singular_values_squared[:k])
    return retained_energy / total_energy if total_energy > 0 else 0.0

def pca_via_svd(M, k):
    """
    INPUT: matrix M, integer k
    OUTPUT: top-k PCA projection of M
    """
    # Zero-center the data
    M_centered = M - np.mean(M, axis=0, keepdims=True)
    
    # Perform SVD on the centered matrix
    U, Sigma, V = svd_manual(M_centered)

    # Project M onto the top-k principal components.
    U_k = U[:, :k]
    Sigma_k = Sigma[:k, :k]

    # M_pca = U_k * Sigma_k
    return np.dot(U_k, Sigma_k)

def distance_correlation(M, M_reduced):
    """
    INPUT: original matrix M and reduced matrix M_reduced
    OUTPUT: distance correlation between pairwise distances in M and M_reduced
    """
    n = M.shape[0]

    # Step 1: Compute pairwise distance matrices.
    D = np.zeros((n, n))
    D_k = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            dist = norm(M[i, :] - M[j, :])
            D[i, j] = dist
            D[j, i] = dist

            dist_k = norm(M_reduced[i, :] - M_reduced[j, :])
            D_k[i, j] = dist_k
            D_k[j, i] = dist_k

    # Step 2: Calculate means.
    D_bar = np.mean(D)
    D_k_bar = np.mean(D_k)

    # Step 3: Center the distance matrices.
    A = D - D_bar
    B = D_k - D_k_bar

    # Step 4: Calculate numerator and denominators.
    numerator = np.sum(A * B)

    denom_A = np.sum(A ** 2)
    denom_B = np.sum(B ** 2)
    denominator = np.sqrt(denom_A * denom_B)

    return numerator / denominator if denominator > 0 else 0.0


if __name__ == "__main__":
    # Given matrix M
    M = np.array([
        [8, 2, 1, 0, 2],
        [2, 3, 0, 6, 0],
        [1, 0, 4, 0, 0],
        [0, 6, 0, 8, 1],
        [2, 0, 0, 1, 7]
    ], dtype=float)

    # ----------------------------------------------
    #           Please do not modify below
    # ----------------------------------------------
    with open("output2.txt", "w") as f:
        # Part (a): Compute M^T M and M M^T
        print("\n(a) M^T M and M M^T:", file=f)
        MTM, MMT = matrix_mult(M)
        print("M^T M:\n", MTM, file=f)
        print("M M^T:\n", MMT, file=f)
        
        # Part (b): Eigenpairs using numpy.linalg.eig()
        print("\n(b) Power iteration vs numpy.linalg.eig():", file=f)
        eigvals_MTM, eigvecs_MTM = np.linalg.eig(MTM)
        eigvals_MMT, eigvecs_MMT = np.linalg.eig(MMT)
        print("Eigenvalues of M^T M:\n", eigvals_MTM, file=f)
        print("Eigenvectors of M^T M:\n", eigvecs_MTM, file=f)

        eigenvalues_manual_MTM = compute_eigenvalues(MTM)
        eigenvalues_manual_MMT = compute_eigenvalues(MMT)
        print("Eigenvalues of M^T M using power iteration:\n", eigenvalues_manual_MTM, file=f)
        print("Eigenvalues of M M^T using power iteration:\n", eigenvalues_manual_MMT, file=f)

        # Part (c): SVD implementation
        print("\n(c) SVD implementation:", file=f)
        U, Sigma, V = svd_manual(M)
        print("U:\n", U, file=f)
        print("Sigma:\n", Sigma, file=f)
        print("V:\n", V, file=f)

        # Part (d): Rank-k approximations
        print("\n(d) Rank-k approximations:", file=f)
        for k in range(1, len(Sigma) + 1):
            M_k = matrix_approximation(U, Sigma, V, k)
            print(f"\nRank-{k} approximation of M:\n", np.round(M_k, 3), file=f)

        # Part (e): Energy retention
        print("\n(e) Energy retention ratios:", file=f)
        for k in range(1, len(Sigma) + 1):
            energy = energy_retained(Sigma, k)
            print(f"  k = {k}: {energy * 100:.2f}% energy retained", file=f)
        
        # Part (f): PCA via SVD
        print("\n(f) PCA via SVD:", file=f)
        for k in range(1, 6):
            M_pca = pca_via_svd(M, k)
            print(f"\nTop-{k} PCA projection of M:\n", np.round(M_pca, 3), file=f)

        # Part (g): Top-k PCA projection vs. Random projection (Distance Correlation)
        print("\n(g) Top-k PCA projection vs. Random projection (Distance Correlation):", file=f)
        np.random.seed(42)  # For reproducibility
        for k in range(1, 6):
            M_pca = pca_via_svd(M, k)
            dist_corr = distance_correlation(M, M_pca)
            M_random = M @ np.random.randn(M.shape[1], k)
            dist_corr_random = distance_correlation(M, M_random)
            print(f"  k = {k}: PCA Dist Corr = {dist_corr:.4f}, Random Dist Corr = {dist_corr_random:.4f}", file=f)
