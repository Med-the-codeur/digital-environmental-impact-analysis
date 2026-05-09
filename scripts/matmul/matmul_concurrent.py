
#Voici une implémentation complète en Python utilisant concurrent.futures.ThreadPoolExecutor.
#Le code est commenté en français et inclut des vérifications de dimensions, une allocation
#sécurisée des résultats et un exemple d'utilisation.


import concurrent.futures
import time

def _mul_row(args):
    """Calcule une ligne complète du produit matriciel A × B."""
    i, A, B, n = args
    p = len(B[0])
    # C[i][j] = sum(A[i][k] * B[k][j] for k in range(n))
    row = [sum(A[i][k] * B[k][j] for k in range(n)) for j in range(p)]
    return i, row

def _transpose_row(args):
    """Calcule une ligne de la matrice transposée (correspond à une colonne de A)."""
    j, A, m = args
    # Ligne j de A^T = Colonne j de A
    col = [A[i][j] for i in range(m)]
    return j, col

def matrix_multiply_mt(A, B, max_workers=None):
    """Produit matriciel multithreadé."""
    if not A or not A[0] or not B or not B[0]:
        raise ValueError("Les matrices ne doivent pas être vides.")
    m, n = len(A), len(A[0])
    n2, p = len(B), len(B[0])
    if n != n2:
        raise ValueError(f"Dimensions incompatibles : A({m}x{n}) × B({n2}x{p})")

    # Pré-allocation du résultat (thread-safe car chaque thread écrit à un index unique)
    C = [[0] * p for _ in range(m)]
    
    # Nombre de threads raisonnable (par défaut : nombre de lignes ou 8)
    workers = max_workers or min(m, 8)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_mul_row, (i, A, B, n)) for i in range(m)]
        for future in concurrent.futures.as_completed(futures):
            idx, row = future.result()
            C[idx] = row
    return C

def matrix_transpose_mt(A, max_workers=None):
    """Transposée multithreadée."""
    if not A or not A[0]:
        raise ValueError("La matrice ne doit pas être vide.")
    m, n = len(A), len(A[0])
    B = [[0] * m for _ in range(n)]
    workers = max_workers or min(n, 8)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_transpose_row, (j, A, m)) for j in range(n)]
        for future in concurrent.futures.as_completed(futures):
            idx, col = future.result()
            B[idx] = col
    return B

# ========================
# EXEMPLE D'UTILISATION
# ========================
if __name__ == "__main__":
    # Matrices de test
    A = [[1, 2, 3],
         [4, 5, 6]]
    B = [[7, 8],
         [9, 10],
         [11, 12]]

    print("Matrice A:")
    for row in A: print(row)
    print("\nMatrice B:")
    for row in B: print(row)

    # Produit
    C = matrix_multiply_mt(A, B)
    print("\nA × B (multithreadé):")
    for row in C: print(row)

    # Transposée
    At = matrix_transpose_mt(A)
    print("\nA^T (multithreadé):")
    for row in At: print(row)

    # Note importante sur les performances
    print("\n Benchmark rapide sur une matrice 500x500...")
    import random
    size = 500
    M = [[random.uniform(0, 1) for _ in range(size)] for _ in range(size)]
    
    start = time.perf_counter()
    matrix_transpose_mt(M, max_workers=8)
    print(f"Transposée multithreadée : {time.perf_counter() - start:.4f} s")

    
