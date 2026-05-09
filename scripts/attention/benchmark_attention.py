"""
Comparison of attention implementations and GES (Greenhouse Gas Emissions) impact
PyTorch implementation vs Custom NumPy implementation
"""

import numpy as np
import time
import csv
import sys
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path

# PyTorch imports
import torch
import torch.nn.functional as F
import math

# ============================================================================
# CONFIGURATION DES DOSSIERS
# ============================================================================

BASE_DIR = Path(__file__).resolve().parents[2]

RESULTS_DATA_DIR = BASE_DIR / "results" / "data"
RESULTS_FIGURES_DIR = BASE_DIR / "results" / "figures"

# Création automatique des dossiers
# RESULTS_DATA_DIR.mkdir(parents=True, exist_ok=True)
# RESULTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# # Ajouter results/data au PYTHONPATH
# sys.path.insert(0, str(RESULTS_DATA_DIR))

# ============================================================================
# IMPORT CUSTOM ATTENTION
# ============================================================================

from attention_custom import scaled_dot_product_attention as custom_attention

# ============================================================================
# CODECARBON
# ============================================================================

from codecarbon import OfflineEmissionsTracker

EMISSIONS_LOG_FILE = "emissions.csv"

# ============================================================================
# PYTORCH IMPLEMENTATION
# ============================================================================

def pytorch_scaled_dot_product(q, k, v, mask=None, country_code="TN"):
    """PyTorch implementation"""

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    q = torch.from_numpy(q).to(device)
    k = torch.from_numpy(k).to(device)
    v = torch.from_numpy(v).to(device)

    d_k = q.size()[-1]

    tracker = OfflineEmissionsTracker(
        country_iso_code=country_code,
        measure_power_secs=1,
        output_dir=str(RESULTS_DATA_DIR),
        output_file=EMISSIONS_LOG_FILE
    )

    tracker.start()

    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)

    if mask is not None:
        mask = torch.from_numpy(mask).to(device)
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)

    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)

    emissions = tracker.stop()

    values_np = values.cpu().numpy()
    attention_np = attention.cpu().numpy()

    return values_np, attention_np, emissions if emissions else 0.0

# ============================================================================
# TEST IMPLEMENTATION
# ============================================================================

def test_implementation(seq_len, d_k, implementation="pytorch", country_code="TN"):
    """Test an implementation and return emissions, time, and results"""

    np.random.seed(42)

    q = np.random.randn(seq_len, d_k).astype(np.float32)
    k = np.random.randn(seq_len, d_k).astype(np.float32)
    v = np.random.randn(seq_len, d_k).astype(np.float32)

    start_time = time.time()

    if implementation == "pytorch":

        values, attention, emissions = pytorch_scaled_dot_product(
            q, k, v, country_code=country_code
        )

    else:

        tracker = OfflineEmissionsTracker(
            country_iso_code=country_code,
            measure_power_secs=1,
            output_dir=str(RESULTS_DATA_DIR),
            output_file=EMISSIONS_LOG_FILE
        )

        tracker.start()

        values, attention = custom_attention(q, k, v)

        emissions = tracker.stop()
        emissions = emissions if emissions else 0.0

    elapsed_time = time.time() - start_time

    return {
        "implementation": implementation,
        "seq_len": seq_len,
        "d_k": d_k,
        "matrix_size": seq_len * d_k,
        "emissions": emissions if emissions else 0.0,
        "time": elapsed_time,
        "timestamp": datetime.now().isoformat()
    }

# ============================================================================
# RUN TESTS
# ============================================================================

def run_comparison_tests(country_code="TN"):
    """Exécute des tests de comparaison"""

    print("=" * 70)
    print("COMPARAISON : PyTorch vs Custom Attention")
    print("=" * 70)
    print(f"Code pays : {country_code}")
    print()

    test_configs = [
        (50, 50),
        (100, 100),
        (200, 200),
        (300, 300),
        (500, 500),
    ]

    results = []

    for seq_len, d_k in test_configs:

        matrix_size = seq_len * d_k

        print(f"Test : seq_len={seq_len}, d_k={d_k}, taille={matrix_size}")

        # ==========================
        # PYTORCH
        # ==========================

        print("  → Implémentation PyTorch...", end=" ", flush=True)

        try:

            result_pt = test_implementation(
                seq_len,
                d_k,
                "pytorch",
                country_code
            )

            results.append(result_pt)

            print(
                f"✓ (émissions: {result_pt['emissions']:.2e} kg CO₂, "
                f"temps: {result_pt['time']:.4f}s)"
            )

        except Exception as e:

            print(f"✗ Erreur: {e}")

        # ==========================
        # CUSTOM
        # ==========================

        print("  → Implémentation Custom...", end=" ", flush=True)

        try:

            result_custom = test_implementation(
                seq_len,
                d_k,
                "custom",
                country_code
            )

            results.append(result_custom)

            print(
                f"✓ (émissions: {result_custom['emissions']:.2e} kg CO₂, "
                f"temps: {result_custom['time']:.4f}s)"
            )

        except Exception as e:

            print(f"✗ Erreur: {e}")

        print()

    return results

# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_results(results, filename="comparison_results.csv"):
    """Sauvegarde les résultats CSV"""

    if not results:
        print("Aucun résultat à sauvegarder")
        return

    try:

        filepath = RESULTS_DATA_DIR / filename

        with open(filepath, 'w', newline='') as f:

            writer = csv.DictWriter(f, fieldnames=results[0].keys())

            writer.writeheader()
            writer.writerows(results)

        print(f"✓ Résultats sauvegardés : {filepath}")

    except Exception as e:

        print(f"✗ Erreur sauvegarde : {e}")

# ============================================================================
# PLOTS
# ============================================================================

def plot_comparisons(results):
    """Génère les graphiques"""

    pytorch_results = [
        r for r in results if r['implementation'] == 'pytorch'
    ]

    custom_results = [
        r for r in results if r['implementation'] == 'custom'
    ]

    if not pytorch_results or not custom_results:
        print("Résultats insuffisants")
        return

    pytorch_results.sort(key=lambda x: x['matrix_size'])
    custom_results.sort(key=lambda x: x['matrix_size'])

    pt_times = [r['time'] for r in pytorch_results]
    pt_sizes = [r['matrix_size'] for r in pytorch_results]
    pt_emissions = [r['emissions'] * 1e6 for r in pytorch_results]
    pt_labels = [f"{r['seq_len']}×{r['d_k']}" for r in pytorch_results]

    custom_times = [r['time'] for r in custom_results]
    custom_sizes = [r['matrix_size'] for r in custom_results]
    custom_emissions = [r['emissions'] * 1e6 for r in custom_results]

    plt.style.use('seaborn-v0_8-whitegrid')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # ==========================
    # TEMPS
    # ==========================

    ax1.plot(
        pt_sizes,
        pt_times,
        marker='o',
        linewidth=2.5,
        markersize=8,
        label='PyTorch'
    )

    ax1.plot(
        custom_sizes,
        custom_times,
        marker='s',
        linewidth=2.5,
        markersize=8,
        label='Custom'
    )

    ax1.set_xlabel("Taille matrice")
    ax1.set_ylabel("Temps (s)")
    ax1.set_title("Temps d'exécution")
    ax1.set_xticks(pt_sizes)
    ax1.set_xticklabels(pt_labels, rotation=45)
    ax1.legend()

    # ==========================
    # EMISSIONS
    # ==========================

    ax2.plot(
        pt_sizes,
        pt_emissions,
        marker='o',
        linewidth=2.5,
        markersize=8,
        label='PyTorch'
    )

    ax2.plot(
        custom_sizes,
        custom_emissions,
        marker='s',
        linewidth=2.5,
        markersize=8,
        label='Custom'
    )

    ax2.set_xlabel("Taille matrice")
    ax2.set_ylabel("Émissions (µg CO₂)")
    ax2.set_title("Émissions GES")
    ax2.set_xticks(pt_sizes)
    ax2.set_xticklabels(pt_labels, rotation=45)
    ax2.legend()

    fig.suptitle(
        "Comparaison Attention : PyTorch vs Custom",
        fontsize=16,
        fontweight='bold'
    )

    plt.tight_layout()

    output_path = RESULTS_FIGURES_DIR / "attention-emissions-comparison.png"

    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    print(f"✓ Graphique sauvegardé : {output_path}")

    plt.show()

# ============================================================================
# SUMMARY
# ============================================================================

def print_summary(results):

    pytorch_results = [
        r for r in results if r['implementation'] == 'pytorch'
    ]

    custom_results = [
        r for r in results if r['implementation'] == 'custom'
    ]

    if not pytorch_results or not custom_results:
        return

    print("\n" + "=" * 70)
    print("RÉSUMÉ")
    print("=" * 70)

    pt_total_emissions = sum(r['emissions'] for r in pytorch_results)
    custom_total_emissions = sum(r['emissions'] for r in custom_results)

    pt_total_time = sum(r['time'] for r in pytorch_results)
    custom_total_time = sum(r['time'] for r in custom_results)

    print("\nPyTorch")
    print(f"  Émissions : {pt_total_emissions:.2e} kg CO₂")
    print(f"  Temps : {pt_total_time:.4f} s")

    print("\nCustom")
    print(f"  Émissions : {custom_total_emissions:.2e} kg CO₂")
    print(f"  Temps : {custom_total_time:.4f} s")

    print("=" * 70)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":

    COUNTRY_CODE = "TN"

    results = run_comparison_tests(country_code=COUNTRY_CODE)

    if results:

        save_results(results)

        print_summary(results)

        print("\nGénération des graphiques...")

        plot_comparisons(results)

    else:

        print("Aucun résultat")