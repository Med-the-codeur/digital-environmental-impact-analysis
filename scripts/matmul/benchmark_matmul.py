"""
Comparaison : PyTorch vs Calcul Matriciel Multithreadé Concurent
Avec analyse statistique (moyenne, écart-type) et graphe 2D
"""

import numpy as np
import time
import csv
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pathlib import Path

# PyTorch
import torch
import torch.nn.functional as F
import math

# Calcul matriciel multithreadé
from matmul_concurrent import matrix_multiply_mt

# CodeCarbon
from codecarbon import OfflineEmissionsTracker

# Configuration pour la sauvegarde des fichiers
BASE_DIR = Path(__file__).resolve().parents[2]

RESULTS_DATA_DIR = BASE_DIR / "results" / "data"

RESULTS_FIGURES_DIR = BASE_DIR / "results" / "figures"

EMISSIONS_LOG_PATH = str(RESULTS_DATA_DIR / "emissions.csv")


def pytorch_matmul(A_np, B_np, country_code="TN"):
    """Multiplication matricielle avec PyTorch"""
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    
    A = torch.from_numpy(A_np).to(device)
    B = torch.from_numpy(B_np).to(device)
    
    tracker = OfflineEmissionsTracker(country_iso_code=country_code, measure_power_secs=1, log_file_path=EMISSIONS_LOG_PATH)
    
    start_time = time.time()
    tracker.start()
    
    C = torch.matmul(A, B)
    
    elapsed_time = time.time() - start_time
    emissions = tracker.stop()
    
    return C.cpu().numpy(), emissions if emissions else 0.0, elapsed_time


def concurrent_matmul(A_list, B_list, country_code="TN"):
    """Multiplication matricielle multithreadée"""
    tracker = OfflineEmissionsTracker(country_iso_code=country_code, measure_power_secs=1, log_file_path=EMISSIONS_LOG_PATH)
    tracker.start()
    
    start_time = time.time()
    C = matrix_multiply_mt(A_list, B_list, max_workers=8)
    elapsed_time = time.time() - start_time
    
    emissions = tracker.stop()
    
    return np.array(C), emissions if emissions else 0.0, elapsed_time


def numpy_to_list(arr):
    """Convertir numpy array en liste imbriquée"""
    return arr.tolist()


def test_implementation_multiple(seq_len, d_k, implementation="pytorch", country_code="TN", num_runs=15):
    """
    Exécute l'implémentation plusieurs fois avec les MÊMES matrices et retourne statistiques
    
    Returns:
        dict avec moyennes et écarts-types
    """
    np.random.seed(42)
    
    # Générer les matrices UNE SEULE FOIS
    A = np.random.randn(seq_len, d_k).astype(np.float32)
    B = np.random.randn(d_k, seq_len).astype(np.float32)
    
    # Pour concurrent, convertir une seule fois
    if implementation == "concurrent":
        A_list = numpy_to_list(A)
        B_list = numpy_to_list(B)
    
    emissions_list = []
    times_list = []
    
    # Faire 15 exécutions avec les mêmes matrices
    for run in range(num_runs):
        if implementation == "pytorch":
            _, emissions, exec_time = pytorch_matmul(A, B, country_code)
        else:  # concurrent
            _, emissions, exec_time = concurrent_matmul(A_list, B_list, country_code)
        
        emissions_list.append(emissions)
        times_list.append(exec_time)
    
    # Calculs statistiques
    mean_emissions = np.mean(emissions_list)
    mean_time = np.mean(times_list)
    std_time = np.std(times_list)
    std_emissions = np.std(emissions_list)
    
    return {
        "implementation": implementation,
        "seq_len": seq_len,
        "d_k": d_k,
        "matrix_size": seq_len * d_k,
        "mean_emissions": mean_emissions,
        "std_emissions": std_emissions,
        "mean_time": mean_time,
        "std_time": std_time,
        "num_runs": num_runs
    }


def run_comparison_tests(country_code="TN", num_runs=15):
    """Exécute tests de comparaison avec statistiques"""
    print("=" * 70)
    print("COMPARAISON : PyTorch vs Calcul Multithreadé Concurent")
    print("=" * 70)
    print(f"Code pays : {country_code}")
    print(f"Nombre d'exécutions par test : {num_runs}")
    print()
    
    test_configs = [
        (200, 200),
        (400, 400),
        (600, 600),
        (800, 800),
        (1000, 1000),
    ]
    
    results = []
    
    for seq_len, d_k in test_configs:
        matrix_size = seq_len * d_k
        print(f"Test : seq_len={seq_len}, d_k={d_k}, taille_totale={matrix_size}")
        
        # PyTorch
        print(f"  PyTorch ({num_runs} exécutions)...", end=" ", flush=True)
        try:
            result_pt = test_implementation_multiple(seq_len, d_k, "pytorch", country_code, num_runs)
            results.append(result_pt)
            print(f"(émissions: {result_pt['mean_emissions']:.2e} kg CO₂, "
                  f"temps: {result_pt['mean_time']:.4f}±{result_pt['std_time']:.4f}s)")
        except Exception as e:
            print(f" Erreur: {e}")
        
        # Concurrent
        print(f"  Multithreadé ({num_runs} exécutions)...", end=" ", flush=True)
        try:
            result_conc = test_implementation_multiple(seq_len, d_k, "concurrent", country_code, num_runs)
            results.append(result_conc)
            print(f"(émissions: {result_conc['mean_emissions']:.2e} kg CO₂, "
                  f"temps: {result_conc['mean_time']:.4f}±{result_conc['std_time']:.4f}s)")
        except Exception as e:
            print(f" Erreur: {e}")
        
        print()
    
    return results


def save_results(results, filename="comparison_pytorch_concurrent.csv"):
    """Sauvegarde les résultats"""
    if not results:
        print("Aucun résultat à sauvegarder")
        return
    
    filepath = Path(__file__).resolve().parents[2] / "results" / "data" / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Résultats sauvegardés : {filepath}")


def _plot_2d_comparison_legacy(results):
    """
    Graphe 2D professionnel :
    - Sous-graphe 1 : temps d'exécution moyen selon la taille de la matrice
    - Sous-graphe 2 : émissions GES moyennes selon la taille de la matrice
    - Les barres représentent les écarts-types
    """
    
    pytorch_results = [r for r in results if r['implementation'] == 'pytorch']
    concurrent_results = [r for r in results if r['implementation'] == 'concurrent']
    
    if not pytorch_results or not concurrent_results:
        print("Résultats insuffisants pour la comparaison")
        return
    
    pytorch_results.sort(key=lambda x: x['matrix_size'])
    concurrent_results.sort(key=lambda x: x['matrix_size'])
    
    # Extraction des données
    pt_times = [r['mean_time'] for r in pytorch_results]
    pt_std_times = [r['std_time'] for r in pytorch_results]
    pt_sizes = [r['matrix_size'] for r in pytorch_results]
    pt_emissions = [r['mean_emissions'] * 1e6 for r in pytorch_results]
    pt_std_emissions = [r['std_emissions'] * 1e6 for r in pytorch_results]
    
    conc_times = [r['mean_time'] for r in concurrent_results]
    conc_std_times = [r['std_time'] for r in concurrent_results]
    conc_sizes = [r['matrix_size'] for r in concurrent_results]
    conc_emissions = [r['mean_emissions'] * 1e6 for r in concurrent_results]
    conc_std_emissions = [r['std_emissions'] * 1e6 for r in concurrent_results]
    
    # Style professionnel
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Figure 2D
    fig, (ax_time, ax_emissions) = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor('white')
    
    for axis in (ax_time, ax_emissions):
        axis.set_facecolor('#f5f5f5')
        axis.grid(True, alpha=0.4, linestyle='--')
        axis.tick_params(axis='both', labelsize=10)
        axis.set_xlabel('Taille de la Matrice (seq_len x d_k)', fontsize=11, fontweight='bold')
    
    # Temps d'exécution moyen
    ax_time.errorbar(pt_sizes, pt_times, yerr=pt_std_times, fmt='o--', color='#E74C3C',
                     ecolor='#C0392B', elinewidth=2, capsize=5, markersize=9,
                     markeredgewidth=2, markeredgecolor='#C0392B', label='PyTorch')
    ax_time.errorbar(conc_sizes, conc_times, yerr=conc_std_times, fmt='s--', color='#3498DB',
                     ecolor='#2980B9', elinewidth=2, capsize=5, markersize=9,
                     markeredgewidth=2, markeredgecolor='#2980B9', label='Multithreadé')
    ax_time.set_ylabel('Temps d\'exécution moyen (secondes)', fontsize=11, fontweight='bold')
    ax_time.set_title('Temps d\'exécution', fontsize=13, fontweight='bold', pad=12)
    ax_time.legend(fontsize=10, framealpha=0.95, edgecolor='black', fancybox=True)
    
    # Émissions GES moyennes
    ax_emissions.errorbar(pt_sizes, pt_emissions, yerr=pt_std_emissions, fmt='o--', color='#E74C3C',
                          ecolor='#C0392B', elinewidth=2, capsize=5, markersize=9,
                          markeredgewidth=2, markeredgecolor='#C0392B', label='PyTorch')
    ax_emissions.errorbar(conc_sizes, conc_emissions, yerr=conc_std_emissions, fmt='s--', color='#3498DB',
                          ecolor='#2980B9', elinewidth=2, capsize=5, markersize=9,
                          markeredgewidth=2, markeredgecolor='#2980B9', label='Multithreadé')
    ax_emissions.set_ylabel('Émissions GES moyennes (µg CO₂)', fontsize=11, fontweight='bold')
    ax_emissions.set_title('Émissions GES', fontsize=13, fontweight='bold', pad=12)
    ax_emissions.legend(fontsize=10, framealpha=0.95, edgecolor='black', fancybox=True)
    
    fig.suptitle('Analyse 2D : PyTorch vs Calcul Multithreadé Concurent\n'
                 '(15 exécutions par test - points = moyenne, barres = écart-type)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Sauvegarde
    output_path = Path(__file__).resolve().parents[2] / "results" / "figures" / "matrix-multiplication-emissions-comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f" Graphique 2D sauvegardé : {output_path}")
    
    plt.show()


def print_summary(results):
    """Affiche résumé statistique"""
    pytorch_results = [r for r in results if r['implementation'] == 'pytorch']
    concurrent_results = [r for r in results if r['implementation'] == 'concurrent']
    
    if not pytorch_results or not concurrent_results:
        return
    
    print("\n" + "=" * 70)
    print("RÉSUMÉ STATISTIQUE (15 exécutions par test)")
    print("=" * 70)
    
    avg_pt_emissions = np.mean([r['mean_emissions'] for r in pytorch_results])
    avg_conc_emissions = np.mean([r['mean_emissions'] for r in concurrent_results])
    
    avg_pt_time = np.mean([r['mean_time'] for r in pytorch_results])
    avg_conc_time = np.mean([r['mean_time'] for r in concurrent_results])
    
    avg_pt_std = np.mean([r['std_time'] for r in pytorch_results])
    avg_conc_std = np.mean([r['std_time'] for r in concurrent_results])
    
    print(f"\nImplémentation PyTorch :")
    print(f"  Émissions (moyenne) : {avg_pt_emissions:.2e} kg CO₂ ({avg_pt_emissions*1e6:.2f} µg CO₂)")
    print(f"  Temps (moyenne) : {avg_pt_time:.4f} secondes")
    print(f"  Écart-type (temps) : {avg_pt_std:.4f} secondes")
    
    print(f"\nImplémentation Multithreadée :")
    print(f"  Émissions (moyenne) : {avg_conc_emissions:.2e} kg CO₂ ({avg_conc_emissions*1e6:.2f} µg CO₂)")
    print(f"  Temps (moyenne) : {avg_conc_time:.4f} secondes")
    print(f"  Écart-type (temps) : {avg_conc_std:.4f} secondes")
    
    if avg_pt_emissions > 0:
        emission_diff = ((avg_conc_emissions - avg_pt_emissions) / avg_pt_emissions) * 100
        print(f"\nDifférence d'émissions : {emission_diff:+.2f}%")
    
    if avg_pt_time > 0:
        time_diff = ((avg_conc_time - avg_pt_time) / avg_pt_time) * 100
        print(f"Différence de temps : {time_diff:+.2f}%")
    
    print("=" * 70 + "\n")


def plot_2d_comparison(results):
    """
    Graphique de comparaison avec un rendu plus propre pour un rapport.
    """
    pytorch_results = [r for r in results if r['implementation'] == 'pytorch']
    concurrent_results = [r for r in results if r['implementation'] == 'concurrent']

    if not pytorch_results or not concurrent_results:
        print("Resultats insuffisants pour la comparaison")
        return

    pytorch_results.sort(key=lambda x: x['matrix_size'])
    concurrent_results.sort(key=lambda x: x['matrix_size'])

    pt_sizes = [r['seq_len'] for r in pytorch_results]
    pt_times = [r['mean_time'] for r in pytorch_results]
    pt_std_times = [r['std_time'] for r in pytorch_results]
    pt_emissions = [r['mean_emissions'] for r in pytorch_results]
    pt_std_emissions = [r['std_emissions'] for r in pytorch_results]

    conc_sizes = [r['seq_len'] for r in concurrent_results]
    conc_times = [r['mean_time'] for r in concurrent_results]
    conc_std_times = [r['std_time'] for r in concurrent_results]
    conc_emissions = [r['mean_emissions'] for r in concurrent_results]
    conc_std_emissions = [r['std_emissions'] for r in concurrent_results]

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'axes.edgecolor': '#D0D7DE',
        'axes.labelcolor': '#24292F',
        'axes.titlecolor': '#24292F',
        'xtick.color': '#57606A',
        'ytick.color': '#57606A',
        'grid.color': '#D8DEE4',
        'grid.linestyle': '-',
        'grid.linewidth': 0.8,
        'legend.frameon': False,
    })

    colors = {
        'pytorch': '#D1495B',
        'concurrent': '#1F77B4',
    }

    def format_size(value, _):
        return f'{value:.0f}'

    def format_metric(value, _):
        if value == 0:
            return '0'
        if abs(value) < 0.01:
            return f'{value:.1e}'
        if abs(value) < 1:
            return f'{value:.3f}'
        return f'{value:.2f}'

    def draw_series(axis, sizes, values, errors, color, marker, label):
        values_arr = np.array(values)
        errors_arr = np.array(errors)
        lower = np.maximum(values_arr - errors_arr, 0)
        upper = values_arr + errors_arr

        axis.plot(
            sizes,
            values,
            color=color,
            marker=marker,
            markersize=7,
            linewidth=2.4,
            label=label,
            markerfacecolor='white',
            markeredgewidth=2,
            zorder=3,
        )
        axis.errorbar(
            sizes,
            values,
            yerr=errors,
            fmt='none',
            ecolor=color,
            elinewidth=1.4,
            capsize=4,
            alpha=0.45,
            zorder=2,
        )
        axis.fill_between(sizes, lower, upper, color=color, alpha=0.08, linewidth=0)

    fig, (ax_time, ax_emissions) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.14, top=0.68, wspace=0.24)

    for axis in (ax_time, ax_emissions):
        axis.set_facecolor('white')
        axis.grid(True, axis='y', alpha=0.8)
        axis.grid(False, axis='x')
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
        axis.spines['left'].set_color('#D0D7DE')
        axis.spines['bottom'].set_color('#D0D7DE')
        axis.tick_params(axis='both', labelsize=10)
        axis.xaxis.set_major_formatter(FuncFormatter(format_size))
        axis.yaxis.set_major_formatter(FuncFormatter(format_metric))
        axis.set_xlabel('Taille de matrice NxN', fontsize=11, labelpad=10)

    draw_series(ax_time, pt_sizes, pt_times, pt_std_times, colors['pytorch'], 'o', 'PyTorch')
    draw_series(ax_time, conc_sizes, conc_times, conc_std_times, colors['concurrent'], 's', 'Concurrent')
    ax_time.set_ylabel("Temps moyen d'execution (s)", fontsize=11, labelpad=10)
    ax_time.set_title("Performance temporelle", fontsize=14, fontweight='bold', loc='left', pad=14)

    draw_series(ax_emissions, pt_sizes, pt_emissions, pt_std_emissions, colors['pytorch'], 'o', 'PyTorch')
    draw_series(
        ax_emissions,
        conc_sizes,
        conc_emissions,
        conc_std_emissions,
        colors['concurrent'],
        's',
        'Concurrent',
    )
    ax_emissions.set_ylabel('Emissions moyennes (kg CO2e)', fontsize=11, labelpad=10)
    ax_emissions.set_title("Impact environnemental estime", fontsize=14, fontweight='bold', loc='left', pad=14)

    handles, labels = ax_time.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.79),
        ncol=2,
        fontsize=10.5,
        handlelength=2.8,
    )
    fig.suptitle(
        'Comparaison PyTorch vs calcul matriciel multithreade',
        fontsize=16,
        fontweight='bold',
        y=0.965,
    )
    fig.text(
        0.5,
        0.905,
        'Points = moyenne des executions, bandes/barres = ecart-type',
        ha='center',
        fontsize=10.5,
        color='#57606A',
    )

    output_path = Path(__file__).resolve().parents[2] / "results" / "figures" / "matrix-multiplication-emissions-comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=260, bbox_inches='tight', facecolor='white')
    print(f"Graphique 2D sauvegarde : {output_path}")

    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    COUNTRY_CODE = "TN"
    NUM_RUNS = 15
    
    print(f"\n Lancement de la comparaison avec {NUM_RUNS} exécutions par test...\n")
    
    results = run_comparison_tests(country_code=COUNTRY_CODE, num_runs=NUM_RUNS)
    
    if results:
        save_results(results)
        print_summary(results)
        print(" Génération du graphique 2D...")
        plot_2d_comparison(results)
    else:
        print("Aucun résultat à traiter")
