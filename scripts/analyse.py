#!/usr/bin/env python3
"""Analyse des sorties ColabFold / AlphaFold2 : pLDDT et PAE.

Usage :
    python scripts/analyse.py results/P0DP23 -o figures/P0DP23
    python scripts/analyse.py results/P0DP23 -o figures/P0DP23 --domains 1-78,82-149

Le script lit, dans le dossier de résultats ColabFold :
  - les fichiers *_scores_rank_00N_*.json (pLDDT, PAE, pTM de chaque modèle) ;
  - le fichier *_unrelaxed_rank_001_*.pdb (structure du meilleur modèle).

Il produit dans le dossier de sortie :
  - plddt_rank_001.csv   : pLDDT par résidu du meilleur modèle ;
  - plddt.png            : profil pLDDT (meilleur modèle + autres modèles) ;
  - pae.png              : matrice PAE du meilleur modèle ;
  - resume.txt           : statistiques résumées (moyennes, classes, régions faibles).
"""

import argparse
import csv
import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")  # pas besoin d'écran : on écrit directement des PNG
import matplotlib.pyplot as plt
import numpy as np

# Seuils de confiance définis par DeepMind / AlphaFold DB
PLDDT_CLASSES = [
    (90, 101, "Très élevée (≥ 90)", "#0053D6"),
    (70, 90, "Élevée (70–90)", "#65CBF3"),
    (50, 70, "Faible (50–70)", "#FFDB13"),
    (0, 50, "Très faible (< 50)", "#FF7D45"),
]


def find_scores(results_dir):
    """Retourne la liste triée (rang, chemin) des fichiers de scores JSON."""
    files = glob.glob(os.path.join(results_dir, "*_scores_rank_*.json"))
    if not files:
        raise SystemExit(f"Aucun fichier *_scores_rank_*.json dans {results_dir}")
    ranked = []
    for f in files:
        rank = int(re.search(r"_rank_(\d+)_", f).group(1))
        ranked.append((rank, f))
    return sorted(ranked)


def plddt_from_pdb(pdb_path):
    """Lit le pLDDT stocké dans la colonne B-factor des atomes CA du PDB."""
    values = []
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                values.append(float(line[60:66]))
    return np.array(values)


def segments(mask):
    """Transforme un masque booléen en liste de segments (début, fin) 1-indexés."""
    segs, start = [], None
    for i, m in enumerate(mask, start=1):
        if m and start is None:
            start = i
        elif not m and start is not None:
            segs.append((start, i - 1))
            start = None
    if start is not None:
        segs.append((start, len(mask)))
    return segs


def parse_domains(text):
    """'1-78,82-149' -> [(1, 78), (82, 149)]"""
    doms = []
    for part in text.split(","):
        a, b = part.split("-")
        doms.append((int(a), int(b)))
    return doms


def plot_plddt(all_scores, out_png):
    fig, ax = plt.subplots(figsize=(9, 3.5))
    for lo, hi, label, color in PLDDT_CLASSES:
        ax.axhspan(lo, min(hi, 100), color=color, alpha=0.15, lw=0)
    for rank, scores in all_scores[1:]:
        ax.plot(np.arange(1, len(scores["plddt"]) + 1), scores["plddt"],
                color="grey", lw=0.8, alpha=0.6)
    rank, best = all_scores[0]
    ax.plot(np.arange(1, len(best["plddt"]) + 1), best["plddt"],
            color="black", lw=1.6, label="rang 1 (meilleur modèle)")
    ax.plot([], [], color="grey", lw=0.8, label="rangs 2–5")
    ax.set_xlim(1, len(best["plddt"]))
    ax.set_ylim(0, 100)
    ax.set_xlabel("Position du résidu")
    ax.set_ylabel("pLDDT")
    ax.set_title("pLDDT par résidu")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_pae(pae, out_png, domains=None):
    n = pae.shape[0]
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    im = ax.imshow(pae, cmap="Greens_r", vmin=0, vmax=30,
                   extent=(0.5, n + 0.5, n + 0.5, 0.5))
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Erreur attendue (Å)")
    if domains:
        for a, b in domains:
            ax.add_patch(plt.Rectangle((a - 0.5, a - 0.5), b - a + 1, b - a + 1,
                                       fill=False, ec="red", lw=1, ls="--"))
    ax.set_xlabel("Résidu aligné (j)")
    ax.set_ylabel("Résidu dont on évalue la position (i)")
    ax.set_title("Predicted Aligned Error (rang 1)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results_dir", help="dossier de sortie de colabfold_batch")
    p.add_argument("-o", "--out", default="figures", help="dossier des analyses")
    p.add_argument("--domains", help="domaines pour la PAE, ex. 1-78,82-149")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ranked = find_scores(args.results_dir)
    all_scores = []
    for rank, path in ranked:
        with open(path) as fh:
            all_scores.append((rank, json.load(fh)))

    best_rank, best = all_scores[0]
    plddt = np.array(best["plddt"])
    pae = np.array(best["pae"])
    n = len(plddt)

    # Vérification croisée : le pLDDT est aussi écrit dans le B-factor du PDB
    pdbs = glob.glob(os.path.join(args.results_dir, "*_unrelaxed_rank_001_*.pdb"))
    pdb_check = ""
    if pdbs:
        from_pdb = plddt_from_pdb(pdbs[0])
        diff = np.abs(from_pdb - plddt).max() if len(from_pdb) == n else float("nan")
        pdb_check = f"Écart max JSON vs B-factor PDB : {diff:.2f}\n"

    # CSV par résidu
    with open(os.path.join(args.out, "plddt_rank_001.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["residu", "plddt", "classe"])
        for i, v in enumerate(plddt, start=1):
            cls = next(lab for lo, hi, lab, _ in PLDDT_CLASSES if lo <= v < hi)
            w.writerow([i, f"{v:.2f}", cls])

    plot_plddt(all_scores, os.path.join(args.out, "plddt.png"))
    domains = parse_domains(args.domains) if args.domains else None
    plot_pae(pae, os.path.join(args.out, "pae.png"), domains)

    # Résumé texte
    lines = [f"Nombre de résidus : {n}", pdb_check.rstrip("\n")]
    lines.append("\nScores par modèle (rang : pLDDT moyen, pTM) :")
    for rank, s in all_scores:
        lines.append(f"  rang {rank} : {np.mean(s['plddt']):.1f}, pTM = {s.get('ptm', float('nan')):.3f}")
    lines.append(f"\npLDDT meilleur modèle : moyenne {plddt.mean():.1f}, "
                 f"médiane {np.median(plddt):.1f}, min {plddt.min():.1f} (résidu {plddt.argmin() + 1})")
    lines.append("Répartition des résidus par classe de confiance :")
    for lo, hi, lab, _ in PLDDT_CLASSES:
        k = int(((plddt >= lo) & (plddt < hi)).sum())
        lines.append(f"  {lab:<22} {k:4d} résidus ({100 * k / n:5.1f} %)")
    low = segments(plddt < 70)
    lines.append("Régions pLDDT < 70 : " + (", ".join(f"{a}-{b}" for a, b in low) or "aucune"))

    lines.append(f"\nPAE : moyenne {pae.mean():.1f} Å, max {pae.max():.1f} Å")
    if domains:
        lines.append("PAE moyenne entre domaines (lignes = résidus évalués, colonnes = alignés sur) :")
        for i, (a, b) in enumerate(domains):
            row = []
            for c, d in domains:
                row.append(f"{pae[a - 1:b, c - 1:d].mean():5.1f}")
            lines.append(f"  domaine {i + 1} ({a}-{b}) : " + "  ".join(row))

    text = "\n".join(l for l in lines if l is not None)
    with open(os.path.join(args.out, "resume.txt"), "w") as fh:
        fh.write(text + "\n")
    print(text)
    print(f"\nFichiers écrits dans {args.out}/")


if __name__ == "__main__":
    main()
