# Projet ADEME – Optimisation de Tournées de Livraison (CVRPTW)

## Description

Ce projet implémente un solveur **ALNS + SA (Adaptive Large Neighborhood Search)** pour résoudre le problème de routage de véhicules avec contraintes de capacité et fenêtres temporelles (CVRPTW). 

Le solveur est conçu dans le cadre de l'appel à manifestation d'intérêt de l'ADEME pour l'optimisation de la mobilité et de la logistique.

## Fonctionnalités

- Résolution du CVRPTW via algorithme ALNS hybride avec recuit simulé
- Interface graphique Streamlit pour la résolution interactive
- Support de multiples formats d'instances (VRPLIB, Solomon, formats personnalisés)
- Visualisation des solutions (cartes, courbes de convergence, diagrammes de Gantt)
- Analyse statistique expérimentale (multi-runs, boxplots, convergence)
- Calcul d'impact environnemental (émissions CO₂)

## Prérequis

- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)

## Installation

1. **Cloner ou télécharger le projet**

2. **Installer les dépendances**

```bash
pip install -r requirements.txt
```

## Lancement de l'application

### Interface graphique (recommandé)

```bash
streamlit run interface_vrp.py
```

L'application s'ouvrira automatiquement dans votre navigateur à l'adresse `http://localhost:8501`.

### Utilisation en ligne de commande

```python
from solver import solve_vrplib_instance

# Résoudre une instance
solver, solution = solve_vrplib_instance(
    instance_path="data/C101.txt",
    max_iterations=500
)

print(f"Coût optimal: {solution.cost:.2f}")
print(f"Nombre de routes: {len([r for r in solution.routes if r])}")
```

## Structure du projet

```
projet_RO_cvrptw/
├── solver.py              # Solveur ALNS principal
├── interface_vrp.py       # Interface Streamlit
├── Livrable_final.ipynb   # Documentation et rapport
├── requirements.txt       # Dépendances Python
├── README.md              # Ce fichier
└── data/                  # Instances de test VRP
    ├── C101.txt
    ├── A-n32-k5.vrp
    └── ...
```

## Guide d'utilisation rapide

### Via l'interface Streamlit

1. **Onglet "Résolution VRP"**
   - Sélectionner une instance dans le dossier `data/`
   - Configurer le nombre d'itérations ALNS
   - Cliquer sur "Démarrer"
   - Visualiser les résultats (routes, convergence, fenêtres temporelles)

2. **Onglet "Statistiques expérimentales"**
   - Sélectionner plusieurs instances
   - Configurer le nombre de runs par instance
   - Lancer la campagne expérimentale
   - Analyser les statistiques (boxplots, convergence moyenne, temps d'exécution)

### Formats d'instances supportés

- Format VRPLIB (`.vrp`, `.vrptw`)
- Format Solomon (`.txt`)
- Formats personnalisés avec sections `CUSTOMER`, `NODE_COORD_SECTION`, `DEMAND_SECTION`, `TIME_WINDOW_SECTION`

## Auteurs

**Groupe 2 - CesiCDP**
- ABDELKKADERMEKKI Mohamed
- BENAZIZ Rayan
- HALLAOUA Sid-Ali
- RECHAM Wissam

## Licence

Ce projet a été développé dans le cadre académique pour l'ADEME.

