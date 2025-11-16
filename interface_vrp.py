"""
interface_vrp.py
----------------
UI Streamlit VRP/ALNS avec :
- Mode Auto (VRPLIB) et Manuel (coords, demandes, TW)
- Boutons Démarrer/Arrêter
- Tableau TW par client (respecté ou non)
- Dashboard expérimental : multi-instances, 20 runs, boxplots, temps vs taille,
  histogrammes, convergence moyenne comparée par instance.
"""

import os
import time
import json
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from solver import ALNSSolver, solve_vrplib_instance, solve_manual_instance

# ----------------------------------------------------------------------
# Page / style
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Projet ADEME – CesiCDP : optimisation de tournées sous contraintes",
    layout="wide",
)

st.markdown(
    """
<style>
.stApp { 
    background:#0D0D0D; 
    color:#e8ecf2; 
}

.block-container { 
    padding-top:1rem; 
}

/* Cartes statistiques */
.stat-card {
    background:#010309;
    border-radius:12px;
    padding:14px 12px;
    margin-bottom:8px;
    box-shadow:0 2px 8px rgba(0,0,0,.45);
}

.stat-card b { 
    color:#9aa3b2; 
    font-size:.86rem;
}

.stat-card span { 
    color:#FCFDE2; 
    font-size:1.02rem;
}

/* Bouton START — bleu premium */
.stButton button.start {
    background:#0E6B71;
    color:white;
    border-radius:10px;
    font-weight:600;
    padding:.45rem 1rem;
    border:none;
}

/* Bouton STOP — rouge moderne */
.stButton button.stop {
    background:#1B6447;
    color:#f2f2f2;
    border-radius:10px;
    font-weight:600;
    padding:.45rem 1rem;
    border:none;
}

/* Ligne de séparation */
hr {
    border:1px solid #1d2230;
}

/* Onglets */
.stTabs [data-baseweb="tab"] {
    color: #c7d0dd !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #ffffff !important;
}

.stTabs [aria-selected="true"] {
    color: #4fd1c5 !important;  /* Accent turquoise */
}

/* Sous-titres */
h2, h3, .stSubheader {
    color: #e8ecf2 !important;
}

/* Header - ajuster z-index pour ne pas masquer le contenu */
header[data-testid="stHeader"],
header[data-testid="stHeader"] > div {
    z-index: 1 !important;
    position: relative !important;
}

/* Contenu principal - s'assurer qu'il est visible */
.main .block-container {
    z-index: 0 !important;
    position: relative !important;
    padding-top: 2rem !important;
}

/* Tabs - s'assurer qu'ils sont visibles */
.stTabs {
    z-index: 0 !important;
    position: relative !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Références benchmark et CO2
# ----------------------------------------------------------------------
BENCHMARK_OPTIMA = {
    "A-n32-k5.vrp": 784.0,
    "X-n101-k25.vrp": 27591.0,
}
EMISSION_FACTOR_KG_CO2_PER_KM = 0.2

# ----------------------------------------------------------------------
# Utils
# ----------------------------------------------------------------------
def list_vrplib_instances(data_dir: str = "data") -> List[str]:
    """Liste toutes les instances VRP (tous types de fichiers sauf .sol)."""
    paths: List[str] = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            # Inclure tous les fichiers sauf .sol
            if not f.lower().endswith(".sol"):
                paths.append(os.path.join(root, f))
    paths.sort()
    return paths


def read_optimal_from_sol(instance_path: str) -> float:
    """
    Lit la valeur optimale depuis le fichier .sol correspondant à l'instance.
    Supporte tous types de fichiers (.vrp, .vrptw, .txt, etc.)
    """
    if not instance_path:
        return 0.0
    
    # Enlever toutes les extensions et ajouter .sol
    base_name = os.path.basename(instance_path)
    while '.' in base_name:
        base_name = base_name.rsplit('.', 1)[0]
    base_name += ".sol"
    
    dir_path = os.path.dirname(instance_path) or "."
    
    # 1. Chercher dans le même dossier
    sol_path = os.path.join(dir_path, base_name)
    if os.path.exists(sol_path):
        return _read_cost_from_sol_file(sol_path)
    
    # 2. Chercher récursivement dans data/
    data_dir = dir_path if dir_path != "." else "data"
    for root, _, files in os.walk(data_dir):
        if base_name in files:
            return _read_cost_from_sol_file(os.path.join(root, base_name))
    
    return 0.0


def _read_cost_from_sol_file(sol_path: str) -> float:
    """
    Lit la valeur du coût depuis un fichier .sol.
    
    Parameters
    ----------
    sol_path : str
        Chemin vers le fichier .sol
        
    Returns
    -------
    float
        Valeur du coût trouvée, ou 0.0 si erreur
    """
    try:
        with open(sol_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("Cost"):
                    # Format: "Cost 784" ou "Cost 27591" ou "Cost 827.3"
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            return float(parts[1])
                        except ValueError:
                            continue
    except (IOError, OSError, UnicodeDecodeError):
        return 0.0
    
    return 0.0


def calculate_environmental_impact(
    solution, 
    distance_matrix: np.ndarray,
    co2_per_km: float = 0.12,  # kg CO₂ par km pour un véhicule de livraison moyen
    vitesse_kmh: float = 50.0  # Vitesse constante en km/h
) -> Dict[str, float]:
    """
    Calcule l'impact environnemental basé sur la distance parcourue.
    
    Parameters
    ----------
    solution : Solution
        Solution VRP avec les routes
    distance_matrix : np.ndarray
        Matrice des distances
    co2_per_km : float
        Émissions de CO₂ en kg par km (défaut: 0.12 kg/km pour véhicule de livraison)
    vitesse_kmh : float
        Vitesse constante en km/h (défaut: 50 km/h)
    
    Returns
    -------
    Dict[str, float]
        Dictionnaire avec distance_totale_km, co2_kg, co2_tonnes, temps_trajet_heures, temps_trajet_minutes
    """
    total_distance = 0.0
    depot_idx = solution.depot
    
    for route in solution.routes:
        if not route:
            continue
        # Distance dépôt -> premier client
        total_distance += distance_matrix[depot_idx, route[0]]
        # Distances entre clients
        for i in range(len(route) - 1):
            total_distance += distance_matrix[route[i], route[i + 1]]
        # Distance dernier client -> dépôt
        total_distance += distance_matrix[route[-1], depot_idx]
    
    # Convertir en km (supposant que les distances sont en unités arbitraires)
    # Si les distances sont déjà en km, pas besoin de conversion
    distance_km = total_distance
    
    # Calcul CO₂
    co2_kg = distance_km * co2_per_km
    co2_tonnes = co2_kg / 1000.0
    
    # Calcul temps de trajet constant (vitesse constante)
    temps_trajet_heures = distance_km / vitesse_kmh if vitesse_kmh > 0 else 0.0
    temps_trajet_minutes = temps_trajet_heures * 60.0
    
    return {
        "distance_totale_km": distance_km,
        "co2_kg": co2_kg,
        "co2_tonnes": co2_tonnes,
        "temps_trajet_heures": temps_trajet_heures,
        "temps_trajet_minutes": temps_trajet_minutes
    }


def plot_routes_2d(
    coords: np.ndarray,
    routes: List[List[int]],
    depot: int = 0,
    title: str = "Routes",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    colors = [
        "#7c3aed",
        "#10b981",
        "#f97316",
        "#ef4444",
        "#3b82f6",
        "#22c55e",
        "#eab308",
        "#0ea5e9",
    ]
    dx, dy = coords[depot, 0], coords[depot, 1]
    ax.scatter([dx], [dy], s=80, c="black", marker="s", label="Dépôt")
    for ridx, r in enumerate(routes):
        if not r:
            continue
        c = colors[ridx % len(colors)]
        xs = [dx] + [coords[i, 0] for i in r] + [dx]
        ys = [dy] + [coords[i, 1] for i in r] + [dy]
        ax.plot(xs, ys, "-o", linewidth=2.0, markersize=4.0, color=c, label=f"Route {ridx + 1}")
    ax.set_title(title, color="#e5e7eb")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_facecolor("#111827")
    fig.patch.set_facecolor("#111827")
    ax.legend(fontsize=8)
    return fig


def plot_convergence(history: dict, optimal: float = 0.0) -> go.Figure:
    it = history.get("iterations", [])
    best = history.get("best_costs", [])
    cur = history.get("current_costs", [])
    fig = go.Figure()
    if best:
        fig.add_trace(
            go.Scatter(
                x=it,
                y=best,
                mode="lines",
                name="Meilleur coût",
                line=dict(color="#93c5fd", width=2.4),
            )
        )
    if cur:
        fig.add_trace(
            go.Scatter(
                x=it,
                y=cur,
                mode="lines",
                name="Coût courant",
                line=dict(color="#f97316", width=1.5, dash="dot"),
            )
        )
    if optimal and optimal > 0:
        fig.add_trace(
            go.Scatter(
                x=it,
                y=[optimal] * len(it),
                mode="lines",
                name="Référence",
                line=dict(color="#ef4444", width=1.3, dash="dash"),
            )
        )
    fig.update_layout(
        template="plotly_dark",
        height=320,
        margin=dict(l=0, r=0, t=28, b=0),
        xaxis_title="Itérations",
        yaxis_title="Coût",
    )
    return fig


def plot_gantt_time_windows(
    routes: List[List[int]],
    time_windows: List[Tuple[float, float]],
    tw_report: Dict[int, Dict[str, Any]],
    service_times: Optional[List[float]] = None,
    title: str = "Diagramme de Gantt - Fenêtres Temporelles",
) -> go.Figure:
    """Crée un diagramme de Gantt pour visualiser les fenêtres temporelles par véhicule."""
    fig = go.Figure()
    colors = ["#7c3aed", "#10b981", "#f97316", "#ef4444", "#3b82f6", "#22c55e", "#eab308", 
              "#0ea5e9", "#ec4899", "#8b5cf6", "#14b8a6", "#f59e0b", "#6366f1", "#a855f7", "#f43f5e"]
    max_time = 0
    
    for ridx, route in enumerate(routes):
        if not route: continue
        vname, color = f"V{ridx + 1}", colors[ridx % len(colors)]
        current_time = 0.0
        
        # Dépôt départ
        fig.add_trace(go.Scatter(x=[0], y=[ridx], mode="markers", marker=dict(size=12, color="red", symbol="square"),
                                 name="Dépôt", showlegend=(ridx == 0), hovertemplate=f"{vname}<br>Dépôt départ<extra></extra>"))
        
        # Clients
        for c in route:
            if c >= len(time_windows): continue
            tw, rep = time_windows[c], tw_report.get(c, {})
            ready = tw[0] if tw[0] != float('inf') else 0
            due = tw[1] if tw[1] != float('inf') else max(ready + 100, 1000)
            arrival = rep.get("arrival", current_time + 10)
            start_s = rep.get("start_service", max(arrival, ready))
            st = service_times[c] if service_times and c < len(service_times) else rep.get("finish", start_s) - start_s
            finish = rep.get("finish", start_s + st)
            
            # Zone TW (bleue claire)
            fig.add_trace(go.Scatter(x=[ready, due, due, ready, ready], 
                                   y=[ridx-0.4, ridx-0.4, ridx+0.4, ridx+0.4, ridx-0.4],
                                   fill="toself", fillcolor="rgba(59, 130, 246, 0.2)",
                                   line=dict(color="rgba(59, 130, 246, 0.5)", width=1),
                                   mode="lines", showlegend=False, hoverinfo="skip"))
            
            # Barre service
            respected = rep.get("respected", True)
            bar_color = color if respected else "#ef4444"
            fig.add_trace(go.Scatter(x=[start_s, finish], y=[ridx, ridx], mode="lines",
                                   line=dict(color=bar_color, width=20), showlegend=False,
                                   hovertemplate=(f"{vname} - Client {c}<br>TW: [{ready:.1f}, {due:.1f}]<br>"
                                                f"Arrivée: {arrival:.1f}<br>Début: {start_s:.1f}<br>Fin: {finish:.1f}<br>"
                                                f"Service: {st:.1f}<br>Statut: {'Respecté' if respected else 'Non respecté'}<extra></extra>")))
            
            # Point arrivée
            fig.add_trace(go.Scatter(x=[arrival], y=[ridx], mode="markers",
                                   marker=dict(size=8, color="#3b82f6", symbol="circle"),
                                   showlegend=False, hovertemplate=f"Arrivée: {arrival:.1f}<extra></extra>"))
            
            current_time = finish
            max_time = max(max_time, due, finish)
        
        # Retour dépôt
        return_time = current_time + 10
        fig.add_trace(go.Scatter(x=[return_time], y=[ridx], mode="markers",
                               marker=dict(size=12, color="red", symbol="square"),
                               showlegend=False, hovertemplate=f"{vname}<br>Retour dépôt: {return_time:.1f}<extra></extra>"))
        max_time = max(max_time, return_time)
    
    fig.update_layout(template="plotly_dark", height=max(400, len(routes) * 30),
                     xaxis_title="Temps", yaxis_title="Véhicule", title=title, showlegend=False,
                     margin=dict(l=80, r=20, t=60, b=40),
                     xaxis=dict(range=[-max_time * 0.05, max_time * 1.05]),
                     yaxis=dict(tickmode="array", tickvals=list(range(len(routes))),
                              ticktext=[f"V{i+1}" for i in range(len(routes))],
                              range=[-0.5, len(routes) - 0.5]))
    return fig


# Etat global pour arrêt
if "stop_flag" not in st.session_state:
    st.session_state.stop_flag = False


def request_stop() -> None:
    st.session_state.stop_flag = True


def should_stop() -> bool:
    return st.session_state.get("stop_flag", False)


def reset_stop() -> None:
    st.session_state.stop_flag = False


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------

tabs = st.tabs(["Résolution VRP", "Statistiques expérimentales"])

# ============================== Onglet 1 ===============================
with tabs[0]:
    col_left, col_right = st.columns([1, 2], gap="large")

    # ---------------------------------------------------------- Config
    with col_left:
        st.subheader("Configuration")

        max_iter = st.number_input(
            "Itérations ALNS",
            min_value=50,
            max_value=2000000,
            step=50,
            value=500,
        )

        instances = list_vrplib_instances("data")
        selected_instance = None
        
        if instances:
            # Organiser les instances par dossier pour un meilleur affichage
            instances_by_folder = {}
            for inst_path in instances:
                folder = os.path.dirname(inst_path) or "data"
                if folder not in instances_by_folder:
                    instances_by_folder[folder] = []
                instances_by_folder[folder].append(inst_path)
            
            # Créer un dictionnaire pour faciliter la recherche
            all_labels = []
            all_paths = []
            for folder, paths in sorted(instances_by_folder.items()):
                for path in sorted(paths):
                    rel_path = os.path.relpath(path, "data")
                    label = f"{rel_path}" if folder != "data" else os.path.basename(path)
                    all_labels.append(label)
                    all_paths.append(path)
            
            # Filtre de recherche
            search_filter = st.text_input(
                "Rechercher une instance (filtre par nom)",
                value="",
                help="Tapez pour filtrer les instances par nom"
            )
            
            # Filtrer les instances selon la recherche
            if search_filter:
                filtered_labels = [l for l in all_labels if search_filter.lower() in l.lower()]
                filtered_paths = [all_paths[all_labels.index(l)] for l in filtered_labels]
            else:
                filtered_labels = all_labels
                filtered_paths = all_paths
            
            if filtered_labels:
                # Sélecteur avec recherche
                selected_label = st.selectbox(
                    f"Instance VRPLIB ({len(filtered_labels)}/{len(all_labels)} trouvée(s))",
                    options=filtered_labels,
                    index=0,
                    help="Sélectionnez une instance à résoudre"
                )
                selected_instance = filtered_paths[filtered_labels.index(selected_label)]
                
                # Afficher le chemin complet
                st.caption(f"Chemin: {selected_instance}")
            else:
                st.warning(f"Aucune instance ne correspond au filtre '{search_filter}'.")
        else:
            st.warning("Aucune instance trouvée dans le dossier data/.")
        
        # Lecture automatique de la valeur optimale depuis le fichier .sol
        opt_guess = 0.0
        if selected_instance:
            opt_guess = read_optimal_from_sol(selected_instance)
            if opt_guess > 0:
                st.info(f"Valeur optimale trouvée dans le fichier .sol : {opt_guess:.2f}")
            else:
                st.caption("Aucun fichier .sol trouvé pour cette instance. Vous pouvez saisir manuellement la valeur optimale.")
        
        optimal = st.number_input(
            "Coût optimal (auto-détecté depuis .sol, modifiable)",
            min_value=0.0,
            step=1.0,
            value=float(opt_guess),
        )

        c1, c2 = st.columns(2)
        with c1:
            start_btn = st.button(
                "Démarrer",
                type="primary",
                key="start_btn",
                help="Lancer la résolution",
            )
        with c2:
            stop_btn = st.button(
                "Arrêter",
                key="stop_btn",
                help="Arrêter la résolution en cours",
            )

    # ---------------------------------------------------------- Résultats
    with col_right:
        st.subheader("Résultats et visualisation")

        if stop_btn:
            request_stop()

        if start_btn:
            reset_stop()

            logs: List[str] = []
            log_box = st.empty()
            conv_placeholder = st.empty()
            stats_placeholder = st.empty()
            routes_placeholder = st.empty()
            tw_table_placeholder = st.empty()
            download_placeholder = st.empty()

            def log_cb(it: int, best: float, cur: float) -> None:
                logs.append(
                    f"it={it:5d} | best={best:10.2f} | current={cur:10.2f}"
                )
                log_box.text("\n".join(logs[-40:]))

            t0 = time.time()

            if not instances:
                st.error("Aucune instance VRPLIB disponible.")
                best_sol = None
                solver = None
                inst_name = ""
                optimal_for_gap = 0.0
            else:
                solver, best_sol = solve_vrplib_instance(
                    instance_path=selected_instance,
                    max_iterations=int(max_iter),
                    log_callback=log_cb,
                    should_stop=should_stop,
                )
                inst_name = os.path.basename(selected_instance)
                optimal_for_gap = float(optimal)

            if best_sol is not None and solver is not None:
                elapsed = time.time() - t0

                nb_routes = len([r for r in best_sol.routes if r])
                nb_clients = solver.n_customers
                best_cost = best_sol.cost
                cap = solver.capacity

                if optimal_for_gap > 0:
                    gap_val = 100.0 * (best_cost - optimal_for_gap) / optimal_for_gap
                    gap_str = f"{gap_val:.2f} %"
                else:
                    gap_str = "-"

                # Calcul de l'impact environnemental
                env_impact = calculate_environmental_impact(best_sol, solver.distance_matrix)
                distance_km = env_impact["distance_totale_km"]
                co2_kg = env_impact["co2_kg"]
                co2_tonnes = env_impact["co2_tonnes"]
                temps_trajet_heures = env_impact["temps_trajet_heures"]
                temps_trajet_minutes = env_impact["temps_trajet_minutes"]
                
                # Affichage en console
                print("\n" + "="*60)
                print("IMPACT ENVIRONNEMENTAL")
                print("="*60)
                print(f"Distance totale parcourue : {distance_km:.2f} km")
                print(f"Temps de trajet constant (50 km/h) : {temps_trajet_heures:.2f} h ({temps_trajet_minutes:.1f} min)")
                print(f"Émissions CO₂ : {co2_kg:.2f} kg ({co2_tonnes:.3f} tonnes)")
                if optimal_for_gap > 0:
                    # Calculer l'impact si on avait utilisé la solution optimale
                    optimal_distance = optimal_for_gap  # Approximation : coût optimal ≈ distance optimale
                    optimal_co2_kg = optimal_distance * 0.12
                    excess_co2_kg = co2_kg - optimal_co2_kg
                    excess_pct = (excess_co2_kg / optimal_co2_kg * 100) if optimal_co2_kg > 0 else 0
                    if excess_co2_kg > 0:
                        print(f"Excès CO₂ vs solution optimale : +{excess_co2_kg:.2f} kg (+{excess_pct:.1f}%)")
                    else:
                        print(f"Économie CO₂ vs solution optimale : {abs(excess_co2_kg):.2f} kg ({abs(excess_pct):.1f}%)")
                print("="*60 + "\n")

                sc = stats_placeholder.columns(4)
                items = [
                    ("Instance", inst_name),
                    ("Clients servis", f"{nb_clients}"),
                    ("Nombre de routes", f"{nb_routes}"),
                    ("Capacité camion", f"{cap}"),
                    ("Meilleur coût", f"{best_cost:.2f}"),
                    ("Temps de calcul", f"{elapsed:.2f} s"),
                    ("Gap vs référence", gap_str),
                    (
                        "Faisabilité capacité",
                        "OK"
                        if best_sol.is_feasible_capacity()
                        else "Non respectée",
                    ),
                    ("Distance totale", f"{distance_km:.2f} km"),
                    ("Temps de trajet constant", f"{temps_trajet_heures:.2f} h ({temps_trajet_minutes:.1f} min)"),
                    ("Émissions CO₂", f"{co2_kg:.2f} kg ({co2_tonnes:.3f} t)"),
                ]
                for i, (k, v) in enumerate(items):
                    sc[i % 4].markdown(
                        f"<div class='stat-card'><b>{k}</b><br><span>{v}</span></div>",
                        unsafe_allow_html=True,
                    )

                conv_fig = plot_convergence(
                    solver.history,
                    optimal=optimal_for_gap,
                )
                conv_placeholder.plotly_chart(conv_fig, use_container_width=True)

                fig_routes = plot_routes_2d(
                    solver.coords,
                    best_sol.routes,
                    depot=0,
                    title=f"Routes pour {inst_name}",
                )
                routes_placeholder.pyplot(fig_routes)

                if solver.time_windows is not None:
                    rows = []
                    for c in range(1, nb_clients + 1):
                        tw = solver.time_windows[c]
                        rep = best_sol.tw_report.get(c, None)
                        rows.append(
                            {
                                "client": c,
                                "ready": tw[0],
                                "due": tw[1],
                                "arrival": None
                                if rep is None
                                else rep["arrival"],
                                "start_service": None
                                if rep is None
                                else rep["start_service"],
                                "finish": None
                                if rep is None
                                else rep["finish"],
                                "respected": None
                                if rep is None
                                else bool(rep["respected"]),
                            }
                        )
                    tw_df = pd.DataFrame(rows)
                    st.markdown(
                        "### Fenêtres temporelles – contrôle par client"
                    )
                    tw_table_placeholder.dataframe(
                        tw_df, use_container_width=True
                    )
                    
                    # Diagramme de Gantt pour les fenêtres temporelles
                    st.markdown("### Diagramme de Gantt - Fenêtres Temporelles")
                    gantt_fig = plot_gantt_time_windows(
                        best_sol.routes,
                        solver.time_windows,
                        best_sol.tw_report,
                        solver.service_times,
                        title=f"Diagramme de Gantt - {inst_name}",
                    )
                    st.plotly_chart(gantt_fig, use_container_width=True)

                # Convertir tw_report en format JSON-sérialisable
                tw_report_serializable = {}
                if best_sol.tw_report:
                    for client_id, info in best_sol.tw_report.items():
                        tw_report_serializable[int(client_id)] = {
                            "arrival": float(info.get("arrival", 0.0)),
                            "ready": float(info.get("ready", 0.0)),
                            "due": float(info.get("due", 0.0)),
                            "start_service": float(info.get("start_service", 0.0)),
                            "finish": float(info.get("finish", 0.0)),
                            "respected": bool(info.get("respected", True)),
                            "wait_time": float(info.get("wait_time", 0.0)),
                        }
                
                export = {
                    "instance": inst_name,
                    "best_cost": float(best_cost),
                    "routes": best_sol.routes,
                    "time_seconds": float(elapsed),
                    "gap": gap_str,
                    "feasible_capacity": bool(
                        best_sol.is_feasible_capacity()
                    ),
                    "feasible_time_windows": bool(
                        best_sol.is_feasible_time_windows()
                    ),
                    "tw_report": tw_report_serializable,
                }
                download_placeholder.download_button(
                    "Télécharger la solution (JSON)",
                    data=json.dumps(export, indent=2),
                    file_name=f"solution_{inst_name}.json",
                    mime="application/json",
                )

# ============================== Onglet 2 ===============================
with tabs[1]:
    st.subheader("Statistiques expérimentales")

    colA, colB = st.columns(2)
    with colA:
        nb_runs = st.number_input(
            "Nombre de runs par instance",
            min_value=1,
            max_value=50,
            value=20,
            step=1,
        )
        max_iter_bench = st.number_input(
            "Itérations ALNS par run",
            min_value=50,
            max_value=100000,
            value=500,
            step=50,
        )
    with colB:
        data_dir = st.text_input("Dossier des instances", value="data")
        # Chercher récursivement toutes les instances
        found_paths = list_vrplib_instances(data_dir)
        found = [os.path.basename(p) for p in found_paths]
        
        # Organiser par dossier
        instances_by_folder = {}
        for inst_path in found_paths:
            folder = os.path.dirname(inst_path) or "data"
            if folder not in instances_by_folder:
                instances_by_folder[folder] = []
            instances_by_folder[folder].append(inst_path)
        
        # Créer liste avec chemins relatifs
        all_instances_with_paths = []
        for folder, paths in sorted(instances_by_folder.items()):
            for path in sorted(paths):
                rel_path = os.path.relpath(path, data_dir) if data_dir != "." else os.path.basename(path)
                all_instances_with_paths.append((rel_path, path))
        
        # Sélecteur d'instances avec recherche
        st.markdown("**Sélection des instances**")
        search_bench = st.text_input(
            "Filtrer les instances",
            value="",
            help="Tapez pour filtrer les instances par nom",
            key="search_bench"
        )
        
        # Filtrer selon la recherche
        if search_bench:
            filtered_instances = [(label, path) for label, path in all_instances_with_paths 
                                if search_bench.lower() in label.lower()]
        else:
            filtered_instances = all_instances_with_paths
        
        # Sélecteur multiple
        instance_labels = [label for label, _ in filtered_instances]
        selected_instances_labels = st.multiselect(
            f"Instances à tester ({len(instance_labels)}/{len(all_instances_with_paths)} disponibles)",
            options=instance_labels,
            default=[],
            help="Sélectionnez une ou plusieurs instances. Si aucune n'est sélectionnée, toutes les instances avec .sol seront utilisées."
        )
        
        # Mapper les labels sélectionnés vers les chemins
        selected_instances_paths = [path for label, path in filtered_instances if label in selected_instances_labels]
        
        # Stocker dans session_state pour utilisation dans start_bench
        st.session_state["selected_instances_paths"] = selected_instances_paths
        st.session_state["data_dir_bench"] = data_dir
        
        # Utiliser toutes les instances qui ont un fichier .sol correspondant (si aucune sélection)
        if not selected_instances_paths:
            used = []
            for inst_path in found_paths:
                if read_optimal_from_sol(inst_path) > 0:
                    used.append(os.path.basename(inst_path))
            st.caption(
                f"Aucune instance sélectionnée. Utilisation automatique de toutes les instances avec .sol ({len(used)} trouvée(s))"
            )
        else:
            # Vérifier lesquelles ont un .sol
            used = []
            for inst_path in selected_instances_paths:
                if read_optimal_from_sol(inst_path) > 0:
                    used.append(os.path.relpath(inst_path, data_dir) if data_dir != "." else os.path.basename(inst_path))
            st.caption(
                f"{len(used)} instance(s) sélectionnée(s) avec fichier .sol (sur {len(selected_instances_paths)} sélectionnée(s))"
            )
        
        st.caption(
            f"Total: {len(found)} instance(s) trouvée(s) dans {data_dir}"
        )

    start_bench = st.button(
        "Lancer la campagne expérimentale", key="start_bench"
    )
    stop_bench = st.button("Arrêter la campagne", key="stop_bench")
    if stop_bench:
        request_stop()

    if start_bench:
        reset_stop()
        # Récupérer la sélection depuis session_state
        selected_instances_paths = st.session_state.get("selected_instances_paths", [])
        data_dir_actual = st.session_state.get("data_dir_bench", data_dir)
        
        # Recalculer les chemins pour être sûr
        found_paths_bench = list_vrplib_instances(data_dir_actual)
        
        # Utiliser les instances sélectionnées ou toutes celles avec .sol
        if selected_instances_paths:
            # Utiliser uniquement les instances sélectionnées qui ont un .sol
            used_bench = []
            for inst_path in selected_instances_paths:
                if read_optimal_from_sol(inst_path) > 0:
                    used_bench.append(inst_path)
        else:
            # Utiliser toutes les instances avec .sol (comportement par défaut)
            used_bench = []
            for inst_path in found_paths_bench:
                if read_optimal_from_sol(inst_path) > 0:
                    used_bench.append(inst_path)
        
        if not used_bench:
            st.error(
                "Aucune instance de référence trouvée avec fichier .sol. "
                "Place par exemple A-n32-k5.vrp ou X-n101-k25.vrp dans data/ avec leurs fichiers .sol correspondants."
            )
        else:
            st.info(f"Démarrage de la campagne sur {len(used_bench)} instance(s)...")
            st.info("Campagne en cours…")
            all_rows = []
            conv_sum = {}
            conv_count = {}
            conv_iter = {}

            for inst_path in used_bench:
                inst = os.path.basename(inst_path)
                # Lecture automatique de la valeur optimale depuis le fichier .sol
                optimal = read_optimal_from_sol(inst_path)
                if optimal <= 0:
                    # Fallback sur BENCHMARK_OPTIMA si pas de .sol
                    optimal = BENCHMARK_OPTIMA.get(inst, 0.0)
                conv_sum[inst] = None
                conv_count[inst] = 0
                conv_iter[inst] = None

                for r in range(1, int(nb_runs) + 1):
                    if should_stop():
                        st.warning("Campagne interrompue.")
                        break
                    t0 = time.time()
                    solver_b = ALNSSolver(instance_path=inst_path)
                    best_b = solver_b.solve(
                        max_iterations=int(max_iter_bench),
                        print_every=0,
                        should_stop=should_stop,
                    )
                    t1 = time.time()
                    gap = (
                        None
                        if optimal <= 0
                        else 100.0 * (best_b.cost - optimal) / optimal
                    )
                    all_rows.append(
                        {
                            "instance": inst,
                            "run": r,
                            "n_clients": solver_b.n_customers,
                            "capacity": solver_b.capacity,
                            "cost": best_b.cost,
                            "optimal": optimal if optimal > 0 else None,
                            "gap": gap,
                            "time": t1 - t0,
                        }
                    )

                    bc = np.array(
                        solver_b.history["best_costs"], dtype=float
                    )
                    it = np.array(
                        solver_b.history["iterations"], dtype=int
                    )
                    if bc.size:
                        if conv_sum[inst] is None:
                            conv_sum[inst] = bc.copy()
                            conv_iter[inst] = it.copy()
                        else:
                            m = min(len(conv_sum[inst]), len(bc))
                            conv_sum[inst][:m] += bc[:m]
                        conv_count[inst] += 1

                if should_stop():
                    break

            df_runs = pd.DataFrame(all_rows)
            st.session_state["runs_df"] = df_runs
            st.session_state["conv_sum"] = conv_sum
            st.session_state["conv_count"] = conv_count
            st.session_state["conv_iter"] = conv_iter
            st.session_state["data_dir"] = data_dir  # Stocker pour utilisation ultérieure
            st.success("Campagne terminée.")

    if "runs_df" in st.session_state:
        df = st.session_state["runs_df"]
        st.markdown("### Résultats détaillés (tous les runs)")
        st.dataframe(df, use_container_width=True)

        st.markdown("### Synthèse statistique complète par instance")
        
        # Calculer toutes les statistiques détaillées
        stats_list = []
        for inst in df["instance"].unique():
            inst_df = df[df["instance"] == inst]
            cost_vals = inst_df["cost"].dropna()
            gap_vals = inst_df["gap"].dropna()
            time_vals = inst_df["time"].dropna()
            
            stats_dict = {
                "instance": inst,
                "n_runs": len(inst_df),
                "n_clients": inst_df["n_clients"].iloc[0],
                "capacity": inst_df["capacity"].iloc[0],
                "optimal": inst_df["optimal"].iloc[0] if inst_df["optimal"].notna().any() else None,
            }
            
            # Statistiques sur les coûts
            if len(cost_vals) > 0:
                stats_dict.update({
                    "cost_mean": cost_vals.mean(),
                    "cost_median": cost_vals.median(),
                    "cost_std": cost_vals.std(),
                    "cost_min": cost_vals.min(),
                    "cost_max": cost_vals.max(),
                    "cost_q1": cost_vals.quantile(0.25),
                    "cost_q3": cost_vals.quantile(0.75),
                    "cost_iqr": cost_vals.quantile(0.75) - cost_vals.quantile(0.25),
                })
            else:
                stats_dict.update({k: None for k in ["cost_mean", "cost_median", "cost_std", "cost_min", "cost_max", "cost_q1", "cost_q3", "cost_iqr"]})
            
            # Statistiques sur les gaps
            if len(gap_vals) > 0:
                stats_dict.update({
                    "gap_mean": gap_vals.mean(),
                    "gap_median": gap_vals.median(),
                    "gap_std": gap_vals.std(),
                    "gap_min": gap_vals.min(),
                    "gap_max": gap_vals.max(),
                    "gap_q1": gap_vals.quantile(0.25),
                    "gap_q3": gap_vals.quantile(0.75),
                    "gap_iqr": gap_vals.quantile(0.75) - gap_vals.quantile(0.25),
                    "gap_pct_gt_5": float(np.mean(gap_vals > 5) * 100.0),
                    "gap_pct_gt_10": float(np.mean(gap_vals > 10) * 100.0),
                    "gap_pct_optimal": float(np.mean(gap_vals <= 0.1) * 100.0) if len(gap_vals) > 0 else 0.0,
                })
            else:
                stats_dict.update({k: None for k in ["gap_mean", "gap_median", "gap_std", "gap_min", "gap_max", "gap_q1", "gap_q3", "gap_iqr", "gap_pct_gt_5", "gap_pct_gt_10", "gap_pct_optimal"]})
            
            # Statistiques sur les temps
            if len(time_vals) > 0:
                stats_dict.update({
                    "time_mean": time_vals.mean(),
                    "time_median": time_vals.median(),
                    "time_std": time_vals.std(),
                    "time_min": time_vals.min(),
                    "time_max": time_vals.max(),
                    "time_q1": time_vals.quantile(0.25),
                    "time_q3": time_vals.quantile(0.75),
                    "time_iqr": time_vals.quantile(0.75) - time_vals.quantile(0.25),
                })
            else:
                stats_dict.update({k: None for k in ["time_mean", "time_median", "time_std", "time_min", "time_max", "time_q1", "time_q3", "time_iqr"]})
            
            stats_list.append(stats_dict)
        
        stats_df = pd.DataFrame(stats_list)
        
        # Afficher les statistiques par catégorie
        st.markdown("#### Statistiques sur les Coûts")
        cost_cols = ["instance", "n_runs", "optimal", "cost_mean", "cost_median", "cost_std", 
                     "cost_min", "cost_max", "cost_q1", "cost_q3", "cost_iqr"]
        cost_stats = stats_df[[c for c in cost_cols if c in stats_df.columns]].copy()
        # Formater les valeurs numériques
        for col in cost_stats.select_dtypes(include=[np.number]).columns:
            cost_stats[col] = cost_stats[col].apply(lambda x: f"{float(x):.2f}" if pd.notna(x) and x is not None else "-")
        st.dataframe(cost_stats, use_container_width=True)
        
        st.markdown("#### Statistiques sur les Gaps (%)")
        gap_cols = ["instance", "n_runs", "gap_mean", "gap_median", "gap_std", 
                     "gap_min", "gap_max", "gap_q1", "gap_q3", "gap_iqr",
                     "gap_pct_gt_5", "gap_pct_gt_10", "gap_pct_optimal"]
        gap_stats = stats_df[[c for c in gap_cols if c in stats_df.columns]].copy()
        # Formater les valeurs numériques
        for col in gap_stats.select_dtypes(include=[np.number]).columns:
            gap_stats[col] = gap_stats[col].apply(lambda x: f"{float(x):.2f}" if pd.notna(x) and x is not None else "-")
        st.dataframe(gap_stats, use_container_width=True)
        
        st.markdown("#### Statistiques sur les Temps d'exécution (s)")
        time_cols = ["instance", "n_runs", "time_mean", "time_median", "time_std", 
                     "time_min", "time_max", "time_q1", "time_q3", "time_iqr"]
        time_stats = stats_df[[c for c in time_cols if c in stats_df.columns]].copy()
        # Formater les valeurs numériques
        for col in time_stats.select_dtypes(include=[np.number]).columns:
            time_stats[col] = time_stats[col].apply(lambda x: f"{float(x):.2f}" if pd.notna(x) and x is not None else "-")
        st.dataframe(time_stats, use_container_width=True)
        
        # Tableau récapitulatif compact
        st.markdown("#### Récapitulatif compact")
        summary_cols = ["instance", "n_runs", "n_clients", "optimal", 
                         "cost_mean", "cost_median", "cost_min", "cost_max",
                         "gap_mean", "gap_median", "gap_min", "gap_max",
                         "time_mean", "time_median", "time_min", "time_max"]
        summary = stats_df[[c for c in summary_cols if c in stats_df.columns]].copy()
        # Formater les valeurs numériques
        for col in summary.select_dtypes(include=[np.number]).columns:
            summary[col] = summary[col].apply(lambda x: f"{float(x):.2f}" if pd.notna(x) and x is not None else "-")
        st.dataframe(summary, use_container_width=True)
        
        # Sauvegarder pour utilisation ultérieure
        st.session_state["stats_df"] = stats_df
        
        # Bouton de téléchargement des statistiques
        st.markdown("---")
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            # Télécharger les statistiques en CSV
            csv_stats = stats_df.to_csv(index=False)
            st.download_button(
                "Télécharger les statistiques (CSV)",
                data=csv_stats,
                file_name=f"statistiques_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="dl_stats_csv"
            )
        with col_dl2:
            # Télécharger les statistiques en JSON
            json_stats = stats_df.to_json(orient="records", indent=2)
            st.download_button(
                "Télécharger les statistiques (JSON)",
                data=json_stats,
                file_name=f"statistiques_{time.strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                key="dl_stats_json"
            )

        st.markdown("#### Boxplot des gaps (20 runs par instance)")
        instances = stats_df["instance"].tolist()
        data_box = [
            df[df["instance"] == inst]["gap"].dropna() for inst in instances
        ]
        # Boxplot agrandi et centré avec vraie boîte à moustaches
        fig_box, ax_box = plt.subplots(figsize=(max(12.0, len(instances) * 1.2), 8.0))
        if data_box:
            # Créer le boxplot avec patch_artist pour rendre la boîte visible
            bp = ax_box.boxplot(
                data_box, 
                labels=instances, 
                patch_artist=True,  # CRUCIAL : permet de voir la boîte
                showmeans=True,     # Affiche la moyenne
                meanline=False,    # Moyenne en marqueur
                showfliers=True,    # Affiche les outliers
                # Médiane : ligne épaisse dans la boîte
                medianprops=dict(color='yellow', linewidth=3, linestyle='-'),
                # Moyenne : marqueur visible
                meanprops=dict(marker='D', markeredgecolor='cyan', markerfacecolor='cyan', 
                             markersize=12, markeredgewidth=2),
                # Boîte : Q1-Q3 avec bordure visible
                boxprops=dict(linewidth=2.5, edgecolor='white'),
                # Moustaches : lignes visibles
                whiskerprops=dict(color='white', linewidth=2.5),
                # Barres aux extrémités
                capprops=dict(color='white', linewidth=2.5),
                # Outliers
                flierprops=dict(marker='o', markerfacecolor='red', markeredgecolor='red', 
                              markersize=8, alpha=0.8)
            )
            # Colorier chaque boîte avec des couleurs vives et visibles
            colors = plt.cm.tab20(np.linspace(0, 1, len(bp['boxes'])))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)  # Transparence pour voir la médiane
                patch.set_edgecolor('white')
                patch.set_linewidth(2.5)
        
        # Améliorer les axes et échelles
        ax_box.set_ylabel("Gap (%)", fontsize=14, color='white', fontweight='bold')
        ax_box.set_xlabel("Instance", fontsize=14, color='white', fontweight='bold')
        ax_box.tick_params(colors='white', labelsize=11)
        # Rotation des labels
        plt.setp(ax_box.get_xticklabels(), rotation=45, ha='right')
        # Grille améliorée
        ax_box.grid(True, linestyle="--", alpha=0.5, color='lightgray', linewidth=1)
        ax_box.set_axisbelow(True)  # Grille derrière les données
        # Fond sombre
        fig_box.patch.set_facecolor("#111827")
        ax_box.set_facecolor("#111827")
        # Centrer et ajuster avec marges
        plt.tight_layout(pad=2.0)
        st.pyplot(fig_box)

        st.markdown("#### Temps d'exécution moyen vs nombre de clients")
        fig_t, ax_t = plt.subplots(figsize=(7.0, 4.0))
        ax_t.scatter(stats_df["n_clients"], stats_df["time_mean"])
        for _, row in stats_df.iterrows():
            ax_t.text(
                row["n_clients"],
                row["time_mean"],
                row["instance"],
                fontsize=8,
            )
        ax_t.set_xlabel("Nombre de clients")
        ax_t.set_ylabel("Temps moyen (s)")
        ax_t.grid(True, linestyle="--", alpha=0.3)
        fig_t.patch.set_facecolor("#111827")
        ax_t.set_facecolor("#111827")
        st.pyplot(fig_t)

        st.markdown(
            "#### Courbes de convergence moyennes (comparaison entre instances)"
        )
        conv_sum = st.session_state["conv_sum"]
        conv_count = st.session_state["conv_count"]
        conv_iter = st.session_state["conv_iter"]

        fig_conv_all = go.Figure()
        for inst in instances:
            sum_vec = conv_sum.get(inst)
            cnt = conv_count.get(inst, 0)
            it_vec = conv_iter.get(inst)
            if sum_vec is None or cnt <= 0 or it_vec is None:
                continue
            mean_best = sum_vec / cnt
            fig_conv_all.add_trace(
                go.Scatter(
                    x=it_vec,
                    y=mean_best,
                    mode="lines",
                    name=inst,
                )
            )
            # Lecture de la valeur optimale depuis .sol
            data_dir_for_opt = st.session_state.get("data_dir", "data")
            inst_path_for_opt = os.path.join(data_dir_for_opt, inst)
            opt_val = read_optimal_from_sol(inst_path_for_opt)
            if opt_val <= 0:
                # Fallback sur BENCHMARK_OPTIMA si pas de .sol
                opt_val = BENCHMARK_OPTIMA.get(inst, 0.0)
            if opt_val > 0:
                fig_conv_all.add_trace(
                    go.Scatter(
                        x=it_vec,
                        y=[opt_val] * len(it_vec),
                        mode="lines",
                        name=f"{inst} (optimal)",
                        line=dict(dash="dot"),
                        opacity=0.4,
                        showlegend=False,
                    )
                )
        fig_conv_all.update_layout(
            template="plotly_dark",
            height=360,
            margin=dict(l=0, r=0, t=28, b=0),
            xaxis_title="Itérations",
            yaxis_title="Best cost moyen",
        )
        st.plotly_chart(fig_conv_all, use_container_width=True)

        st.markdown("#### Histogramme des coûts par instance")
        inst_h = st.selectbox(
            "Instance pour histogramme des coûts",
            options=instances,
            key="hist_sel",
        )
        fig_h, ax_h = plt.subplots(figsize=(7.0, 4.0))
        vals = df[df["instance"] == inst_h]["cost"].values
        ax_h.hist(vals, bins=min(20, len(vals)))
        ax_h.set_xlabel("Coût")
        ax_h.set_ylabel("Fréquence")
        ax_h.grid(True, linestyle="--", alpha=0.3)
        fig_h.patch.set_facecolor("#111827")
        ax_h.set_facecolor("#111827")
        st.pyplot(fig_h)

    else:
        st.info("Aucune campagne n'a encore été lancée.")
