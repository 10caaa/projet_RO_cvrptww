
"""
======================================================================
SOLVEUR ALNS POUR VRP AVEC FENÊTRES TEMPORELLES
======================================================================

Ce module implémente un solveur ALNS (Adaptive Large Neighborhood Search)
pour résoudre le problème de routage de véhicules (VRP) avec fenêtres 
temporelles (Time Windows).

Structure du code:
1. Parser de fichiers VRP - lit les instances depuis des fichiers
2. Classe Solution - représente une solution VRP avec évaluation
3. Classe LocalSearch - opérateurs d'optimisation locale (2-opt, relocate, swap)
4. Classe ALNSSolver - solveur principal utilisant ALNS
5. Fonctions utilitaires - interfaces pour résoudre des instances

Algorithme ALNS:
- Destruction: retire des clients de la solution (random, worst, shaw, related)
- Réparation: réinsère les clients (greedy, regret2)
- Intensification: recherche locale périodique
- Acceptation: recuit simulé pour accepter/rejeter les solutions
"""

import math  # Pour calculer les distances euclidiennes (hypot)
import random  # Pour la sélection aléatoire et le recuit simulé
import time  # Pour mesurer le temps d'exécution
from typing import List, Tuple, Optional, Callable, Dict, Any
from copy import deepcopy  # Pour copier profondément les solutions

import numpy as np  # Pour les matrices de distances et calculs vectoriels
import vrplib  # Bibliothèque pour lire les fichiers VRP standard


# ======================================================================
# PARSER COMPACT POUR FICHIERS VRP AVEC FENÊTRES TEMPORELLES
# ======================================================================

def parse_vrp_file(file_path: str) -> Dict[str, Any]:
    """
    Parse un fichier VRP (.vrp, .vrptw, .txt) et extrait toutes les données.
    
    Cette fonction lit un fichier VRP et extrait:
    - Les coordonnées des nœuds (dépôt + clients)
    - Les demandes de chaque client
    - La capacité des véhicules
    - Les fenêtres temporelles (ready time, due time)
    - Les temps de service
    
    Fonctionnement:
    1. Lit le fichier ligne par ligne
    2. Détecte les sections (NODE_COORD_SECTION, DEMAND_SECTION, etc.)
    3. Parse chaque section selon son format
    4. Retourne un dictionnaire structuré avec toutes les données
    
    Args:
        file_path: Chemin vers le fichier VRP à parser
        
    Returns:
        Dictionnaire contenant toutes les données de l'instance:
        - 'node_coord': liste de [x, y] pour chaque nœud
        - 'demand': liste des demandes (0 pour le dépôt)
        - 'capacity': capacité des véhicules
        - 'time_window': liste de [ready, due] pour chaque nœud
        - 'service_time': liste des temps de service
    """
    # Ouvrir le fichier et lire toutes les lignes non vides
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [l.strip() for l in f if l.strip()]  # Enlever espaces et lignes vides
    
    # Initialiser la structure de données avec des valeurs par défaut
    data = {'name': '', 'comment': '', 'type': '', 'dimension': 0, 'capacity': 0,
            'edge_weight_type': '', 'node_coord': [], 'demand': [], 
            'time_window': [], 'service_time': [], 'depot': 1}
    
    section = None  # Section actuelle du fichier qu'on est en train de parser
    node_map = {}  # Mapping entre ID nœud dans fichier et index dans notre liste
    
    # Parcourir chaque ligne du fichier
    for line in lines:
        if not line: continue  # Ignorer les lignes vides
        
        # ============================================================
        # PARSER LES MÉTADONNÉES (format "CLÉ: VALEUR")
        # ============================================================
        # Les métadonnées sont au début du fichier (NAME, CAPACITY, etc.)
        if ':' in line:
            k, v = line.split(':', 1)  # Séparer clé et valeur
            k, v = k.strip().upper(), v.strip()  # Normaliser en majuscules
            # Extraire les différentes métadonnées
            if k == 'NAME': data['name'] = v
            elif k == 'COMMENT': data['comment'] = v
            elif k == 'TYPE': data['type'] = v
            elif k == 'DIMENSION': data['dimension'] = int(v)  # Nombre de nœuds
            elif k == 'CAPACITY': data['capacity'] = int(v)  # Capacité véhicule
            elif k == 'EDGE_WEIGHT_TYPE': data['edge_weight_type'] = v
            elif k == 'SERVICE_TIME': data['_global_st'] = float(v)  # Temps service global
        
        # ============================================================
        # DÉTECTER LES SECTIONS DU FICHIER
        # ============================================================
        # Les sections indiquent où commencent les données (coords, demandes, etc.)
        lu = line.upper().strip()  # Ligne en majuscules pour comparaison
        if lu.startswith('NODE_COORD_SECTION'): 
            section = 'nodes'  # Section des coordonnées (x, y)
        elif lu.startswith('DEMAND_SECTION'): 
            section = 'demands'  # Section des demandes
        elif lu.startswith('TIME_WINDOW_SECTION'): 
            section = 'time_windows'  # Section des fenêtres temporelles
        elif lu.startswith('DEPOT_SECTION'): 
            section = 'depot'  # Section du dépôt
        elif lu.startswith('VEHICLE'): 
            section = 'vehicle'  # Section des véhicules
        elif 'CUSTOMER' in lu or 'CUST NO.' in lu: 
            section = 'customer'  # Section format client (id x y demand ready due service)
            continue  # Skip la ligne d'en-tête
        elif lu == 'EOF': 
            break  # Fin du fichier
        
        # ============================================================
        # PARSER LES DONNÉES SELON LA SECTION DÉTECTÉE
        # ============================================================
        
        # Section NODE_COORD_SECTION: Format "id x y"
        elif section == 'nodes':
            p = line.split()  # Séparer les valeurs par espaces
            if len(p) >= 3:
                try:
                    nid, x, y = int(p[0]), float(p[1]), float(p[2])
                    # Mapper l'ID du fichier vers notre index interne
                    node_map[nid] = len(data['node_coord'])
                    data['node_coord'].append([x, y])  # Stocker coordonnées
                    data['demand'].append(0)  # Demande par défaut (sera mis à jour)
                    data['time_window'].append([0.0, float('inf')])  # TW infinie par défaut
                    data['service_time'].append(0.0)  # Temps service par défaut
                except: pass  # Ignorer les lignes mal formatées
        
        # Section CUSTOMER: Format "id x y demand ready due service"
        elif section == 'customer':
            p = line.split()
            if len(p) >= 7:
                try:
                    # Extraire toutes les informations d'un coup
                    nid, x, y, d, r, du, s = int(p[0]), float(p[1]), float(p[2]), int(p[3]), float(p[4]), float(p[5]), float(p[6])
                    data['node_coord'].append([x, y])
                    data['demand'].append(d)  # Demande du client
                    data['time_window'].append([r, du])  # Fenêtre [ready, due]
                    data['service_time'].append(s)  # Temps de service
                except: pass
        
        # Section DEMAND_SECTION: Format "id demand"
        elif section == 'demands':
            p = line.split()
            if len(p) >= 2:
                try:
                    nid, d = int(p[0]), int(p[1])
                    # Si le nœud existe déjà (dans node_map), mettre à jour sa demande
                    if nid in node_map:
                        data['demand'][node_map[nid]] = d
                    else:
                        # Sinon, stocker temporairement comme tuple (id, demand)
                        data['demand'].append((nid, d))
                except: pass
        
        # Section TIME_WINDOW_SECTION: Format "id ready due"
        elif section == 'time_windows':
            p = line.split()
            if len(p) >= 3:
                try:
                    nid, r, du = int(p[0]), float(p[1]), float(p[2])
                    # Mettre à jour la fenêtre temporelle si le nœud existe
                    if nid in node_map:
                        data['time_window'][node_map[nid]] = [r, du]
                except: pass
        
        # Section DEPOT_SECTION: L'ID du dépôt
        elif section == 'depot' and line != '-1':
            try: 
                data['depot'] = int(line)  # ID du dépôt
            except: pass
        
        # Section VEHICLE: Extraire la capacité depuis la ligne suivante
        elif section == 'vehicle' and 'CAPACITY' in lu:
            idx = lines.index(line) + 1
            if idx < len(lines):
                p = lines[idx].split()
                if len(p) >= 2: 
                    data['capacity'] = int(p[1])  # Capacité du véhicule
    
    # Service time global
    if '_global_st' in data:
        data['service_time'] = [data.pop('_global_st')] * len(data['node_coord'])
    
    # Nom par défaut
    if not data['name']: data['name'] = lines[0] if lines else ''
    if not data['capacity']: data['capacity'] = 200
    
    # Convertir demandes si format (id, demand)
    if data['demand'] and isinstance(data['demand'][0], tuple):
        demand_dict = {id: d for id, d in data['demand']}
        data['demand'] = [demand_dict.get(i, 0) for i in range(len(data['node_coord']))]
    
    # Nettoyer TW et service_time
    if data['time_window'] and all(tw[1] == float('inf') for tw in data['time_window']):
        data.pop('time_window', None)
    if data['service_time'] and all(st == 0.0 for st in data['service_time']):
        data.pop('service_time', None)
    
    return data


# ======================================================================
# Représentation d'une solution VRP avec TW
# ======================================================================

class Solution:
    """
    Solution VRP (routes + coût + faisabilité).

    Attributs principaux
    --------------------
    routes : List[List[int]]
        Routes, chaque route liste des indices clients (1..n). L'indice 0 est le dépôt.
    distance_matrix : np.ndarray
        Distances entre tous les nœuds (0..n).
    demands : np.ndarray
        Demande de chaque nœud (0 pour le dépôt).
    capacity : int
        Capacité du véhicule.
    depot : int
        Index du dépôt (0 par défaut).
    time_windows : Optional[List[Tuple[float, float]]]
        Fenêtres temporelles TW[i] = (ready, due) pour chaque nœud (0 = dépôt).
    service_times : Optional[List[float]]
        Temps de service sur chaque nœud (0 pour le dépôt).
    penalty_late : float
        Pénalité appliquée par unité de temps si la TW n'est pas respectée.
    cost : float
        Coût (distance + pénalités).
    tw_report : Dict[int, Dict[str, Any]]
        Rapport détaillé TW par client (arrival, start_service, finish, respected, ...).
    """

    __slots__ = [
        "routes",
        "distance_matrix",
        "demands",
        "capacity",
        "depot",
        "time_windows",
        "service_times",
        "penalty_late",
        "cost",
        "tw_report",
    ]

    def __init__(
        self,
        routes: List[List[int]],
        distance_matrix: np.ndarray,
        demands: np.ndarray,
        capacity: int,
        depot: int = 0,
        time_windows: Optional[List[Tuple[float, float]]] = None,
        service_times: Optional[List[float]] = None,
        penalty_late: float = 1e6,
    ) -> None:
        self.routes = deepcopy(routes)
        self.distance_matrix = distance_matrix
        self.demands = demands
        self.capacity = capacity
        self.depot = depot
        self.time_windows = time_windows
        self.service_times = service_times
        self.penalty_late = penalty_late

        self.tw_report: Dict[int, Dict[str, Any]] = {}
        self.cost = self.calculate_cost()

    # ------------------------------------------------------------------
    def copy(self) -> "Solution":
        """Copie profonde de la solution."""
        return Solution(
            routes=self.routes,
            distance_matrix=self.distance_matrix,
            demands=self.demands,
            capacity=self.capacity,
            depot=self.depot,
            time_windows=self.time_windows,
            service_times=self.service_times,
            penalty_late=self.penalty_late,
        )

    def route_load(self, route_idx: int) -> int:
        """Charge totale d'une route."""
        if route_idx < 0 or route_idx >= len(self.routes):
            return 0
        return int(sum(self.demands[c] for c in self.routes[route_idx]))

    def is_feasible_capacity(self) -> bool:
        """Toutes les routes respectent la capacité ?"""
        for r in self.routes:
            if sum(self.demands[c] for c in r) > self.capacity:
                return False
        return True

    def is_feasible_time_windows(self) -> bool:
        """Toutes les fenêtres temporelles sont respectées ?"""
        if not self.time_windows:
            return True
        for info in self.tw_report.values():
            if info.get("respected") is False:
                return False
        return True

    def is_feasible(self) -> bool:
        """Faisabilité globale (capacité + TW)."""
        return self.is_feasible_capacity() and self.is_feasible_time_windows()

    # ------------------------------------------------------------------
    def _evaluate_tw_for_route(
        self, route: List[int]
    ) -> Tuple[float, Dict[int, Dict[str, Any]]]:
        """
        Évalue les fenêtres temporelles pour une route complète.
        
        Cette méthode simule le parcours d'une route et vérifie si toutes
        les fenêtres temporelles sont respectées. Elle calcule pour chaque client:
        - L'heure d'arrivée
        - L'heure de début de service (peut attendre si arrivée avant ready)
        - L'heure de fin de service
        - Si la fenêtre est respectée (arrival <= due)
        
        Contraintes :
        - Interdiction de livrer hors de la fenêtre (arrivée > due = infaisable)
        - Possibilité d'attendre sur place l'ouverture de la fenêtre
        - Pas de pénalité dans le coût ici, juste vérification de faisabilité
        
        Args:
            route: Liste des indices clients dans l'ordre de visite
            
        Returns:
            Tuple (0.0, report) où report est un dictionnaire avec pour chaque client:
            - "arrival": heure d'arrivée
            - "ready": heure d'ouverture de la fenêtre
            - "due": heure de fermeture de la fenêtre
            - "start_service": heure de début de service (max(arrival, ready))
            - "finish": heure de fin de service
            - "respected": bool indiquant si arrival <= due
            - "wait_time": temps d'attente si arrivée avant ready
        """
        report: Dict[int, Dict[str, Any]] = {}
        # Si pas de fenêtres temporelles, tout est faisable
        if not self.time_windows:
            return 0.0, report

        # Initialiser: partir du dépôt à t=0
        time_current = 0.0  # Temps actuel (heure de fin de service précédent)
        current = self.depot  # Position actuelle (dépôt au début)

        # Parcourir chaque client de la route dans l'ordre
        for c in route:
            # ============================================================
            # CALCULER L'HEURE D'ARRIVÉE AU CLIENT
            # ============================================================
            # Temps de trajet depuis le client précédent (ou dépôt)
            travel = self.distance_matrix[current, c]
            arrival = time_current + travel  # Heure d'arrivée au client

            # ============================================================
            # RÉCUPÉRER LES CONTRAINTES TEMPORELLES DU CLIENT
            # ============================================================
            ready, due = self.time_windows[c]  # [ready_time, due_time]
            service = self.service_times[c] if self.service_times is not None else 0.0

            # ============================================================
            # CALCULER L'HEURE DE DÉBUT DE SERVICE
            # ============================================================
            # Possibilité d'attendre sur place l'ouverture de la fenêtre
            # Si on arrive avant ready, on attend jusqu'à ready
            start_service = max(arrival, ready)
            
            # ============================================================
            # VÉRIFIER SI LA FENÊTRE EST RESPECTÉE
            # ============================================================
            # Interdiction de livrer hors de la fenêtre
            # Si arrival > due, c'est infaisable (violation)
            respected = arrival <= due

            # ============================================================
            # CALCULER L'HEURE DE FIN DE SERVICE
            # ============================================================
            finish = start_service + service  # Fin = début + temps service

            # ============================================================
            # STOCKER LE RAPPORT POUR CE CLIENT
            # ============================================================
            report[c] = {
                "arrival": arrival,  # Heure d'arrivée
                "ready": ready,  # Heure d'ouverture fenêtre
                "due": due,  # Heure de fermeture fenêtre
                "start_service": start_service,  # Heure début service
                "finish": finish,  # Heure fin service
                "respected": respected,  # Fenêtre respectée ?
                "wait_time": max(0.0, ready - arrival) if arrival < ready else 0.0,  # Temps d'attente
            }

            # Mettre à jour pour le client suivant
            time_current = finish  # On repart après avoir fini le service
            current = c  # On est maintenant à la position de ce client

        # Retour au dépôt (sans vérification TW car dépôt n'a pas de contrainte)
        time_current += self.distance_matrix[current, self.depot]
        # Pas de pénalité dans le coût ici, juste rapport
        return 0.0, report

    def calculate_cost(self) -> float:
        """
        Calcule le coût total de la solution.
        
        Le coût comprend:
        1. La distance totale parcourue par tous les véhicules
        2. Les pénalités pour violations de fenêtres temporelles
        
        Cette méthode met aussi à jour self.tw_report avec les détails
        de chaque client (arrivée, respect des TW, etc.)
        
        Returns:
            Coût total (distance + pénalités)
        """
        total = 0.0  # Distance totale
        self.tw_report = {}  # Réinitialiser le rapport TW
        penalty = 0.0  # Pénalités pour violations TW

        # Parcourir chaque route de la solution
        for route in self.routes:
            if not route:  # Ignorer les routes vides
                continue

            # ============================================================
            # CALCULER LA DISTANCE DE LA ROUTE
            # ============================================================
            # Distance dépôt -> premier client
            total += self.distance_matrix[self.depot, route[0]]
            # Distances entre clients consécutifs
            for i in range(len(route) - 1):
                total += self.distance_matrix[route[i], route[i + 1]]
            # Distance dernier client -> dépôt
            total += self.distance_matrix[route[-1], self.depot]

            # ============================================================
            # ÉVALUER LES FENÊTRES TEMPORELLES ET CALCULER PÉNALITÉS
            # ============================================================
            if self.time_windows is not None:
                # Évaluer la route et obtenir le rapport détaillé
                _, rep = self._evaluate_tw_for_route(route)
                self.tw_report.update(rep)  # Stocker le rapport
                
                # Ajouter pénalité pour chaque violation TW
                for c, info in rep.items():
                    if not info.get("respected", True):
                        # Pénalité proportionnelle au retard
                        # Si on arrive après due, pénalité = penalty_late * (arrival - due)
                        arrival = info.get("arrival", 0.0)
                        due = info.get("due", float('inf'))
                        if arrival > due:
                            # Plus le retard est grand, plus la pénalité est élevée
                            penalty += self.penalty_late * (arrival - due)

        # Coût total = distance + pénalités
        self.cost = float(total + penalty)
        return self.cost

    def __repr__(self) -> str:
        return (
            f"Solution(routes={self.routes}, cost={self.cost:.2f}, "
            f"feasible_cap={self.is_feasible_capacity()}, "
            f"feasible_tw={self.is_feasible_time_windows()})"
        )


# ======================================================================
# Recherche locale : 2-opt + relocate
# ======================================================================

class LocalSearch:
    """Optimisation locale : 2-opt, relocate, swap (comme dans le code fourni)."""

    def __init__(
        self, 
        distance_matrix: np.ndarray, 
        demands: np.ndarray, 
        capacity: int,
        time_windows: Optional[List[Tuple[float, float]]] = None,
        service_times: Optional[List[float]] = None,
        depot: int = 0,
        n_customers: Optional[int] = None
    ) -> None:
        self.distance_matrix = distance_matrix
        self.demands = demands
        self.capacity = capacity
        self.time_windows = time_windows
        self.service_times = service_times
        self.depot = depot
        # Calculer n_customers si non fourni (len(demands) - 1 car le dépôt est à l'index 0)
        self.n_customers = n_customers if n_customers is not None else len(demands) - 1

    def _route_load(self, route: List[int]) -> int:
        return int(sum(self.demands[c] for c in route))
    
    def _is_route_feasible_tw(self, route: List[int]) -> bool:
        """Vérifie si une route respecte les fenêtres temporelles."""
        if not self.time_windows or not route:
            return True
        
        time_current = 0.0
        current = self.depot
        
        for c in route:
            travel = self.distance_matrix[current, c]
            arrival = time_current + travel
            ready, due = self.time_windows[c]
            
            # Si on arrive après la fenêtre, la route n'est pas faisable
            if arrival > due:
                return False
            
            service = self.service_times[c] if self.service_times else 0.0
            start_service = max(arrival, ready)
            finish = start_service + service
            time_current = finish
            current = c
        
        return True
    
    def _is_insertion_feasible_tw(self, route: List[int], customer: int, pos: int) -> bool:
        """Vérifie si l'insertion de customer à la position pos dans route est faisable au niveau TW."""
        if not self.time_windows:
            return True
        
        # Créer la nouvelle route avec insertion
        new_route = route[:pos] + [customer] + route[pos:]
        return self._is_route_feasible_tw(new_route)

    def two_opt_intra(self, solution: Solution) -> bool:
        """
        Opérateur 2-opt au sein de chaque route.
        
        L'algorithme 2-opt améliore une route en inversant un segment.
        Principe:
        - On prend deux arêtes (i->i+1) et (j->j+1) dans la route
        - On les remplace par (i->j) et (i+1->j+1) en inversant le segment entre i+1 et j
        - Si cela réduit la distance, on accepte le changement
        
        Exemple: Route [A, B, C, D, E]
        - Si on prend i=0 (A->B) et j=2 (C->D)
        - Nouvelle route: [A, C, B, D, E] (segment B,C inversé)
        
        Optimisé pour grandes instances: limite la recherche pour performance.
        
        Args:
            solution: Solution à améliorer
            
        Returns:
            True si une amélioration a été trouvée, False sinon
        """
        improved = False
        # Détecter si on a de grandes routes (plus de 20 clients)
        is_large = len(solution.routes) > 0 and any(len(r) > 20 for r in solution.routes)
        max_search = 5 if is_large else 15  # Limiter la recherche pour grandes instances
        
        # Parcourir chaque route
        for route_idx, route in enumerate(solution.routes):
            if len(route) < 4:  # 2-opt nécessite au moins 4 clients
                continue
            
            # Limiter la recherche pour grandes routes (performance)
            max_i = min(len(route) - 2, 20 if is_large else len(route) - 2)
            # Tester toutes les paires d'arêtes (i, j) avec i < j
            for i in range(max_i):
                max_j = min(i + max_search, len(route))  # Limiter j pour performance
                for j in range(i + 2, max_j):  # j doit être au moins i+2
                    # ============================================================
                    # CALCULER LE GAIN DE DISTANCE
                    # ============================================================
                    # Distance actuelle: arêtes (i->i+1) et (j->j+1 ou j->dépôt)
                    if j == len(route) - 1:  # j est le dernier client
                        # Cas spécial: dernière arête va au dépôt
                        current = (self.distance_matrix[route[i], route[i + 1]] + 
                                 self.distance_matrix[route[j], 0])
                        new = (self.distance_matrix[route[i], route[j]] + 
                              self.distance_matrix[route[i + 1], 0])
                    else:
                        # Cas général: deux arêtes internes
                        current = (self.distance_matrix[route[i], route[i + 1]] + 
                                 self.distance_matrix[route[j], route[j + 1]])
                        new = (self.distance_matrix[route[i], route[j]] + 
                              self.distance_matrix[route[i + 1], route[j + 1]])
                    
                    # ============================================================
                    # ACCEPTER SI AMÉLIORATION
                    # ============================================================
                    if new < current - 0.001:  # Seuil pour éviter erreurs numériques
                        # Créer la nouvelle route avec inversion du segment [i+1, j]
                        # Exemple: route=[A,B,C,D,E], i=0, j=2
                        # new_route = [A] + [C,B] (inversé) + [D,E] = [A,C,B,D,E]
                        new_route = route[:i + 1] + list(reversed(route[i + 1:j + 1])) + route[j + 1:]
                        
                        # Vérifier faisabilité TW avant d'accepter
                        if self._is_route_feasible_tw(new_route):
                            solution.routes[route_idx] = new_route
                            improved = True
                            solution.calculate_cost()  # Recalculer le coût
                            return improved  # Retour immédiat pour performance
        
        return improved

    def two_opt(self, route: List[int]) -> Tuple[List[int], bool]:
        """2-opt pour une route (version compatible avec l'ancien code)."""
        if len(route) < 4:
            return route, False
        # Créer une solution temporaire pour utiliser two_opt_intra
        temp_sol = Solution([route], self.distance_matrix, self.demands, self.capacity)
        improved = self.two_opt_intra(temp_sol)
        if improved:
            return temp_sol.routes[0], True
        return route, False

    def relocate(self, solution: Solution) -> bool:
        """Relocalise un client - optimisé pour grandes instances."""
        improved = False
        is_large = self.n_customers > 100
        
        # Limiter le nombre de routes à considérer pour grandes instances
        max_routes = min(len(solution.routes), 10 if is_large else len(solution.routes))
        
        for i in range(max_routes):
            route_i = solution.routes[i]
            if len(route_i) == 0:
                continue
            
            # Limiter les positions à considérer
            max_pos_i = min(len(route_i), 5 if is_large else len(route_i))
            for pos_i in range(max_pos_i):
                customer = route_i[pos_i]
                
                # Coût avant retrait
                cost_before = self._get_removal_cost(route_i, pos_i)
                
                # Limiter les routes cibles
                max_routes_j = min(len(solution.routes), 10 if is_large else len(solution.routes))
                for j in range(max_routes_j):
                    if i == j:
                        continue
                    
                    route_j = solution.routes[j]
                    
                    # Vérifier la capacité
                    demand_j = sum(self.demands[c] for c in route_j)
                    if demand_j + self.demands[customer] > self.capacity:
                        continue
                    
                    # Trouver la meilleure position faisable dans route_j
                    best_pos_j = None
                    best_cost_after = float('inf')
                    
                    # Limiter les positions à tester
                    max_pos_j = min(len(route_j) + 1, 5 if is_large else len(route_j) + 1)
                    for pos_j in range(max_pos_j):
                        # Vérifier faisabilité TW avant de considérer cette position
                        if not self._is_insertion_feasible_tw(route_j, customer, pos_j):
                            continue
                        
                        cost = self._get_insertion_cost(route_j, customer, pos_j)
                        if cost < best_cost_after:
                            best_cost_after = cost
                            best_pos_j = pos_j
                    
                    # Si amélioration et position faisable trouvée
                    if best_pos_j is not None and cost_before + best_cost_after < -0.001:
                        route_i.pop(pos_i)
                        route_j.insert(best_pos_j, customer)
                        solution.calculate_cost()
                        improved = True
                        return improved  # Retour immédiat pour performance
        
        return improved

    def swap(self, solution: Solution) -> bool:
        """Échange de clients - optimisé pour grandes instances."""
        improved = False
        is_large = self.n_customers > 100
        
        # Limiter le nombre de routes
        max_routes = min(len(solution.routes), 5 if is_large else len(solution.routes))
        
        for i in range(max_routes):
            for j in range(i + 1, max_routes):
                route_i = solution.routes[i]
                route_j = solution.routes[j]
                
                if len(route_i) == 0 or len(route_j) == 0:
                    continue
                
                # Limiter les positions
                max_pos_i = min(len(route_i), 3 if is_large else len(route_i))
                max_pos_j = min(len(route_j), 3 if is_large else len(route_j))
                for pos_i in range(max_pos_i):
                    for pos_j in range(max_pos_j):
                        customer_i = route_i[pos_i]
                        customer_j = route_j[pos_j]
                        
                        # Vérifier les contraintes de capacité
                        demand_i = sum(self.demands[c] for c in route_i)
                        demand_j = sum(self.demands[c] for c in route_j)
                        
                        new_demand_i = demand_i - self.demands[customer_i] + self.demands[customer_j]
                        new_demand_j = demand_j - self.demands[customer_j] + self.demands[customer_i]
                        
                        if new_demand_i > self.capacity or new_demand_j > self.capacity:
                            continue
                        
                        # Créer les nouvelles routes avec échange
                        new_route_i = route_i[:pos_i] + [customer_j] + route_i[pos_i + 1:]
                        new_route_j = route_j[:pos_j] + [customer_i] + route_j[pos_j + 1:]
                        
                        # Vérifier faisabilité TW avant d'accepter
                        if not self._is_route_feasible_tw(new_route_i) or not self._is_route_feasible_tw(new_route_j):
                            continue
                        
                        # Calculer le coût avant
                        cost_before = solution.cost
                        
                        # Échanger
                        route_i[pos_i], route_j[pos_j] = route_j[pos_j], route_i[pos_i]
                        new_cost = solution.calculate_cost()
                        
                        if new_cost < cost_before - 0.001:
                            improved = True
                            return improved  # Retour immédiat pour performance
                        else:
                            # Annuler
                            route_i[pos_i], route_j[pos_j] = route_j[pos_j], route_i[pos_i]
                            solution.cost = cost_before
        
        return improved

    def _get_removal_cost(self, route: List[int], pos: int) -> float:
        """Calcule le coût de retrait d'un client (comme dans le code fourni)."""
        if len(route) == 1:
            return -(self.distance_matrix[0, route[0]] * 2)
        elif pos == 0:
            return -(self.distance_matrix[0, route[0]] + 
                    self.distance_matrix[route[0], route[1]]) + \
                    self.distance_matrix[0, route[1]]
        elif pos == len(route) - 1:
            return -(self.distance_matrix[route[-2], route[-1]] + 
                    self.distance_matrix[route[-1], 0]) + \
                    self.distance_matrix[route[-2], 0]
        else:
            return -(self.distance_matrix[route[pos-1], route[pos]] + 
                    self.distance_matrix[route[pos], route[pos+1]]) + \
                    self.distance_matrix[route[pos-1], route[pos+1]]

    def _get_insertion_cost(self, route: List[int], customer: int, pos: int) -> float:
        """Calcule le coût d'insertion d'un client (comme dans le code fourni)."""
        if len(route) == 0:
            return self.distance_matrix[0, customer] * 2
        elif pos == 0:
            return (self.distance_matrix[0, customer] + 
                   self.distance_matrix[customer, route[0]] - 
                   self.distance_matrix[0, route[0]])
        elif pos == len(route):
            return (self.distance_matrix[route[-1], customer] + 
                   self.distance_matrix[customer, 0] - 
                   self.distance_matrix[route[-1], 0])
        else:
            return (self.distance_matrix[route[pos-1], customer] + 
                   self.distance_matrix[customer, route[pos]] - 
                   self.distance_matrix[route[pos-1], route[pos]])

    def optimize(self, solution: Solution, max_iterations: int = 30) -> Tuple[Solution, bool]:
        """
        Applique les optimisations locales - adapté à la taille de l'instance.
        """
        initial_cost = solution.cost
        is_large = self.n_customers > 100
        improved = True
        iterations = 0
        
        # Réduire drastiquement les itérations pour grandes instances
        max_iter = min(max_iterations, 5 if is_large else max_iterations)
        
        while improved and iterations < max_iter:
            improved = False
            iterations += 1
            
            # Pour grandes instances, utiliser seulement les opérateurs rapides
            if is_large:
                if self.two_opt_intra(solution):
                    improved = True
                    continue
                if self.relocate(solution):
                    improved = True
                    continue
            else:
                # Pour petites instances, utiliser tous les opérateurs
                if self.two_opt_intra(solution):
                    improved = True
                if self.relocate(solution):
                    improved = True
                if self.swap(solution):
                    improved = True
        
        solution.calculate_cost()
        final_cost = solution.cost
        was_improved = final_cost < initial_cost - 1e-9
        return solution, was_improved

    def intensive_optimize(self, solution: Solution, max_passes: int = 5) -> Tuple[Solution, bool]:
        """
        Optimisation intensive - adaptée à la taille de l'instance.
        """
        initial_cost = solution.cost
        is_large = self.n_customers > 100
        best_sol = solution.copy()
        any_improvement = False
        
        # Réduire drastiquement les passes pour grandes instances
        max_p = min(max_passes, 1 if is_large else max_passes)
        
        for pass_num in range(max_p):
            improved = False
            
            if is_large:
                # Pour grandes instances : seulement opérateurs rapides, une seule passe
                if self.two_opt_intra(best_sol):
                    improved = True
                if self.relocate(best_sol):
                    improved = True
            else:
                # Pour petites instances : tous les opérateurs
                if self.two_opt_intra(best_sol):
                    improved = True
                if self.relocate(best_sol):
                    improved = True
                if self.swap(best_sol):
                    improved = True
            
            if improved:
                any_improvement = True
            else:
                break
        
        final_cost = best_sol.cost
        was_improved = final_cost < initial_cost - 1e-9 or any_improvement
        return best_sol, was_improved


# ======================================================================
# ALNS VRP + TW
# ======================================================================

class ALNSSolver:
    """
    ALNS pour VRP capacitaire, avec possibilité de pénaliser les fenêtres temporelles.
    """

    def __init__(
        self,
        instance_path: Optional[str] = None,
        coords: Optional[np.ndarray] = None,
        demands: Optional[np.ndarray] = None,
        capacity: Optional[int] = None,
        name: Optional[str] = None,
        time_windows: Optional[List[Tuple[float, float]]] = None,
        service_times: Optional[List[float]] = None,
        penalty_late: float = 1e6,
    ) -> None:
        # Chargement depuis fichier VRP
        if instance_path is not None:
            # Utiliser notre parser pour tous les fichiers
            try:
                inst = parse_vrp_file(instance_path)
            except Exception as e:
                # Fallback sur vrplib seulement pour .vrp et .vrptw
                if instance_path.lower().endswith(('.vrp', '.vrptw')):
                    try:
                        inst = vrplib.read_instance(instance_path)
                    except:
                        raise ValueError(f"Impossible de parser l'instance {instance_path}: {e}")
                else:
                    raise ValueError(f"Impossible de parser l'instance {instance_path}: {e}")
            
            self.name = instance_path.split("/")[-1].split("\\")[-1]
            # Enlever toutes les extensions
            while '.' in self.name:
                self.name = self.name.rsplit('.', 1)[0]
            
            self.coords = np.array(inst["node_coord"], dtype=float)
            self.demands = np.array(inst.get("demand", [0] * len(self.coords)), dtype=int)
            self.capacity = int(inst.get("capacity", 10**9))
            
            # Charger fenêtres temporelles
            if time_windows is None:
                tw = inst.get("time_window") or inst.get("time_windows")
                if tw and isinstance(tw, list) and len(tw) > 0:
                    if isinstance(tw[0], (list, tuple)) and len(tw[0]) >= 2:
                        time_windows = [(float(t[0]), float(t[1])) for t in tw]
                        if all(t[1] == float('inf') for t in time_windows):
                            time_windows = None
                    else:
                        time_windows = None
                else:
                    time_windows = None
            
            # Charger temps de service
            if service_times is None:
                st = inst.get("service_time") or inst.get("service_times")
                if st and isinstance(st, list) and len(st) > 0:
                    service_times = [float(s) for s in st]
                    if all(s == 0.0 for s in service_times):
                        service_times = None
                else:
                    service_times = None
        else:
            # Mode manuel
            if coords is None or demands is None or capacity is None:
                raise ValueError("coords, demands, capacity requis pour le mode manuel.")
            self.name = name or "instance_manuel"
            self.coords = np.array(coords, dtype=float)
            self.demands = np.array(demands, dtype=int)
            self.capacity = int(capacity)

        self.n_customers = len(self.coords) - 1
        self.distance_matrix = self._compute_distances(self.coords)

        # Fenêtres temporelles
        self.time_windows = time_windows
        self.service_times = service_times
        self.penalty_late = penalty_late

        self.ls = LocalSearch(
            self.distance_matrix, 
            self.demands, 
            self.capacity,
            time_windows=self.time_windows,
            service_times=self.service_times,
            depot=0,
            n_customers=self.n_customers
        )

        self.history = {
            "iterations": [],
            "best_costs": [],
            "current_costs": [],
            "times": [],
        }
        self._t0: Optional[float] = None

        # Recuit simulé - paramètres ajustés pour meilleure exploration
        self.T = 15000.0  # Température initiale plus élevée
        self.alpha = 0.998  # Refroidissement plus lent (était 0.995)
        self.T_min = 0.1

    # ------------------------------------------------------------------
    @staticmethod
    def _compute_distances(coords: np.ndarray) -> np.ndarray:
        n = len(coords)
        D = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                d = float(
                    int(round(math.hypot(coords[i, 0] - coords[j, 0], coords[i, 1] - coords[j, 1])))
                )
                D[i, j] = D[j, i] = d
        return D

    # ------------------------------------------------------------------
    # Constructions initiales
    def _construct_savings(self) -> Solution:
        routes = [[i] for i in range(1, self.n_customers + 1)]
        savings = []
        for i in range(1, self.n_customers + 1):
            for j in range(i + 1, self.n_customers + 1):
                s = self.distance_matrix[0, i] + self.distance_matrix[0, j] - self.distance_matrix[i, j]
                savings.append((s, i, j))
        savings.sort(reverse=True)
        owner = {c: idx for idx, r in enumerate(routes) for c in r}
        for s, i, j in savings:
            ri, rj = owner.get(i), owner.get(j)
            if ri is None or rj is None or ri == rj:
                continue
            A, B = routes[ri], routes[rj]
            load = sum(self.demands[c] for c in A) + sum(self.demands[c] for c in B)
            if load > self.capacity:
                continue
            if A[-1] == i and B[0] == j:
                routes[ri] = A + B
                routes[rj] = []
                for c in B:
                    owner[c] = ri
            elif B[-1] == j and A[0] == i:
                routes[rj] = B + A
                routes[ri] = []
                for c in A:
                    owner[c] = rj
        routes = [r for r in routes if r]
        return Solution(
            routes,
            self.distance_matrix,
            self.demands,
            self.capacity,
            depot=0,
            time_windows=self.time_windows,
            service_times=self.service_times,
            penalty_late=self.penalty_late,
        )

    def _construct_nearest_neighbor(self) -> Solution:
        customers = list(range(1, self.n_customers + 1))
        random.shuffle(customers)
        remaining = set(customers)
        routes: List[List[int]] = []
        while remaining:
            first = remaining.pop()
            route = [first]
            load = self.demands[first]
            cur = first
            while remaining:
                best, bestd = None, float("inf")
                for c in list(remaining):
                    if load + self.demands[c] > self.capacity:
                        continue
                    # Vérifier faisabilité TW si applicable
                    if self.time_windows:
                        feasible, _ = self._is_insertion_feasible_tw(route, c, len(route))
                        if not feasible:
                            continue
                    d = self.distance_matrix[cur, c]
                    if d < bestd:
                        best, bestd = c, d
                if best is None:
                    break
                route.append(best)
                remaining.remove(best)
                load += self.demands[best]
                cur = best
            routes.append(route)
        return Solution(
            routes,
            self.distance_matrix,
            self.demands,
            self.capacity,
            depot=0,
            time_windows=self.time_windows,
            service_times=self.service_times,
            penalty_late=self.penalty_late,
        )

    def _initial_solution(self) -> Solution:
        """Génère une solution initiale rapidement, adaptée à la taille."""
        is_large = self.n_customers > 100
        
        if is_large:
            # Pour grandes instances : solution simple et rapide
            solutions = []
            solutions.append(self._construct_savings())
            solutions.append(self._construct_nearest_neighbor())
            best_init = min(solutions, key=lambda s: s.cost)
            # Optimisation minimale
            best_init, _ = self.ls.optimize(best_init, max_iterations=2)
        else:
            # Pour petites instances : solution de qualité
            solutions = []
            solutions.append(self._construct_savings())
            for _ in range(3):
                solutions.append(self._construct_nearest_neighbor())
            best_init = min(solutions, key=lambda s: s.cost)
            best_init, _ = self.ls.optimize(best_init, max_iterations=15)
            # Essayer aussi d'intensifier les autres bonnes solutions
            solutions.sort(key=lambda s: s.cost)
            for sol in solutions[:2]:  # Top 2 seulement
                improved, _ = self.ls.optimize(sol, max_iterations=10)
                if improved.cost < best_init.cost:
                    best_init = improved
        return best_init

    # ------------------------------------------------------------------
    # Opérateurs ALNS
    def _random_removal(self, sol: Solution, n: int) -> List[int]:
        allc = [c for r in sol.routes for c in r]
        n = min(n, len(allc))
        removed = random.sample(allc, n)
        for k in range(len(sol.routes)):
            sol.routes[k] = [c for c in sol.routes[k] if c not in removed]
        return removed

    def _worst_removal(self, sol: Solution, n: int) -> List[int]:
        contrib = []
        for r in sol.routes:
            for p, c in enumerate(r):
                if len(r) == 1:
                    cost = self.distance_matrix[0, c] * 2
                elif p == 0:
                    cost = (
                        self.distance_matrix[0, c]
                        + self.distance_matrix[c, r[1]]
                        - self.distance_matrix[0, r[1]]
                    )
                elif p == len(r) - 1:
                    cost = (
                        self.distance_matrix[r[-2], c]
                        + self.distance_matrix[c, 0]
                        - self.distance_matrix[r[-2], 0]
                    )
                else:
                    cost = (
                        self.distance_matrix[r[p - 1], c]
                        + self.distance_matrix[c, r[p + 1]]
                        - self.distance_matrix[r[p - 1], r[p + 1]]
                    )
                contrib.append((cost, c))
        contrib.sort(reverse=True)
        removed = [c for _, c in contrib[:n]]
        for k in range(len(sol.routes)):
            sol.routes[k] = [c for c in sol.routes[k] if c not in removed]
        return removed

    def _shaw_removal(self, sol: Solution, n: int) -> List[int]:
        """Retire clients similaires (basé sur distance et demande)."""
        allc = [c for r in sol.routes for c in r]
        if not allc:
            return []
        n = min(n, len(allc))
        seed = random.choice(allc)
        removed = [seed]
        similarities = []
        for c in allc:
            if c != seed:
                dist = self.distance_matrix[seed, c]
                demand_diff = abs(self.demands[seed] - self.demands[c])
                sim = dist + demand_diff * 0.1
                similarities.append((sim, c))
        similarities.sort()
        for i in range(min(n - 1, len(similarities))):
            removed.append(similarities[i][1])
        for k in range(len(sol.routes)):
            sol.routes[k] = [c for c in sol.routes[k] if c not in removed]
        return removed

    def _related_removal(self, sol: Solution, n: int) -> List[int]:
        """Retire clients proches géographiquement."""
        allc = [c for r in sol.routes for c in r]
        if not allc:
            return []
        n = min(n, len(allc))
        seed = random.choice(allc)
        removed = [seed]
        distances = [(self.distance_matrix[seed, c], c) for c in allc if c != seed]
        distances.sort()
        for i in range(min(n - 1, len(distances))):
            removed.append(distances[i][1])
        for k in range(len(sol.routes)):
            sol.routes[k] = [c for c in sol.routes[k] if c not in removed]
        return removed

    def _insertion_cost(self, route: List[int], c: int, pos: int) -> float:
        if not route:
            return self.distance_matrix[0, c] * 2
        if pos == 0:
            return (
                self.distance_matrix[0, c]
                + self.distance_matrix[c, route[0]]
                - self.distance_matrix[0, route[0]]
            )
        if pos == len(route):
            return (
                self.distance_matrix[route[-1], c]
                + self.distance_matrix[c, 0]
                - self.distance_matrix[route[-1], 0]
            )
        return (
            self.distance_matrix[route[pos - 1], c]
            + self.distance_matrix[c, route[pos]]
            - self.distance_matrix[route[pos - 1], route[pos]]
        )
    
    def _is_insertion_feasible_tw(self, route: List[int], c: int, pos: int) -> Tuple[bool, float]:
        """
        Vérifie si l'insertion de c à la position pos dans route est faisable au niveau TW.
        Retourne (faisable, temps_arrivée_estimé).
        """
        if not self.time_windows:
            return True, 0.0
        
        # Simuler le temps le long de la route jusqu'à pos
        time_current = 0.0
        current = 0  # dépôt
        
        # Parcourir la route jusqu'à pos
        for i in range(pos):
            client = route[i]
            travel = self.distance_matrix[current, client]
            arrival = time_current + travel
            ready, due = self.time_windows[client]
            service = self.service_times[client] if self.service_times else 0.0
            start_service = max(arrival, ready)
            finish = start_service + service
            time_current = finish
            current = client
        
        # Vérifier l'insertion de c
        travel_to_c = self.distance_matrix[current, c]
        arrival_c = time_current + travel_to_c
        ready_c, due_c = self.time_windows[c]
        
        # Vérifier si faisable
        if arrival_c > due_c:
            return False, arrival_c
        
        # Vérifier l'impact sur les clients suivants
        service_c = self.service_times[c] if self.service_times else 0.0
        start_service_c = max(arrival_c, ready_c)
        finish_c = start_service_c + service_c
        time_after_c = finish_c
        
        # Vérifier les clients suivants
        prev_client = c
        for i in range(pos, len(route)):
            client = route[i]
            travel = self.distance_matrix[prev_client, client]
            arrival = time_after_c + travel
            ready, due = self.time_windows[client]
            if arrival > due:
                return False, arrival
            service = self.service_times[client] if self.service_times else 0.0
            start_service = max(arrival, ready)
            finish = start_service + service
            time_after_c = finish
            prev_client = client
        
        return True, arrival_c

    def _greedy_repair(self, sol: Solution, customers: List[int]) -> None:
        for c in customers:
            best = (float("inf"), None, None)
            for ridx, r in enumerate(sol.routes):
                load = sum(self.demands[i] for i in r)
                if load + self.demands[c] > self.capacity:
                    continue
                for pos in range(len(r) + 1):
                    # Vérifier faisabilité TW si applicable
                    if self.time_windows:
                        feasible, _ = self._is_insertion_feasible_tw(r, c, pos)
                        if not feasible:
                            continue
                    delta = self._insertion_cost(r, c, pos)
                    if delta < best[0]:
                        best = (delta, ridx, pos)
            if best[1] is None:
                sol.routes.append([c])
            else:
                sol.routes[best[1]].insert(best[2], c)

    def _regret2_repair(self, sol: Solution, customers: List[int]) -> None:
        remain = customers[:]
        while remain:
            best_c, best_choice, best_regret = None, None, -1.0
            for c in remain:
                cand = []
                for ridx, r in enumerate(sol.routes):
                    load = sum(self.demands[i] for i in r)
                    if load + self.demands[c] > self.capacity:
                        continue
                    for pos in range(len(r) + 1):
                        # Vérifier faisabilité TW si applicable
                        if self.time_windows:
                            feasible, _ = self._is_insertion_feasible_tw(r, c, pos)
                            if not feasible:
                                continue
                        cost = self._insertion_cost(r, c, pos)
                        cand.append((cost, ridx, pos))
                if not cand:
                    # Nouvelle route - vérifier TW si applicable
                    if self.time_windows:
                        ready_c, due_c = self.time_windows[c]
                        travel = self.distance_matrix[0, c]
                        if travel > due_c:
                            continue  # Impossible même dans une nouvelle route
                    cand = [(self.distance_matrix[0, c] * 2, -1, 0)]
                cand.sort(key=lambda x: x[0])
                regret = (cand[1][0] - cand[0][0]) if len(cand) > 1 else 0
                if regret > best_regret:
                    best_regret = regret
                    best_c = c
                    best_choice = cand[0]
            if best_c is None:
                # Aucun client faisable, forcer l'insertion du premier
                if remain:
                    best_c = remain[0]
                    sol.routes.append([best_c])
                    remain.remove(best_c)
                else:
                    break
            else:
                delta, ridx, pos = best_choice
                if ridx == -1:
                    sol.routes.append([best_c])
                else:
                    sol.routes[ridx].insert(pos, best_c)
                remain.remove(best_c)

    # ------------------------------------------------------------------
    def _log(self, it: int, best_cost: float, cur_cost: float) -> None:
        assert self._t0 is not None
        self.history["iterations"].append(it)
        self.history["best_costs"].append(best_cost)
        self.history["current_costs"].append(cur_cost)
        self.history["times"].append(time.time() - self._t0)

    def solve(
        self,
        max_iterations: int = 2000,
        log_callback: Optional[Callable[[int, float, float], None]] = None,
        print_every: int = 50,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> Solution:
        """
        Boucle principale ALNS.

        Parameters
        ----------
        max_iterations : int
            Nombre d'itérations.
        log_callback : callable
            Optionnel, appelé à chaque itération avec (it, best_cost, current_cost).
        print_every : int
            Fréquence d'affichage dans le terminal (0 = jamais).
        should_stop : callable
            Optionnel, retourne True pour demander un arrêt anticipé (bouton arrêt).
        """
        self._t0 = time.time()
        current = self._initial_solution()
        # Intensification initiale plus agressive
        best, _ = self.ls.optimize(current.copy(), max_iterations=50)  # Augmenté de 30 à 50
        # Double intensification pour grandes instances
        if self.n_customers > 100:
            best, _ = self.ls.intensive_optimize(best.copy(), max_passes=3)
        self._log(0, best.cost, current.cost)
        if log_callback:
            log_callback(0, best.cost, current.cost)
        print(
            f"[ALNS] Instance {self.name} - départ : "
            f"coût init={current.cost:.2f}, best={best.cost:.2f}"
        )

        last_intensification = 0  # Itération de la dernière intensification
        
        # ============================================================
        # BOUCLE PRINCIPALE ALNS
        # ============================================================
        # À chaque itération:
        # 1. DESTRUCTION: Retirer des clients de la solution actuelle
        # 2. RÉPARATION: Réinsérer les clients retirés
        # 3. INTENSIFICATION: Améliorer localement la nouvelle solution
        # 4. ACCEPTATION: Accepter/rejeter selon recuit simulé
        for it in range(1, max_iterations + 1):
            # Vérifier si arrêt demandé (bouton arrêt dans interface)
            if should_stop and should_stop():
                print(f"[ALNS] Arrêt demandé à l'itération {it}.")
                break

            # ============================================================
            # ÉTAPE 1: DÉTERMINER LE TAUX DE DESTRUCTION ADAPTATIF
            # ============================================================
            # Le pourcentage de destruction s'adapte selon:
            # - La taille de l'instance (grandes vs petites)
            # - Le temps depuis la dernière amélioration
            # Plus on n'a pas amélioré depuis longtemps, plus on détruit
            is_large = self.n_customers > 100
            no_improvement_count = it - last_intensification if last_intensification > 0 else 0
            
            # Pour grandes instances, réduire drastiquement le pourcentage de destruction
            # (sinon trop coûteux en temps)
            if is_large:
                if no_improvement_count < 50:
                    destroy_pct = 0.05  # Très faible pour grandes instances (5%)
                elif no_improvement_count < 150:
                    destroy_pct = 0.10  # 10%
                else:
                    destroy_pct = 0.15  # 15% maximum
            else:
                # Petites instances: on peut se permettre plus de destruction
                if no_improvement_count < 100:
                    destroy_pct = 0.15  # 15%
                elif no_improvement_count < 300:
                    destroy_pct = 0.25  # 25%
                else:
                    destroy_pct = 0.35  # 35% maximum
            
            # Calculer le nombre de clients à retirer
            n_remove = max(1, int(self.n_customers * destroy_pct))
            # Limiter le nombre maximum de clients à retirer pour grandes instances
            if is_large:
                n_remove = min(n_remove, 20)  # Maximum 20 clients pour grandes instances
            
            # Créer une copie de la solution actuelle pour la modifier
            cand = current.copy()
            
            # ============================================================
            # ÉTAPE 2: DESTRUCTION - RETIRER DES CLIENTS
            # ============================================================
            # Sélection probabiliste de l'opérateur de destruction
            # 4 opérateurs disponibles:
            # - random_removal: Retire n clients aléatoirement (exploration)
            # - worst_removal: Retire les n clients avec plus forte contribution coût (exploitation)
            # - shaw_removal: Retire des clients similaires (distance + demande)
            # - related_removal: Retire des clients proches géographiquement
            destroy_choice = random.random()
            if self.n_customers > 100:
                # Pour grandes instances, favoriser worst et shaw (plus efficaces)
                if destroy_choice < 0.2:
                    removed = self._random_removal(cand, n_remove)  # 20% random
                elif destroy_choice < 0.5:
                    removed = self._worst_removal(cand, n_remove)  # 30% worst
                elif destroy_choice < 0.8:
                    removed = self._shaw_removal(cand, n_remove)  # 30% shaw
                else:
                    removed = self._related_removal(cand, n_remove)  # 20% related
            else:
                # Pour petites instances, distribution équilibrée
                if destroy_choice < 0.25:
                    removed = self._random_removal(cand, n_remove)  # 25% random
                elif destroy_choice < 0.5:
                    removed = self._worst_removal(cand, n_remove)  # 25% worst
                elif destroy_choice < 0.75:
                    removed = self._shaw_removal(cand, n_remove)  # 25% shaw
                else:
                    removed = self._related_removal(cand, n_remove)  # 25% related
            
            # ============================================================
            # ÉTAPE 3: RÉPARATION - RÉINSÉRER LES CLIENTS RETIRÉS
            # ============================================================
            # Sélection probabiliste de l'opérateur de réparation
            # 2 opérateurs disponibles:
            # - greedy_repair: Réinsère chaque client à la meilleure position (rapide)
            # - regret2_repair: Réinsère d'abord les clients avec plus grand regret
            #   (regret = différence entre 2ème et 1ère meilleure insertion)
            #   Priorise les clients difficiles à placer
            repair_choice = random.random()
            if repair_choice < 0.7:  # 70% regret2 (meilleure qualité)
                self._regret2_repair(cand, removed)
            else:
                self._greedy_repair(cand, removed)  # 30% greedy (plus rapide)

            # INTENSIFICATION 1: Recherche locale adaptée à la taille
            is_large = self.n_customers > 100
            if is_large:
                # Grandes instances : optimisation très rare et minimale
                if it % 20 == 0:
                    cand, _ = self.ls.optimize(cand, max_iterations=2)
            else:
                # Petites instances : optimisation fréquente
                if it % 5 == 0:
                    cand, _ = self.ls.optimize(cand, max_iterations=10)

            # INTENSIFICATION 2: Intensification périodique adaptée
            intensification_interval = 100 if is_large else 50
            if it - last_intensification >= intensification_interval and it > 0:
                max_opt_iter = 5 if is_large else 30
                intensive_best, improved = self.ls.optimize(best.copy(), max_iterations=max_opt_iter)
                if improved:
                    improvement = best.cost - intensive_best.cost
                    best = intensive_best
                    print(
                        f"[ALNS] Intensification périodique it={it}: "
                        f"amélioration={improvement:.2f}, best={best.cost:.2f}"
                    )
                last_intensification = it

            # ============================================================
            # ÉTAPE 4: ACCEPTATION - RECUIT SIMULÉ
            # ============================================================
            # Calculer la différence de coût entre nouvelle et ancienne solution
            delta = cand.cost - current.cost  # Positif si pire, négatif si meilleure
            
            # Règle d'acceptation du recuit simulé:
            # - Toujours accepter si meilleure (delta < 0)
            # - Accepter une solution pire avec probabilité exp(-delta/T)
            #   Plus la température T est élevée, plus on accepte de mauvaises solutions
            #   Plus delta est grand (solution beaucoup pire), moins on accepte
            accept = (
                delta < -1e-9  # Toujours accepter si meilleure (seuil pour erreurs numériques)
                or (self.T > 0 and random.random() < math.exp(-delta / self.T))  # Accepter pire avec proba
            )

            if accept:
                # Accepter la nouvelle solution comme solution courante
                current = cand
                
                # ============================================================
                # NOUVELLE MEILLEURE SOLUTION TROUVÉE
                # ============================================================
                if current.cost + 1e-9 < best.cost:
                    # INTENSIFICATION 3: Intensification immédiate adaptée
                    # Quand on trouve une meilleure solution, on l'améliore encore plus
                    is_large = self.n_customers > 100
                    if is_large:
                        # Grandes instances : optimisation minimale (rapide)
                        best, _ = self.ls.optimize(current.copy(), max_iterations=3)
                    else:
                        # Petites instances : optimisation agressive (qualité)
                        best, _ = self.ls.optimize(current.copy(), max_iterations=30)
                        best, _ = self.ls.intensive_optimize(best.copy(), max_passes=2)
                    improvement = current.cost - best.cost
                    print(
                        f"[ALNS] Nouvelle meilleure solution it={it}: "
                        f"amélioration={improvement:.2f}, best={best.cost:.2f}"
                    )
                    last_intensification = it  # Mettre à jour le compteur

            # ============================================================
            # LOGGING ET AFFICHAGE
            # ============================================================
            # Enregistrer l'itération dans l'historique
            self._log(it, best.cost, current.cost)
            if log_callback:
                log_callback(it, best.cost, current.cost)  # Callback pour interface graphique

            # Afficher périodiquement dans le terminal
            if print_every and it % print_every == 0:
                assert self._t0 is not None
                print(
                    f"[ALNS] it={it} best={best.cost:.2f} current={current.cost:.2f} "
                    f"T={self.T:.2f} t={time.time()-self._t0:.1f}s"
                )

            # ============================================================
            # REFROIDISSEMENT - DIMINUER LA TEMPÉRATURE
            # ============================================================
            # À chaque itération, on refroidit (diminue T)
            # T = T * alpha où alpha < 1 (ex: 0.998)
            # Plus T diminue, moins on accepte de mauvaises solutions
            # Cela permet de converger vers une bonne solution
            self.T = max(self.T_min, self.T * self.alpha)  # Ne pas descendre en dessous de T_min

        # INTENSIFICATION 4: Post-optimisation finale adaptée à la taille
        if max_iterations > 0:
            is_large = self.n_customers > 100
            if is_large:
                # Grandes instances : post-optimisation minimale
                best, _ = self.ls.optimize(best.copy(), max_iterations=3)
            else:
                # Petites instances : post-optimisation agressive
                print(f"[ALNS] Démarrage post-optimisation intensive...")
                max_intensification_passes = 5
                min_improvement_threshold = 0.01
                consecutive_no_improvement = 0
                max_consecutive_no_improvement = 3
                
                for pass_num in range(max_intensification_passes):
                    initial_cost = best.cost
                    
                    # Pass 1: Optimisation standard
                    best, improved1 = self.ls.optimize(best.copy(), max_iterations=30)
                    
                    # Pass 2: Optimisation intensive si amélioration
                    if improved1:
                        best, improved2 = self.ls.intensive_optimize(best.copy(), max_passes=2)
                    else:
                        improved2 = False
                    
                    improvement = initial_cost - best.cost
                    
                    if improvement > min_improvement_threshold:
                        print(
                            f"[ALNS] Post-optimisation passe {pass_num + 1}: "
                            f"amélioration={improvement:.2f}, nouveau best={best.cost:.2f}"
                        )
                        consecutive_no_improvement = 0
                    else:
                        consecutive_no_improvement += 1
                        if pass_num > 0:
                            print(
                            f"[ALNS] Post-optimisation passe {pass_num + 1}: "
                            f"amélioration minime ({improvement:.2f}), continue..."
                        )
                    
                    # Arrêter si stagnation
                    if consecutive_no_improvement >= max_consecutive_no_improvement:
                        if pass_num > 0:
                            print(
                                f"[ALNS] Post-optimisation: arrêt après {pass_num + 1} passes "
                                f"(stagnation {consecutive_no_improvement} passes consécutives)"
                            )
                        break

        if self._t0:
            print(f"[ALNS] Fin - best={best.cost:.2f} temps={time.time()-self._t0:.1f}s")
        return best


# ======================================================================
# Fonctions utilitaires pour l'interface / scripts
# ======================================================================

def solve_vrplib_instance(
    instance_path: str,
    max_iterations: int = 500,
    log_callback: Optional[Callable[[int, float, float], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> Tuple[ALNSSolver, Solution]:
    """
    Résout une instance VRPLIB (.vrp) en mode automatique.
    """
    solver = ALNSSolver(instance_path=instance_path)
    best = solver.solve(
        max_iterations=max_iterations,
        log_callback=log_callback,
        should_stop=should_stop,
    )
    return solver, best


def solve_manual_instance(
    coords: np.ndarray,
    demands: np.ndarray,
    capacity: int,
    max_iterations: int = 500,
    name: str = "instance_manuel",
    time_windows: Optional[List[Tuple[float, float]]] = None,
    service_times: Optional[List[float]] = None,
    penalty_late: float = 1e6,
    log_callback: Optional[Callable[[int, float, float], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> Tuple[ALNSSolver, Solution]:
    """
    Résout une instance fournie manuellement (coords, demandes, capacité, TW).
    """
    solver = ALNSSolver(
        instance_path=None,
        coords=coords,
        demands=demands,
        capacity=capacity,
        name=name,
        time_windows=time_windows,
        service_times=service_times,
        penalty_late=penalty_late,
    )
    best = solver.solve(
        max_iterations=max_iterations,
        log_callback=log_callback,
        should_stop=should_stop,
    )
    return solver, best