"""Módulo PERT/CPM: tiempos esperados, holguras, ruta crítica y probabilidad.

Cada actividad es un diccionario:
    {"id": "A", "pred": [], "a": 2, "m": 4, "b": 6}   (PERT, tres tiempos)
    {"id": "A", "pred": [], "t": 4}                   (CPM, tiempo fijo)
"""
import networkx as nx
from scipy.stats import norm


def analizar_proyecto(actividades, plazo=None):
    """Calcula ES, EF, LS, LF, holgura y ruta(s) crítica(s) del proyecto."""
    G = nx.DiGraph()
    for act in actividades:
        if "t" in act:
            te, var = float(act["t"]), 0.0
        else:
            a, m, b = act["a"], act["m"], act["b"]
            if not a <= m <= b:
                raise ValueError(f"{act['id']}: se requiere a ≤ m ≤ b")
            te = (a + 4 * m + b) / 6
            var = ((b - a) / 6) ** 2
        G.add_node(act["id"], te=te, var=var)
    for act in actividades:
        for p in act.get("pred", []):
            if p not in G:
                raise ValueError(f"Predecesora desconocida: {p}")
            G.add_edge(p, act["id"])
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("La red tiene ciclos: revise las precedencias")

    orden = list(nx.topological_sort(G))
    # Pasada hacia adelante
    for n in orden:
        G.nodes[n]["ES"] = max((G.nodes[p]["EF"] for p in G.predecessors(n)), default=0.0)
        G.nodes[n]["EF"] = G.nodes[n]["ES"] + G.nodes[n]["te"]
    duracion = max(G.nodes[n]["EF"] for n in G)
    # Pasada hacia atrás
    for n in reversed(orden):
        G.nodes[n]["LF"] = min((G.nodes[s]["LS"] for s in G.successors(n)), default=duracion)
        G.nodes[n]["LS"] = G.nodes[n]["LF"] - G.nodes[n]["te"]
        G.nodes[n]["holgura"] = G.nodes[n]["LS"] - G.nodes[n]["ES"]

    criticas = {n for n in G if abs(G.nodes[n]["holgura"]) < 1e-9}
    H = G.subgraph(criticas)
    inicios = [n for n in H if G.nodes[n]["ES"] == 0]
    finales = [n for n in H if abs(G.nodes[n]["EF"] - duracion) < 1e-9]
    rutas = {tuple(r) for i in inicios for f in finales if i != f
             for r in nx.all_simple_paths(H, i, f)}
    rutas |= {(n,) for n in inicios if n in finales}  # actividad única crítica
    rutas = sorted(list(r) for r in rutas)

    # Varianza: se toma la ruta crítica de mayor varianza (criterio conservador)
    var_ruta = max(sum(G.nodes[n]["var"] for n in r) for r in rutas)
    sigma = var_ruta ** 0.5

    resultado = {
        "tabla": [{"id": n, **{k: G.nodes[n][k] for k in
                               ("te", "var", "ES", "EF", "LS", "LF", "holgura")}}
                  for n in orden],
        "duracion": duracion, "rutas_criticas": rutas,
        "varianza": var_ruta, "sigma": sigma, "grafo": G,
    }
    if plazo is not None:
        resultado["plazo"] = plazo
        if sigma > 0:
            z = (plazo - duracion) / sigma
            resultado["z"] = z
            resultado["probabilidad"] = float(norm.cdf(z))
        else:
            resultado["probabilidad"] = 1.0 if plazo >= duracion else 0.0
    return resultado


def plazo_para_probabilidad(resultado, probabilidad):
    """Plazo T que se cumple con la probabilidad dada: T = μ + z·σ."""
    return float(resultado["duracion"] + norm.ppf(probabilidad) * resultado["sigma"])
