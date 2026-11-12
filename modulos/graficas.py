"""Gráficas de apoyo (Matplotlib) usadas por la interfaz y por el informe."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sympy as sp

from modulos.inventarios import eoq


def grafica_pnl(objetivo, restricciones, solucion, rango=(0, 25)):
    """Curvas de nivel de f(x, y), recta de la restricción y punto óptimo."""
    x, y = sp.symbols("x y")
    f = sp.lambdify((x, y), sp.sympify(objetivo), "numpy")
    X, Y = np.meshgrid(np.linspace(*rango, 300), np.linspace(*rango, 300))
    fig, ax = plt.subplots(figsize=(6, 4.6))
    cs = ax.contour(X, Y, f(X, Y), levels=14, colors="0.45", linewidths=0.8)
    ax.clabel(cs, fontsize=7, fmt="%.0f")
    for r in restricciones:
        lhs, rhs = r.split("<=") if "<=" in r else r.split(">=")
        g = sp.sympify(lhs) - sp.sympify(rhs)
        G = sp.lambdify((x, y), g, "numpy")(X, Y)
        ax.contour(X, Y, G, levels=[0], colors="black", linewidths=2)
        ax.contourf(X, Y, G, levels=[G.min(), 0], colors=["0.85"], alpha=0.6)
    xs, ys = solucion["x"]["x"], solucion["x"]["y"]
    ax.plot(xs, ys, "ko", ms=8)
    ax.annotate(f"Óptimo ({xs:.0f}, {ys:.0f})\nf = {solucion['f']:.0f}", (xs, ys),
                xytext=(xs + 1.2, ys + 3), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="black"))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Curvas de nivel de f(x, y) y región factible (gris)")
    fig.tight_layout()
    return fig


def grafica_costos_colas(tabla, s_opt):
    """Costo total por número de servidores (solo configuraciones estables)."""
    est = [f for f in tabla if f["estable"]]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ss = [f["s"] for f in est]
    ax.bar(ss, [f["costo_total"] for f in est],
           color=["black" if s == s_opt else "0.7" for s in ss], width=0.55)
    for f in est:
        ax.text(f["s"], f["costo_total"] + 2, f"{f['costo_total']:.2f}",
                ha="center", fontsize=9)
    ax.set_xticks(ss)
    ax.set_xlabel("Número de servidores (s)")
    ax.set_ylabel("Costo total (USD/h)")
    ax.set_title("Costo total por número de cajeros (negro = óptimo)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def grafica_eoq(D, S, h):
    """Curvas de costo de pedir, mantener y total en función de Q."""
    r = eoq(D, S, h)
    Q = np.linspace(r["Q"] * 0.25, r["Q"] * 2.5, 300)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(Q, D / Q * S, "--", color="0.4", label="Costo de pedir (D/Q)·S")
    ax.plot(Q, Q / 2 * h, ":", color="0.4", label="Costo de mantener (Q/2)·h")
    ax.plot(Q, D / Q * S + Q / 2 * h, "k-", lw=2, label="Costo total")
    ax.axvline(r["Q"], color="black", lw=0.8)
    ax.annotate(f"Q* = {r['Q']:.1f}\nCT = {r['costo_total']:.2f}",
                (r["Q"], r["costo_total"]), xytext=(r["Q"] * 1.25, r["costo_total"] * 1.6),
                arrowprops=dict(arrowstyle="->"), fontsize=9)
    ax.set_xlabel("Tamaño del lote Q (unidades)")
    ax.set_ylabel("Costo anual (USD)")
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def grafica_pert(resultado):
    """Red del proyecto por niveles; la ruta crítica se dibuja en negro grueso."""
    G = resultado["grafo"]
    criticas = {n for r in resultado["rutas_criticas"] for n in r}
    for capa, nodos in enumerate(nx.topological_generations(G)):
        for n in nodos:
            G.nodes[n]["capa"] = capa
    pos = nx.multipartite_layout(G, subset_key="capa")
    fig, ax = plt.subplots(figsize=(6, 3.4))
    aristas_c = [(u, v) for u, v in G.edges if u in criticas and v in criticas]
    aristas_n = [e for e in G.edges if e not in aristas_c]
    nx.draw_networkx_edges(G, pos, edgelist=aristas_n, edge_color="0.6", ax=ax,
                           arrowsize=14, node_size=1500)
    nx.draw_networkx_edges(G, pos, edgelist=aristas_c, edge_color="black", width=2.5,
                           ax=ax, arrowsize=16, node_size=1500)
    nx.draw_networkx_nodes(G, pos, node_color=["black" if n in criticas else "white"
                                               for n in G], edgecolors="black",
                           node_size=1500, ax=ax)
    etiquetas = {n: f"{n}\nte={G.nodes[n]['te']:.0f}\nH={G.nodes[n]['holgura']:.0f}"
                 for n in G}
    for n, (px, py) in pos.items():
        ax.text(px, py, etiquetas[n], ha="center", va="center", fontsize=8,
                color="white" if n in criticas else "black")
    ax.set_title(f"Red del proyecto — duración esperada {resultado['duracion']:.0f} semanas")
    ax.axis("off")
    ax.margins(0.18)
    fig.tight_layout()
    return fig
