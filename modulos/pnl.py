"""Módulo de Programación No Lineal (PNL).

La función objetivo y las restricciones se escriben como texto, por ejemplo:
    objetivo = "40*x + 30*y - x**2 - 2*y**2"
    restricciones = ["x + y <= 20", "x >= 0", "y >= 0"]
SymPy las convierte en funciones y calcula gradientes y Hessiana;
SciPy resuelve el problema numéricamente (SLSQP).
"""
import re

import numpy as np
import sympy as sp
from scipy.optimize import minimize

TOL = 1e-6


def _parsear_restriccion(texto, simbolos):
    """Convierte 'x + y <= 20' en (expresión g, tipo) con g(x) <= 0, >= 0 o == 0."""
    m = re.match(r"^(.*?)(<=|>=|==|=)(.*)$", texto.replace(" ", ""))
    if not m:
        raise ValueError(f"Restricción no válida: {texto!r}")
    izq, op, der = m.groups()
    g = sp.sympify(izq, locals=simbolos) - sp.sympify(der, locals=simbolos)
    tipo = {"<=": "<=", ">=": ">=", "==": "==", "=": "=="}[op]
    return sp.simplify(g), tipo


def resolver_pnl(objetivo, restricciones=(), sentido="max", x0=None):
    """Resuelve max/min f(x) sujeto a restricciones y analiza las condiciones KKT.

    Devuelve un diccionario con la solución, el valor óptimo, los
    multiplicadores de Lagrange de las restricciones activas y la
    clasificación de la Hessiana (cóncava/convexa).
    """
    if sentido not in ("max", "min"):
        raise ValueError("sentido debe ser 'max' o 'min'")
    f_expr = sp.sympify(objetivo)
    variables = sorted(f_expr.free_symbols, key=lambda s: s.name)
    simbolos = {v.name: v for v in variables}
    n = len(variables)

    f_num = sp.lambdify([variables], f_expr, "numpy")
    grad_expr = [sp.diff(f_expr, v) for v in variables]
    grad_num = sp.lambdify([variables], grad_expr, "numpy")
    hess_num = sp.lambdify([variables], sp.hessian(f_expr, variables), "numpy")
    signo = -1.0 if sentido == "max" else 1.0  # SciPy siempre minimiza

    rest = [(_parsear_restriccion(r, simbolos)) for r in restricciones]
    cons, g_info = [], []
    for g, tipo in rest:
        g_num = sp.lambdify([variables], g, "numpy")
        dg_num = sp.lambdify([variables], [sp.diff(g, v) for v in variables], "numpy")
        g_info.append((g, tipo, g_num, dg_num))
        if tipo == "==":
            cons.append({"type": "eq", "fun": g_num})
        elif tipo == "<=":  # SciPy usa fun(x) >= 0
            cons.append({"type": "ineq", "fun": lambda x, gn=g_num: -gn(x)})
        else:
            cons.append({"type": "ineq", "fun": g_num})

    x0 = np.ones(n) if x0 is None else np.asarray(x0, dtype=float)
    res = minimize(lambda x: signo * f_num(x), x0,
                   jac=lambda x: signo * np.asarray(grad_num(x), dtype=float),
                   constraints=cons, method="SLSQP",
                   options={"ftol": 1e-12, "maxiter": 500})
    if not res.success:
        raise RuntimeError(f"El optimizador no convergió: {res.message}")
    x = res.x

    # Multiplicadores KKT: grad f = sum(lambda_i * grad g_i) sobre las activas
    activas = [(str(g), t, dg) for g, t, gn, dg in g_info if abs(gn(x)) < 1e-5]
    multiplicadores = {}
    if activas:
        A = np.column_stack([np.asarray(dg(x), dtype=float) for _, _, dg in activas])
        lam, *_ = np.linalg.lstsq(A, np.asarray(grad_num(x), dtype=float), rcond=None)
        for (nombre, tipo, _), l in zip(activas, lam):
            multiplicadores[f"{nombre} {tipo} 0"] = float(l)

    eig = np.linalg.eigvalsh(np.asarray(hess_num(x), dtype=float))
    if np.all(eig < -TOL):
        curvatura = "definida negativa (f cóncava)"
    elif np.all(eig > TOL):
        curvatura = "definida positiva (f convexa)"
    else:
        curvatura = "indefinida o semidefinida"

    return {
        "variables": [v.name for v in variables],
        "x": {v.name: float(xi) for v, xi in zip(variables, x)},
        "f": float(f_num(x)),
        "multiplicadores": multiplicadores,
        "autovalores_hessiana": [float(e) for e in eig],
        "curvatura": curvatura,
        "iteraciones": int(res.nit),
    }


def punto_estacionario(objetivo):
    """Resuelve grad f = 0 de forma analítica (problema sin restricciones)."""
    f_expr = sp.sympify(objetivo)
    variables = sorted(f_expr.free_symbols, key=lambda s: s.name)
    sol = sp.solve([sp.diff(f_expr, v) for v in variables], variables, dict=True)
    return [{str(k): float(v) for k, v in s.items()} for s in sol]
