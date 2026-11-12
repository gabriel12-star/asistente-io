"""Módulo de Líneas de Espera: modelos M/M/1, M/M/s y M/M/1/K.

Convención: lam = tasa de llegadas (λ), mu = tasa de servicio por servidor (μ),
ambas en las mismas unidades de tiempo (por ejemplo, clientes por hora).
Los tiempos W y Wq resultan en esa misma unidad de tiempo.
"""
from math import factorial


def _validar(lam, mu):
    if lam <= 0 or mu <= 0:
        raise ValueError("λ y μ deben ser positivos")


def mm1(lam, mu):
    """Modelo M/M/1 (un servidor, capacidad infinita)."""
    _validar(lam, mu)
    rho = lam / mu
    if rho >= 1:
        raise ValueError(f"Sistema inestable: ρ = {rho:.3f} ≥ 1")
    L = rho / (1 - rho)
    Lq = rho ** 2 / (1 - rho)
    return {"modelo": "M/M/1", "rho": rho, "P0": 1 - rho, "L": L, "Lq": Lq,
            "W": 1 / (mu - lam), "Wq": lam / (mu * (mu - lam))}


def mms(lam, mu, s):
    """Modelo M/M/s (s servidores en paralelo, capacidad infinita)."""
    _validar(lam, mu)
    if int(s) != s or s < 1:
        raise ValueError("s debe ser un entero ≥ 1")
    s = int(s)
    r = lam / mu
    rho = r / s
    if rho >= 1:
        raise ValueError(f"Sistema inestable: ρ = {rho:.3f} ≥ 1")
    suma = sum(r ** n / factorial(n) for n in range(s))
    P0 = 1 / (suma + r ** s / (factorial(s) * (1 - rho)))
    Lq = P0 * r ** s * rho / (factorial(s) * (1 - rho) ** 2)
    Wq = Lq / lam
    W = Wq + 1 / mu
    L = lam * W  # Ley de Little
    Pw = P0 * r ** s / (factorial(s) * (1 - rho))  # prob. de esperar (Erlang C)
    return {"modelo": f"M/M/{s}", "s": s, "rho": rho, "P0": P0, "L": L,
            "Lq": Lq, "W": W, "Wq": Wq, "P_espera": Pw}


def mm1k(lam, mu, K):
    """Modelo M/M/1/K (un servidor, máximo K clientes en el sistema)."""
    _validar(lam, mu)
    if int(K) != K or K < 1:
        raise ValueError("K debe ser un entero ≥ 1")
    rho = lam / mu
    if abs(rho - 1) < 1e-12:
        P0 = 1 / (K + 1)
        L = K / 2
    else:
        P0 = (1 - rho) / (1 - rho ** (K + 1))
        L = rho / (1 - rho) - (K + 1) * rho ** (K + 1) / (1 - rho ** (K + 1))
    PK = P0 * rho ** K  # probabilidad de rechazo (sistema lleno)
    lam_ef = lam * (1 - PK)  # tasa efectiva de entrada
    Lq = L - (1 - P0)
    return {"modelo": f"M/M/1/{K}", "rho": rho, "P0": P0, "PK": PK,
            "lam_efectiva": lam_ef, "L": L, "Lq": Lq,
            "W": L / lam_ef, "Wq": Lq / lam_ef}


def costo_por_servidores(lam, mu, costo_servidor, costo_espera, s_max=None):
    """Evalúa CT(s) = Cs·s + Ce·L para s = 1..s_max y halla el s de costo mínimo.

    Devuelve (tabla, s_optimo). En la tabla, los valores de s con
    ρ ≥ 1 aparecen marcados como inestables.
    """
    s_min = int(lam // mu) + 1  # primer s estable
    s_max = s_max or s_min + 3
    tabla = []
    for s in range(1, s_max + 1):
        if lam / (s * mu) >= 1:
            tabla.append({"s": s, "rho": lam / (s * mu), "estable": False})
            continue
        m = mms(lam, mu, s)
        m["estable"] = True
        m["costo_total"] = costo_servidor * s + costo_espera * m["L"]
        tabla.append(m)
    estables = [f for f in tabla if f["estable"]]
    s_opt = min(estables, key=lambda f: f["costo_total"])["s"]
    return tabla, s_opt
