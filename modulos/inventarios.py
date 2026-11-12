"""Módulo de Inventarios: EOQ, EPQ, EOQ con faltantes, punto de reorden
y modelo del vendedor de periódicos (newsvendor).

Notación: D = demanda anual, S = costo por pedido, h = costo de mantener
una unidad por año, b = costo de faltante por unidad-año, p = tasa anual
de producción.
"""
from math import sqrt

from scipy.stats import norm


def _positivos(**kw):
    for nombre, valor in kw.items():
        if valor <= 0:
            raise ValueError(f"{nombre} debe ser positivo")


def eoq(D, S, h, dias_anio=365):
    """Lote económico de pedido (reposición instantánea, sin faltantes)."""
    _positivos(D=D, S=S, h=h)
    Q = sqrt(2 * D * S / h)
    costo_pedir = D / Q * S
    costo_mantener = Q / 2 * h
    N = D / Q
    return {"modelo": "EOQ", "Q": Q, "N": N, "ciclo_dias": dias_anio / N,
            "costo_pedir": costo_pedir, "costo_mantener": costo_mantener,
            "costo_total": costo_pedir + costo_mantener}


def epq(D, S, h, p, dias_anio=365):
    """Lote económico de producción (reposición gradual a tasa p > D)."""
    _positivos(D=D, S=S, h=h, p=p)
    if p <= D:
        raise ValueError("La tasa de producción p debe ser mayor que la demanda D")
    factor = 1 - D / p
    Q = sqrt(2 * D * S / (h * factor))
    I_max = Q * factor
    costo_pedir = D / Q * S
    costo_mantener = I_max / 2 * h
    return {"modelo": "EPQ", "Q": Q, "I_max": I_max, "N": D / Q,
            "ciclo_dias": dias_anio * Q / D,
            "dias_produccion": dias_anio * Q / p,
            "costo_pedir": costo_pedir, "costo_mantener": costo_mantener,
            "costo_total": costo_pedir + costo_mantener}


def eoq_faltantes(D, S, h, b):
    """EOQ con faltantes planeados (pedidos pendientes) de costo b por unidad-año."""
    _positivos(D=D, S=S, h=h, b=b)
    Q = sqrt(2 * D * S / h) * sqrt((h + b) / b)
    I_max = Q * b / (h + b)
    faltante_max = Q - I_max
    costo_pedir = D / Q * S
    costo_mantener = h * I_max ** 2 / (2 * Q)
    costo_faltante = b * faltante_max ** 2 / (2 * Q)
    return {"modelo": "EOQ con faltantes", "Q": Q, "I_max": I_max,
            "faltante_max": faltante_max, "costo_pedir": costo_pedir,
            "costo_mantener": costo_mantener, "costo_faltante": costo_faltante,
            "costo_total": costo_pedir + costo_mantener + costo_faltante}


def punto_reorden(D, tiempo_entrega_dias, dias_anio=365,
                  sigma_diaria=0.0, nivel_servicio=None):
    """R = d·L + z·σd·√L. Si no se da nivel de servicio, no hay inventario de seguridad."""
    d = D / dias_anio
    z = float(norm.ppf(nivel_servicio)) if nivel_servicio else 0.0
    seguridad = z * sigma_diaria * sqrt(tiempo_entrega_dias)
    return {"demanda_diaria": d, "z": z, "inventario_seguridad": seguridad,
            "R": d * tiempo_entrega_dias + seguridad}


def newsvendor(media, desviacion, costo_faltante, costo_sobrante):
    """Cantidad óptima para demanda normal: F(Q) = Cu / (Cu + Co)."""
    _positivos(desviacion=desviacion, costo_faltante=costo_faltante,
               costo_sobrante=costo_sobrante)
    razon = costo_faltante / (costo_faltante + costo_sobrante)
    z = float(norm.ppf(razon))
    return {"modelo": "Newsvendor", "razon_critica": razon, "z": z,
            "Q": media + z * desviacion}
