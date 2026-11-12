"""Pruebas automatizadas: casos de estudio del informe y propiedades teóricas.

Ejecutar con:  pytest -v
"""
import pytest

from ia.clasificador import clasificar_reglas
from ia.extractor import extraer_reglas
from modulos import colas, inventarios, pert_cpm, pnl

TOL = 1e-3

ACTIVIDADES = [
    {"id": "A", "pred": [], "a": 2, "m": 4, "b": 6},
    {"id": "B", "pred": [], "a": 3, "m": 5, "b": 13},
    {"id": "C", "pred": ["A"], "a": 2, "m": 3, "b": 4},
    {"id": "D", "pred": ["A", "B"], "a": 4, "m": 6, "b": 8},
    {"id": "E", "pred": ["C", "D"], "a": 1, "m": 2, "b": 3},
]


# ----------------------------- Casos de estudio -----------------------------
def test_caso_pnl():
    r = pnl.resolver_pnl("40*x + 30*y - x**2 - 2*y**2", ["x + y <= 20"])
    assert r["x"]["x"] == pytest.approx(15, abs=TOL)
    assert r["x"]["y"] == pytest.approx(5, abs=TOL)
    assert r["f"] == pytest.approx(475, abs=TOL)
    assert list(r["multiplicadores"].values())[0] == pytest.approx(10, abs=TOL)


def test_caso_colas():
    tabla, s_opt = colas.costo_por_servidores(12, 8, 20, 30, s_max=4)
    assert s_opt == 3
    f3 = next(f for f in tabla if f["s"] == 3)
    assert f3["Lq"] == pytest.approx(0.2368, abs=TOL)
    assert f3["costo_total"] == pytest.approx(112.105, abs=TOL)
    assert not tabla[0]["estable"]  # s = 1 es inestable


def test_caso_inventarios():
    r = inventarios.eoq(12000, 50, 2)
    assert r["Q"] == pytest.approx(774.597, abs=TOL)
    assert r["costo_total"] == pytest.approx(1549.193, abs=TOL)
    assert inventarios.punto_reorden(12000, 5)["R"] == pytest.approx(164.384, abs=TOL)


def test_caso_pert():
    r = pert_cpm.analizar_proyecto(ACTIVIDADES, plazo=16)
    assert r["rutas_criticas"] == [["B", "D", "E"]]
    assert r["duracion"] == pytest.approx(14)
    assert r["probabilidad"] == pytest.approx(0.8633, abs=TOL)


# --------------------------- Propiedades teóricas ---------------------------
@pytest.mark.parametrize("lam,mu,s", [(12, 8, 2), (12, 8, 3), (5, 2, 4), (3, 4, 1)])
def test_ley_de_little(lam, mu, s):
    r = colas.mms(lam, mu, s)
    assert r["L"] == pytest.approx(lam * r["W"])
    assert r["Lq"] == pytest.approx(lam * r["Wq"])


def test_mms_con_un_servidor_igual_a_mm1():
    a, b = colas.mms(6, 8, 1), colas.mm1(6, 8)
    for k in ("P0", "L", "Lq", "W", "Wq"):
        assert a[k] == pytest.approx(b[k])


def test_rechaza_sistema_inestable():
    with pytest.raises(ValueError):
        colas.mm1(12, 8)
    with pytest.raises(ValueError):
        colas.mms(16, 8, 2)


def test_mm1k_tiende_a_mm1_con_k_grande():
    assert colas.mm1k(6, 8, 200)["L"] == pytest.approx(colas.mm1(6, 8)["L"], rel=1e-6)


def test_eoq_costo_pedir_igual_mantener():
    r = inventarios.eoq(5000, 80, 4)
    assert r["costo_pedir"] == pytest.approx(r["costo_mantener"])


def test_epq_y_faltantes_superan_eoq():
    q = inventarios.eoq(12000, 50, 2)["Q"]
    assert inventarios.epq(12000, 50, 2, 30000)["Q"] > q
    assert inventarios.eoq_faltantes(12000, 50, 2, 6)["Q"] > q


def test_pert_holgura_y_consistencia():
    r = pert_cpm.analizar_proyecto(ACTIVIDADES)
    assert any(abs(f["holgura"]) < 1e-9 for f in r["tabla"])
    for f in r["tabla"]:
        assert f["ES"] + f["te"] == pytest.approx(f["EF"])
        assert f["holgura"] >= -1e-9


def test_pert_detecta_ciclos():
    with pytest.raises(ValueError):
        pert_cpm.analizar_proyecto([{"id": "A", "pred": ["B"], "t": 1},
                                    {"id": "B", "pred": ["A"], "t": 1}])


def test_pnl_numerico_coincide_con_lagrange():
    r = pnl.resolver_pnl("40*x + 30*y - x**2 - 2*y**2", ["x + y <= 20"])
    lam = 10  # solución analítica del informe
    assert r["x"]["x"] == pytest.approx((40 - lam) / 2, abs=TOL)
    assert r["x"]["y"] == pytest.approx((30 - lam) / 4, abs=TOL)
    assert r["x"]["x"] + r["x"]["y"] <= 20 + TOL
    assert "cóncava" in r["curvatura"]


# ----------------------- Componente de IA (respaldo) ------------------------
ENUNCIADOS = {
    "colas": [
        "A un banco llegan 12 clientes por hora y cada cajero atiende 8 por hora.",
        "En una ventanilla de peajes llegan 90 vehículos por hora y se atienden 100.",
        "¿Cuál es el tiempo de espera en la fila si llegan 5 pacientes por hora?",
        "Un call center recibe llamadas; cada asesor atiende 10 por hora. ¿Cuántos servidores?",
        "La tasa de llegadas a la cafetería es 30 clientes por hora y hay 2 cajeros.",
    ],
    "inventarios": [
        "La demanda anual es de 12.000 unidades y el costo de ordenar es 50 USD.",
        "Calcule el lote económico (EOQ) con costo de mantener de 2 USD por unidad.",
        "Una tienda quiere saber su punto de reorden con tiempo de entrega de 5 días.",
        "Una fábrica produce 30000 unidades al año; determine el EPQ y su inventario.",
        "Se permiten faltantes con costo de 6 USD; halle el pedido óptimo de inventario.",
    ],
    "pnl": [
        "Maximizar f(x,y) = 40x + 30y - x^2 - 2y^2 sujeto a x + y <= 20.",
        "Minimizar el costo x² + y² con la restricción x + y >= 10 usando Lagrange.",
        "Aplique las condiciones KKT a la función objetivo no lineal dada.",
        "La utilidad presenta rendimientos decrecientes; maximizar sujeto a capacidad.",
        "Encuentre el máximo de f(x) = 10x - x^2 (programación no lineal).",
    ],
    "pert_cpm": [
        "Un proyecto tiene 5 actividades con tiempos optimista, probable y pesimista.",
        "Determine la ruta crítica del proyecto y la holgura de cada actividad.",
        "La actividad C tiene como predecesora a A; calcule la duración del proyecto.",
        "Con PERT, ¿qué probabilidad hay de terminar en 16 semanas?",
        "Aplique CPM a la red de actividades de la construcción.",
    ],
}


def test_clasificador_precision_minima():
    total = aciertos = 0
    for modulo, frases in ENUNCIADOS.items():
        for frase in frases:
            total += 1
            aciertos += clasificar_reglas(frase)["modulo"] == modulo
    assert aciertos / total >= 0.9, f"Precisión: {aciertos}/{total}"


def test_extractor_ejemplo_del_informe():
    r = extraer_reglas("A un banco llegan 12 clientes por hora y cada cajero "
                       "atiende 8 por hora; hay 2 cajeros", "colas")
    assert r["modelo"] == "M/M/s"
    assert (r["parametros"]["lambda"], r["parametros"]["mu"], r["parametros"]["s"]) == (12, 8, 2)


def test_extractor_numeros_en_formato_colombiano():
    r = extraer_reglas("La demanda anual es de 12.000 unidades, el costo de ordenar "
                       "es 50 por pedido y el costo de mantener es 2,5 por unidad", "inventarios")
    assert (r["parametros"]["D"], r["parametros"]["S"], r["parametros"]["h"]) == (12000, 50, 2.5)
