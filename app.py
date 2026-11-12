"""Interfaz web del Asistente de Investigación de Operaciones.

Ejecutar con:  streamlit run app.py
"""
import json

import pandas as pd
import streamlit as st

from ia.clasificador import clasificar
from ia.explicador import explicar
from ia.extractor import extraer, validar
from ia.llm import llm_disponible
from modulos import graficas
from motor import resolver

NOMBRES = {"pnl": "Programación No Lineal", "colas": "Líneas de espera",
           "inventarios": "Inventarios", "pert_cpm": "PERT/CPM"}

st.set_page_config(page_title="Asistente de IO", layout="wide")
st.title("Asistente inteligente de Investigación de Operaciones")
st.sidebar.header("Configuración")
modo = st.sidebar.radio("Modo de ingreso", ["Lenguaje natural", "Ingreso manual"])
st.sidebar.caption("IA: modelo de lenguaje conectado" if llm_disponible()
                   else "IA: respaldo por reglas (sin ANTHROPIC_API_KEY)")


def mostrar_resultados(modulo, modelo, params):
    """Calcula, muestra tablas/gráficas y la explicación de la IA."""
    try:
        r = resolver(modulo, modelo, params)
    except (ValueError, RuntimeError, KeyError) as error:
        st.error(f"No fue posible calcular: {error}")
        return
    st.subheader("Resultados")
    if modulo == "pert_cpm":
        st.dataframe(pd.DataFrame(r["tabla"]).round(3), hide_index=True)
        st.write(f"**Ruta crítica:** {' y '.join('-'.join(x) for x in r['rutas_criticas'])}"
                 f" — duración {r['duracion']:.2f}, σ = {r['sigma']:.3f}")
        if "probabilidad" in r:
            st.metric("Probabilidad de cumplir el plazo", f"{r['probabilidad']:.1%}")
        st.pyplot(graficas.grafica_pert(r))
    elif modulo == "colas" and "tabla" in r:
        st.dataframe(pd.DataFrame(r["tabla"]).round(4), hide_index=True)
        st.success(f"Número óptimo de servidores: {r['s_optimo']}")
        st.pyplot(graficas.grafica_costos_colas(r["tabla"], r["s_optimo"]))
    else:
        st.json({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
        if modulo == "inventarios" and r["modelo"] == "EOQ":
            st.pyplot(graficas.grafica_eoq(params["D"], params["S"], params["h"]))
        if modulo == "pnl" and set(r["x"]) == {"x", "y"}:
            st.pyplot(graficas.grafica_pnl(params["objetivo"],
                                           params.get("restricciones", []), r))
    st.subheader("Interpretación")
    st.info(explicar(modulo, r))


if modo == "Lenguaje natural":
    texto = st.text_area("Describa su problema", height=150, placeholder=(
        "Ej.: A un banco llegan 12 clientes por hora y cada cajero atiende 8 por "
        "hora; hay 2 cajeros."))
    if st.button("Analizar con IA") and texto.strip():
        clase = clasificar(texto)
        if not clase["modulo"]:
            st.warning("No se reconoció el tipo de problema. Use el ingreso manual.")
        else:
            st.session_state["extraccion"] = extraer(texto, clase["modulo"])
    ext = st.session_state.get("extraccion")
    if ext:
        st.write(f"**Módulo:** {NOMBRES[ext['modulo']]} — **Modelo:** {ext['modelo']} "
                 f"(fuente: {ext['fuente']})")
        st.caption("Revise y corrija los parámetros antes de calcular.")
        editado = st.text_area("Parámetros extraídos (JSON)",
                               json.dumps(ext["parametros"], ensure_ascii=False, indent=2),
                               height=220)
        try:
            params = json.loads(editado)
            faltantes, errores = validar(ext["modulo"], params)
        except json.JSONDecodeError:
            params, faltantes, errores = None, [], ["El JSON no es válido"]
        for f in faltantes:
            st.warning(f"Falta el dato «{f}»: el enunciado no lo trae; complételo.")
        for e in errores:
            st.error(e)
        if st.button("Confirmar y calcular", disabled=bool(faltantes or errores)):
            mostrar_resultados(ext["modulo"], ext["modelo"], params)
else:
    pest = st.tabs(list(NOMBRES.values()))

    with pest[0]:
        obj = st.text_input("Función objetivo f(x, y)", "40*x + 30*y - x**2 - 2*y**2")
        rest = st.text_area("Restricciones (una por línea)", "x + y <= 20")
        sentido = st.selectbox("Sentido", ["max", "min"])
        if st.button("Resolver PNL"):
            mostrar_resultados("pnl", "PNL", {"objetivo": obj, "sentido": sentido,
                                              "restricciones": [r for r in rest.splitlines() if r.strip()]})

    with pest[1]:
        c1, c2, c3 = st.columns(3)
        lam = c1.number_input("λ (llegadas por hora)", 0.1, value=12.0)
        mu = c2.number_input("μ (servicios por hora y servidor)", 0.1, value=8.0)
        s = c3.number_input("Servidores s", 1, value=2, step=1)
        c4, c5, c6 = st.columns(3)
        K = c4.number_input("Capacidad K (0 = infinita)", 0, value=0, step=1)
        cs = c5.number_input("Costo por servidor (0 = no optimizar)", 0.0, value=20.0)
        ce = c6.number_input("Costo de espera por cliente", 0.0, value=30.0)
        if st.button("Resolver cola"):
            mostrar_resultados("colas", "", {"lambda": lam, "mu": mu, "s": int(s),
                                             "K": int(K) or None, "costo_servidor": cs or None,
                                             "costo_espera": ce or None})

    with pest[2]:
        c1, c2, c3 = st.columns(3)
        D = c1.number_input("Demanda anual D", 1.0, value=12000.0)
        S = c2.number_input("Costo por pedido S", 0.01, value=50.0)
        h = c3.number_input("Costo de mantener h", 0.01, value=2.0)
        c4, c5, c6 = st.columns(3)
        p = c4.number_input("Producción anual p (0 = EOQ)", 0.0, value=0.0)
        b = c5.number_input("Costo de faltante b (0 = sin faltantes)", 0.0, value=0.0)
        L = c6.number_input("Tiempo de entrega (días)", 0.0, value=5.0)
        if st.button("Resolver inventario"):
            mostrar_resultados("inventarios", "", {"D": D, "S": S, "h": h, "p": p or None,
                                                   "b": b or None, "tiempo_entrega_dias": L or None})

    with pest[3]:
        base = pd.DataFrame({"id": list("ABCDE"), "pred": ["", "", "A", "A,B", "C,D"],
                             "a": [2, 3, 2, 4, 1], "m": [4, 5, 3, 6, 2],
                             "b": [6, 13, 4, 8, 3]})
        tabla = st.data_editor(base, num_rows="dynamic", hide_index=True)
        plazo = st.number_input("Plazo para calcular probabilidad", 0.0, value=16.0)
        if st.button("Resolver proyecto"):
            acts = [{"id": str(f.id), "a": float(f.a), "m": float(f.m), "b": float(f.b),
                     "pred": [x.strip() for x in str(f.pred).split(",") if x.strip()]}
                    for f in tabla.itertuples()]
            mostrar_resultados("pert_cpm", "PERT", {"actividades": acts,
                                                    "plazo": plazo or None})
