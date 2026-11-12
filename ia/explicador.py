"""Explicador: redacta la interpretación de resultados YA calculados por Python."""
import json

from ia.llm import llamar_llm

SISTEMA = """Eres un tutor de Investigación de Operaciones. Recibirás los
resultados numéricos de un modelo, calculados por un programa verificado.
Escribe en español, en 4 a 6 frases: qué significan los resultados y una
recomendación concreta para decidir. Usa EXACTAMENTE los números recibidos
(redondeados a 2 decimales); no recalcules ni inventes datos."""


def _limpiar(obj):
    """Quita objetos no serializables (por ejemplo, el grafo de NetworkX)."""
    if isinstance(obj, dict):
        return {k: _limpiar(v) for k, v in obj.items() if k != "grafo"}
    if isinstance(obj, list):
        return [_limpiar(v) for v in obj]
    return obj


def explicar(modulo, resultados):
    datos = json.dumps(_limpiar(resultados), ensure_ascii=False, default=float)
    texto = llamar_llm(SISTEMA, f"Módulo: {modulo}\nResultados: {datos}")
    return texto if texto else explicar_plantilla(modulo, resultados)


def explicar_plantilla(modulo, r):
    """Explicación de respaldo cuando no hay modelo de lenguaje disponible."""
    if modulo == "pnl":
        xs = ", ".join(f"{k} = {v:.2f}" for k, v in r["x"].items())
        txt = f"La solución óptima es {xs}, con valor objetivo {r['f']:.2f}. "
        for rest, lam in r["multiplicadores"].items():
            txt += (f"La restricción «{rest}» está activa con multiplicador "
                    f"λ = {lam:.2f}: relajarla en una unidad cambiaría el objetivo "
                    f"en aproximadamente {lam:.2f} (precio sombra). ")
        txt += f"La Hessiana es {r['curvatura']}"
        if "cóncava" in r["curvatura"] or "convexa" in r["curvatura"]:
            txt += ", por lo que el óptimo encontrado es global."
        return txt
    if modulo == "colas":
        if "s_optimo" in r:
            f = next(x for x in r["tabla"] if x["s"] == r["s_optimo"])
            return (f"La configuración de menor costo es s = {r['s_optimo']} servidores, "
                    f"con costo total de {f['costo_total']:.2f} por unidad de tiempo, "
                    f"utilización ρ = {f['rho']:.2f} y espera promedio en cola de "
                    f"{f['Wq'] * 60:.2f} minutos (si las tasas son por hora).")
        return (f"Con el modelo {r['modelo']} la utilización es ρ = {r['rho']:.2f}. "
                f"En promedio hay {r['L']:.2f} clientes en el sistema y {r['Lq']:.2f} en "
                f"cola; cada cliente espera {r['Wq'] * 60:.2f} minutos en cola "
                f"(si las tasas son por hora).")
    if modulo == "inventarios":
        txt = (f"Con el modelo {r['modelo']} conviene pedir lotes de {r['Q']:.1f} "
               f"unidades, con un costo anual total de {r['costo_total']:.2f}. ")
        if "R" in r:
            txt += f"Emita cada pedido cuando el inventario baje a {r['R']:.0f} unidades."
        return txt
    if modulo == "pert_cpm":
        rutas = " y ".join("-".join(x) for x in r["rutas_criticas"])
        txt = (f"La ruta crítica es {rutas}, con duración esperada de "
               f"{r['duracion']:.2f} y desviación estándar {r['sigma']:.3f}. ")
        if "probabilidad" in r:
            txt += (f"La probabilidad de terminar en {r['plazo']} es "
                    f"{r['probabilidad'] * 100:.1f} %. Cualquier retraso en una "
                    f"actividad crítica retrasa todo el proyecto.")
        return txt
    return json.dumps(_limpiar(r), ensure_ascii=False, default=float)
