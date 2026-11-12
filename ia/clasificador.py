"""Clasificador: decide a qué módulo de IO pertenece un enunciado."""
import unicodedata

from ia.llm import extraer_json, llamar_llm

MODULOS = ("pnl", "colas", "inventarios", "pert_cpm")

SISTEMA = """Eres un experto en Investigación de Operaciones. Clasifica el
problema del usuario en UNO de estos módulos:
- "pnl": programación no lineal (función objetivo o restricciones no lineales,
  x², rendimientos decrecientes, Lagrange, KKT).
- "colas": líneas de espera (llegadas, tasa de servicio, servidores, cajeros, espera).
- "inventarios": EOQ, EPQ, lote económico, demanda anual, costo de pedir o mantener.
- "pert_cpm": proyectos con actividades, precedencias, duraciones, ruta crítica.
Responde SOLO con JSON: {"modulo": "...", "confianza": 0.0-1.0}"""

# Palabras clave del respaldo sin conexión (sin tildes, en minúscula)
CLAVES = {
    "pnl": ["maximizar", "minimizar", "no lineal", "lagrange", "kkt", "x^2", "x**2",
            "x²", "y^2", "y²", "rendimientos decrecientes", "funcion objetivo",
            "restriccion", "utilidad f", "f(x"],
    "colas": ["llegan", "llegada", "cola", "espera", "cajero", "servidor", "atiende",
              "ventanilla", "por hora", "tasa de servicio", "clientes", "fila"],
    "inventarios": ["inventario", "demanda anual", "lote", "eoq", "epq", "pedido",
                    "ordenar", "mantener", "almacenamiento", "reorden", "faltante",
                    "unidades al ano", "unidades/ano"],
    "pert_cpm": ["actividad", "proyecto", "predecesora", "ruta critica", "pert",
                 "cpm", "optimista", "pesimista", "semanas", "holgura", "duracion"],
}


def normalizar(texto):
    """Minúsculas y sin tildes, para comparar palabras clave."""
    t = unicodedata.normalize("NFD", texto.lower())
    return "".join(c for c in t if unicodedata.category(c) != "Mn")


def clasificar_reglas(texto):
    t = normalizar(texto)
    puntajes = {m: sum(t.count(k) for k in claves) for m, claves in CLAVES.items()}
    mejor = max(puntajes, key=puntajes.get)
    total = sum(puntajes.values())
    return {"modulo": mejor if total else None,
            "confianza": round(puntajes[mejor] / total, 2) if total else 0.0,
            "fuente": "reglas"}


def clasificar(texto):
    """Clasifica con el modelo de lenguaje; si no está disponible, usa reglas."""
    datos = extraer_json(llamar_llm(SISTEMA, texto, max_tokens=100))
    if datos and datos.get("modulo") in MODULOS:
        datos["fuente"] = "llm"
        return datos
    return clasificar_reglas(texto)
