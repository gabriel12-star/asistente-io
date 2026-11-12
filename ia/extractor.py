"""Extractor: convierte el enunciado en parámetros JSON y los valida.

Nunca inventa datos: lo que no aparece en el texto se reporta en
"faltantes" para que el usuario lo complete antes de calcular.
"""
import json
import re

from ia.clasificador import normalizar
from ia.llm import extraer_json, llamar_llm

ESQUEMAS = {
    "colas": '{"modelo": "M/M/1|M/M/s|M/M/1/K", "parametros": {"lambda": num, '
             '"mu": num, "s": int, "K": int|null, "costo_servidor": num|null, '
             '"costo_espera": num|null}}',
    "inventarios": '{"modelo": "EOQ|EPQ|faltantes", "parametros": {"D": num, "S": num, '
                   '"h": num, "p": num|null, "b": num|null, "tiempo_entrega_dias": num|null}}',
    "pnl": '{"modelo": "PNL", "parametros": {"objetivo": "expresión en sintaxis Python", '
           '"restricciones": ["x + y <= 20", ...], "sentido": "max|min"}}',
    "pert_cpm": '{"modelo": "PERT|CPM", "parametros": {"actividades": [{"id": "A", '
                '"pred": [], "a": num, "m": num, "b": num}], "plazo": num|null}}',
}

SISTEMA = """Extrae los parámetros del problema de Investigación de Operaciones.
Módulo: {modulo}. Responde SOLO con JSON con este esquema:
{esquema}
Reglas: convierte todas las tasas a la MISMA unidad de tiempo (por ejemplo, por
hora; si dice "atiende a un cliente cada 5 minutos", mu = 12 por hora). Usa
punto decimal. Si un dato obligatorio NO aparece en el texto, pon null y agrégalo
a la lista "faltantes". Nunca inventes valores."""

OBLIGATORIOS = {
    "colas": ["lambda", "mu"],
    "inventarios": ["D", "S", "h"],
    "pnl": ["objetivo"],
    "pert_cpm": ["actividades"],
}

NUM = r"(\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:[.,]\d+)?)"
PALABRAS = {"un": 1, "uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4,
            "cinco": 5, "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10}


def a_numero(texto):
    """'12.000' -> 12000 ; '2,5' -> 2.5 ; 'dos' -> 2."""
    texto = texto.strip()
    if texto in PALABRAS:
        return PALABRAS[texto]
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?", texto):
        texto = texto.replace(".", "")
    valor = float(texto.replace(",", "."))
    return int(valor) if valor.is_integer() else valor


def _buscar(patron, texto):
    m = re.search(patron, texto)
    return a_numero(m.group(1)) if m else None


# ------------------------- respaldo basado en reglas -------------------------
def _extraer_colas(t):
    n = NUM + "|" + "|".join(PALABRAS)
    num = rf"\b({n})"
    lam = _buscar(rf"(?:llegan?|arriban?|ingresan?|tasa de llegadas? (?:es )?de)\s+"
                  rf"(?:en promedio |aproximadamente |unos? )?{NUM}", t)
    mu = _buscar(rf"(?:atiende|atienden|sirve|procesa|tasa de servicio (?:es )?de)\s+"
                 rf"(?:en promedio |a )?{NUM}", t)
    minutos = _buscar(rf"(?:tarda|demora|tiempo (?:promedio )?de servicio (?:es )?de)\s+"
                      rf"(?:en promedio )?{NUM}\s*min", t)
    if mu is None and minutos:
        mu = 60 / minutos
    s = _buscar(rf"{num}\s+(?:cajeros?|servidores?|cajas|operadores?|asesores?|"
                rf"ventanillas?|medicos?|mecanicos?|agentes?|puestos?)", t)
    K = _buscar(rf"(?:capacidad maxima de|maximo|caben|cabe)\s+{NUM}", t)
    cs = _buscar(rf"costo (?:de|por) (?:un |cada )?(?:cajero|servidor|asesor|"
                 rf"operador|agente)[^0-9]{{0,20}}{NUM}", t)
    ce = _buscar(rf"costo (?:de|por) (?:la )?espera[^0-9]{{0,30}}{NUM}", t)
    modelo = "M/M/1/K" if K else ("M/M/s" if (s or 1) > 1 or cs else "M/M/1")
    return modelo, {"lambda": lam, "mu": mu, "s": s or 1, "K": K,
                    "costo_servidor": cs, "costo_espera": ce}


def _extraer_inventarios(t):
    D = _buscar(rf"demanda (?:anual )?(?:es |de )?(?:de )?{NUM}", t)
    S = _buscar(rf"costo (?:de |por )?(?:ordenar|pedido|pedir|preparacion|hacer un pedido)"
                rf"[^0-9]{{0,25}}{NUM}", t)
    h = _buscar(rf"(?:mantener|almacenamiento|mantenimiento)[^0-9]{{0,30}}{NUM}", t)
    L = _buscar(rf"tiempo de entrega (?:es )?(?:de )?{NUM}", t)
    p = _buscar(rf"(?:produccion (?:anual )?(?:es )?(?:de )?|produce )(?:a )?{NUM}", t)
    b = _buscar(rf"faltante[^0-9]{{0,30}}{NUM}", t)
    modelo = "EPQ" if p else ("faltantes" if b else "EOQ")
    return modelo, {"D": D, "S": S, "h": h, "p": p, "b": b, "tiempo_entrega_dias": L}


def _extraer_pnl(texto, t):
    from sympy.parsing.sympy_parser import (implicit_multiplication_application,
                                            parse_expr, standard_transformations)
    tr = standard_transformations + (implicit_multiplication_application,)
    limpio = texto.replace("²", "^2").replace("≤", "<=").replace("≥", ">=").replace("−", "-")
    obj = None
    m = re.search(r"=\s*([0-9xyz\s+\-*^.,()]+?)(?=[;\n]|\s+y\s+|\s*\(|$|\s+[a-wA-Z])", limpio)
    if m:
        try:
            expr = m.group(1).replace("^", "**").replace(",", ".").strip()
            obj = str(parse_expr(expr, transformations=tr))
        except Exception:
            obj = None
    restricciones = []
    for izq, op, der in re.findall(r"([0-9xyz\s+\-*.]+?)\s*(<=|>=)\s*(\d+(?:[.,]\d+)?)", limpio):
        try:
            lhs = str(parse_expr(izq.strip(), transformations=tr))
            restricciones.append(f"{lhs} {op} {der.replace(',', '.')}")
        except Exception:
            pass
    sentido = "min" if re.search(r"minimi|costo minimo", t) else "max"
    return "PNL", {"objetivo": obj, "restricciones": restricciones, "sentido": sentido}


def _extraer_pert(texto, t):
    actividades = []
    for linea in re.split(r"[\n;]", texto):
        m = re.match(r"\s*(?:actividad\s+)?([A-Z])\b(.*)", linea)
        if not m:
            continue
        ident, resto = m.groups()
        pred = re.search(r"(?:predecesoras?:?|despues de|después de|tras|luego de|"
                         r"requiere|depende de)\s*([A-Z](?:\s*(?:,|y)\s*[A-Z])*)", resto)
        preds = re.findall(r"[A-Z]", pred.group(1)) if pred else []
        resto_num = resto[:pred.start()] + resto[pred.end():] if pred else resto
        nums = [a_numero(x) for x in re.findall(r"\d+(?:[.,]\d+)?", resto_num)]
        if len(nums) >= 3:
            actividades.append({"id": ident, "pred": preds,
                                "a": nums[0], "m": nums[1], "b": nums[2]})
        elif len(nums) == 1:
            actividades.append({"id": ident, "pred": preds, "t": nums[0]})
    plazo = _buscar(rf"(?:en|antes de|plazo de)\s+{NUM}\s+(?:semanas|dias|meses)", t)
    modelo = "PERT" if any("a" in a for a in actividades) else "CPM"
    return modelo, {"actividades": actividades or None, "plazo": plazo}


def extraer_reglas(texto, modulo):
    t = normalizar(texto)
    if modulo == "colas":
        modelo, p = _extraer_colas(t)
    elif modulo == "inventarios":
        modelo, p = _extraer_inventarios(t)
    elif modulo == "pnl":
        modelo, p = _extraer_pnl(texto, t)
    else:
        modelo, p = _extraer_pert(texto, t)
    return {"modulo": modulo, "modelo": modelo, "parametros": p, "fuente": "reglas"}


# ------------------------------ función pública ------------------------------
def extraer(texto, modulo):
    """Extrae parámetros con el modelo de lenguaje o, si no hay acceso, con reglas."""
    sistema = SISTEMA.format(modulo=modulo, esquema=ESQUEMAS[modulo])
    datos = extraer_json(llamar_llm(sistema, texto))
    if datos and isinstance(datos.get("parametros"), dict):
        datos.update(modulo=modulo, fuente="llm")
    else:
        datos = extraer_reglas(texto, modulo)
    datos["faltantes"], datos["errores"] = validar(modulo, datos["parametros"])
    return datos


def validar(modulo, p):
    """Comprueba datos obligatorios, signos y estabilidad. Devuelve (faltantes, errores)."""
    faltantes = [k for k in OBLIGATORIOS[modulo] if p.get(k) in (None, "", [])]
    errores = []
    for k, v in p.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v <= 0:
            errores.append(f"{k} debe ser positivo (valor recibido: {v})")
    if modulo == "colas" and not faltantes and not p.get("K"):
        s = p.get("s") or 1
        rho = p["lambda"] / (s * p["mu"])
        if rho >= 1 and not p.get("costo_servidor"):
            errores.append(f"Sistema inestable con s = {s}: ρ = {rho:.3f} ≥ 1")
    if modulo == "inventarios" and p.get("p") and p.get("D") and p["p"] <= p["D"]:
        errores.append("La tasa de producción debe superar la demanda")
    return faltantes, errores


if __name__ == "__main__":
    ejemplo = ("A un banco llegan 12 clientes por hora y cada cajero atiende "
               "8 por hora; hay 2 cajeros.")
    print(json.dumps(extraer(ejemplo, "colas"), ensure_ascii=False, indent=2))
