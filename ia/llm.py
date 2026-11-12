"""Conexión con el modelo de lenguaje.

Si existe la variable de entorno ANTHROPIC_API_KEY se usa la API de
Claude; si no, las funciones devuelven None y los módulos de IA usan
su respaldo basado en reglas (el aplicativo funciona sin conexión).
"""
import json
import os
import re

MODELO = os.getenv("IO_MODELO", "claude-sonnet-5-5")


def llm_disponible():
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def llamar_llm(sistema, mensaje, max_tokens=800):
    """Envía un mensaje al modelo y devuelve el texto, o None si no hay acceso."""
    if not llm_disponible():
        return None
    try:
        import anthropic
        cliente = anthropic.Anthropic()
        resp = cliente.messages.create(
            model=MODELO, max_tokens=max_tokens, temperature=0,
            system=sistema, messages=[{"role": "user", "content": mensaje}])
        return "".join(b.text for b in resp.content if b.type == "text")
    except Exception as error:  # red, cuota, clave inválida...
        print(f"[IA] Fallo al llamar al modelo, se usa el respaldo: {error}")
        return None


def extraer_json(texto):
    """Obtiene el primer objeto JSON de una respuesta (tolera bloques ```json)."""
    if texto is None:
        return None
    m = re.search(r"\{.*\}", texto, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
