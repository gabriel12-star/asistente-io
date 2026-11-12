"""Motor de cálculo: recibe los parámetros confirmados y llama al módulo correcto."""
from modulos import colas, inventarios, pert_cpm, pnl


def resolver(modulo, modelo, p):
    if modulo == "pnl":
        return pnl.resolver_pnl(p["objetivo"], p.get("restricciones", []),
                                p.get("sentido", "max"))
    if modulo == "colas":
        if p.get("K"):
            return colas.mm1k(p["lambda"], p["mu"], p["K"])
        if p.get("costo_servidor") and p.get("costo_espera"):
            tabla, s_opt = colas.costo_por_servidores(
                p["lambda"], p["mu"], p["costo_servidor"], p["costo_espera"])
            return {"tabla": tabla, "s_optimo": s_opt}
        if (p.get("s") or 1) > 1:
            return colas.mms(p["lambda"], p["mu"], p["s"])
        return colas.mm1(p["lambda"], p["mu"])
    if modulo == "inventarios":
        if p.get("p"):
            r = inventarios.epq(p["D"], p["S"], p["h"], p["p"])
        elif p.get("b"):
            r = inventarios.eoq_faltantes(p["D"], p["S"], p["h"], p["b"])
        else:
            r = inventarios.eoq(p["D"], p["S"], p["h"])
        if p.get("tiempo_entrega_dias"):
            r.update(inventarios.punto_reorden(p["D"], p["tiempo_entrega_dias"]))
        return r
    if modulo == "pert_cpm":
        return pert_cpm.analizar_proyecto(p["actividades"], p.get("plazo"))
    raise ValueError(f"Módulo desconocido: {modulo}")
