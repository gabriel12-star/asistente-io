"""Ejecuta los cuatro casos de estudio del informe y guarda las gráficas.

Ejecutar con:  python ejemplos.py
"""
import os

from ia.clasificador import clasificar
from ia.explicador import explicar
from ia.extractor import extraer
from modulos import colas, graficas, inventarios, pert_cpm, pnl
from motor import resolver

os.makedirs("figuras", exist_ok=True)

print("=== 1. PNL: asignación de recursos ===")
print("Punto estacionario sin restricción:",
      pnl.punto_estacionario("40*x + 30*y - x**2 - 2*y**2"))
r1 = pnl.resolver_pnl("40*x + 30*y - x**2 - 2*y**2", ["x + y <= 20"])
print(f"x* = {r1['x']['x']:.4f}, y* = {r1['x']['y']:.4f}, f* = {r1['f']:.4f}")
print("Multiplicadores:", {k: round(v, 4) for k, v in r1["multiplicadores"].items()})
print("Hessiana:", r1["curvatura"], r1["autovalores_hessiana"])
graficas.grafica_pnl("40*x + 30*y - x**2 - 2*y**2", ["x + y <= 20"], r1) \
    .savefig("figuras/pnl.png", dpi=200)

print("\n=== 2. Colas: número óptimo de cajeros ===")
tabla, s_opt = colas.costo_por_servidores(12, 8, 20, 30, s_max=4)
print(f"{'s':>2} {'rho':>6} {'P0':>7} {'Lq':>7} {'L':>7} {'Wq(min)':>8} {'CT':>8}")
for f in tabla:
    if not f["estable"]:
        print(f"{f['s']:>2} {f['rho']:>6.3f} inestable (rho >= 1)")
    else:
        print(f"{f['s']:>2} {f['rho']:>6.3f} {f['P0']:>7.4f} {f['Lq']:>7.4f} "
              f"{f['L']:>7.4f} {f['Wq'] * 60:>8.2f} {f['costo_total']:>8.2f}")
print("Número óptimo de cajeros:", s_opt)
graficas.grafica_costos_colas(tabla, s_opt).savefig("figuras/colas.png", dpi=200)

print("\n=== 3. Inventarios: EOQ del distribuidor ===")
r3 = inventarios.eoq(12000, 50, 2)
rp = inventarios.punto_reorden(12000, 5)
print(f"Q* = {r3['Q']:.2f} unidades | N = {r3['N']:.2f} pedidos/año | "
      f"ciclo = {r3['ciclo_dias']:.2f} días")
print(f"Costo pedir = {r3['costo_pedir']:.2f} | costo mantener = "
      f"{r3['costo_mantener']:.2f} | CT = {r3['costo_total']:.2f}")
print(f"d = {rp['demanda_diaria']:.2f} unid/día | R = {rp['R']:.2f} unidades")
graficas.grafica_eoq(12000, 50, 2).savefig("figuras/eoq.png", dpi=200)

print("\n=== 4. PERT/CPM: implementación de un sistema ===")
acts = [{"id": "A", "pred": [], "a": 2, "m": 4, "b": 6},
        {"id": "B", "pred": [], "a": 3, "m": 5, "b": 13},
        {"id": "C", "pred": ["A"], "a": 2, "m": 3, "b": 4},
        {"id": "D", "pred": ["A", "B"], "a": 4, "m": 6, "b": 8},
        {"id": "E", "pred": ["C", "D"], "a": 1, "m": 2, "b": 3}]
r4 = pert_cpm.analizar_proyecto(acts, plazo=16)
print(f"{'Act':>3} {'te':>5} {'var':>6} {'ES':>5} {'EF':>5} {'LS':>5} {'LF':>5} {'H':>4}")
for f in r4["tabla"]:
    print(f"{f['id']:>3} {f['te']:>5.2f} {f['var']:>6.3f} {f['ES']:>5.1f} {f['EF']:>5.1f} "
          f"{f['LS']:>5.1f} {f['LF']:>5.1f} {f['holgura']:>4.1f}")
print(f"Ruta crítica: {r4['rutas_criticas']} | duración = {r4['duracion']:.2f} | "
      f"varianza = {r4['varianza']:.4f} | sigma = {r4['sigma']:.4f}")
print(f"Z = {r4['z']:.4f} | P(T <= 16) = {r4['probabilidad']:.4f}")
print(f"Plazo con 95 % de confianza: {pert_cpm.plazo_para_probabilidad(r4, 0.95):.2f} semanas")
graficas.grafica_pert(r4).savefig("figuras/pert.png", dpi=200)

print("\n=== 5. Flujo completo con IA (lenguaje natural) ===")
texto = ("A un banco llegan 12 clientes por hora y cada cajero atiende 8 por hora; "
         "hay 2 cajeros.")
clase = clasificar(texto)
ext = extraer(texto, clase["modulo"])
print("Clasificación:", clase)
print("Extracción:", {k: ext[k] for k in ("modelo", "parametros", "faltantes")})
res = resolver(ext["modulo"], ext["modelo"], ext["parametros"])
print("Explicación:", explicar(ext["modulo"], res))
