"""
Vandermonde.py
--------------
Interpolación por método de Vandermonde.

Cambios respecto a la versión anterior:

1. Ordenamiento automático de x (e y consistente). Antes se rechazaban
   datos no ordenados con un ValueError; ahora se ordenan internamente.

2. Validación específica de x repetidos. Antes se rechazaban con el
   mismo mensaje confuso de "no están en orden creciente"; ahora con
   un mensaje claro que identifica los duplicados.

3. Advertencia (no rechazo) cuando la matriz está mal condicionada.
   El profe en el PDF dice que Vandermonde es "no recomendable" por
   este motivo; advertimos al usuario sin abortar el cálculo.

4. Validación de tipos numéricos en x e y. Si vienen NaN o inf por
   algún error del usuario, se detecta antes en lugar de propagar.

5. La firma de retorno se mantiene: (resultado_str, info_dict).
   Solo se agrega 'advertencias' al info_dict para que la GUI pueda
   mostrarlas si quiere.
"""

import numpy as np
import time
import matplotlib.pyplot as plt

from Python.supCp3 import SUBinterpol_lagrange
from Python.supCp3 import SUBinterpol_newton
from Python.supCp3 import SUBspln_cubico
from Python.supCp3 import SUBspline_lineal


# Umbral de "condicionamiento alto": con condición >= COND_WARN, los
# coeficientes pueden tener errores numéricos significativos en double.
# log10(1e10) ≈ 10 dígitos perdidos de los ~16 disponibles.
COND_WARN = 1e10


def _validar_y_ordenar(ValoresX, ValoresY):
    """Valida los datos de entrada y los ordena por x.

    Retorna (x_ordenado, y_ordenado, advertencias).
    Lanza ValueError SOLO en casos matemáticamente irreparables:
    - menos de 2 puntos
    - x e y de longitudes distintas
    - x repetidos
    - valores no finitos (NaN, inf)
    """
    if ValoresX is None or ValoresY is None:
        raise ValueError("Debe proporcionar listas de valores x e y.")

    x = np.array(ValoresX, dtype=float)
    y = np.array(ValoresY, dtype=float)

    # Mínimo de puntos
    if x.ndim == 0 or len(x) < 2:
        raise ValueError("Debe ingresar al menos dos puntos para interpolar.")

    # Longitudes coincidentes
    if len(x) != len(y):
        raise ValueError(
            f"Las listas de x e y deben tener la misma longitud "
            f"(recibí len(x)={len(x)}, len(y)={len(y)})."
        )

    # Valores finitos
    if not np.all(np.isfinite(x)):
        raise ValueError("Los valores de x contienen NaN o infinito.")
    if not np.all(np.isfinite(y)):
        raise ValueError("Los valores de y contienen NaN o infinito.")

    # Detectar repetidos en x. np.unique con return_counts nos dice cuáles
    # y cuántas veces aparece cada uno.
    valores, conteos = np.unique(x, return_counts=True)
    duplicados = valores[conteos > 1]
    if len(duplicados) > 0:
        # Mensaje específico que el usuario pueda accionar
        lista = ", ".join(f"x={v}" for v in duplicados)
        raise ValueError(
            f"Los valores de x deben ser únicos. Encontré valor(es) "
            f"repetido(s): {lista}. La interpolación polinómica requiere "
            f"que cada x aparezca una sola vez."
        )

    # Ordenar por x. argsort devuelve los índices que ordenan x;
    # aplicamos esa permutación a y también para mantener correspondencia.
    advertencias = []
    if not np.all(np.diff(x) > 0):
        idx = np.argsort(x)
        x = x[idx]
        y = y[idx]
        advertencias.append(
            "Los datos se reordenaron internamente por x ascendente."
        )

    return x, y, advertencias


def interpolacion_vandermonde(ValoresX=None, ValoresY=None,
                               show_report=True, auto_compare=True):
    """
    Interpolación de Vandermonde: monta la matriz de potencias y resuelve.

    Devuelve (resultado_str, info_dict) compatible con la GUI.
    info_dict incluye 'advertencias' (lista de strings) además de los
    campos del resto de métodos.
    """
    # --- Validación ---
    x, y, advertencias = _validar_y_ordenar(ValoresX, ValoresY)

    # --- Cálculo principal ---
    start_time = time.perf_counter()
    V = np.vander(x, increasing=True)
    cond_num = np.linalg.cond(V)
    a = np.linalg.solve(V, y)
    end_time = time.perf_counter()
    tiempo_ejecucion = end_time - start_time

    # Advertencia de condicionamiento alto.
    # No abortamos: el usuario puede querer ver justamente este efecto.
    if cond_num >= COND_WARN:
        advertencias.append(
            f"Matriz de Vandermonde mal condicionada (cond = {cond_num:.2e}). "
            f"Los coeficientes pueden tener errores numéricos significativos. "
            f"Considera usar Newton o splines para mayor estabilidad."
        )

    # --- Construcción del polinomio para mostrar ---
    # numpy.poly1d espera coeficientes en orden DESCENDENTE de potencias.
    # Nuestra 'a' viene en ascendente, por eso invertimos con [::-1].
    x_plot = np.linspace(min(x), max(x), 100)
    y_vander = np.polyval(a[::-1], x_plot)
    poly = np.poly1d(a[::-1])
    poly_str = str(poly)

    # --- Resultado para la GUI ---
    puntos_fmt = ", ".join(f"({xi:g}, {yi:g})" for xi, yi in zip(x, y))
    resultado = (
        f"Puntos ingresados (ordenados): {puntos_fmt}\n"
        f"Coeficientes del polinomio (orden ascendente de potencias):\n"
        f"  {a}\n"
        f"Polinomio interpolador p(x):\n{poly_str}\n"
        f"Tiempo de ejecución: {tiempo_ejecucion:.6f} segundos\n"
        f"Número de condición de la matriz: {cond_num:.2e}"
    )
    if advertencias:
        resultado += "\n\nAdvertencias:\n" + "\n".join(f"- {a}" for a in advertencias)

    info = {
        "tiempo": tiempo_ejecucion,
        "coeficientes": a.tolist(),
        "polinomio_str": poly_str,
        "polinomio_obj": poly,
        "condicion": float(cond_num),
        "n_tramos": None,
        "polinomios_por_tramo": None,
        "advertencias": advertencias,
    }

    # --- Gráfica del informe (sin cambios funcionales, solo más robusta) ---
    if show_report:
        try:
            fig, (ax_plot, ax_table) = plt.subplots(
                ncols=2, figsize=(12, 6),
                gridspec_kw={'width_ratios': [3, 2]}
            )
            ax_plot.plot(x, y, 'ro', label='Puntos dados')
            ax_plot.plot(x_plot, y_vander, 'b-', label='Vandermonde')
            ax_plot.set_title("Interpolación con Vandermonde")
            ax_plot.set_xlabel("x")
            ax_plot.set_ylabel("y")
            ax_plot.legend()
            ax_plot.grid()

            # Tabla de resumen. Si hay advertencias las incluimos al final.
            rows = [
                ["Tiempo (s)", f"{tiempo_ejecucion:.6f}"],
                ["Número de condición", f"{cond_num:.3e}"],
                ["Coeficientes (orden asc.)",
                 ", ".join([f"{c:.6g}" for c in a])],
            ]
            if advertencias:
                rows.append(["Advertencias", advertencias[0][:80] + "..."])

            ax_table.axis('off')
            table = ax_table.table(
                cellText=rows, colLabels=["Propiedad", "Valor"],
                loc='center', cellLoc='left'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2)

            plt.tight_layout()

            # Polinomio en figura pequeña aparte
            fig_pol = plt.figure(figsize=(8, 1.5))
            plt.axis('off')
            plt.text(0.01, 0.5, f"Polinomio p(x): {poly_str}",
                     va='center', family='monospace', fontsize=9)
            plt.tight_layout()
            plt.show()
        except Exception:
            # No queremos que un fallo de matplotlib tumbe el método
            pass

    return resultado, info


def comparar_metodos(ValoresX, ValoresY, show_report=True):
    """Comparación de Vandermonde con los demás métodos.

    Esta función auxiliar la dejamos como estaba, solo aplicando
    validación y ordenamiento al inicio.
    """
    x, y, _ = _validar_y_ordenar(ValoresX, ValoresY)
    x_plot = np.linspace(min(x), max(x), max(100, int(500)))

    # Vandermonde
    t0 = time.perf_counter()
    V = np.vander(x, increasing=True)
    cond_num = np.linalg.cond(V)
    a = np.linalg.solve(V, y)
    t1 = time.perf_counter()
    t_vander = t1 - t0
    y_vander = np.polyval(a[::-1], x_plot)

    # Otros métodos
    t0 = time.perf_counter()
    ILG = SUBinterpol_lagrange.interpol_lagrange(x, y)
    t1 = time.perf_counter()
    t_lagrange = t1 - t0

    t0 = time.perf_counter()
    INT = SUBinterpol_newton.interpol_newton(x, y)
    t1 = time.perf_counter()
    t_newton = t1 - t0

    t0 = time.perf_counter()
    SPCC = SUBspln_cubico.SUBSUBspline_cubico(x, y)
    t1 = time.perf_counter()
    t_spline_cub = t1 - t0

    t0 = time.perf_counter()
    SPL = SUBspline_lineal.SUBSUBspline_lineal(x, y)
    t1 = time.perf_counter()
    t_spline_lin = t1 - t0

    def reeval_on_common(res):
        rx = np.array(res[0])
        ry = np.array(res[1])
        return np.interp(x_plot, rx, ry)

    y_ilg = reeval_on_common(ILG)
    y_int = reeval_on_common(INT)
    y_spcc = reeval_on_common(SPCC)
    y_spl = reeval_on_common(SPL)

    def metrics(y_ref, y_cmp):
        diff = y_ref - y_cmp
        return float(np.max(np.abs(diff))), float(np.sqrt(np.mean(diff**2)))

    m_lagrange = metrics(y_vander, y_ilg)
    m_newton = metrics(y_vander, y_int)
    m_splc = metrics(y_vander, y_spcc)
    m_spll = metrics(y_vander, y_spl)

    if show_report:
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(x, y, 'ro', label='Puntos dados')
            plt.plot(x_plot, y_vander, 'b-', label='Vandermonde')
            plt.plot(x_plot, y_spl, 'g--', label='Spline lineal')
            plt.plot(x_plot, y_ilg, 'm--', label='Lagrange')
            plt.plot(x_plot, y_int, 'c--', label='Newton')
            plt.plot(x_plot, y_spcc, 'y--', label='Spline Cúbico')
            plt.title("Comparación General")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.legend()
            plt.grid()
            plt.show()
        except Exception:
            pass

    informe = {
        "Vandermonde": {"tiempo": t_vander, "condicion": cond_num,
                        "coeficientes": a.tolist()},
        "Lagrange": {"tiempo": t_lagrange,
                     "max_err_vs_vander": m_lagrange[0],
                     "rmse_vs_vander": m_lagrange[1],
                     "polinomio": ILG[2] if len(ILG) > 2 else None},
        "Newton": {"tiempo": t_newton,
                   "max_err_vs_vander": m_newton[0],
                   "rmse_vs_vander": m_newton[1],
                   "polinomio": INT[2] if len(INT) > 2 else None},
        "Spline_lineal": {"tiempo": t_spline_lin,
                          "max_err_vs_vander": m_spll[0],
                          "rmse_vs_vander": m_spll[1]},
        "Spline_cubico": {"tiempo": t_spline_cub,
                          "max_err_vs_vander": m_splc[0],
                          "rmse_vs_vander": m_splc[1]},
    }
    return informe