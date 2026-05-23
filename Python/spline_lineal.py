"""
spline_lineal.py
----------------
Interpolación por spline (trazador) lineal.

Para n+1 puntos (x_i, y_i), se construyen n polinomios de grado 1
S_i(x) = m_i·x + b_i, cada uno válido en el intervalo [x_i, x_{i+1}],
tales que S_i(x_i) = y_i y S_i(x_{i+1}) = y_{i+1}.

Es el caso más simple de spline: continuidad de la función pero NO de
las derivadas. La fórmula explícita por tramo es:
    m_i = (y_{i+1} - y_i) / (x_{i+1} - x_i)
    b_i = y_i - m_i · x_i

Cambios respecto a la versión anterior:

1. Validaciones delegadas en _validar_y_ordenar (importado de Vandermonde).
2. Eliminada la rama del input() que colgaba la GUI sin argumentos.
3. Eliminado código muerto después del return.
4. Se mantiene scipy.interpolate.interp1d para la evaluación (estable,
   probado, no introduce bugs).
5. info_dict incluye 'coeficientes_por_tramo' con los (m_i, b_i)
   numéricos, además de los polinomios como strings.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.interpolate import interp1d

from Python.Vandermonde import _validar_y_ordenar
from Python.supCp3 import SUBinterpol_lagrange
from Python.supCp3 import SUBinterpol_newton
from Python.supCp3 import SUBspln_cubico
from Python.supCp3 import Subvandermonde


def spline_lineal_con_polinomios(ValoresX=None, ValoresY=None,
                                   show_report=True, auto_compare=True):
    """
    Interpolación por spline lineal.

    Devuelve (resultado_str, info_dict) compatible con la GUI.
    """
    eval_grid = 500

    # --- Validación ---
    x, y, advertencias = _validar_y_ordenar(ValoresX, ValoresY)

    # --- Cálculo ---
    start_time = time.perf_counter()
    spline = interp1d(x, y, kind='linear')

    # Coeficientes por tramo
    n_tramos = len(x) - 1
    coeficientes_por_tramo = []
    polinomios = []
    for i in range(n_tramos):
        x0, x1 = x[i], x[i + 1]
        y0, y1 = y[i], y[i + 1]
        m = (y1 - y0) / (x1 - x0)
        b = y0 - m * x0
        coeficientes_por_tramo.append((float(m), float(b)))

        # Representación en forma de Newton (la que el profe enseña)
        polinomio = (
            f"S_{i}(x) = {y0:g} + {m:g}·(x - {x0:g})\n"
            f"    para x ∈ [{x0:g}, {x1:g}]"
        )
        polinomios.append(polinomio)

    # Evaluación para gráfica
    x_plot = np.linspace(min(x), max(x), 500)
    y_spline = spline(x_plot)

    end_time = time.perf_counter()
    tiempo_ejecucion = end_time - start_time

    # --- Resultado para la GUI ---
    puntos_fmt = ", ".join(f"({xi:g}, {yi:g})" for xi, yi in zip(x, y))
    resultado = (
        f"Puntos ingresados (ordenados): {puntos_fmt}\n\n"
        f"Número de tramos: {n_tramos}\n\n"
        f"Polinomios por tramo:\n\n" + "\n\n".join(polinomios) +
        f"\n\nTiempo de ejecución: {tiempo_ejecucion:.6f} segundos"
    )
    if advertencias:
        resultado += "\n\nAdvertencias:\n" + "\n".join(f"- {a}" for a in advertencias)

    info = {
        "tiempo": tiempo_ejecucion,
        "n_tramos": n_tramos,
        "polinomios_por_tramo": polinomios,
        "coeficientes_por_tramo": coeficientes_por_tramo,
        "polinomio_str": None,
        "polinomio_obj": None,
        "polinomios_base": None,
        "coeficientes": None,
        "condicion": None,
        "advertencias": advertencias,
    }

    # --- Gráfica del informe ---
    if show_report:
        try:
            fig, (ax_plot, ax_table) = plt.subplots(
                ncols=2, figsize=(12, 6),
                gridspec_kw={'width_ratios': [3, 2]}
            )
            ax_plot.plot(x, y, 'ro', label='Puntos dados', markersize=8)
            ax_plot.plot(x_plot, y_spline, 'b-', label='Spline Lineal',
                         linewidth=2)
            ax_plot.set_title("Interpolación con Spline Lineal")
            ax_plot.set_xlabel("x")
            ax_plot.set_ylabel("y")
            ax_plot.legend()
            ax_plot.grid()

            primer_pol = polinomios[0] if polinomios else ""
            rows = [
                ["Tiempo (s)", f"{tiempo_ejecucion:.6f}"],
                ["Número de tramos", f"{n_tramos}"],
                ["Primer tramo (trunc)",
                 primer_pol[:80] + ("..." if len(primer_pol) > 80 else "")],
            ]
            ax_table.axis('off')
            table = ax_table.table(
                cellText=rows, colLabels=["Propiedad", "Valor"],
                loc='center', cellLoc='left'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2)
            plt.tight_layout()

            # Polinomios completos en figura aparte
            fig_pol = plt.figure(figsize=(10, 2 + 0.5 * len(polinomios)))
            plt.axis('off')
            text = "\n\n".join(polinomios)
            plt.text(0.01, 0.99, text, va='top', family='monospace', fontsize=9)
            plt.tight_layout()
            plt.show()
        except Exception:
            pass

    # --- Comparación automática (mismo patrón que los otros métodos) ---
    if auto_compare:
        try:
            eval_pts = np.linspace(min(x), max(x), max(100, int(eval_grid)))
            y_ref = spline(eval_pts)

            other_results = {}
            for name, fn in [
                ("Lagrange", lambda: SUBinterpol_lagrange.interpol_lagrange(x, y)),
                ("Newton", lambda: SUBinterpol_newton.interpol_newton(x, y)),
                ("Spline_cubico", lambda: SUBspln_cubico.SUBSUBspline_cubico(x, y)),
                ("Vandermonde", lambda: Subvandermonde.interpol_vandermonde(x, y)),
            ]:
                try:
                    other_results[name] = fn()
                except Exception:
                    other_results[name] = None

            def reeval_on_common(res):
                if res is None:
                    return None
                rx = np.array(res[0])
                ry = np.array(res[1])
                return np.interp(eval_pts, rx, ry)

            metrics = {}
            for name, res in other_results.items():
                y_cmp = reeval_on_common(res)
                if y_cmp is None:
                    metrics[name] = {'max_err': None, 'rmse': None}
                else:
                    diff = y_ref - y_cmp
                    metrics[name] = {
                        'max_err': float(np.max(np.abs(diff))),
                        'rmse': float(np.sqrt(np.mean(diff**2))),
                    }

            if show_report:
                try:
                    fig, (ax_plot2, ax_table) = plt.subplots(
                        ncols=2, figsize=(12, 5),
                        gridspec_kw={'width_ratios': [3, 2]}
                    )
                    ax_plot2.plot(x, y, 'ro', label='Puntos dados')
                    ax_plot2.plot(eval_pts, y_ref, 'k-',
                                  label='Spline lineal (referencia)')
                    colors = {'Lagrange': 'm--', 'Newton': 'c--',
                              'Spline_cubico': 'y--', 'Vandermonde': 'b:'}
                    for name, res in other_results.items():
                        ycmp = reeval_on_common(res)
                        if ycmp is not None:
                            ax_plot2.plot(eval_pts, ycmp,
                                          colors.get(name, '--'), label=name)
                    ax_plot2.set_title('Comparación respecto a Spline lineal')
                    ax_plot2.set_xlabel('x')
                    ax_plot2.set_ylabel('y')
                    ax_plot2.legend()
                    ax_plot2.grid()

                    rows = []
                    for name in ['Lagrange', 'Newton',
                                 'Spline_cubico', 'Vandermonde']:
                        m = metrics.get(name, {})
                        max_err = m.get('max_err')
                        rmse = m.get('rmse')
                        rows.append([
                            name,
                            f"{max_err:.6g}" if max_err is not None else 'N/A',
                            f"{rmse:.6g}" if rmse is not None else 'N/A',
                        ])
                    ax_table.axis('off')
                    table = ax_table.table(
                        cellText=rows,
                        colLabels=['Método', 'Max err', 'RMSE'],
                        loc='center'
                    )
                    table.auto_set_font_size(False)
                    table.set_fontsize(9)
                    table.scale(1, 2)
                    plt.tight_layout()
                    plt.show()
                except Exception:
                    pass
        except Exception:
            pass

    return resultado, info


if __name__ == "__main__":
    res, info = spline_lineal_con_polinomios(
        [1, 2, 3, 4],
        [3.9, 4, 3.8, 4.3],
        show_report=False, auto_compare=False
    )
    print(res)