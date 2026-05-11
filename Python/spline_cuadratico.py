"""
spline_cuadratico.py
--------------------
Interpolación por spline (trazador) cuadrático.

Para n+1 puntos (x_i, y_i) con i=0,...,n, se construyen n polinomios
cuadráticos
    S_i(x) = a_i·x² + b_i·x + c_i
uno por cada tramo [x_i, x_{i+1}], con las siguientes condiciones
(página 23 del PDF del profe):

  1. Cada S_i pasa por sus dos extremos:
       S_i(x_i)=y_i,  S_i(x_{i+1})=y_{i+1}
  2. Continuidad de S' en los nodos internos:
       S'_i(x_i) = S'_{i-1}(x_i)
  3. Condición de borde: S''_0(x_0) = 0
     Esto equivale a 2·a_0 = 0, es decir, el primer tramo se degenera
     a lineal. Es la convención del MATLAB del profe (Spline.m).

Total: n + n + (n-1) + 1 = 3n ecuaciones con 3n incógnitas (donde
n es el número de tramos = puntos - 1).

Se resuelve por np.linalg.solve replicando Spline.m caso d=2.
"""

import numpy as np
import matplotlib.pyplot as plt
import time

from Python.Vandermonde import _validar_y_ordenar
from Python.supCp3 import SUBinterpol_lagrange
from Python.supCp3 import SUBinterpol_newton
from Python.supCp3 import SUBspline_lineal
from Python.supCp3 import SUBspln_cubico
from Python.supCp3 import Subvandermonde


def _resolver_spline_cuadratico(x, y):
    """Monta y resuelve el sistema 3n×3n del spline cuadrático.

    Replica Spline.m del profe (caso d=2). El vector solución tiene
    los coeficientes ordenados como:
        [a_0, b_0, c_0, a_1, b_1, c_1, ..., a_{n-1}, b_{n-1}, c_{n-1}]
    donde cada tramo es S_i(x) = a_i·x² + b_i·x + c_i, con
    a_0 = 0 por la condición S''_0(x_0) = 0.

    Retorna una matriz Tabla de tamaño (n, 3): cada fila son los 3
    coeficientes [a_i, b_i, c_i] de un tramo, en orden descendente
    de potencias.
    """
    n = len(x) - 1  # número de tramos
    cua = x ** 2

    A = np.zeros((3 * n, 3 * n))
    b_vec = np.zeros(3 * n)

    # --- Bloque 1: cada tramo pasa por su punto izquierdo ---
    # S_i(x_i) = y_i, para i = 0, 1, ..., n-1
    h = 0
    c = 0
    for i in range(n):
        A[h, c]     = cua[i]
        A[h, c + 1] = x[i]
        A[h, c + 2] = 1.0
        b_vec[h] = y[i]
        c += 3
        h += 1

    # --- Bloque 2: cada tramo pasa por su punto derecho ---
    # S_i(x_{i+1}) = y_{i+1}, para i = 0, 1, ..., n-1
    c = 0
    for i in range(n):
        A[h, c]     = cua[i + 1]
        A[h, c + 1] = x[i + 1]
        A[h, c + 2] = 1.0
        b_vec[h] = y[i + 1]
        c += 3
        h += 1

    # --- Bloque 3: continuidad de la primera derivada en nodos internos ---
    # S'_i(x_{i+1}) = S'_{i+1}(x_{i+1}), para i = 0, ..., n-2
    # Es decir: 2·a_i·x + b_i - 2·a_{i+1}·x - b_{i+1} = 0
    c = 0
    for i in range(1, n):
        # Nodo interno x[i], que conecta tramos (i-1) e (i)
        A[h, c]     = 2 * x[i]
        A[h, c + 1] = 1.0
        A[h, c + 3] = -2 * x[i]
        A[h, c + 4] = -1.0
        b_vec[h] = 0.0
        c += 3
        h += 1

    # --- Bloque 4: condición de borde S''_0(x_0) = 0 ---
    # Como S'' = 2a, esto es 2·a_0 = 0.
    A[h, 0] = 2.0
    b_vec[h] = 0.0

    # Resolver el sistema
    val = np.linalg.solve(A, b_vec)

    # Reshape a (n, 3): cada fila son los 3 coeficientes de un tramo
    Tabla = val.reshape(n, 3)
    return Tabla


def _evaluar_spline_cuadratico(Tabla, x_nodos, x_eval):
    """Evalúa el spline cuadrático en los puntos x_eval."""
    n_tramos = Tabla.shape[0]
    y_eval = np.zeros_like(x_eval, dtype=float)

    for k, xv in enumerate(x_eval):
        # Encontrar el tramo donde cae xv
        tramo = n_tramos - 1
        for i in range(n_tramos):
            if xv <= x_nodos[i + 1]:
                tramo = i
                break

        a, b, c = Tabla[tramo]
        y_eval[k] = a * xv**2 + b * xv + c

    return y_eval


def _formatear_polinomio_cuadratico(a, b, c, var="x", precision=4):
    """Formatea un polinomio cuadrático a·x²+b·x+c como string."""
    coef = [a, b, c]
    grados = [2, 1, 0]
    partes = []

    for valor, grado in zip(coef, grados):
        if abs(valor) < 10**(-precision - 2):
            continue
        signo = "+" if valor >= 0 else "-"
        abs_val = abs(valor)
        if grado == 0:
            term = f"{abs_val:.{precision}f}"
        elif grado == 1:
            term = f"{abs_val:.{precision}f}{var}"
        else:
            term = f"{abs_val:.{precision}f}{var}^{grado}"
        if not partes:
            partes.append(term if valor >= 0 else f"-{term}")
        else:
            partes.append(f"{signo} {term}")

    if not partes:
        return "0"
    return " ".join(partes)


def spline_cuadratico(ValoresX=None, ValoresY=None,
                       show_report=True, auto_compare=True):
    """
    Interpolación por spline cuadrático.

    Devuelve (resultado_str, info_dict) compatible con la GUI.
    """
    eval_grid = 500

    # --- Validación ---
    x, y, advertencias = _validar_y_ordenar(ValoresX, ValoresY)

    # Validación específica: cuadrático necesita al menos 3 puntos
    if len(x) < 3:
        raise ValueError(
            "El spline cuadrático requiere al menos 3 puntos para que "
            "el sistema de ecuaciones sea consistente. Para 2 puntos "
            "use spline lineal o interpolación de Newton."
        )

    # --- Cálculo ---
    start_time = time.perf_counter()
    Tabla = _resolver_spline_cuadratico(x, y)
    n_tramos = Tabla.shape[0]

    # Construir representación legible
    polinomios = []
    coeficientes_por_tramo = []
    for i in range(n_tramos):
        a, b_, c = Tabla[i]
        coeficientes_por_tramo.append((float(a), float(b_), float(c)))
        poly_str = _formatear_polinomio_cuadratico(a, b_, c)
        polinomio = (
            f"S_{i}(x) = {poly_str}\n"
            f"    para x ∈ [{x[i]:g}, {x[i+1]:g}]"
        )
        polinomios.append(polinomio)

    # Evaluación para gráfica
    x_plot = np.linspace(min(x), max(x), 500)
    y_spline = _evaluar_spline_cuadratico(Tabla, x, x_plot)

    end_time = time.perf_counter()
    tiempo_ejecucion = end_time - start_time

    # --- Resultado para la GUI ---
    puntos_fmt = ", ".join(f"({xi:g}, {yi:g})" for xi, yi in zip(x, y))
    resultado = (
        f"Puntos ingresados (ordenados): {puntos_fmt}\n\n"
        f"Número de tramos: {n_tramos}\n"
        f"Condición de borde: S''(x_0) = 0  →  a_0 = 0  "
        f"(primer tramo lineal)\n\n"
        f"Polinomios por tramo:\n\n" + "\n\n".join(polinomios) +
        f"\n\nTabla de coeficientes (a·x² + b·x + c):\n"
    )
    resultado += f"  {'Tramo':>5}  {'a':>12}  {'b':>12}  {'c':>12}\n"
    for i in range(n_tramos):
        a, b_, c = Tabla[i]
        resultado += f"  {i:>5}  {a:>12.6f}  {b_:>12.6f}  {c:>12.6f}\n"
    resultado += f"\nTiempo de ejecución: {tiempo_ejecucion:.6f} segundos"

    if advertencias:
        resultado += "\n\nAdvertencias:\n" + "\n".join(f"- {a}" for a in advertencias)

    info = {
        "tiempo": tiempo_ejecucion,
        "n_tramos": n_tramos,
        "polinomios_por_tramo": polinomios,
        "coeficientes_por_tramo": coeficientes_por_tramo,
        "tabla_coeficientes": Tabla.tolist(),
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
            ax_plot.plot(x_plot, y_spline, 'b-', label='Spline Cuadrático',
                         linewidth=2)
            ax_plot.set_title("Interpolación con Spline Cuadrático")
            ax_plot.set_xlabel("x")
            ax_plot.set_ylabel("y")
            ax_plot.legend()
            ax_plot.grid()

            rows = [
                ["Tiempo (s)", f"{tiempo_ejecucion:.6f}"],
                ["Número de tramos", f"{n_tramos}"],
                ["Primer tramo (trunc)",
                 polinomios[0][:80] + ("..." if len(polinomios[0]) > 80 else "")],
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

    # --- Comparación automática ---
    if auto_compare:
        try:
            eval_pts = np.linspace(min(x), max(x), max(100, int(eval_grid)))
            y_ref = _evaluar_spline_cuadratico(Tabla, x, eval_pts)

            other_results = {}
            for name, fn in [
                ("Lagrange", lambda: SUBinterpol_lagrange.interpol_lagrange(x, y)),
                ("Newton", lambda: SUBinterpol_newton.interpol_newton(x, y)),
                ("Spline_lineal", lambda: SUBspline_lineal.SUBSUBspline_lineal(x, y)),
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
                        'rmse': float(np.sqrt(np.mean(diff ** 2))),
                    }

            if show_report:
                try:
                    fig, (ax_plot2, ax_table) = plt.subplots(
                        ncols=2, figsize=(12, 5),
                        gridspec_kw={'width_ratios': [3, 2]}
                    )
                    ax_plot2.plot(x, y, 'ro', label='Puntos dados')
                    ax_plot2.plot(eval_pts, y_ref, 'k-',
                                  label='Spline cuadrático (referencia)')
                    colors = {'Lagrange': 'm--', 'Newton': 'c--',
                              'Spline_lineal': 'g--', 'Spline_cubico': 'y--',
                              'Vandermonde': 'b:'}
                    for name, res in other_results.items():
                        ycmp = reeval_on_common(res)
                        if ycmp is not None:
                            ax_plot2.plot(eval_pts, ycmp,
                                          colors.get(name, '--'), label=name)
                    ax_plot2.set_title('Comparación respecto a Spline cuadrático')
                    ax_plot2.set_xlabel('x')
                    ax_plot2.set_ylabel('y')
                    ax_plot2.legend()
                    ax_plot2.grid()

                    rows = []
                    for name in ['Lagrange', 'Newton', 'Spline_lineal',
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
    # Ejemplo del profe (mismo de ejemnewt.m, caso d=2)
    res, info = spline_cuadratico(
        [1, 2, 3, 4],
        [3.9, 4, 3.8, 4.3],
        show_report=False, auto_compare=False
    )
    print(res)