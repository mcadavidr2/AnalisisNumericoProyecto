"""
interpolacion_newton.py
-----------------------
Interpolación por polinomio de Newton (diferencias divididas).

Replica el flujo del MATLAB del profe:
  - Newtonint.m: construye la tabla de diferencias divididas
  - Newtonor.m: expande el polinomio a su forma estándar

Cambios respecto a la versión anterior:

1. Validación delegada en _validar_y_ordenar (importada de Vandermonde),
   mismas reglas: orden, duplicados, NaN, longitudes.

2. Eliminada la rama del input() que colgaba la GUI sin argumentos.

3. Eliminado código muerto después del return.

4. La tabla de diferencias divididas se almacena en convención triangular
   INFERIOR como en el MATLAB del profe. Los coeficientes b_i del
   polinomio están en la DIAGONAL (no en la primera fila). Esto permite
   mostrar la tabla con el mismo formato del PDF del profe (página 18).

5. NUEVO: se muestra la tabla de diferencias divididas con formato
   alineado, equivalente al de las diapositivas del profe.

6. NUEVO: se muestran AMBAS formas del polinomio:
   - Forma factorizada (canónica de Newton)
   - Forma expandida (potencias estándar) construida con Newtonor del profe

7. Firma y formato del info_dict compatibles con Vandermonde y Lagrange.
"""

import numpy as np
import time
import matplotlib.pyplot as plt

from Python.Vandermonde import _validar_y_ordenar
from Python.supCp3 import SUBinterpol_lagrange
from Python.supCp3 import SUBspln_cubico
from Python.supCp3 import SUBspline_lineal
from Python.supCp3 import Subvandermonde


def _tabla_diferencias_divididas(x, y):
    """Construye la tabla de diferencias divididas en convención triangular
    INFERIOR (la del MATLAB Newtonint.m del profe).

    Estructura de la tabla T (n x (n+1)):
      T[:, 0] = x_i
      T[:, 1] = y_i = f[x_i]
      T[:, 2] = primeras diferencias divididas
      T[:, 3] = segundas diferencias divididas
      ...
    Los coeficientes b_i del polinomio están en la DIAGONAL:
      b_0 = T[0, 1], b_1 = T[1, 2], b_2 = T[2, 3], ...
    """
    n = len(x)
    T = np.zeros((n, n + 1))
    T[:, 0] = x
    T[:, 1] = y

    # Replica del MATLAB:
    #   for j=3:n+1
    #     for i=j-1:n
    #       T(i,j) = (T(i,j-1) - T(i-1,j-1)) / (T(i,1) - T(i-j+2,1))
    # Pasando a Python (0-indexed), j de MATLAB es j-1 aquí.
    for j in range(2, n + 1):
        for i in range(j - 1, n):
            T[i, j] = (T[i, j-1] - T[i-1, j-1]) / (T[i, 0] - T[i - (j - 1), 0])

    return T


def _coeficientes_newton(T):
    """Extrae los coeficientes b_i del polinomio de Newton desde la
    DIAGONAL de la tabla (la convención del profe).
    """
    n = T.shape[0]
    # b_i = T[i, i+1]  para i = 0, 1, ..., n-1
    return np.array([T[i, i + 1] for i in range(n)])


def _expandir_newton(x, b):
    """Replica el MATLAB Newtonor.m del profe: expande el polinomio
    de Newton desde su forma factorizada
        p(x) = b_0 + b_1(x-x_0) + b_2(x-x_0)(x-x_1) + ...
    a su forma estándar
        p(x) = a_n x^n + a_{n-1} x^{n-1} + ... + a_0
    devolviendo los coeficientes a_i en orden DESCENDENTE de potencias.
    """
    n = len(x)
    # Acumulador: polinomio (x-x_0)(x-x_1)...(x-x_{i-1})
    # Empieza siendo el polinomio constante 1.
    acum = np.array([1.0])
    pol = b[0] * acum.copy()

    for i in range(n - 1):
        # Equivalente a 'pol = [0 pol]' del MATLAB: agregamos coeficiente
        # cero al inicio (sube de grado) para preparar la suma.
        pol = np.concatenate([[0.0], pol])

        # Acumular el siguiente factor (x - x_i)
        factor = np.array([1.0, -x[i]])
        acum = np.convolve(acum, factor)

        # Sumar b_{i+1} * acum, alineando con pol al final
        pad = len(pol) - len(acum)
        acum_pad = np.concatenate([np.zeros(pad), acum]) if pad > 0 else acum
        pol = pol + b[i + 1] * acum_pad

    return pol


def _formatear_tabla_diferencias(T, x, precision=4):
    """Formatea la tabla de diferencias divididas como string alineado,
    estilo de las diapositivas del profe (página 18).
    """
    n = T.shape[0]
    headers = ["n", "x_i", "f[x_i]"]
    for k in range(1, n):
        if k == 1:
            headers.append("1ra")
        elif k == 2:
            headers.append("2da")
        elif k == 3:
            headers.append("3ra")
        else:
            headers.append(f"{k}ta")

    # Ancho de columna fijo para alineación
    w = precision + 6
    lineas = []
    lineas.append(" ".join(f"{h:>{w}}" for h in headers))

    for i in range(n):
        fila = [f"{i:>{w}}"]
        fila.append(f"{x[i]:>{w}.{precision}f}")
        # Las columnas de diferencias: T[i, 1], T[i, 2], ..., T[i, i+1]
        # (lo demás es triangular superior, queda vacío)
        for j in range(1, i + 2):
            fila.append(f"{T[i, j]:>{w}.{precision}f}")
        # Rellenar con espacios las celdas vacías (triángulo superior)
        for _ in range(n - 1 - i):
            fila.append(" " * w)
        lineas.append(" ".join(fila))

    return "\n".join(lineas)


def _formatear_polinomio(coef, var="x", precision=6):
    """Formatea coeficientes en orden DESCENDENTE como string legible."""
    n = len(coef)
    grado_max = n - 1
    partes = []

    for i, c in enumerate(coef):
        grado = grado_max - i
        if abs(c) < 10**(-precision):
            continue
        signo = "+" if c >= 0 else "-"
        valor = abs(c)
        if grado == 0:
            term = f"{valor:.{precision-2}g}"
        elif grado == 1:
            term = f"{valor:.{precision-2}g} {var}"
        else:
            term = f"{valor:.{precision-2}g} {var}^{grado}"
        if not partes:
            partes.append(term if c >= 0 else f"-{term}")
        else:
            partes.append(f"{signo} {term}")

    if not partes:
        return "0"
    return " ".join(partes)


def _formatear_polinomio_newton_factorizado(x, b, precision=4):
    """Forma factorizada: p(x) = b_0 + b_1(x-x_0) + b_2(x-x_0)(x-x_1) + ..."""
    partes = [f"{b[0]:.{precision}f}"]
    factores = ""
    for i in range(1, len(b)):
        # Acumula el factor (x - x_{i-1})
        factores += f"(x - {x[i-1]:g})"
        signo = "+" if b[i] >= 0 else "-"
        partes.append(f"{signo} {abs(b[i]):.{precision}f}{factores}")
    return " ".join(partes)


def interpolacion_newton(ValoresX=None, ValoresY=None,
                          show_report=True, auto_compare=True):
    """
    Interpolación por polinomio de Newton con diferencias divididas.

    Devuelve (resultado_str, info_dict) compatible con la GUI.
    """
    eval_grid = 500

    # --- Validación ---
    x, y, advertencias = _validar_y_ordenar(ValoresX, ValoresY)

    # --- Cálculo ---
    start_time = time.time()

    # Paso 1: construir la tabla de diferencias divididas (Newtonint del profe)
    T = _tabla_diferencias_divididas(x, y)

    # Paso 2: extraer los coeficientes b_i (la diagonal de la tabla)
    b = _coeficientes_newton(T)

    # Paso 3: expandir a forma estándar (Newtonor del profe)
    coef_expandidos = _expandir_newton(x, b)
    poly_obj = np.poly1d(coef_expandidos)

    end_time = time.time()
    tiempo_ejecucion = end_time - start_time

    # --- Strings para mostrar ---
    tabla_str = _formatear_tabla_diferencias(T, x)
    forma_factorizada = _formatear_polinomio_newton_factorizado(x, b)
    forma_expandida = _formatear_polinomio(coef_expandidos)

    # --- Gráfica ---
    x_plot = np.linspace(min(x), max(x), 500)
    y_newton = np.polyval(coef_expandidos, x_plot)

    # --- Resultado para la GUI ---
    puntos_fmt = ", ".join(f"({xi:g}, {yi:g})" for xi, yi in zip(x, y))
    resultado = (
        f"Puntos ingresados (ordenados): {puntos_fmt}\n\n"
        f"Tabla de diferencias divididas:\n"
        f"{tabla_str}\n\n"
        f"Coeficientes b_i (diagonal de la tabla):\n  {b}\n\n"
        f"Forma factorizada de Newton:\n  P(x) = {forma_factorizada}\n\n"
        f"Polinomio expandido:\n  P(x) = {forma_expandida}\n\n"
        f"Tiempo de ejecución: {tiempo_ejecucion:.6f} segundos"
    )
    if advertencias:
        resultado += "\n\nAdvertencias:\n" + "\n".join(f"- {a}" for a in advertencias)

    info = {
        "tiempo": tiempo_ejecucion,
        "coeficientes": coef_expandidos.tolist(),
        "coeficientes_newton": b.tolist(),
        "tabla_diferencias": T.tolist(),
        "polinomio_str": forma_expandida,
        "polinomio_factorizado": forma_factorizada,
        "polinomio_obj": poly_obj,
        "polinomios_base": None,
        "polinomios_por_tramo": None,
        "n_tramos": None,
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
            ax_plot.plot(x, y, 'ro', label='Puntos dados')
            ax_plot.plot(x_plot, y_newton, 'b-', label='Polinomio de Newton')
            ax_plot.set_title("Interpolación de Newton")
            ax_plot.set_xlabel("x")
            ax_plot.set_ylabel("y")
            ax_plot.legend()
            ax_plot.grid()

            rows = [
                ["Tiempo (s)", f"{tiempo_ejecucion:.6f}"],
                ["Grado", f"{len(x)-1}"],
                ["Polinomio (trunc)",
                 forma_expandida[:120] +
                 ("..." if len(forma_expandida) > 120 else "")],
            ]
            ax_table.axis('off')
            table = ax_table.table(
                cellText=rows, colLabels=["Propiedad", "Valor"],
                loc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2)
            plt.tight_layout()

            # Figura adicional con la tabla de diferencias divididas
            fig_tab = plt.figure(figsize=(10, 1 + 0.4 * len(x)))
            plt.axis('off')
            plt.text(0.01, 0.99, tabla_str, va='top',
                     family='monospace', fontsize=9)
            plt.tight_layout()
            plt.show()
        except Exception:
            pass

    # --- Comparación automática (mismo patrón que Lagrange) ---
    if auto_compare:
        try:
            eval_pts = np.linspace(min(x), max(x), max(100, int(eval_grid)))
            y_ref = np.polyval(coef_expandidos, eval_pts)

            other_results = {}
            for name, fn in [
                ("Lagrange", lambda: SUBinterpol_lagrange.interpol_lagrange(x, y)),
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
                        'rmse': float(np.sqrt(np.mean(diff**2))),
                    }

            if show_report:
                try:
                    fig, (ax_plot, ax_table) = plt.subplots(
                        ncols=2, figsize=(12, 5),
                        gridspec_kw={'width_ratios': [3, 2]}
                    )
                    ax_plot.plot(x, y, 'ro', label='Puntos dados')
                    ax_plot.plot(eval_pts, y_ref, 'k-', label='Newton (referencia)')
                    colors = {'Lagrange': 'm--', 'Spline_lineal': 'g--',
                              'Spline_cubico': 'y--', 'Vandermonde': 'b:'}
                    for name, res in other_results.items():
                        ycmp = reeval_on_common(res)
                        if ycmp is not None:
                            ax_plot.plot(eval_pts, ycmp,
                                         colors.get(name, '--'), label=name)
                    ax_plot.set_title('Comparación respecto a Newton')
                    ax_plot.set_xlabel('x')
                    ax_plot.set_ylabel('y')
                    ax_plot.legend()
                    ax_plot.grid()

                    rows = []
                    for name in ['Lagrange', 'Spline_lineal',
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
                        cellText=rows, colLabels=['Método', 'Max err', 'RMSE'],
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
    # Caso del PDF del profe (página 7-9): debe dar el polinomio
    # p(x) = 0.4124 x^3 + 0.9394 x^2 - 5.836 x + 0.0047
    res, info = interpolacion_newton(
        [-2, -1, 2, 3],
        [12.13533528, 6.367879441, -4.610943901, 2.085536923],
        show_report=False, auto_compare=False
    )
    print(res)