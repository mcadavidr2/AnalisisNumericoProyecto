"""
interpolacion_lagrange.py
-------------------------
Interpolación de Lagrange.

Cambios respecto a la versión anterior:

1. Importa _validar_y_ordenar desde Vandermonde para no duplicar
   lógica. Mismas validaciones que Vandermonde:
   - mínimo de puntos
   - longitudes coincidentes
   - x repetidos detectados con mensaje específico
   - NaN/inf rechazados
   - reordenamiento automático con advertencia

2. Se eliminó la rama del input() que colgaba la GUI cuando se
   llamaba sin argumentos. Ahora exige x e y siempre.

3. Se elimina código muerto que estaba después del return.

4. NUEVO: polinomio expandido. Antes solo mostraba los L_i base y
   la suma simbólica. Ahora también muestra el polinomio expandido
   (orden descendente de potencias) construido vía multiplicación
   polinómica directa, replicando el método del MATLAB del profe
   (conv en Lagrange.m).

5. Se arregla 'tiempo' duplicado en info_dict.
"""

import numpy as np
import time
import matplotlib.pyplot as plt

from Python.Vandermonde import _validar_y_ordenar
from Python.supCp3 import SUBinterpol_newton
from Python.supCp3 import SUBspln_cubico
from Python.supCp3 import SUBspline_lineal
from Python.supCp3 import Subvandermonde


def _construir_polinomio_lagrange(x, y):
    """Construye el polinomio interpolador de Lagrange en su forma
    expandida, retornando los coeficientes en orden DESCENDENTE de
    potencias (compatible con np.poly1d y np.polyval).

    Replica lo que hace el MATLAB del profe en Lagrange.m:
        Para cada i:
            Li = producto de (x - x_j) / (x_i - x_j) para j != i
            Tabla[i] = y_i * Li
        polinomio = suma(Tabla)
    """
    n = len(x)
    # Coeficientes acumulados del polinomio total. Lo iremos sumando.
    coef_total = np.zeros(n)

    for i in range(n):
        # Construir L_i(x) como polinomio. Empieza siendo el polinomio
        # constante 1 (lista [1] en convención descendente).
        Li = np.array([1.0])
        den = 1.0
        for j in range(n):
            if j != i:
                # Factor (x - x_j) en convención descendente: [1, -x_j]
                factor = np.array([1.0, -x[j]])
                # np.convolve multiplica polinomios
                Li = np.convolve(Li, factor)
                den *= (x[i] - x[j])

        # Aporte de este i al polinomio total: y_i * L_i / den
        coef_total += (y[i] / den) * Li

    return coef_total


def _formatear_polinomio(coef, var="x", precision=6):
    """Formatea un array de coeficientes (orden DESCENDENTE) como
    string legible, ej: '0.4124 x^3 + 0.9394 x^2 - 5.836 x + 0.0047'.
    """
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
            # primer término no usa el "+"
            partes.append(term if c >= 0 else f"-{term}")
        else:
            partes.append(f"{signo} {term}")

    if not partes:
        return "0"
    return " ".join(partes)


def interpolacion_lagrange(ValoresX=None, ValoresY=None,
                            show_report=True, auto_compare=True):
    """
    Interpolación por polinomio de Lagrange.

    Devuelve (resultado_str, info_dict) compatible con la GUI.
    """
    eval_grid = 500

    # --- Validación (delega en Vandermonde) ---
    x, y, advertencias = _validar_y_ordenar(ValoresX, ValoresY)

    # --- Cálculo del polinomio en forma expandida ---
    start_time = time.time()
    coef_expandidos = _construir_polinomio_lagrange(x, y)
    poly_obj = np.poly1d(coef_expandidos)
    end_time = time.time()
    tiempo_ejecucion = end_time - start_time

    polinomio_expandido = _formatear_polinomio(coef_expandidos)

    # --- Polinomios base L_i(x) en forma simbólica (la canónica de Lagrange) ---
    polinomios_base = []
    for i in range(len(x)):
        factores = []
        for j in range(len(x)):
            if j != i:
                factores.append(
                    f"(x - {x[j]:g}) / ({x[i]:g} - {x[j]:g})"
                )
        L_i = f"L_{i}(x) = " + " · ".join(factores)
        polinomios_base.append(L_i)

    # Forma simbólica del polinomio total (ponderaciones por y_i)
    forma_simbolica = " + ".join(
        f"{y[i]:g}·L_{i}(x)" for i in range(len(x))
    )

    # --- Evaluación para gráfica (usando coeficientes expandidos) ---
    x_plot = np.linspace(min(x), max(x), 500)
    y_lagrange = np.polyval(coef_expandidos, x_plot)

    # --- Resultado para la GUI ---
    puntos_fmt = ", ".join(f"({xi:g}, {yi:g})" for xi, yi in zip(x, y))
    resultado = (
        f"Puntos ingresados (ordenados): {puntos_fmt}\n\n"
        f"Polinomios base de Lagrange:\n" +
        "\n".join(f"  {L}" for L in polinomios_base) +
        f"\n\nForma canónica:\n  P(x) = {forma_simbolica}\n\n"
        f"Polinomio expandido:\n  P(x) = {polinomio_expandido}\n\n"
        f"Tiempo de ejecución: {tiempo_ejecucion:.6f} segundos"
    )
    if advertencias:
        resultado += "\n\nAdvertencias:\n" + "\n".join(f"- {a}" for a in advertencias)

    info = {
        "tiempo": tiempo_ejecucion,
        "coeficientes": coef_expandidos.tolist(),
        "polinomio_str": polinomio_expandido,
        "polinomio_obj": poly_obj,
        "polinomios_base": polinomios_base,
        "polinomios_por_tramo": None,
        "n_tramos": None,
        "condicion": None,
        "advertencias": advertencias,
    }

    # --- Gráfica del informe (sin cambios funcionales) ---
    if show_report:
        try:
            fig, (ax_plot, ax_table) = plt.subplots(
                ncols=2, figsize=(12, 6),
                gridspec_kw={'width_ratios': [3, 2]}
            )
            ax_plot.plot(x, y, 'ro', label='Puntos dados')
            ax_plot.plot(x_plot, y_lagrange, 'b-', label='Polinomio de Lagrange')
            ax_plot.set_title("Interpolación de Lagrange")
            ax_plot.set_xlabel("x")
            ax_plot.set_ylabel("y")
            ax_plot.legend()
            ax_plot.grid()

            rows = [
                ["Tiempo (s)", f"{tiempo_ejecucion:.6f}"],
                ["Grado", f"{len(x)-1}"],
                ["Polinomio (trunc)",
                 polinomio_expandido[:120] +
                 ("..." if len(polinomio_expandido) > 120 else "")],
            ]
            ax_table.axis('off')
            table = ax_table.table(
                cellText=rows, colLabels=["Propiedad", "Valor"],
                loc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2)

            fig_pol = plt.figure(figsize=(10, 2 + 0.5 * len(polinomios_base)))
            plt.axis('off')
            text = "\n".join(polinomios_base)
            plt.text(0.01, 0.99, text, va='top', family='monospace', fontsize=9)
            plt.tight_layout()
            plt.show()
        except Exception:
            pass

    # --- Comparación automática con otros métodos ---
    if auto_compare:
        try:
            eval_pts = np.linspace(min(x), max(x), max(100, int(eval_grid)))
            y_ref = np.polyval(coef_expandidos, eval_pts)

            other_results = {}
            for name, fn in [
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
                        'rmse': float(np.sqrt(np.mean(diff**2))),
                    }

            if show_report:
                try:
                    fig, (ax_plot, ax_table) = plt.subplots(
                        ncols=2, figsize=(12, 5),
                        gridspec_kw={'width_ratios': [3, 2]}
                    )
                    ax_plot.plot(x, y, 'ro', label='Puntos dados')
                    ax_plot.plot(eval_pts, y_ref, 'k-', label='Lagrange (referencia)')
                    colors = {'Newton': 'c--', 'Spline_lineal': 'g--',
                              'Spline_cubico': 'y--', 'Vandermonde': 'b:'}
                    for name, res in other_results.items():
                        ycmp = reeval_on_common(res)
                        if ycmp is not None:
                            ax_plot.plot(eval_pts, ycmp,
                                         colors.get(name, '--'), label=name)
                    ax_plot.set_title('Comparación respecto a Lagrange')
                    ax_plot.set_xlabel('x')
                    ax_plot.set_ylabel('y')
                    ax_plot.legend()
                    ax_plot.grid()

                    rows = []
                    for name in ['Newton', 'Spline_lineal',
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
    # Caso del PDF del profe
    res, info = interpolacion_lagrange(
        [-2, -1, 2, 3],
        [12.13533528, 6.367879441, -4.610943901, 2.085536923],
        show_report=False, auto_compare=False
    )
    print(res)