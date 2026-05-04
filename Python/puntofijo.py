"""
puntofijo.py
------------
Iteración de punto fijo compatible con la GUI.

A diferencia de bisección y regla falsa, este es un método iterativo
puro: arranca con x_0 y aplica g(x) repetidamente:
    x_{n+1} = g(x_n)
La sucesión converge a un punto fijo x* (donde g(x*) = x*) si g es
contractiva en la región de interés (|g'(x)| < 1 cerca del punto fijo).

Cambios respecto a la versión anterior:

1. Las evaluaciones de g(x) están envueltas en _safe_call, que atrapa
   OverflowError, ZeroDivisionError, ValueError y FloatingPointError
   y devuelve nan en su lugar. Antes, g(x) = x**2 desde x0 = 2 reventaba
   con OverflowError ANTES de poder llegar al chequeo isfinite, y el
   método se interrumpía con excepción que la GUI mostraba como popup.

2. Si g(x) produce nan/inf, registramos fila con error = inf y salimos
   limpiamente. Antes era ValueError.

3. Tipos de error unificados a los 4 de la imagen:
       'abs'   -> |x_n - x_{n-1}|
       'rel'   -> |(x_n - x_{n-1}) / x_n|
       'rel2'  -> |(x_n - x_{n-1}) / x_{n-1}|
       'rond'  -> |g(x_n) - x_n|     (residual de punto fijo;
                                       NO |f(x)| como en los demás métodos)

4. Firma y forma de la tabla SIN cambios:
       (x*, g(x*), iteraciones, tabla)
       fila = [iter, x_n, g(x_n), error]
"""

import numpy as np


def _safe_call(g, x):
    """Llama g(x) atrapando los errores aritméticos típicos.

    Devuelve nan si g(x) lanza una excepción aritmética o si el resultado
    no se puede convertir a float. NO atrapa errores genéricos (NameError,
    TypeError por bugs del usuario, etc.) para no ocultar problemas de
    input.
    """
    try:
        val = g(x)
    except (OverflowError, ZeroDivisionError, ValueError, FloatingPointError):
        return float("nan")
    try:
        return float(val)
    except Exception:
        return float("nan")


def _calc_error(error_type, x_new, x_old, gxn):
    """Calcula el error según el tipo. Para 'rond' usa el residual de punto
    fijo |g(x_n) - x_n|, NO |f(x)| (porque en punto fijo no hay f).
    """
    diff = abs(x_new - x_old)
    if error_type == "abs":
        return diff
    if error_type == "rel":
        return diff / abs(x_new) if x_new != 0 else diff
    if error_type == "rel2":
        return diff / abs(x_old) if x_old != 0 else diff
    if error_type == "rond":
        # Residual: qué tan lejos está x_old de ser punto fijo de g.
        # gxn = g(x_old), entonces el residual es |g(x_old) - x_old|.
        return abs(gxn - x_old) if np.isfinite(gxn) else float("inf")
    return diff


def fixed_point_iteration(
    f,                          # la GUI manda 'g' como 'f' por compatibilidad
    x0,
    tolerance,
    max_iterations,
    error_type="rel",
    show_report=True,
    eval_grid=500,
    auto_compare=False,
):
    """
    Devuelve:
        (x*, g(x*), iteraciones, tabla)
    Tabla por fila:
        [iter, x_n, g(x_n), error]
    """
    g = f                       # claridad: en este método 'f' es realmente g(x)
    x_current = float(x0)
    tol = float(tolerance)
    nmax = int(max_iterations)

    iteration_data = []

    for k in range(nmax):
        g_current = _safe_call(g, x_current)

        # Si g divergió o retornó nan, registramos fila y salimos
        if not np.isfinite(g_current):
            iteration_data.append([k, x_current, g_current, float("inf")])
            break

        x_next = g_current
        error = _calc_error(error_type, x_next, x_current, g_current)

        iteration_data.append([k, x_current, g_current, error])

        if error < tol:
            x_current = x_next
            break

        x_current = x_next

    # Resultado final
    root = x_current
    g_root = _safe_call(g, root)
    iterations = len(iteration_data)

    return root, g_root, iterations, iteration_data