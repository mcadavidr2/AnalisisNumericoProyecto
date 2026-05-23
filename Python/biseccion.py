"""
biseccion.py
------------
Método de bisección compatible con la GUI.

Cambios respecto a la versión anterior:

1. Si f(a)·f(b) >= 0 ya NO se levanta ValueError. Se devuelve una tabla
   con una sola fila marcada como NO-CONVERGENTE (error = inf) para que
   la GUI pueda informar el problema y, aun así, graficar f(x) en [a, b].
   Antes la excepción mataba el flujo y no se podía graficar nada.
   (Este era el bug de f(x) = x**2 en intervalos simétricos.)

2. Si f(a) == 0 o f(b) == 0, se devuelve la raíz exacta inmediatamente,
   tal como hace el código MATLAB original (Biseccion.m del profe).

3. Tipos de error unificados a los 4 de la imagen del enunciado:
       'abs'  -> E_abs   = |x_n - x_{n-1}|
       'rel'  -> E_rel   = |(x_n - x_{n-1}) / x_n|
       'rel2' -> E_rel2  = |(x_n - x_{n-1}) / x_{n-1}|
       'rond' -> E_rond  = |f(x_n)|

4. La firma de retorno y la forma de la tabla NO cambian:
       (raíz, f(raíz), iteraciones, tabla)
       fila = [k, a, f(a), pm, f(pm), b, f(b), error]
"""

import numpy as np


def _calc_error(error_type, x_new, x_old, f_new):
    """Calcula el error según error_type, usando la convención de la imagen."""
    diff = abs(x_new - x_old)
    if error_type == "abs":
        return diff
    if error_type == "rel":
        return diff / abs(x_new) if x_new != 0 else diff
    if error_type == "rel2":
        return diff / abs(x_old) if x_old != 0 else diff
    if error_type == "rond":
        return abs(f_new)
    return diff   # fallback robusto


def biseccion(
    f,
    lower_bound,
    upper_bound,
    tolerance,
    max_iterations,
    error_type="rel",
    show_report=True,
    eval_grid=500,
    auto_compare=False,
):
    """
    Devuelve:
        (raíz, f(raíz), iteraciones, tabla)
    Tabla por fila:
        [Iter, a, f(a), pm, f(pm), b, f(b), Error]
    """

    a = float(lower_bound)
    b = float(upper_bound)
    tol = float(tolerance)
    nmax = int(max_iterations)

    # --- Salvaguardas de entrada ---
    if a == b:
        # Intervalo degenerado: una fila informativa y salimos
        try:
            fa = float(f(a))
        except Exception:
            fa = float("nan")
        return a, fa, 1, [[0, a, fa, a, fa, b, fa, float("inf")]]

    if a > b:
        a, b = b, a   # ordenamos por seguridad

    f_a = float(f(a))
    f_b = float(f(b))

    # Caso: alguno de los extremos ya es raíz exacta (igual que MATLAB)
    if f_a == 0.0:
        return a, 0.0, 1, [[0, a, 0.0, a, 0.0, b, f_b, 0.0]]
    if f_b == 0.0:
        return b, 0.0, 1, [[0, a, f_a, b, 0.0, b, 0.0, 0.0]]

    # Caso: NO cambia de signo en [a, b]. En lugar de levantar excepción,
    # devolvemos una fila "no convergente" para que la GUI pueda graficar
    # f(x) y mostrar al usuario por qué no se pudo aplicar bisección.
    if f_a * f_b > 0:
        pm0 = (a + b) / 2.0
        try:
            f_pm0 = float(f(pm0))
        except Exception:
            f_pm0 = float("nan")
        return pm0, f_pm0, 1, [[0, a, f_a, pm0, f_pm0, b, f_b, float("inf")]]

    # --- Iteración principal ---
    matriz = []

    pm = (a + b) / 2.0
    f_pm = float(f(pm))

    # Primera fila: no hay pm anterior. Si error_type == 'abs', usamos
    # |b - a| como cota inicial (tamaño del intervalo). Para los demás
    # tipos, calculamos contra 'a' que es lo más razonable que tenemos.
    if error_type == "abs":
        error = abs(b - a)
    else:
        error = _calc_error(error_type, pm, a, f_pm)

    matriz.append([0, a, f_a, pm, f_pm, b, f_b, error])

    iter_count = 0
    while iter_count < nmax and error > tol and f_pm != 0.0:
        # Elegir el subintervalo donde cambia el signo
        if f_a * f_pm < 0:
            b = pm
            f_b = f_pm
        else:
            a = pm
            f_a = f_pm

        pm_prev = pm
        pm = (a + b) / 2.0
        f_pm = float(f(pm))

        error = _calc_error(error_type, pm, pm_prev, f_pm)
        iter_count += 1
        matriz.append([iter_count, a, f_a, pm, f_pm, b, f_b, error])

    root = pm
    f_root = f_pm
    iterations = len(matriz)
    return root, f_root, iterations, matriz