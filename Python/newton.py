"""
newton.py
---------
Método de Newton compatible con la GUI.

Fórmula:
    x_{n+1} = x_n - f(x_n) / f'(x_n)

La derivada se calcula NUMÉRICAMENTE con diferencias centradas:
    f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
Esto difiere del MATLAB del profe que usa derivada simbólica
(syms x; diff(f)). El error de truncamiento de las diferencias
centradas es O(h²); con h = 1e-6 el error es del orden 1e-12,
suficiente para los problemas del Cap 1.

Cambios respecto a la versión anterior:

1. Detección de derivada nula con tolerancia (abs(df) < EPS_DF) en lugar
   de igualdad estricta (df == 0). Antes, derivadas de magnitud 1e-15
   (ruido numérico) pasaban el chequeo y producían x_next astronómico.

2. f(x) y f'(x) envueltos en _safe_call para atrapar OverflowError,
   ZeroDivisionError, ValueError y FloatingPointError.

3. Tipos de error unificados a los 4 de la imagen:
       'abs', 'rel', 'rel2', 'rond' (donde rond = |f(x_n)|, el residual).

4. Firma y forma de la tabla SIN cambios:
       (raíz, f(raíz), iteraciones, tabla)
       fila = [iter, x_n, f(x_n), f'(x_n), error]
"""

import numpy as np


# Tolerancia para considerar la derivada efectivamente cero.
# Con derivadas numéricas, el ruido puede producir valores ~1e-15
# que sin esta protección causarían x_next astronómico.
EPS_DF = 1e-12


def _safe_call(f, x):
    """Llama f(x) atrapando errores aritméticos típicos. Ver puntofijo.py."""
    try:
        val = f(x)
    except (OverflowError, ZeroDivisionError, ValueError, FloatingPointError):
        return float("nan")
    try:
        return float(val)
    except Exception:
        return float("nan")


def _df_central(f, x, h=1e-6):
    """Derivada numérica por diferencias centradas.

    Devuelve nan si las evaluaciones de f en x±h fallan.
    Error de truncamiento O(h²); con h = 1e-6 esto da ~1e-12,
    suficiente para Newton en aritmética double.
    """
    fp = _safe_call(f, x + h)
    fm = _safe_call(f, x - h)
    if not (np.isfinite(fp) and np.isfinite(fm)):
        return float("nan")
    return (fp - fm) / (2.0 * h)


def _calc_error(error_type, x_new, x_old, f_new):
    diff = abs(x_new - x_old)
    if error_type == "abs":
        return diff
    if error_type == "rel":
        return diff / abs(x_new) if x_new != 0 else diff
    if error_type == "rel2":
        return diff / abs(x_old) if x_old != 0 else diff
    if error_type == "rond":
        return abs(f_new) if np.isfinite(f_new) else float("inf")
    return diff


def newton_method(
    f,
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
        (raíz, f(raíz), iteraciones, tabla)
    Tabla por fila:
        [iter, x_n, f(x_n), f'(x_n), error]
    """
    x_current = float(x0)
    tol = float(tolerance)
    nmax = int(max_iterations)

    iteration_data = []

    for k in range(nmax):
        f_current  = _safe_call(f, x_current)
        df_current = _df_central(f, x_current)

        # Derivada efectivamente cero o evaluación no finita
        if not np.isfinite(df_current) or abs(df_current) < EPS_DF:
            iteration_data.append([k, x_current, f_current, df_current, float("inf")])
            break

        if not np.isfinite(f_current):
            iteration_data.append([k, x_current, f_current, df_current, float("inf")])
            break

        x_next = x_current - f_current / df_current

        # Si el paso de Newton produce inf/nan, tampoco podemos seguir
        if not np.isfinite(x_next):
            iteration_data.append([k, x_current, f_current, df_current, float("inf")])
            break

        error = _calc_error(error_type, x_next, x_current, f_current)
        iteration_data.append([k, x_current, f_current, df_current, error])

        if error < tol:
            x_current = x_next
            break

        x_current = x_next

    root = x_current
    f_root = _safe_call(f, root)
    iterations = len(iteration_data)

    return root, f_root, iterations, iteration_data