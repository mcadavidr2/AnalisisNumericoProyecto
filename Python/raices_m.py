"""
raices_m.py
-----------
Método de Newton para raíces múltiples (variante de Schroder),
compatible con la GUI.

Fórmula:
    x_{n+1} = x_n - f(x_n) · f'(x_n) / [f'(x_n)² - f(x_n) · f''(x_n)]

Esta variante de Newton recupera convergencia CUADRÁTICA en raíces
múltiples (donde Newton clásico solo logra convergencia lineal). El
precio: requiere conocer f'' además de f'. Aquí ambas se calculan
numéricamente por diferencias centradas.

Nota sobre la segunda derivada:
    El step h para f'' es 1e-3, distinto del 1e-6 de f'. Esto se debe a
    que el error de redondeo de las diferencias centradas para f'' crece
    como 1/h² (vs 1/h para f'), así que un h más grande reduce el ruido
    aunque aumente ligeramente el error de truncamiento. Es un compromiso
    estándar en cálculo numérico.

Cambios respecto a la versión anterior:

1. h de f'' subido de 1e-4 a 1e-3 para reducir ruido numérico.

2. Detección de denominador casi cero con tolerancia (no igualdad).

3. f, f', f'' envueltos en _safe_call (mismo patrón que el resto).

4. Tipos de error unificados a los 4 de la imagen:
       'abs', 'rel', 'rel2', 'rond'.

5. Firma y forma de la tabla SIN cambios:
       (raíz, f(raíz), iteraciones, tabla)
       fila = [iter, x_n, f(x_n), f'(x_n), f''(x_n), error]
"""

import numpy as np


# Tolerancia para considerar el denominador efectivamente cero.
EPS_DENOM = 1e-14


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
    """Primera derivada por diferencias centradas. Error O(h²)."""
    fp = _safe_call(f, x + h)
    fm = _safe_call(f, x - h)
    if not (np.isfinite(fp) and np.isfinite(fm)):
        return float("nan")
    return (fp - fm) / (2.0 * h)


def _d2f_central(f, x, h=1e-3):
    """Segunda derivada por diferencias centradas.

    Step más grande que para f' porque el error de redondeo crece como
    1/h². h = 1e-3 es un compromiso estándar para double precision.
    """
    fp = _safe_call(f, x + h)
    f0 = _safe_call(f, x)
    fm = _safe_call(f, x - h)
    if not (np.isfinite(fp) and np.isfinite(f0) and np.isfinite(fm)):
        return float("nan")
    return (fp - 2.0 * f0 + fm) / (h * h)


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


def multiple_roots(
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
        [iter, x_n, f(x_n), f'(x_n), f''(x_n), error]
    """
    x_current = float(x0)
    tol = float(tolerance)
    nmax = int(max_iterations)

    iteration_data = []

    for k in range(nmax):
        f_val   = _safe_call(f, x_current)
        df_val  = _df_central(f, x_current)
        d2f_val = _d2f_central(f, x_current)

        # Si cualquier evaluación falla, terminamos limpiamente
        if not (np.isfinite(f_val) and np.isfinite(df_val) and np.isfinite(d2f_val)):
            iteration_data.append([k, x_current, f_val, df_val, d2f_val, float("inf")])
            break

        denom = df_val**2 - f_val * d2f_val
        if abs(denom) < EPS_DENOM:
            # Si el denominador colapsó pero f(x_n) ya es chico, estamos
            # en la raíz numéricamente — no podemos refinar más por
            # cancelación de derivadas, pero la respuesta es válida.
            # Registramos la fila con error = |f(x_n)| (residual real)
            # en lugar de inf, para no descartar una solución correcta.
            if abs(f_val) < tol:
                iteration_data.append([k, x_current, f_val, df_val, d2f_val, abs(f_val)])
            else:
                iteration_data.append([k, x_current, f_val, df_val, d2f_val, float("inf")])
            break

        x_next = x_current - f_val * df_val / denom

        if not np.isfinite(x_next):
            iteration_data.append([k, x_current, f_val, df_val, d2f_val, float("inf")])
            break

        error = _calc_error(error_type, x_next, x_current, f_val)
        iteration_data.append([k, x_current, f_val, df_val, d2f_val, error])

        if error < tol:
            x_current = x_next
            break

        x_current = x_next

    root = x_current
    f_root = _safe_call(f, root)
    iterations = len(iteration_data)

    return root, f_root, iterations, iteration_data