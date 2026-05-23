"""
secante.py
----------
Método de la Secante compatible con la GUI.

Fórmula:
    x_{n+1} = x_n - f(x_n) · (x_n - x_{n-1}) / (f(x_n) - f(x_{n-1}))

Necesita DOS puntos iniciales x_0 y x_1. Convergencia superlineal de
orden phi ≈ 1.618 (razón áurea). Es como Newton pero aproximando la
derivada con la pendiente de la recta secante entre las dos últimas
iteraciones, así no necesita derivar.

Geométricamente la fórmula es idéntica a regla falsa, pero la lógica
de actualización es distinta: secante usa siempre las dos últimas
iteraciones, sin requerir cambio de signo. Es más rápido que regla
falsa pero puede divergir donde aquélla siempre converge.

Cambios respecto a la versión anterior:

1. Detección de denominador casi cero con tolerancia (abs(denom) < EPS_DENOM)
   en lugar de igualdad estricta. Antes, denom = 1e-300 dejaba pasar el
   chequeo y producía x_next astronómico.

2. f(x) envuelta en _safe_call (mismo patrón que punto fijo y Newton)
   para atrapar OverflowError, ZeroDivisionError, ValueError y
   FloatingPointError.

3. Caso x_0 == x_1 manejado limpiamente: una fila informativa con
   error = inf y salida.

4. Tipos de error unificados a los 4 de la imagen:
       'abs', 'rel', 'rel2', 'rond'.

5. Firma y forma de la tabla SIN cambios:
       (raíz, f(raíz), iteraciones, tabla)
       fila = [iter, x_{n-1}, x_n, f(x_n), error]
"""

import numpy as np


# Tolerancia para considerar que la pendiente aproximada es cero.
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


def secante(
    f,
    x0,
    x1,
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
        [iter, x_{n-1}, x_n, f(x_n), error]
    """
    x_prev = float(x0)
    x_curr = float(x1)
    tol = float(tolerance)
    nmax = int(max_iterations)

    f_prev = _safe_call(f, x_prev)
    f_curr = _safe_call(f, x_curr)

    iteration_data = []

    # --- Salvaguardas iniciales ---

    # Caso x0 == x1: no hay forma de calcular pendiente
    if x_prev == x_curr:
        iteration_data.append([0, x_prev, x_curr, f_curr, float("inf")])
        return x_curr, f_curr, 1, iteration_data

    # Caso uno de los dos puntos iniciales no se pudo evaluar
    if not (np.isfinite(f_prev) and np.isfinite(f_curr)):
        iteration_data.append([0, x_prev, x_curr, f_curr, float("inf")])
        return x_curr, f_curr, 1, iteration_data

    # --- Iteración principal ---
    for k in range(nmax):
        # Si en alguna iteración previa se invalidó f, salimos
        if not (np.isfinite(f_prev) and np.isfinite(f_curr)):
            iteration_data.append([k, x_prev, x_curr, f_curr, float("inf")])
            break

        denom = f_curr - f_prev

        # Salvaguarda contra pendiente casi nula
        if abs(denom) < EPS_DENOM:
            iteration_data.append([k, x_prev, x_curr, f_curr, float("inf")])
            break

        x_next = x_curr - f_curr * (x_curr - x_prev) / denom

        # Defensa extra: si por alguna razón x_next no es finito
        if not np.isfinite(x_next):
            iteration_data.append([k, x_prev, x_curr, f_curr, float("inf")])
            break

        f_next = _safe_call(f, x_next)
        error = _calc_error(error_type, x_next, x_curr, f_next)

        iteration_data.append([k, x_prev, x_curr, f_curr, error])

        if error < tol:
            x_curr = x_next
            f_curr = f_next
            break

        # Avanzar una iteración
        x_prev, x_curr = x_curr, x_next
        f_prev, f_curr = f_curr, f_next

    return x_curr, f_curr, len(iteration_data), iteration_data