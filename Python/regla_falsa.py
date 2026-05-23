"""
regla_falsa.py
--------------
Método de Regla Falsa (posición falsa) compatible con la GUI.

Estructura idéntica a bisección. Diferencia clave: la estimación de la
raíz dentro del intervalo no es el punto medio sino la intersección
con el eje x de la recta que une (a, f(a)) y (b, f(b)):

    xr = b - f(b) · (b - a) / (f(b) - f(a))

Cambios respecto a la versión anterior:

1. f(a)·f(b) >= 0 ya NO levanta ValueError. Se devuelve fila con
   error = inf para que la GUI pueda graficar f(x) e informar.

2. Si f(a) == 0 o f(b) == 0 se devuelve la raíz exacta inmediatamente
   (igual que el MATLAB del profe).

3. Salvaguarda contra denominador cero o casi cero (|denom| < 1e-14).
   Cuando ocurre, salimos limpiamente con fila no convergente.
   Esto puede pasar por aritmética de punto flotante o por inputs
   patológicos como funciones constantes.

4. Tipos de error unificados a los 4 de la imagen:
       'abs', 'rel', 'rel2', 'rond'.

5. Firma y forma de la tabla SIN cambios:
       (raíz, f(raíz), iteraciones, tabla)
       fila = [Iter, a, f(a), b, f(b), xr, f(xr), Error]
"""

import numpy as np


# Tolerancia para considerar el denominador "cero" en la fórmula.
# Si f(b) - f(a) es de este orden, dividir produce números enormes
# y conviene detenerse antes en lugar de seguir con basura.
EPS_DENOM = 1e-14


def _calc_error(error_type, x_new, x_old, f_new):
    diff = abs(x_new - x_old)
    if error_type == "abs":
        return diff
    if error_type == "rel":
        return diff / abs(x_new) if x_new != 0 else diff
    if error_type == "rel2":
        return diff / abs(x_old) if x_old != 0 else diff
    if error_type == "rond":
        return abs(f_new)
    return diff


def false_position_method(
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
        [Iter, a, f(a), b, f(b), xr, f(xr), Error]
    """
    a = float(lower_bound)
    b = float(upper_bound)
    tol = float(tolerance)
    nmax = int(max_iterations)

    # --- Salvaguardas de entrada ---
    if a == b:
        try:
            fa = float(f(a))
        except Exception:
            fa = float("nan")
        return a, fa, 1, [[0, a, fa, b, fa, a, fa, float("inf")]]

    if a > b:
        a, b = b, a

    f_a = float(f(a))
    f_b = float(f(b))

    # Raíces exactas en los extremos
    if f_a == 0.0:
        return a, 0.0, 1, [[0, a, 0.0, b, f_b, a, 0.0, 0.0]]
    if f_b == 0.0:
        return b, 0.0, 1, [[0, a, f_a, b, 0.0, b, 0.0, 0.0]]

    # No hay cambio de signo: fila no convergente, sin excepción
    if f_a * f_b > 0:
        xr0 = (a + b) / 2.0
        try:
            f_xr0 = float(f(xr0))
        except Exception:
            f_xr0 = float("nan")
        return xr0, f_xr0, 1, [[0, a, f_a, b, f_b, xr0, f_xr0, float("inf")]]

    # --- Iteración principal ---
    iteration_data = []
    xr_prev = None
    xr = None
    f_xr = None

    for k in range(nmax):
        denom = f_b - f_a

        # Salvaguarda contra denominador casi nulo
        if abs(denom) < EPS_DENOM:
            # Si ya teníamos un xr de iteraciones anteriores, lo conservamos.
            # Si no (caso primera iteración con f constante), usamos el medio.
            if xr is None:
                xr = (a + b) / 2.0
                try:
                    f_xr = float(f(xr))
                except Exception:
                    f_xr = float("nan")
            iteration_data.append([k, a, f_a, b, f_b, xr, f_xr, float("inf")])
            break

        # Fórmula de la regla falsa
        xr = b - f_b * (b - a) / denom
        try:
            f_xr = float(f(xr))
        except Exception:
            f_xr = float("nan")

        # Cálculo del error según el tipo
        if xr_prev is None:
            # Primera iteración: |b - a| como referencia
            base_err = abs(b - a)
        else:
            base_err = abs(xr - xr_prev)

        if error_type == "abs":
            error = base_err
        elif error_type == "rel":
            error = base_err / abs(xr) if xr != 0 else base_err
        elif error_type == "rel2":
            if xr_prev is not None and xr_prev != 0:
                error = base_err / abs(xr_prev)
            else:
                error = base_err
        elif error_type == "rond":
            error = abs(f_xr) if np.isfinite(f_xr) else float("inf")
        else:
            error = base_err

        iteration_data.append([k, a, f_a, b, f_b, xr, f_xr, error])

        # Salida limpia si convergimos o si caímos en una raíz exacta
        if error < tol or f_xr == 0.0:
            break

        # Mantener el cambio de signo: actualizar el extremo correcto
        if f_a * f_xr < 0:
            b = xr
            f_b = f_xr
        else:
            a = xr
            f_a = f_xr

        xr_prev = xr

    return xr, f_xr, len(iteration_data), iteration_data