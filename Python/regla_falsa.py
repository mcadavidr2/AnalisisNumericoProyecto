import numpy as np

def false_position_method(
    f,
    lower_bound,
    upper_bound,
    tolerance,
    max_iterations,
    error_type="rel",    # 'abs', 'rel' o 'cond'
    show_report=True,
    eval_grid=500,
    auto_compare=False
):
    """
    Método de Regla Falsa (posición falsa) compatible con la GUI.

    Parámetros:
        f             : función f(x) ya construida por la interfaz.
        lower_bound   : límite inferior (a).
        upper_bound   : límite superior (b).
        tolerance     : tolerancia deseada.
        max_iterations: máximo de iteraciones.
        error_type    : 'abs', 'rel' o 'cond'.
        show_report, eval_grid, auto_compare: ignorados (para compatibilidad).

    Devuelve:
        (raiz, f(raiz), iteraciones, tabla)

    Tabla (por fila):
        [Iter, A, F(A), B, F(B), Xr, F(Xr), Error]
    """

    a = float(lower_bound)
    b = float(upper_bound)

    f_a = f(a)
    f_b = f(b)

    if f_a * f_b >= 0:
        raise ValueError("f(a) y f(b) deben tener signos opuestos.")

    iteration_data = []
    xr_prev = None

    for k in range(int(max_iterations)):
        # Fórmula clásica de posición falsa
        denom = (f_b - f_a)
        if denom == 0:
            # Guardamos una fila con error infinito y salimos
            xr = (a + b) / 2.0
            f_xr = f(xr)
            error = float("inf")
            iteration_data.append([k, a, f_a, b, f_b, xr, f_xr, error])
            break

        xr = b - f_b * (b - a) / denom
        f_xr = f(xr)

        # Cálculo del error según el tipo
        if xr_prev is None:
            # primera iteración: usamos longitud de intervalo como referencia
            base_err = abs(b - a)
        else:
            base_err = abs(xr - xr_prev)

        if error_type == "abs":
            error = base_err
        elif error_type == "rel":
            if xr != 0:
                error = abs(base_err / xr)
            else:
                error = base_err
        elif error_type == "cond":
            # por ejemplo, residuo |f(xr)|
            error = abs(f_xr)
        else:
            error = base_err

        # Guardar fila en el formato que espera la GUI:
        # [Iter, A, F(A), B, F(B), Xr, F(Xr), Error]
        iteration_data.append([k, a, f_a, b, f_b, xr, f_xr, error])

        # Condición de parada
        if error < tolerance:
            break

        # Actualizar intervalo preservando el cambio de signo
        if f_a * f_xr < 0:
            b = xr
            f_b = f_xr
        else:
            a = xr
            f_a = f_xr

        xr_prev = xr

    # Resultado final
    root = xr
    f_root = f_xr
    iterations = len(iteration_data)

    return root, f_root, iterations, iteration_data