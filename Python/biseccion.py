import numpy as np

def biseccion(
    f,
    lower_bound,
    upper_bound,
    tolerance,
    max_iterations,
    error_type="rel",     # 'abs', 'rel' o 'cond'
    show_report=True,
    eval_grid=500,
    auto_compare=False
):
    """
    Método de bisección compatible con la GUI.

    Parámetros:
        f             : función f(x) ya construida por la interfaz.
        lower_bound   : límite inferior (a).
        upper_bound   : límite superior (b).
        tolerance     : tolerancia deseada.
        max_iterations: máximo de iteraciones.
        error_type    : 'abs', 'rel' o 'cond'.
        show_report, eval_grid, auto_compare: ignorados (solo compatibilidad).

    Devuelve:
        (raiz, f(raiz), iteraciones, tabla)

    Tabla (por fila):
        [Iteración, a, f(a), pm, f(pm), b, f(b), Error]
    """

    a = float(lower_bound)
    b = float(upper_bound)

    f_a = f(a)
    f_b = f(b)

    if f_a * f_b >= 0:
        raise ValueError("f(a) y f(b) deben tener signos opuestos.")

    matriz = []

    # Primer punto medio
    pm = (a + b) / 2.0
    f_pm = f(pm)

    # Error inicial (longitud del intervalo /2, por ejemplo)
    if error_type == "abs":
        error = abs(b - a)
    elif error_type == "rel":
        if pm != 0:
            error = abs((b - a) / pm)
        else:
            error = abs(b - a)
    elif error_type == "cond":
        # por ejemplo, basado en el residuo |f(pm)|
        error = abs(f_pm)
    else:
        error = abs(b - a)

    iter_count = 0
    matriz.append([iter_count, a, f_a, pm, f_pm, b, f_b, error])

    while iter_count < max_iterations and error > tolerance:
        # Elegir subintervalo donde cambia el signo
        if f_a * f_pm < 0:
            b = pm
            f_b = f(b)
        else:
            a = pm
            f_a = f(a)

        pm_prev = pm
        pm = (a + b) / 2.0
        f_pm = f(pm)

        # Cálculo del error según tipo
        if error_type == "abs":
            error = abs(pm - pm_prev)
        elif error_type == "rel":
            if pm != 0:
                error = abs((pm - pm_prev) / pm)
            else:
                error = abs(pm - pm_prev)
        elif error_type == "cond":
            error = abs(f_pm)
        else:
            error = abs(pm - pm_prev)

        iter_count += 1
        matriz.append([iter_count, a, f_a, pm, f_pm, b, f_b, error])

        if error <= tolerance:
            break

    root = pm
    f_root = f_pm
    iterations = len(matriz)

    return root, f_root, iterations, matriz
