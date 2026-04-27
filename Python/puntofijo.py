import numpy as np

def fixed_point_iteration(
    f,
    x0,
    tolerance,
    max_iterations,
    error_type="rel",   # 'abs', 'rel' o 'cond'
    show_report=True,
    eval_grid=500,
    auto_compare=False
):
    """
    Iteración de punto fijo compatible con la GUI.

    f             : función g(x); la GUI la construye a partir de la cadena ingresada.
    x0            : valor inicial.
    tolerance     : tolerancia.
    max_iterations: máximo de iteraciones.
    error_type    : 'abs', 'rel' o 'cond'.
    show_report, eval_grid, auto_compare: ignorados (compatibilidad).

    Devuelve:
        (x*, g(x*), iteraciones, tabla)

    Tabla (por fila):
        [ iter, x_n, g(x_n), error ]
    """

    x_current = float(x0)
    iteration_data = []

    for k in range(int(max_iterations)):
        g_current = f(x_current)   # aquí f es realmente g(x)

        # Protegemos contra overflows / NaN
        if not np.isfinite(g_current):
            raise ValueError(
                "La iteración de punto fijo generó un valor no finito "
                "(overflow o NaN). Revisa g(x) o el valor inicial."
            )

        x_next = g_current

        # Cálculo del error según el tipo
        if error_type == "abs":
            error = abs(x_next - x_current)
        elif error_type == "rel":
            if x_next != 0:
                error = abs((x_next - x_current) / x_next)
            else:
                error = abs(x_next - x_current)
        elif error_type == "cond":
            error = abs(x_next - x_current)  # por ejemplo, residuo de x = g(x)
        else:
            error = abs(x_next - x_current)

        iteration_data.append([k, x_current, g_current, error])

        if error < tolerance:
            x_current = x_next
            break

        x_current = x_next

    root = x_current
    g_root = f(root)
    iterations = len(iteration_data)

    return root, g_root, iterations, iteration_data