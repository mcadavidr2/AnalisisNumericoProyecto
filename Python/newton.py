import numpy as np

def newton_method(
    f,
    x0,
    tolerance,
    max_iterations,
    error_type="rel",
    show_report=True,
    eval_grid=500,
    auto_compare=False
):
    """
    Método de Newton compatible con la interfaz gráfica.

    Parámetros:
        f             : función f(x) ya construida por la GUI (lambda o def).
        x0            : valor inicial.
        tolerance     : tolerancia (float).
        max_iterations: máximo número de iteraciones (int).
        error_type    : 'abs', 'rel' o 'cond' (la GUI manda este valor).
        show_report   : (no usado aquí, pero se deja para compatibilidad).
        eval_grid     : (no usado aquí).
        auto_compare  : (no usado aquí).

    Devuelve:
        (raiz, f(raiz), iteraciones, tabla)
        donde tabla es una lista de filas:
        [iter, x_actual, f(x_actual), f'(x_actual), error]
    """

    # Derivada numérica central
    def df(x):
        h = 1e-6
        return (f(x + h) - f(x - h)) / (2.0 * h)

    x_current = float(x0)
    iteration_data = []

    for k in range(int(max_iterations)):
        f_current = f(x_current)
        df_current = df(x_current)

        # derivada casi cero → peligro
        if df_current == 0:
            # agregamos la fila con error "infinito" para que aparezca en la tabla
            iteration_data.append([k, x_current, f_current, df_current, float("inf")])
            break

        x_next = x_current - f_current / df_current

        # cálculo del error según tipo
        if error_type == "abs":
            error = abs(x_next - x_current)
        elif error_type == "rel":
            if x_next != 0:
                error = abs((x_next - x_current) / x_next)
            else:
                error = abs(x_next - x_current)
        elif error_type == "cond":
            # por ejemplo: error basado en |f(x)|
            error = abs(f_current)
        else:
            # fallback: error absoluto
            error = abs(x_next - x_current)

        iteration_data.append([k, x_current, f_current, df_current, error])

        if error < tolerance:
            x_current = x_next
            break

        x_current = x_next

    root = x_current
    f_root = f(root)
    iterations = len(iteration_data)

    return root, f_root, iterations, iteration_data
