import numpy as np

def multiple_roots(
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
    Método de raíces múltiples compatible con la GUI.

    Parámetros:
        f             : función f(x) con raíz múltiple.
        x0            : valor inicial.
        tolerance     : tolerancia.
        max_iterations: máximo de iteraciones.
        error_type    : 'abs', 'rel' o 'cond'.
        show_report, eval_grid, auto_compare: ignorados (compatibilidad).

    Devuelve:
        (raiz, f(raiz), iteraciones, tabla)

    Tabla (por fila):
        [iter, x_n, f(x_n), f'(x_n), f''(x_n), error]
    """

    # Derivada numérica primera
    def df(x):
        h = 1e-6
        return (f(x + h) - f(x - h)) / (2.0 * h)

    # Derivada numérica segunda
    def d2f(x):
        h = 1e-4
        return (f(x + h) - 2.0 * f(x) + f(x - h)) / (h ** 2)

    x_current = float(x0)
    iteration_data = []

    for k in range(int(max_iterations)):
        f_val = f(x_current)
        df_val = df(x_current)
        d2f_val = d2f(x_current)

        # Denominador de la fórmula de raíces múltiples:
        # x_{n+1} = x_n - f f' / ( (f')^2 - f f'' )
        denom = df_val ** 2 - f_val * d2f_val
        if denom == 0:
            # Guardamos fila con error infinito y salimos
            error = float("inf")
            iteration_data.append([k, x_current, f_val, df_val, d2f_val, error])
            break

        x_next = x_current - f_val * df_val / denom

        # Cálculo del error según el tipo
        if error_type == "abs":
            error = abs(x_next - x_current)
        elif error_type == "rel":
            if x_next != 0:
                error = abs((x_next - x_current) / x_next)
            else:
                error = abs(x_next - x_current)
        elif error_type == "cond":
            # por ejemplo, residuo |f(x_n)|
            error = abs(f_val)
        else:
            error = abs(x_next - x_current)

        # Fila con el formato que espera la GUI:
        # [iter, x_n, f(x_n), f'(x_n), f''(x_n), error]
        iteration_data.append([k, x_current, f_val, df_val, d2f_val, error])

        if error < tolerance:
            x_current = x_next
            break

        x_current = x_next

    root = x_current
    f_root = f(root)
    iterations = len(iteration_data)

    return root, f_root, iterations, iteration_data
