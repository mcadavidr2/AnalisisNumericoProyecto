def secante(
    f,
    x0,
    x1,
    tolerance,
    max_iterations,
    error_type="rel",   # 'abs', 'rel' o 'cond'
    show_report=True,
    eval_grid=500,
    auto_compare=False
):
    """
    Método de la Secante compatible con la GUI.

    Parámetros:
        f             : función f(x), ya construida por la interfaz.
        x0, x1        : valores iniciales.
        tolerance     : tolerancia (float).
        max_iterations: máximo de iteraciones (int).
        error_type    : 'abs', 'rel' o 'cond'.
        show_report, eval_grid, auto_compare: ignorados, solo para compatibilidad.

    Devuelve:
        (raiz, f(raiz), iteraciones, tabla)

    Tabla (por fila):
        [ iter, x_{i-1}, x_i, f(x_i), error ]
    """

    # Aseguramos tipos numéricos
    x_prev = float(x0)
    x_curr = float(x1)

    f_prev = f(x_prev)
    f_curr = f(x_curr)

    iteration_data = []

    for k in range(int(max_iterations)):
        # Evitar división por cero en la fórmula de la secante
        denom = (f_curr - f_prev)
        if denom == 0:
            # Guardamos fila con error infinito y salimos
            error = float("inf")
            iteration_data.append([k, x_prev, x_curr, f_curr, error])
            break

        x_next = x_curr - f_curr * (x_curr - x_prev) / denom
        f_next = f(x_next)

        # Cálculo del error según el tipo
        if error_type == "abs":
            error = abs(x_next - x_curr)
        elif error_type == "rel":
            if x_next != 0:
                error = abs((x_next - x_curr) / x_next)
            else:
                error = abs(x_next - x_curr)
        elif error_type == "cond":
            # Por ejemplo, basándonos en el residuo |f(x_i)|
            error = abs(f_curr)
        else:
            error = abs(x_next - x_curr)

        # Fila con formato esperado por la GUI:
        # [iter, x_{i-1}, x_i, f(x_i), error]
        iteration_data.append([k, x_prev, x_curr, f_curr, error])

        if error < tolerance:
            x_curr = x_next
            f_curr = f_next
            break

        # Avanzar los valores
        x_prev, x_curr = x_curr, x_next
        f_prev, f_curr = f_curr, f_next

    root = x_curr
    f_root = f(root)
    iterations = len(iteration_data)

    return root, f_root, iterations, iteration_data
