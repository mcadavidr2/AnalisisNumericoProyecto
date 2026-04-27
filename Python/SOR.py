import numpy as np
import ast
try:
    from Python.gui_helpers import compute_spectral_radius
except Exception:
    from gui_helpers import compute_spectral_radius

def safe_divide(a, b):
    if b == 0:
        raise ZeroDivisionError(f"Intento de dividir por cero: {a} / {b}")
    return a / b

def str_to_numpy_matrix(matrix_str):
    """
    Convierte una cadena de texto que representa una matriz o vector en un objeto numpy array.
    """
    try:
        matrix_list = ast.literal_eval(matrix_str)
        matrix_np = np.array(matrix_list, dtype=np.float64)
        return matrix_np
    except Exception as e:
        print(f"Error al convertir cadena a matriz numpy: {e}")
        return None

def sor_method(A, b, x0, w, tolerance, max_iterations, error_type='rel', show_report=True, auto_compare=True):
    # si se solicita informe comparativo delegar a supCp2
    if show_report:
        try:
            from Python.supCp2 import subSOR as ss
            return ss.sor_method(A, b, x0, w, tolerance, max_iterations, error_type, show_report=True, auto_compare=auto_compare)
        except Exception:
            try:
                import supCp2.subSOR as ss
                return ss.sor_method(A, b, x0, w, tolerance, max_iterations, error_type, show_report=True, auto_compare=auto_compare)
            except Exception:
                pass
    matrix_a = A if isinstance(A, np.ndarray) else str_to_numpy_matrix(A)
    vector_b = b if isinstance(b, np.ndarray) else str_to_numpy_matrix(b)
    initial_guess = x0 if isinstance(x0, np.ndarray) else str_to_numpy_matrix(x0)

    results_matrix = []
    solution_vector = initial_guess.copy()

    for iteration_count in range(max_iterations):
        previous_solution = solution_vector.copy()

        for i in range(matrix_a.shape[0]):
            sigma = 0
            for j in range(matrix_a.shape[1]):
                if i != j:
                    sigma += matrix_a[i, j] * solution_vector[j]

            solution_vector[i] = (1 - w) * previous_solution[i] + (w / matrix_a[i, i]) * (vector_b[i] - sigma)

        absolute_error = np.linalg.norm(solution_vector - previous_solution)
        denom = np.linalg.norm(solution_vector)
        relative_error = safe_divide(absolute_error, denom) if denom != 0 else float('inf')

        if error_type == "rela":
            error = relative_error
        else:
            error = absolute_error

        results_matrix.append([iteration_count, solution_vector.tolist(), float(absolute_error), float(relative_error)])

        if error < tolerance:
            break

    # calcular radio espectral
    rho, _ = compute_spectral_radius(matrix_a, method='sor', omega=w)
    can_conv = False if rho is None else (rho < 1)
    summary = (
        f"Radio espectral: {rho:.6f}" if rho is not None else "Radio espectral: Desconocido",
        f"Converge (rho<1)?: {'Sí' if can_conv else 'No'}",
    )
    summary_text = "\n".join(summary)
    return (summary_text, results_matrix)

if __name__ == '__main__':
    # modo consola: pedir entradas y mostrar tabla en consola
    matrix_a_str = input("Ingresa la matriz A (e.g., [[4,1],[1,3]]): ")
    vector_b_str = input("Ingresa el vector b (e.g., [1,2]): ")
    initial_guess_str = input("Introduzca la estimación inicial x0 (e.g., [0,0]): ")
    relaxation_factor = float(input("Introduzca el factor de relajación omega: "))
    tolerance = float(input("Ingresa la tolerancia: "))
    max_iterations = int(input("Ingresa el número máximo de iteraciones: "))
    error_type = input("Ingresa el tipo de error (rela para relativo o abs para absoluto): ")

    res = sor_method(matrix_a_str, vector_b_str, initial_guess_str, relaxation_factor, tolerance, max_iterations, error_type)
    # Imprimir resumen y tabla
    # Normalizar retorno: puede ser (summary, table) o (summary, report, table)
    if isinstance(res, (list, tuple)):
        summary = res[0] if len(res) > 0 else ''
        table = res[-1] if len(res) > 1 else []
    else:
        summary = res
        table = []

    print(summary)
    try:
        from tabulate import tabulate
        print(tabulate(table, headers=["Iteración", "Solución", "Error absoluto", "Error relativo"]))
    except Exception:
        for row in table:
            print(row)
