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
        raise ValueError(f"Error al convertir cadena a matriz numpy: {e}")

def gauss_seidel_method(A, b, x0, tolerance, max_iterations, error_type='rel', show_report=False, auto_compare=True):
    # si se solicita informe comparativo delegar a supCp2
    if show_report:
        try:
            from Python.supCp2 import subgauss_seidel as sg
            return sg.gauss_seidel_method(A, b, x0, tolerance, max_iterations, error_type, show_report=True, auto_compare=auto_compare)
        except Exception:
            try:
                import supCp2.subgauss_seidel as sg
                return sg.gauss_seidel_method(A, b, x0, tolerance, max_iterations, error_type, show_report=True, auto_compare=auto_compare)
            except Exception:
                pass
    matrix_a = A if isinstance(A, np.ndarray) else str_to_numpy_matrix(A)
    vector_b = b if isinstance(b, np.ndarray) else str_to_numpy_matrix(b)
    initial_guess = x0 if isinstance(x0, np.ndarray) else str_to_numpy_matrix(x0)
    results_matrix = []

    diagonal_matrix = np.diag(np.diag(matrix_a))
    lu_matrix = matrix_a - diagonal_matrix
    solution_vector = initial_guess

    for iteration_count in range(max_iterations):
        previous_solution = solution_vector.copy()
        for j in range(matrix_a.shape[0]):
            solution_vector[j] = safe_divide(
                (vector_b[j] - np.dot(lu_matrix[j, :], solution_vector)),
                diagonal_matrix[j, j]
            )
        solution_vector = np.round(solution_vector, decimals=5)
        absolute_error = np.linalg.norm(solution_vector - previous_solution)

        if np.linalg.norm(solution_vector) != 0:
            relative_error = absolute_error / np.linalg.norm(solution_vector)
        else:
            relative_error = float('inf')


        if error_type == "rela":
            error = relative_error
        else:
            error = absolute_error
        results_matrix.append([iteration_count,solution_vector.copy().tolist(),round(float(absolute_error), 6),round(float(relative_error), 6)])
        if error < tolerance:
            break


    headers = ["Iteración", "Solución", "Error absoluto", "Error relativo"]
    rho, _ = compute_spectral_radius(matrix_a, method='gauss_seidel')
    can_conv = False if rho is None else (rho < 1)

    summary = (
        f"Radio espectral: {rho:.6f}" if rho is not None else "Radio espectral: Desconocido",
        f"Converge (rho<1)?: {'Sí' if can_conv else 'No'}",
    )
    summary_text = "\n".join(summary)
    return (summary_text, results_matrix)