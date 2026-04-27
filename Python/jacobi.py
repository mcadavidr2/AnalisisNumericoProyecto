import numpy as np
import ast
try:
    from Python.gui_helpers import compute_spectral_radius
except Exception:
    from gui_helpers import compute_spectral_radius

def str_to_numpy_matrix(matrix_str):
    """
    Convierte una cadena de texto que representa una matriz o vector en un objeto numpy array.
    """
    try:
        matrix_list = ast.literal_eval(matrix_str)
        matrix_np = np.array(matrix_list, dtype=np.float64)
        
        return matrix_np
    except Exception as e:
        print(f"Error al convertir la cadena a matriz numpy: {e}")
        return None
    

def jacobi(A, b, x0, tolerance, max_iterations, error_type='rel', show_report=False,  auto_compare=True):
    # si se solicita informe comparativo, delegar a la versión en supCp2 (si existe)
    if show_report:
        try:
            from Python.supCp2 import subjacobi as sj
            return sj.jacobi(A, b, x0, tolerance, max_iterations, error_type, show_report=True, auto_compare=auto_compare)
        except Exception:
            try:
                import supCp2.subjacobi as sj
                return sj.jacobi(A, b, x0, tolerance, max_iterations, error_type, show_report=True, auto_compare=auto_compare)
            except Exception:
                pass
    # A, b, x0 pueden ser numpy arrays (desde GUI2) o strings (ejecución directa)
    matrix_a = A if isinstance(A, np.ndarray) else str_to_numpy_matrix(A)
    vector_b = b if isinstance(b, np.ndarray) else str_to_numpy_matrix(b)
    initial_guess = x0 if isinstance(x0, np.ndarray) else str_to_numpy_matrix(x0)
    results_matrix = []
    diagonal_matrix = np.diag(np.diag(matrix_a))
    lu_matrix = matrix_a - diagonal_matrix
    solution_vector = initial_guess
    
    for iteration_count in range(max_iterations):
        diagonal_inverse = np.linalg.inv(diagonal_matrix)
        previous_solution = solution_vector
        solution_vector = np.dot(diagonal_inverse, np.dot(-lu_matrix, solution_vector)) + np.dot(diagonal_inverse, vector_b)
        absolute_error = np.linalg.norm(solution_vector - previous_solution)
        
        if np.linalg.norm(solution_vector) != 0:
            relative_error = absolute_error / np.linalg.norm(solution_vector)
        else:
            print("Error: División por 0")
            relative_error = float('inf')
        
        results_matrix.append([iteration_count, solution_vector.copy().tolist(), absolute_error, relative_error]) 
        if error_type=="rela":
            error = relative_error
        else:
            error = absolute_error
            
        if error < tolerance:
            break

    headers = ["Iteración", "Solución", "Error absoluto", "Error relativo"]
    # Calcular radio espectral
    rho, _ = compute_spectral_radius(matrix_a, method='jacobi')
    can_conv = False if rho is None else (rho < 1)

    summary = (
        f"Radio espectral: {rho:.6f}" if rho is not None else "Radio espectral: Desconocido",
        f"Converge (rho<1)?: {'Sí' if can_conv else 'No'}",
    )

    # Devolver (summary_text, table_rows)
    summary_text = "\n".join(summary)
    return (summary_text, results_matrix)