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
    """Convierte una cadena que representa una matriz o vector en numpy array."""
    try:
        matrix_list = ast.literal_eval(matrix_str)
        matrix_np = np.array(matrix_list, dtype=np.float64)
        return matrix_np
    except Exception as e:
        raise ValueError(f"Error al convertir cadena a matriz numpy: {e}")

def compute_error(prev_sol, curr_sol, error_type):
    """Calcula el error según el tipo especificado."""
    diff = curr_sol - prev_sol
    norm_diff_inf = np.linalg.norm(diff, ord=np.inf)
    
    if error_type == 'abs':
        return norm_diff_inf
    
    elif error_type == 'rel':  # ||Xn-Xn-1||∞ / ||Xn||∞
        norm_curr = np.linalg.norm(curr_sol, ord=np.inf)
        if norm_curr == 0:
            return float('inf')
        return norm_diff_inf / norm_curr
    
    elif error_type == 'rel2':  # ||Xn-Xn-1||∞ / ||Xn-1||∞
        norm_prev = np.linalg.norm(prev_sol, ord=np.inf)
        if norm_prev == 0:
            return float('inf')
        return norm_diff_inf / norm_prev
    
    elif error_type == 'rel3':  # ||(Xn-Xn-1) ./ Xn||∞
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_error_vec = np.divide(diff, curr_sol, out=np.full_like(diff, np.inf), where=(curr_sol != 0))
        return np.linalg.norm(rel_error_vec, ord=np.inf)
    
    elif error_type == 'rel4':  # ||(Xn-Xn-1) ./ Xn-1||∞
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_error_vec = np.divide(diff, prev_sol, out=np.full_like(diff, np.inf), where=(prev_sol != 0))
        return np.linalg.norm(rel_error_vec, ord=np.inf)
    
    else:
        raise ValueError(f"Tipo de error no reconocido: {error_type}")

def gauss_seidel_method(A, b, x0, tolerance, max_iterations, error_type='rel', show_report=False, auto_compare=True):
    # Si se solicita informe comparativo delegar a supCp2
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
    solution_vector = initial_guess.copy().astype(np.float64)
    
    # Verificar diagonal
    if np.any(np.diag(diagonal_matrix) == 0):
        return ("Error: Elementos cero en la diagonal. Gauss-Seidel no puede aplicarse.", [])
    
    for iteration_count in range(max_iterations):
        previous_solution = solution_vector.copy()
        
        for j in range(matrix_a.shape[0]):
            solution_vector[j] = safe_divide(
                (vector_b[j] - np.dot(lu_matrix[j, :], solution_vector)),
                diagonal_matrix[j, j]
            )
        
        # Calcular error usando la función unificada
        error = compute_error(previous_solution, solution_vector, error_type)
        
        # Calcular errores para mostrar en tabla
        diff_norm_inf = np.linalg.norm(solution_vector - previous_solution, ord=np.inf)
        norm_prev = np.linalg.norm(previous_solution, ord=np.inf)
        abs_error = diff_norm_inf
        rel_error = diff_norm_inf / norm_prev if norm_prev != 0 else float('inf')
        
        # Agregar fila con 6 columnas: Iter, Solución, Error_abs, Error_rel, Error_parada, Tipo_error
        results_matrix.append([
            iteration_count,
            solution_vector.copy().tolist(),
            round(float(abs_error), 8),
            round(float(rel_error), 8),
            round(float(error), 8),
            error_type
        ])
        
        if error < tolerance:
            break
    
    rho, _ = compute_spectral_radius(matrix_a, method='gauss_seidel')
    can_conv = False if rho is None else (rho < 1)
    
    summary = (
        f"Radio espectral: {rho:.6f}" if rho is not None else "Radio espectral: Desconocido",
        f"Converge (rho<1)?: {'Sí' if can_conv else 'No'}",
        f"Tipo de error usado: {error_type}",
        f"Tolerancia: {tolerance}"
    )
    summary_text = "\n".join(summary)
    
    return (summary_text, results_matrix)