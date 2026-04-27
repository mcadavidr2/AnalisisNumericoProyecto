import numpy as np
import ast
import time
import matplotlib.pyplot as plt
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
    

def _compute_once(A_local, b_local, x0_local, tol, max_it, err_type):
    A_n = A_local.copy()
    b_n = b_local.copy()
    x = x0_local.copy()
    rows = []
    start = time.perf_counter()
    D = np.diag(np.diag(A_n))
    R = A_n - D
    for k in range(max_it):
        x_prev = x.copy()
        x = np.dot(np.linalg.inv(D), -np.dot(R, x_prev)) + np.dot(np.linalg.inv(D), b_n)
        abs_err = np.linalg.norm(x - x_prev)
        rel_err = abs_err / np.linalg.norm(x) if np.linalg.norm(x) != 0 else float('inf')
        rows.append([k, x.copy().tolist(), float(abs_err), float(rel_err)])
        err = rel_err if err_type == 'rel' else abs_err
        if err < tol:
            break
    end = time.perf_counter()
    metrics = {'iterations': len(rows), 'abs_error': float(rows[-1][2]) if rows else None, 'rel_error': float(rows[-1][3]) if rows else None, 'time': end - start}
    return rows, metrics


def jacobi(A, b, x0, tolerance, max_iterations, error_type='rel', show_report=False, auto_compare=True):
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

    # use module-level _compute_once for comparisons

    # resultados principales (ya calculados en results_matrix), pero obtener métricas
    main_metrics = {'iterations': len(results_matrix), 'abs_error': float(results_matrix[-1][2]) if results_matrix else None, 'rel_error': float(results_matrix[-1][3]) if results_matrix else None}

    # Calcular radio espectral
    rho, _ = compute_spectral_radius(matrix_a, method='jacobi')
    can_conv = False if rho is None else (rho < 1)
    summary = (
        f"Radio espectral: {rho:.6f}" if rho is not None else "Radio espectral: Desconocido",
        f"Converge (rho<1)?: {'Sí' if can_conv else 'No'}",
    )
    summary_text = "\n".join(summary)

    # Si se solicitó informe comparativo construir informe ejecutando los métodos disponibles
    if show_report:
        # definir tipos de error a probar
        ets = [error_type] if error_type is not None else ['abs', 'rel']
        informe = {}
        # incluir Jacobi
        for et in ets:
                rows_j, metrics_j = _compute_once(matrix_a, vector_b, initial_guess, tolerance, max_iterations, et)
                informe.setdefault('Jacobi', {})[et] = {'rows': rows_j, 'metrics': metrics_j}

        # intentar importar y ejecutar Gauss-Seidel y SOR desde el mismo paquete
        try:
            from Python.supCp2 import subgauss_seidel as sg
            from Python.supCp2 import subSOR as ss
        except Exception:
            try:
                import supCp2.subgauss_seidel as sg
                import supCp2.subSOR as ss
            except Exception:
                sg = None
                ss = None

        if sg is not None:
            for et in ets:
                r_gs, m_gs = sg._compute_once(matrix_a, vector_b, initial_guess, tolerance, max_iterations, et)
                informe.setdefault('Gauss-Seidel', {})[et] = {'rows': r_gs, 'metrics': m_gs}

        if ss is not None:
            for et in ets:
                r_sor, m_sor = ss._compute_once(matrix_a, vector_b, initial_guess, tolerance, max_iterations, et)
                informe.setdefault('SOR', {})[et] = {'rows': r_sor, 'metrics': m_sor}

        # identificar mejor método por cada tipo de error (menor iteraciones, en caso de empate menor abs_error)
        resumen_best = {}
        for et in ets:
            best = None
            for method, data in informe.items():
                if et not in data:
                    continue
                m = data[et]
                if best is None or m['iterations'] < best['iterations'] or (m['iterations'] == best['iterations'] and m['abs_error'] < best['abs_error']):
                    best = {'method': method, **m}
            resumen_best[et] = best

        # Generar una figura única con subplots (uno por tipo de error) y una tabla resumen
        try:
            n = len(ets)
            fig = plt.figure(figsize=(10, 3 * n + 2))
            gs = fig.add_gridspec(n + 1, 1, height_ratios=[3] * n + [1])

            for i, et in enumerate(ets):
                ax = fig.add_subplot(gs[i, 0])
                for method in informe:
                    data = informe[method].get(et)
                    if not data:
                        continue
                    rows = data['rows']
                    its = [r[0] for r in rows]
                    vals = [r[3] for r in rows] if et == 'rel' else [r[2] for r in rows]
                    ax.semilogy(its, vals, marker='o', label=method)
                ax.set_title(f'Convergencia por iteración ({et})')
                ax.set_xlabel('Iteración')
                ax.set_ylabel('Error relativo' if et == 'rel' else 'Error absoluto')
                ax.grid(True, which='both', ls='--')
                ax.legend()

            # Tabla resumen en la parte inferior
            ax_table = fig.add_subplot(gs[-1, 0])
            ax_table.axis('off')
            header = ['Método']
            for et in ets:
                header += [f'{et}-it', f'{et}-abs', f'{et}-rel']

            rows_table = []
            methods = list(informe.keys())
            for method in methods:
                row = [method]
                for et in ets:
                    data = informe[method].get(et)
                    if data:
                        m = data['metrics']
                        row += [str(m.get('iterations')), f"{m.get('abs_error'):.3g}" if m.get('abs_error') is not None else '', f"{m.get('rel_error'):.3g}" if m.get('rel_error') is not None else '']
                    else:
                        row += ['', '', '']
                rows_table.append(row)

            table = ax_table.table(cellText=rows_table, colLabels=header, loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.5)
            plt.tight_layout()
            plt.show()
        except Exception:
            pass

        report = {'metrics': informe, 'best': resumen_best}
        # Siempre devolver también la tabla de iteraciones como último elemento
        return (summary_text, report, results_matrix)

    return (summary_text, results_matrix)