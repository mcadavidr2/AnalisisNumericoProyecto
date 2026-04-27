import tkinter as tk
from tkinter import ttk
import numpy as np


def compute_spectral_radius(matrix_a, method='jacobi', omega=None):
    """
    Calcula el radio espectral de la matriz de iteración para cada método.
    method: 'jacobi', 'gauss_seidel' o 'sor'
    omega: factor de relajación para SOR
    Devuelve (spectral_radius, iteration_matrix) o (None, None) si falla.
    """
    try:
        A = np.array(matrix_a, dtype=np.float64)
        D = np.diag(np.diag(A))
        L = np.tril(A, -1)
        U = np.triu(A, 1)

        if method == 'jacobi':
            # T_J = -D^{-1}(L+U)
            R = A - D
            T = -np.linalg.inv(D).dot(R)
        elif method == 'gauss_seidel':
            # T_GS = -(D+L)^{-1} U
            T = -np.linalg.inv(D + L).dot(U)
        elif method == 'sor':
            if omega is None:
                raise ValueError('Omega requerido para SOR')
            # T_SOR = (D + w L)^{-1} ((1-w)D - w U)
            T = np.linalg.inv(D + omega * L).dot((1 - omega) * D - omega * U)
        else:
            return (None, None)

        eigs = np.linalg.eigvals(T)
        rho = max(np.abs(eigs))
        return (float(rho), T)
    except Exception:
        return (None, None)


def show_results_window(title, headers, rows, spectral_radius, can_converge):
    root = tk.Tk()
    root.title(title)

    frame = ttk.Frame(root, padding=10)
    frame.grid(row=0, column=0, sticky='nsew')

    # Treeview para la tabla
    tree = ttk.Treeview(frame, columns=[f'c{i}' for i in range(len(headers))], show='headings', height=10)
    for i, h in enumerate(headers):
        tree.heading(f'c{i}', text=h)
        tree.column(f'c{i}', width=150, anchor='center')

    for r in rows:
        # convertir cualquier elemento complej o np.array a string legible
        row_vals = []
        for item in r:
            if isinstance(item, (list, tuple, np.ndarray)):
                row_vals.append(str(item))
            else:
                row_vals.append(str(item))
        tree.insert('', 'end', values=row_vals)

    tree.grid(row=0, column=0, sticky='nsew')

    # Labels para radio espectral y convergencia
    info_frame = ttk.Frame(root, padding=(10, 8))
    info_frame.grid(row=1, column=0, sticky='ew')

    rho_text = 'Desconocido' if spectral_radius is None else f'{spectral_radius:.6f}'
    rho_label = ttk.Label(info_frame, text=f'Radio espectral: {rho_text}')
    rho_label.grid(row=0, column=0, sticky='w')

    conv_text = 'Sí' if can_converge else 'No'
    conv_label = ttk.Label(info_frame, text=f'¿Converge según el radio espectral?: {conv_text}')
    conv_label.grid(row=1, column=0, sticky='w')

    # Botón cerrar
    button = ttk.Button(root, text='Cerrar', command=root.destroy)
    button.grid(row=2, column=0, pady=8)

    # Ajustes de redimensionado
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    root.mainloop()
