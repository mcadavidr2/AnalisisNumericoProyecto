import numpy as np
import time
import matplotlib.pyplot as plt

def interpol_newton(xin, yin):
    x = np.array(xin)
    y = np.array(yin)
    
    start_time = time.time()

    # Tabla de diferencias divididas
    n_points = len(x)
    F = np.zeros((n_points, n_points))
    F[:, 0] = y

    for j in range(1, n_points):
        for i in range(n_points - j):
            F[i, j] = (F[i + 1, j - 1] - F[i, j - 1]) / (x[i + j] - x[i])

    # Coeficientes del polinomio (primera fila de F)
    a = F[0, :]

    # --- Formatear el polinomio como string ---
    polinomio = f"P(x) = {a[0]:.6f}"
    termino = ""
    for i in range(1, n_points):
        termino += f"(x - {x[i-1]:.6f})"
        signo = "+" if a[i] >= 0 else "-"
        polinomio += f" {signo} {abs(a[i]):.6f}{termino}"

    end_time = time.time()
    tiempo_ejecucion = end_time - start_time

    # --- Evaluar el polinomio (para gráfica) ---
    def P_newton(x_eval):
        result = a[-1]
        for i in range(len(a) - 2, -1, -1):
            result = result * (x_eval - x[i]) + a[i]
        return result

    x_plot = np.linspace(min(x), max(x), 100)
    y_newton = [P_newton(xi) for xi in x_plot]

    # Retornar: x_plot, y_plot, polinomio_str, tiempo, coeficientes
    return [x_plot, y_newton, polinomio, tiempo_ejecucion, a.tolist()]