import numpy as np
import time
import matplotlib.pyplot as plt

def interpol_newton(xin, yin):
    x = xin
    y = yin
    # --- Construcción del polinomio de Newton ---
    start_time = time.time()

    # Tabla de diferencias divididas
    n_points = len(x)
    F = np.zeros((n_points, n_points))
    F[:, 0] = y  # Primera columna son las y_i

    for j in range(1, n_points):
        for i in range(n_points - j):
            F[i, j] = (F[i + 1, j - 1] - F[i, j - 1]) / (x[i + j] - x[i])

    # Coeficientes del polinomio (primera fila de F)
    a = F[0, :]

    # --- Formatear el polinomio como string ---
    polinomio = f"P(x) = {a[0]:.4f}"
    termino = ""
    for i in range(1, n_points):
        termino += f"(x - {x[i-1]:.4f})"
        polinomio += f" + {a[i]:.4f}" + termino

    end_time = time.time()
    tiempo_ejecucion = end_time - start_time

    # --- Evaluar el polinomio (para gráfica) ---
    def P_newton(x_eval):
        result = a[-1]
        for i in range(len(a) - 2, -1, -1):
            result = result * (x_eval - x[i]) + a[i]
        return result

    # --- Resultados ---

    # --- Gráfica ---
    x_plot = np.linspace(min(x), max(x), 100)
    y_newton = [P_newton(xi) for xi in x_plot]

    return [x_plot, y_newton, polinomio, tiempo_ejecucion]

# Ejecutar
#interpolacion_newton()