import numpy as np
import time
import matplotlib.pyplot as plt

def interpol_lagrange(inx, iny):
    x = np.array(inx)
    y = np.array(iny)

    start_time = time.time()

    # Función para calcular el polinomio base L_i(x)
    def L(i, x_eval):
        result = 1.0
        for j in range(len(x)):
            if j != i:
                result *= (x_eval - x[j]) / (x[i] - x[j])
        return result

    # Función para evaluar el polinomio interpolador P(x)
    def P_lagrange(x_eval):
        return sum(y[i] * L(i, x_eval) for i in range(len(x)))

    # --- Formatear el polinomio como string ---
    polinomio = "P(x) = "
    for i in range(len(x)):
        term = f"{y[i]:.6f}·L_{i}(x)"
        if i > 0:
            polinomio += " + " + term
        else:
            polinomio += term

    # Polinomios base L_i(x)
    polinomios_base = []
    for i in range(len(x)):
        L_i = f"L_{i}(x) = "
        factores = []
        for j in range(len(x)):
            if j != i:
                factores.append(f"(x - {x[j]:.6f})/({x[i]:.6f} - {x[j]:.6f})")
        L_i += "·".join(factores)
        polinomios_base.append(L_i)

    end_time = time.time()
    tiempo_ejecucion = end_time - start_time

    # --- Evaluación para gráfica ---
    x_plot = np.linspace(min(x), max(x), 100)
    y_lagrange = [P_lagrange(xi) for xi in x_plot]

    # Retornar: x_plot, y_plot, polinomio_str, tiempo, polinomios_base
    return [x_plot, y_lagrange, polinomio, tiempo_ejecucion, polinomios_base]