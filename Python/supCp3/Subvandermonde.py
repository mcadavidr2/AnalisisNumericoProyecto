import numpy as np
import time
import matplotlib.pyplot as plt

def interpol_vandermonde(inx, iny):
    x = np.array(inx)
    y = np.array(iny)

    start_time = time.time()
    V = np.vander(x, increasing=True)
    cond_num = np.linalg.cond(V)
    a = np.linalg.solve(V, y)
    
    end_time = time.time()
    tiempo_ejecucion = end_time - start_time
    
    x_plot = np.linspace(min(x), max(x), 100)
    y_vander = np.polyval(a[::-1], x_plot)
    
    # Construir polinomio como string
    poly_str = "P(x) = "
    for i, coef in enumerate(a):
        if i == 0:
            poly_str += f"{coef:.6f}"
        else:
            poly_str += f" + {coef:.6f}·x^{i}"
    
    # Retornar: x_plot, y_plot, coeficientes, tiempo, cond_num, polinomio_str
    return [x_plot, y_vander, a.tolist(), tiempo_ejecucion, cond_num, poly_str]