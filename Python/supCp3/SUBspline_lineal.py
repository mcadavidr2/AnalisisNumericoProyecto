import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def SUBSUBspline_lineal(inx,iny):
    x=inx
    y=iny

    # --- Construcción del Spline Lineal ---
    spline = interp1d(x, y, kind='linear')

    # --- Mostrar polinomios por tramo ---
   # print("\n🔹 POLINOMIOS POR TRAMO:")
    polinomios = []
    for i in range(len(x) - 1):
        x0, x1 = x[i], x[i + 1]
        y0, y1 = y[i], y[i + 1]
        pendiente = (y1 - y0) / (x1 - x0)
        polinomio = f"S_{i}(x) = {y0:.4f} + {pendiente:.4f}(x - {x0:.4f})"
        dominio = f"para x ∈ [{x0:.4f}, {x1:.4f}]"
        polinomios.append([polinomio, dominio])
        #print(f"- {polinomio} \t{dominio}")

    # --- Evaluación y gráfica ---
    x_plot = np.linspace(min(x), max(x), 100)
    y_spline = spline(x_plot)

    return [x_plot, y_spline, polinomios]

# Ejecutar
#spline_lineal_con_polinomios()