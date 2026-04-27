import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

def SUBSUBspline_cubico(xin, yin):
    x= xin
    y= yin

    # --- Construcción del Spline Cúbico Natural ---
    spline = CubicSpline(x, y, bc_type='natural')  # bc_type='natural' -> Segunda derivada = 0 en los extremos

    # --- Coeficientes de los polinomios por tramo ---
    coeficientes = spline.c 
    for i in range(len(x) - 1):
        a, b, c, d = coeficientes[:, i]
        x_i = x[i]
    x_plot = np.linspace(min(x), max(x), 500)
    y_spline = spline(x_plot)

    return [x_plot, y_spline,[a, b, c, d]]