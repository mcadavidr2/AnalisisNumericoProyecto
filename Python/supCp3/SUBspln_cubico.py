import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

def SUBSUBspline_cubico(xin, yin):
    x = np.array(xin)
    y = np.array(yin)

    # --- Construcción del Spline Cúbico Natural ---
    spline = CubicSpline(x, y, bc_type='natural')
    
    # --- Coeficientes por tramo ---
    coeficientes = spline.c  # shape (4, n-1)
    
    polinomios_por_tramo = []
    for i in range(len(x) - 1):
        a, b, c, d = coeficientes[:, i]
        x_i = x[i]
        
        # Guardar coeficientes individuales
        polinomios_por_tramo.append({
            'tramo': i,
            'intervalo': [float(x[i]), float(x[i+1])],
            'a': float(a),  # coeficiente cúbico
            'b': float(b),  # coeficiente cuadrático  
            'c': float(c),  # coeficiente lineal
            'd': float(d),  # término independiente
            'polinomio': f"S_{i}(x) = {d:.6f} + {c:.6f}(x-{x_i:.6f}) + {b:.6f}(x-{x_i:.6f})² + {a:.6f}(x-{x_i:.6f})³"
        })
    
    # --- Evaluación para gráfica ---
    x_plot = np.linspace(min(x), max(x), 500)
    y_spline = spline(x_plot)
    
    # Retornar: [x_plot, y_plot, polinomios_por_tramo, spline, coeficientes_completos]
    return [x_plot, y_spline, polinomios_por_tramo, spline, coeficientes]