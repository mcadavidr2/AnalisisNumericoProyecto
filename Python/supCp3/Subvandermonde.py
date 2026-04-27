import numpy as np
import time
import matplotlib.pyplot as plt

def interpol_vandermonde(inx,iny):
    x = inx
    y = iny

    start_time = time.time()
    V = np.vander(x, increasing=True)
    cond_num = np.linalg.cond(V)
    a = np.linalg.solve(V, y)
    
    end_time = time.time()
    tiempo_ejecucion = end_time - start_time
    x_plot = np.linspace(min(x), max(x), 100)
    y_vander = np.polyval(a[::-1], x_plot)

    return [x_plot, y_vander,tiempo_ejecucion,cond_num]