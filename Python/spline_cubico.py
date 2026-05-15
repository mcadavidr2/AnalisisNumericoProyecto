import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.interpolate import CubicSpline
from Python.supCp3 import SUBinterpol_lagrange
from Python.supCp3 import SUBinterpol_newton
from Python.supCp3.SUBspln_cubico import SUBSUBspline_cubico
from Python.supCp3.SUBspline_lineal import SUBSUBspline_lineal  # ✅ Corregido
from Python.supCp3 import Subvandermonde

def spline_cubico(ValoresX=None, ValoresY=None, show_report=True, auto_compare=True):
    eval_grid = 500
    
    # Entrada de datos si no se pasan argumentos
    if ValoresX is None or ValoresY is None:
        x = input("Ingrese los valores de x separados por coma: ")
        y = input("Ingrese los valores de y separados por coma: ")
        x = np.array([float(val) for val in x.split(",")])
        y = np.array([float(val) for val in y.split(",")])
    else:
        x = np.array(ValoresX)
        y = np.array(ValoresY)

    # Validación
    if x.ndim == 0 or len(x) < 2:
        raise ValueError("Debe ingresar al menos dos puntos para interpolar.")
    if not np.all(np.diff(x) > 0):
        raise ValueError("Los valores de x deben estar en orden creciente")
    if len(x) != len(y):
        raise ValueError("Las listas de x e y deben tener la misma longitud.")

    # --- Construcción del Spline Cúbico Natural ---
    start_time = time.perf_counter()
    spline = CubicSpline(x, y, bc_type='natural')

    # --- Coeficientes de los polinomios por tramo ---
    coeficientes = spline.c  # (4, n-1): [d, c, b, a] para cada tramo i (en scipy)
    polinomios = []
    for i in range(len(x) - 1):
        # En scipy, los coeficientes están en orden: [a, b, c, d] donde:
        # S(x) = a + b*(x-x_i) + c*(x-x_i)² + d*(x-x_i)³
        d, c, b, a = coeficientes[:, i]  # Scipy devuelve [d, c, b, a]
        x_i = x[i]
        polinomio = (
            f"S_{i}(x) = {a:.6f} + {b:.6f}(x-{x_i:.6f}) + {c:.6f}(x-{x_i:.6f})² + {d:.6f}(x-{x_i:.6f})³\n"
            f"    para x ∈ [{x_i:.6f}, {x[i+1]:.6f}]"
        )
        polinomios.append(polinomio)

    # --- Evaluación y gráfica ---
    x_plot = np.linspace(min(x), max(x), 500)
    y_spline = spline(x_plot)
    end_time = time.perf_counter()
    tiempo_ejecucion = end_time - start_time

    resultado = (
        f"Puntos ingresados: {list(zip(ValoresX, ValoresY))}\n"
        f"Polinomios por tramo:\n\n" + "\n\n".join(polinomios) +
        f"\n\nTiempo de ejecución: {tiempo_ejecucion:.6f} segundos"
    )

    info = {
        "tiempo": tiempo_ejecucion,
        "n_tramos": len(polinomios),
        "polinomios_por_tramo": polinomios,
        "polinomio_str": None,
        "polinomio_obj": spline,
        "coeficientes": coeficientes.tolist() if hasattr(coeficientes, 'tolist') else None,
        "condicion": None,
    }

    if show_report:
        try:
            fig, (ax_plot, ax_table) = plt.subplots(ncols=2, figsize=(12, 6), gridspec_kw={'width_ratios': [3, 2]})
            # plot
            ax_plot.plot(x, y, 'ro', label='Puntos dados', markersize=8)
            ax_plot.plot(x_plot, y_spline, 'b-', label='Spline Cúbico Natural', linewidth=2)
            ax_plot.set_title("Interpolación con Spline Cúbico Natural")
            ax_plot.set_xlabel("x")
            ax_plot.set_ylabel("y")
            ax_plot.legend()
            ax_plot.grid()

            # tabla
            rows = [
                ["Tiempo (s)", f"{tiempo_ejecucion:.6f}"],
                ["Número de tramos", f"{len(polinomios)}"],
                ["Primer polinomio (trunc)", polinomios[0][:100] + ("..." if len(polinomios[0]) > 100 else "")]
            ]
            col_labels = ["Propiedad", "Valor"]
            ax_table.axis('off')
            table = ax_table.table(cellText=rows, colLabels=col_labels, loc='center', cellLoc='left')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2)

            # polinomios completos en figura adicional
            fig_pol = plt.figure(figsize=(10, 2 + 0.5 * len(polinomios)))
            plt.axis('off')
            text = "\n\n".join(polinomios)
            plt.text(0.01, 0.99, text, va='top', family='monospace', fontsize=9)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error en show_report: {e}")

    # Comparación automática
    if auto_compare:
        try:
            eval_pts = np.linspace(min(x), max(x), max(100, int(eval_grid)))
            # Referencia: el spline cúbico actual
            y_ref = spline(eval_pts)

            other_results = {}
            
            # Lagrange
            try:
                res = SUBinterpol_lagrange.interpol_lagrange(x, y)
                other_results['Lagrange'] = res  # [x_plot, y_plot, polinomio_str, tiempo, polinomios_base]
            except Exception as e:
                print(f"Error en Lagrange: {e}")
                other_results['Lagrange'] = None
            
            # Newton
            try:
                res = SUBinterpol_newton.interpol_newton(x, y)
                other_results['Newton'] = res  # [x_plot, y_plot, polinomio_str, tiempo, coeficientes]
            except Exception as e:
                print(f"Error en Newton: {e}")
                other_results['Newton'] = None
            
            # Spline Lineal
            try:
                res = SUBSUBspline_lineal(x, y)  # ✅ Ahora importado correctamente
                other_results['Spline_lineal'] = res  # [x_plot, y_plot, polinomios, spline]
            except Exception as e:
                print(f"Error en Spline_lineal: {e}")
                other_results['Spline_lineal'] = None
            
            # Vandermonde
            try:
                res = Subvandermonde.interpol_vandermonde(x, y)
                other_results['Vandermonde'] = res  # [x_plot, y_plot, coeficientes, tiempo, cond_num]
            except Exception as e:
                print(f"Error en Vandermonde: {e}")
                other_results['Vandermonde'] = None

            def reeval_on_common(res):
                if res is None:
                    return None
                if len(res) < 2:
                    return None
                rx = np.array(res[0])
                ry = np.array(res[1])
                # Interpolar en eval_pts
                return np.interp(eval_pts, rx, ry)

            metrics = {}
            for name, res in other_results.items():
                y_cmp = reeval_on_common(res)
                if y_cmp is None:
                    metrics[name] = {'max_err': None, 'rmse': None}
                else:
                    diff = y_ref - y_cmp
                    metrics[name] = {
                        'max_err': float(np.max(np.abs(diff))),
                        'rmse': float(np.sqrt(np.mean(diff**2)))
                    }

            if show_report:
                try:
                    fig, (ax_plot2, ax_table) = plt.subplots(ncols=2, figsize=(12, 5), gridspec_kw={'width_ratios': [3, 2]})
                    ax_plot2.plot(x, y, 'ro', label='Puntos dados', markersize=8)
                    ax_plot2.plot(eval_pts, y_ref, 'k-', linewidth=2, label='Spline cúbico (referencia)')
                    
                    colors = {
                        'Lagrange': 'm--',
                        'Newton': 'c--',
                        'Spline_lineal': 'g--',
                        'Vandermonde': 'b--'
                    }
                    
                    for name, res in other_results.items():
                        ycmp = reeval_on_common(res)
                        if ycmp is not None:
                            ax_plot2.plot(eval_pts, ycmp, colors.get(name, '--'), label=name, alpha=0.7)
                    
                    ax_plot2.set_title('Comparación de métodos de interpolación')
                    ax_plot2.set_xlabel('x')
                    ax_plot2.set_ylabel('y')
                    ax_plot2.legend()
                    ax_plot2.grid()

                    # Tabla de métricas
                    col_labels = ['Método', 'Max error', 'RMSE']
                    rows = []
                    for name in ['Lagrange', 'Newton', 'Spline_lineal', 'Vandermonde']:
                        m = metrics.get(name, {})
                        max_err = f"{m['max_err']:.6e}" if m['max_err'] is not None else 'N/A'
                        rmse = f"{m['rmse']:.6e}" if m['rmse'] is not None else 'N/A'
                        rows.append([name, max_err, rmse])
                    
                    ax_table.axis('off')
                    table = ax_table.table(cellText=rows, colLabels=col_labels, loc='center')
                    table.auto_set_font_size(False)
                    table.set_fontsize(9)
                    table.scale(1, 2)
                    plt.tight_layout()
                    plt.show()
                except Exception as e:
                    print(f"Error en gráfica de comparación: {e}")
        except Exception as e:
            print(f"Error en auto_compare: {e}")

    return resultado, info


if __name__ == "__main__":
    resultado, info = spline_cubico()
    print(resultado)