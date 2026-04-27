import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.interpolate import CubicSpline
from Python.supCp3 import SUBinterpol_lagrange
from Python.supCp3 import SUBinterpol_newton
from Python.supCp3 import SUBspln_cubico
from Python.supCp3 import SUBspline_lineal
from Python.supCp3 import Subvandermonde

def spline_cubico(ValoresX=None, ValoresY=None, show_report=True, auto_compare=True):
    eval_grid=500
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
    spline = CubicSpline(x, y, bc_type='natural')  # Segunda derivada = 0 en los extremos

    # --- Coeficientes de los polinomios por tramo ---
    coeficientes = spline.c  # (4, n-1): [a_i, b_i, c_i, d_i] para cada tramo i
    polinomios = []
    for i in range(len(x) - 1):
        a, b, c, d = coeficientes[:, i]
        x_i = x[i]
        polinomio = (
            f"S_{i}(x) = {a:.4f} + {b:.4f}(x-{x_i:.4f}) + {c:.4f}(x-{x_i:.4f})² + {d:.4f}(x-{x_i:.4f})³\n"
            f"    para x ∈ [{x_i:.4f}, {x[i+1]:.4f}]"
        )
        polinomios.append(polinomio)

    # --- Evaluación y gráfica ---
    x_plot = np.linspace(min(x), max(x), 500)
    y_spline = spline(x_plot)

    x_plot = np.linspace(min(x), max(x), 500)
    y_spline = spline(x_plot)
    end_time = time.perf_counter()
    tiempo_ejecucion = end_time - start_time

    resultado = (
        f"Puntos ingresados: {list(zip(ValoresX,ValoresY))}\n"
        f"Polinomios por tramo:\n\n" + "\n\n".join(polinomios)
    )

    info = {
        "tiempo": tiempo_ejecucion,
        "n_tramos": len(polinomios),
        "polinomios_por_tramo": polinomios,
        "polinomio_str": None,
        "polinomio_obj": None,
        "coeficientes": coeficientes.tolist() if hasattr(coeficientes, 'tolist') else None,
        "condicion": None,
    }

    if show_report:
        try:
            fig, (ax_plot, ax_table) = plt.subplots(ncols=2, figsize=(12, 6), gridspec_kw={'width_ratios':[3,2]})
            # plot
            ax_plot.plot(x, y, 'ro', label='Puntos dados', markersize=8)
            ax_plot.plot(x_plot, y_spline, 'b-', label='Spline Cúbico', linewidth=2)
            ax_plot.set_title("Interpolación con Spline Cúbico Natural")
            ax_plot.set_xlabel("x")
            ax_plot.set_ylabel("y")
            ax_plot.legend()
            ax_plot.grid()

            # tabla
            rows = [
                ["Tiempo (s)", f"{tiempo_ejecucion:.6f}"],
                ["Número de tramos", f"{len(polinomios)}"],
                ["Primer polinomio (trunc)", polinomios[0][:100] + ("..." if len(polinomios[0])>100 else "")]
            ]
            col_labels = ["Propiedad", "Valor"]
            ax_table.axis('off')
            table = ax_table.table(cellText=rows, colLabels=col_labels, loc='center', cellLoc='left')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1,2)

            # polinomios completos en figura adicional
            fig_pol = plt.figure(figsize=(10, 2+0.5*len(polinomios)))
            plt.axis('off')
            text = "\n\n".join(polinomios)
            plt.text(0.01, 0.99, text, va='top', family='monospace', fontsize=9)
            plt.tight_layout()
            plt.show()
        except Exception:
            pass

    # Comparación automática: llamar a los otros métodos del paquete supCp3
    if auto_compare:
        try:
            eval_pts = np.linspace(min(x), max(x), max(100, int(eval_grid)))
            y_ref = np.interp(eval_pts, x_plot, y_spline)

            other_results = {}
            try:
                other_results['Lagrange'] = SUBinterpol_lagrange.interpol_lagrange(x, y)
            except Exception:
                other_results['Lagrange'] = None
            try:
                other_results['Newton'] = SUBinterpol_newton.interpol_newton(x, y)
            except Exception:
                other_results['Newton'] = None
            try:
                other_results['Spline_lineal'] = SUBspline_lineal.SUBSUBspline_lineal(x, y)
            except Exception:
                other_results['Spline_lineal'] = None
            try:
                other_results['Vandermonde'] = Subvandermonde.interpol_vandermonde(x, y)
            except Exception:
                other_results['Vandermonde'] = None

            def reeval_on_common(res):
                if res is None:
                    return None
                rx = np.array(res[0])
                ry = np.array(res[1])
                return np.interp(eval_pts, rx, ry)

            metrics = {}
            for name, res in other_results.items():
                y_cmp = reeval_on_common(res)
                if y_cmp is None:
                    metrics[name] = {'max_err': None, 'rmse': None}
                else:
                    diff = y_ref - y_cmp
                    metrics[name] = {'max_err': float(np.max(np.abs(diff))), 'rmse': float(np.sqrt(np.mean(diff**2)))}

            if show_report:
                try:
                    fig, (ax_plot2, ax_table) = plt.subplots(ncols=2, figsize=(12,5), gridspec_kw={'width_ratios':[3,2]})
                    ax_plot2.plot(x, y, 'ro', label='Puntos dados')
                    ax_plot2.plot(eval_pts, y_ref, 'k-', label='Spline cúbico (referencia)')
                    colors = {'Lagrange':'m--','Newton':'c--','Spline_lineal':'g--'}
                    for name, res in other_results.items():
                        ycmp = reeval_on_common(res)
                        if ycmp is not None:
                            ax_plot2.plot(eval_pts, ycmp, colors.get(name,'--'), label=name)
                    ax_plot2.set_title('Comparación respecto a Spline cúbico')
                    ax_plot2.set_xlabel('x')
                    ax_plot2.set_ylabel('y')
                    ax_plot2.legend()
                    ax_plot2.grid()

                    col_labels = ['Método','Max err','RMSE']
                    rows = []
                    for name in ['Lagrange','Newton','Spline_lineal','Vandermonde']:
                        m = metrics.get(name, {})
                        rows.append([name, f"{m['max_err']:.6g}" if m['max_err'] is not None else 'N/A', f"{m['rmse']:.6g}" if m['rmse'] is not None else 'N/A'])
                    ax_table.axis('off')
                    table = ax_table.table(cellText=rows, colLabels=col_labels, loc='center')
                    table.auto_set_font_size(True)
                    table.set_fontsize(9)
                    table.scale(1,2)
                    plt.tight_layout()
                    plt.show()
                except Exception:
                    pass
        except Exception:
            pass

    return resultado, info

    # Comparación con otros métodos (opcional)
    comparar = input("\n¿Desea comparar con otros métodos? (s/n): ").strip().lower()
    if comparar == 's':
        ILG = SUBinterpol_lagrange.interpol_lagrange(x, y)
        INT = SUBinterpol_newton.interpol_newton(x, y)
        SPL = SUBspline_lineal.SUBSUBspline_lineal(x, y)
        VAN = Subvandermonde.interpol_vandermonde(x, y)

        plt.figure(figsize=(10, 6))
        plt.plot(x, y, 'ro', label='Puntos dados')
        plt.plot(x_plot, y_spline, 'b-', label='Spline Cúbico')
        plt.plot(VAN[0], VAN[1], 'k-', label='Vandermonde')
        plt.plot(SPL[0], SPL[1], 'g--', label='Spline lineal')
        plt.plot(ILG[0], ILG[1], 'm--', label='Lagrange')
        plt.plot(INT[0], INT[1], 'c--', label='Newton')
        plt.title("Comparación General")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid()
        plt.show(block=False)
        resultado += "\nComparación general mostrada en la gráfica."

    

# Ejecutar solo si es script principal
if __name__ == "__main__":
    print(spline_cubico())