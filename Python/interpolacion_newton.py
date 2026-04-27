import numpy as np
import time
import matplotlib.pyplot as plt
from Python.supCp3 import SUBinterpol_lagrange
#from Python.supCp3 import SUBinterpol_newton
from Python.supCp3 import SUBspln_cubico
from Python.supCp3 import SUBspline_lineal
from Python.supCp3 import Subvandermonde

def interpolacion_newton(ValoresX=None, ValoresY=None, show_report=True, auto_compare=True):
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
        polinomio += f" + {a[i]:.4f}{termino}"

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

    resultado = (
        f"Puntos ingresados: {list(zip(ValoresX,ValoresY))}\n"
        f"Polinomio de Newton:\n{polinomio}\n\n"
        f"Tiempo de ejecución: {tiempo_ejecucion:.6f} segundos"
    )

    info = {
        "tiempo": tiempo_ejecucion,
        "coeficientes": a.tolist(),
        "polinomio_str": polinomio,
        "polinomio_obj": None,
        "polinomios_base": None,
        "polinomios_por_tramo": None,
        "n_tramos": None,
        "condicion": None,
    }

    # Construir objeto polinomio estándar para inspección (coeficientes en base monómica)
    try:
        coeffs = np.polyfit(x, y, len(x) - 1)
        poly_obj = np.poly1d(coeffs)
        info['polinomio_obj'] = poly_obj
        info['coeficientes'] = coeffs.tolist()
    except Exception:
        # si falla, dejar polinomio_obj como None
        pass

    if show_report:
        try:
            fig, (ax_plot, ax_table) = plt.subplots(ncols=2, figsize=(12,6), gridspec_kw={'width_ratios':[3,2]})
            ax_plot.plot(x, y, 'ro', label='Puntos dados')
            ax_plot.plot(x_plot, y_newton, 'b-', label='Polinomio de Newton')
            ax_plot.set_title("Interpolación de Newton")
            ax_plot.set_xlabel("x")
            ax_plot.set_ylabel("y")
            ax_plot.legend()
            ax_plot.grid()

            rows = [
                ["Tiempo (s)", f"{tiempo_ejecucion:.6f}"],
                ["Grado", f"{len(x)-1}"],
                ["Polinomio (trunc)", polinomio[:120] + ("..." if len(polinomio)>120 else "")]
            ]
            col_labels = ["Propiedad","Valor"]
            ax_table.axis('off')
            table = ax_table.table(cellText=rows, colLabels=col_labels, loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1,2)

            # polinomio completo en figura adicional
            fig_pol = plt.figure(figsize=(10,2))
            plt.axis('off')
            plt.text(0.01, 0.5, polinomio, va='center', family='monospace', fontsize=9)
            plt.tight_layout()
            plt.show()
        except Exception:
            pass

    # Comparación automática: llamar a los otros métodos del paquete supCp3
    if auto_compare:
        try:
            eval_pts = np.linspace(min(x), max(x), max(100, int(eval_grid)))
            # referencia: Newton (método actual)
            y_ref = np.array([P_newton(xi) for xi in eval_pts])

            other_results = {}
            try:
                other_results['Lagrange'] = SUBinterpol_lagrange.interpol_lagrange(x, y)
            except Exception:
                other_results['Lagrange'] = None
            try:
                other_results['Spline_lineal'] = SUBspline_lineal.SUBSUBspline_lineal(x, y)
            except Exception:
                other_results['Spline_lineal'] = None
            try:
                other_results['Spline_cubico'] = SUBspln_cubico.SUBSUBspline_cubico(x, y)
            except Exception:
                other_results['Spline_cubico'] = None
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
                    fig, (ax_plot, ax_table) = plt.subplots(ncols=2, figsize=(12,5), gridspec_kw={'width_ratios':[3,2]})
                    ax_plot.plot(x, y, 'ro', label='Puntos dados')
                    ax_plot.plot(eval_pts, y_ref, 'k-', label='Newton (referencia)')
                    colors = {'Lagrange':'m--','Spline_lineal':'g--','Spline_cubico':'y--'}
                    for name, res in other_results.items():
                        ycmp = reeval_on_common(res)
                        if ycmp is not None:
                            ax_plot.plot(eval_pts, ycmp, colors.get(name,'--'), label=name)
                    ax_plot.set_title('Comparación respecto a Newton')
                    ax_plot.set_xlabel('x')
                    ax_plot.set_ylabel('y')
                    ax_plot.legend()
                    ax_plot.grid()

                    col_labels = ['Método','Max err','RMSE']
                    rows = []
                    for name in ['Lagrange','Spline_lineal','Spline_cubico','vandermonde']:
                        m = metrics.get(name, {})
                        rows.append([name, f"{m['max_err']:.6g}" if m['max_err'] is not None else 'N/A', f"{m['rmse']:.6g}" if m['rmse'] is not None else 'N/A'])
                    ax_table.axis('off')
                    table = ax_table.table(cellText=rows, colLabels=col_labels, loc='center')
                    table.auto_set_font_size(False)
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
        SPCC = SUBspln_cubico.SUBSUBspline_cubico(x, y)
        SPL = SUBspline_lineal.SUBSUBspline_lineal(x, y)
        VAN = Subvandermonde.interpol_vandermonde(x, y)

        plt.figure(figsize=(10, 6))
        plt.plot(x, y, 'ro', label='Puntos dados')
        plt.plot(x_plot, y_newton, 'b-', label='Newton')
        plt.plot(SPL[0], SPL[1], 'g--', label='Spline lineal')
        plt.plot(ILG[0], ILG[1], 'm--', label='Lagrange')
        plt.plot(VAN[0], VAN[1], 'k-', label='Vandermonde')
        plt.plot(SPCC[0], SPCC[1], 'y--', label='Spline Cúbico')
        plt.title("Comparación General")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid()
        plt.show(block=False)
        resultado += "\nComparación general mostrada en la gráfica."

if __name__ == "__main__":
    print(interpolacion_newton())