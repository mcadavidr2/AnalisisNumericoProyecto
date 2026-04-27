import numpy as np
import time
import matplotlib.pyplot as plt
#from Python.supCp3 import SUBinterpol_lagrange
from Python.supCp3 import SUBinterpol_newton
from Python.supCp3 import SUBspln_cubico
from Python.supCp3 import SUBspline_lineal
from Python.supCp3 import Subvandermonde

def interpolacion_lagrange(ValoresX=None, ValoresY=None, show_report=True, auto_compare=True):
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

    # --- Construcción del polinomio de Lagrange ---
    start_time = time.time()

    # Polinomios base
    def L(i, x_eval):
        result = 1.0
        for j in range(len(x)):
            if j != i:
                result *= (x_eval - x[j]) / (x[i] - x[j])
        return result

    # Polinomio interpolador
    def P_lagrange(x_eval):
        return sum(y[i] * L(i, x_eval) for i in range(len(x)))

    # Formato de polinomio como string
    polinomio = "P(x) = "
    for i in range(len(x)):
        term = f"{y[i]:.4f} * L_{i}(x)"
        if i > 0:
            polinomio += " + " + term
        else:
            polinomio += term

    # Polinomios base como string
    polinomios_base = []
    for i in range(len(x)):
        L_i = f"L_{i}(x) = "
        factores = []
        for j in range(len(x)):
            if j != i:
                factores.append(f"(x - {x[j]:.4f}) / ({x[i]:.4f} - {x[j]:.4f})")
        L_i += " * ".join(factores)
        polinomios_base.append(L_i)

    end_time = time.time()
    tiempo_ejecucion = end_time - start_time

    # --- Evaluación para gráfica ---
    x_plot = np.linspace(min(x), max(x), 100)
    y_lagrange = [P_lagrange(xi) for xi in x_plot]

    resultado = (
        f"Puntos ingresados: {list(zip(ValoresX,ValoresY))}\n"
        f"Polinomios base de Lagrange:\n\n" + "\n".join(polinomios_base) + "\n\n"
        f"Polinomio interpolador de Lagrange:\n{polinomio}\n\n"
        f"Tiempo de ejecución: {tiempo_ejecucion:.6f} segundos"
    )

    info = {
        "tiempo": tiempo_ejecucion,
        "tiempo": tiempo_ejecucion,
        "coeficientes": None,
        "polinomio_str": polinomio,
        "polinomio_obj": None,
        "polinomios_base": polinomios_base,
        "polinomios_por_tramo": None,
        "n_tramos": None,
        "condicion": None,
    }

    # Construir un objeto polinomio (coeficientes estándar) para facilitar la inspección
    try:
        coeffs = np.polyfit(x, y, len(x) - 1)
        poly_obj = np.poly1d(coeffs)
        info['polinomio_obj'] = poly_obj
        info['coeficientes'] = coeffs.tolist()
    except Exception:
        pass

    if show_report:
        try:
            fig, (ax_plot, ax_table) = plt.subplots(ncols=2, figsize=(12,6), gridspec_kw={'width_ratios':[3,2]})
            # plot
            ax_plot.plot(x, y, 'ro', label='Puntos dados')
            ax_plot.plot(x_plot, y_lagrange, 'b-', label='Polinomio de Lagrange')
            ax_plot.set_title("Interpolación de Lagrange")
            ax_plot.set_xlabel("x")
            ax_plot.set_ylabel("y")
            ax_plot.legend()
            ax_plot.grid()

            # tabla resumen
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

            # polinomios base completos en figura adicional
            fig_pol = plt.figure(figsize=(10, 2+0.5*len(polinomios_base)))
            plt.axis('off')
            text = "\n".join(polinomios_base)
            plt.text(0.01, 0.99, text, va='top', family='monospace', fontsize=9)
            plt.tight_layout()
            plt.show()
        except Exception:
            pass

    # Comparación automática: llamar a los otros métodos del paquete supCp3
    if auto_compare:
        try:
            # Malla común para evaluación
            eval_pts = np.linspace(min(x), max(x), max(100, int(eval_grid)))

            # Valor de referencia: método actual (Lagrange) evaluado en eval_pts
            y_ref = np.array([P_lagrange(xi) for xi in eval_pts])

            # Llamadas a otros métodos
            other_results = {}
            try:
                other_results['Newton'] = SUBinterpol_newton.interpol_newton(x, y)
            except Exception:
                other_results['Newton'] = None
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

            # Helper para re-evaluar en la malla común
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
                    maxerr = float(np.max(np.abs(diff)))
                    rmse = float(np.sqrt(np.mean(diff**2)))
                    metrics[name] = {'max_err': maxerr, 'rmse': rmse}

            # Mostrar comparación: plot y_ref y demás + tabla de errores
            if show_report:
                try:
                    fig, (ax_plot, ax_table) = plt.subplots(ncols=2, figsize=(12,5), gridspec_kw={'width_ratios':[3,2]})
                    ax_plot.plot(x, y, 'ro', label='Puntos dados')
                    ax_plot.plot(eval_pts, y_ref, 'k-', label='Lagrange (referencia)')
                    colors = {'Newton':'c--','Spline_lineal':'g--','Spline_cubico':'y--'}
                    for name, res in other_results.items():
                        ycmp = reeval_on_common(res)
                        if ycmp is not None:
                            ax_plot.plot(eval_pts, ycmp, colors.get(name,'--'), label=name)
                    ax_plot.set_title('Comparación respecto a Lagrange')
                    ax_plot.set_xlabel('x')
                    ax_plot.set_ylabel('y')
                    ax_plot.legend()
                    ax_plot.grid()

                    # Tabla
                    col_labels = ['Método','Max err','RMSE']
                    rows = []
                    for name in ['Newton','Spline_lineal','Spline_cubico','Vandermonde']:
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
        SPCC = SUBspln_cubico.SUBSUBspline_cubico(x, y)
        SPL = SUBspline_lineal.SUBSUBspline_lineal(x, y)
        VAN = Subvandermonde.interpol_vandermonde(x, y)
        INT = SUBinterpol_newton.interpol_newton(x, y)

        plt.figure(figsize=(10, 6))
        plt.plot(x, y, 'ro', label='Puntos dados')
        plt.plot(x_plot, y_lagrange, 'b-', label='Lagrange')
        plt.plot(VAN[0], VAN[1], 'k-', label='Vandermonde')
        plt.plot(SPL[0], SPL[1], 'g--', label='Spline lineal')
        plt.plot(INT[0], INT[1], 'c--', label='Newton')
        plt.plot(SPCC[0], SPCC[1], 'y--', label='Spline Cúbico')
        plt.title("Comparación General")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid()
        plt.show(block=False)
        resultado += "\nComparación general mostrada en la gráfica."

    

# Ejecutar solo si es script principal
if __name__ == "__main__":
    print(interpolacion_lagrange())