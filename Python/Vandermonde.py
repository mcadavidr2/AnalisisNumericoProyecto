import numpy as np
import time
import matplotlib.pyplot as plt
from Python.supCp3 import SUBinterpol_lagrange
from Python.supCp3 import SUBinterpol_newton
from Python.supCp3 import SUBspln_cubico
from Python.supCp3 import SUBspline_lineal
#from Python.supCp3 import Subvandermonde


def interpolacion_vandermonde(ValoresX=None, ValoresY=None, show_report=True,  auto_compare=True):
    # Validación de entrada
    x = np.array(ValoresX)
    y = np.array(ValoresY)
    if x.ndim == 0 or len(x) < 2:
        raise ValueError("Debe ingresar al menos dos puntos para interpolar.")
    if not np.all(np.diff(x) > 0):
        raise ValueError("Los valores de x deben estar en orden creciente")
    if len(x) != len(y):
        raise ValueError("Las listas de x e y deben tener la misma longitud.")

    # --- Análisis de tiempo ---
    start_time = time.perf_counter()
    V = np.vander(x, increasing=True)
    cond_num = np.linalg.cond(V)
    a = np.linalg.solve(V, y)
    end_time = time.perf_counter()
    tiempo_ejecucion = end_time - start_time

    # --- Gráfica ---
    x_plot = np.linspace(min(x), max(x), 100)
    y_vander = np.polyval(a[::-1], x_plot)  # polyval espera orden descendente

    # Representación legible del polinomio (numpy.poly1d espera coef orden descendente)
    poly = np.poly1d(a[::-1])
    poly_str = str(poly)

    # Prepara resultados para la GUI (sin coeficiente de referencia)
    resultado = (
        f"Puntos ingresados: {list(zip(x, y))}\n"
        f"Coeficientes del polinomio (Vandermonde) [orden ascendente de potencias]: {a}\n"
        f"Polinomio interpolador p(x): {poly_str}\n"
        f"Tiempo de ejecución: {tiempo_ejecucion:.6f} segundos\n"
        f"Número de condición de la matriz: {cond_num:.2e} (valores altos indican inestabilidad)\n"
    )

    # Devolvemos la cadena para compatibilidad plus un dict con datos estructurados
    info = {
        "tiempo": tiempo_ejecucion,
        "coeficientes": a.tolist(),
        "polinomio_str": poly_str,
        "polinomio_obj": poly,
        "condicion": cond_num,
        "n_tramos": None,
        "polinomios_por_tramo": None,
    }

    # Mostrar informe y gráfica en la misma ventana (dos paneles) si se solicita
    if show_report:
        try:
            fig, (ax_plot, ax_table) = plt.subplots(ncols=2, figsize=(12, 6), gridspec_kw={'width_ratios': [3, 2]})

            # Plot en el panel izquierdo
            ax_plot.plot(x, y, 'ro', label='Puntos dados')
            ax_plot.plot(x_plot, y_vander, 'b-', label='Vandermonde')
            ax_plot.set_title("Interpolación con Vandermonde")
            ax_plot.set_xlabel("x")
            ax_plot.set_ylabel("y")
            ax_plot.legend()
            ax_plot.grid()

            # Tabla de resumen en el panel derecho
            rows = [
                ["Tiempo (s)", f"{tiempo_ejecucion:.6f}"],
                ["Número de condición", f"{cond_num:.3e}"],
                ["Coeficientes (orden asc.)", ", ".join([f"{c:.6g}" for c in a])],
            ]
            col_labels = ["Propiedad", "Valor"]
            ax_table.axis('off')
            table = ax_table.table(cellText=rows, colLabels=col_labels, loc='center', cellLoc='left')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2)

            # Mostrar el polinomio completo abajo de la figura si cabe (como texto)
            plt.tight_layout()
            # Añadir una figura de texto pequeña debajo usando su propia figura para evitar recorte si es muy largo
            fig_pol = plt.figure(figsize=(8, 1.5))
            plt.axis('off')
            plt.text(0.01, 0.5, f"Polinomio p(x): {poly_str}", va='center', family='monospace', fontsize=9)
            plt.tight_layout()
            plt.show()
        except Exception:
            pass

    return resultado, info

def comparar_metodos(ValoresX, ValoresY, show_report=True):
    x = np.array(ValoresX)
    y = np.array(ValoresY)
    x_plot = np.linspace(min(x), max(x), max(100, int(500)))

    # Vandermonde
    t0 = time.perf_counter()
    V = np.vander(x, increasing=True)
    cond_num = np.linalg.cond(V)
    a = np.linalg.solve(V, y)
    t1 = time.perf_counter()
    t_vander = t1 - t0
    y_vander = np.polyval(a[::-1], x_plot)

    # Otros métodos con medición de tiempo
    t0 = time.perf_counter()
    ILG = SUBinterpol_lagrange.interpol_lagrange(x, y)
    t1 = time.perf_counter()
    t_lagrange = t1 - t0

    t0 = time.perf_counter()
    INT = SUBinterpol_newton.interpol_newton(x, y)
    t1 = time.perf_counter()
    t_newton = t1 - t0

    t0 = time.perf_counter()
    SPCC = SUBspln_cubico.SUBSUBspline_cubico(x, y)
    t1 = time.perf_counter()
    t_spline_cub = t1 - t0

    t0 = time.perf_counter()
    SPL = SUBspline_lineal.SUBSUBspline_lineal(x, y)
    t1 = time.perf_counter()
    t_spline_lin = t1 - t0

    # Interpolar/reevaluar todas las soluciones en la malla común x_plot
    def reeval_on_common(res):
        # res esperado: [x_plot_method, y_plot_method, ...]
        rx = np.array(res[0])
        ry = np.array(res[1])
        return np.interp(x_plot, rx, ry)

    y_ilg = reeval_on_common(ILG)
    y_int = reeval_on_common(INT)
    y_spcc = reeval_on_common(SPCC)
    y_spl = reeval_on_common(SPL)

    # Métricas de error frente a Vandermonde
    def metrics(y_ref, y_cmp):
        diff = y_ref - y_cmp
        maxerr = np.max(np.abs(diff))
        rmse = np.sqrt(np.mean(diff ** 2))
        return maxerr, rmse

    m_lagrange = metrics(y_vander, y_ilg)
    m_newton = metrics(y_vander, y_int)
    m_splc = metrics(y_vander, y_spcc)
    m_spll = metrics(y_vander, y_spl)

    # Gráfica comparativa
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'ro', label='Puntos dados')
    plt.plot(x_plot, y_vander, 'b-', label='Vandermonde')
    plt.plot(x_plot, y_spl, 'g--', label='Spline lineal')
    plt.plot(x_plot, y_ilg, 'm--', label='Lagrange')
    plt.plot(x_plot, y_int, 'c--', label='Newton')
    plt.plot(x_plot, y_spcc, 'y--', label='Spline Cúbico')
    plt.title("Comparación General")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid()
    plt.show()

    # Preparar informe en memoria (sin escribir archivos)
    informe = {
        "Vandermonde": {"tiempo": t_vander, "condicion": cond_num, "coeficientes": a.tolist()},
        "Lagrange": {"tiempo": t_lagrange, "max_err_vs_vander": float(m_lagrange[0]), "rmse_vs_vander": float(m_lagrange[1]), "polinomio": ILG[2] if len(ILG) > 2 else None},
        "Newton": {"tiempo": t_newton, "max_err_vs_vander": float(m_newton[0]), "rmse_vs_vander": float(m_newton[1]), "polinomio": INT[2] if len(INT) > 2 else None},
        "Spline_lineal": {"tiempo": t_spline_lin, "max_err_vs_vander": float(m_spll[0]), "rmse_vs_vander": float(m_spll[1])},
        "Spline_cubico": {"tiempo": t_spline_cub, "max_err_vs_vander": float(m_splc[0]), "rmse_vs_vander": float(m_splc[1])},
    }

    # Mostrar informe en una ventana separada (figura matplotlib) si se solicita
    if show_report:
        try:
            # Crear una figura con dos columnas: izquierda gráfico, derecha tabla
            fig, (ax_plot, ax_table) = plt.subplots(ncols=2, figsize=(12, 5), gridspec_kw={'width_ratios': [3, 2]})

            # Plotear en ax_plot
            ax_plot.plot(x, y, 'ro', label='Puntos dados')
            ax_plot.plot(x_plot, y_vander, 'b-', label='Vandermonde')
            ax_plot.plot(x_plot, y_spl, 'g--', label='Spline lineal')
            ax_plot.plot(x_plot, y_ilg, 'm--', label='Lagrange')
            ax_plot.plot(x_plot, y_int, 'c--', label='Newton')
            ax_plot.plot(x_plot, y_spcc, 'y--', label='Spline Cúbico')
            ax_plot.set_title("Comparación General")
            ax_plot.set_xlabel("x")
            ax_plot.set_ylabel("y")
            ax_plot.legend()
            ax_plot.grid()

            # Preparar tabla con métricas
            col_labels = ["Método", "Tiempo (s)", "Max err", "RMSE", "Notas"]
            rows = []
            v = informe['Vandermonde']
            rows.append(["Vandermonde", f"{v['tiempo']:.6f}", "0", "0", f"cond={float(v['condicion']):.3e}"])
            L = informe['Lagrange']
            rows.append(["Lagrange", f"{L['tiempo']:.6f}", f"{L['max_err_vs_vander']:.6g}", f"{L['rmse_vs_vander']:.6g}", "polinomio"])
            N = informe['Newton']
            rows.append(["Newton", f"{N['tiempo']:.6f}", f"{N['max_err_vs_vander']:.6g}", f"{N['rmse_vs_vander']:.6g}", "polinomio"])
            SL = informe['Spline_lineal']
            rows.append(["Spline lineal", f"{SL['tiempo']:.6f}", f"{SL['max_err_vs_vander']:.6g}", f"{SL['rmse_vs_vander']:.6g}", "por tramos"])
            SC = informe['Spline_cubico']
            rows.append(["Spline cúbico", f"{SC['tiempo']:.6f}", f"{SC['max_err_vs_vander']:.6g}", f"{SC['rmse_vs_vander']:.6g}", "por tramos"])

            ax_table.axis('off')
            table = ax_table.table(cellText=rows, colLabels=col_labels, loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2)
            plt.tight_layout()
            plt.show()
        except Exception:
            pass

    return informe
