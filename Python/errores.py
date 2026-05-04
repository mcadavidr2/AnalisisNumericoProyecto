"""
errores.py
----------
Cálculo unificado de los 4 tipos de error usados en el Capítulo 1
(métodos para raíces de ecuaciones no lineales):

    E_abs   = |x_n - x_{n-1}|
    E_rel   = |(x_n - x_{n-1}) / x_n|         (relativo respecto a x_n)
    E_rel2  = |(x_n - x_{n-1}) / x_{n-1}|     (relativo respecto a x_{n-1})
    E_rond  = |f(x_n)|                        (residual / "error de redondeo")

Diseño:
    Los métodos del Cap 1 guardan en su tabla la sucesión x_n y, según el
    método, también la sucesión f(x_n) o g(x_n). Este módulo POST-PROCESA
    esas tablas para calcular los 4 errores sin tener que volver a correr
    el método. Esto permite graficar los 4 tipos en un solo plot.

Para Punto Fijo, donde no hay una "f" sino una g, el residual se redefine
como |g(x_n) - x_n| (qué tan lejos está x_n de ser un punto fijo).

Las primeras posiciones donde no se puede calcular un error porque no hay
iteración previa se rellenan con np.nan; matplotlib los ignora al graficar.
"""

import numpy as np


# Mapeo central: nombre del método -> (columna_de_x_n, columna_de_residual)
# residual_col = None significa "calcular como |g(x) - x|" (caso Punto Fijo)
COLUMN_MAP = {
    "Bisección":        (3, 4),
    "Regla Falsa":      (5, 6),
    "Newton":           (1, 2),
    "Secante":          (2, 3),
    "Punto Fijo":       (1, None),
    "Raíces Múltiples": (1, 2),
}

# Etiquetas legibles para leyendas y combobox
ERROR_LABELS = {
    "abs":  "E_abs = |xₙ − xₙ₋₁|",
    "rel":  "E_rel = |(xₙ − xₙ₋₁)/xₙ|",
    "rel2": "E_rel₂ = |(xₙ − xₙ₋₁)/xₙ₋₁|",
    "rond": "E_rond = |f(xₙ)|",
}


def compute_errors(x_seq, residual_seq=None):
    """
    Calcula los 4 tipos de error iteración a iteración.

    Parámetros
    ----------
    x_seq : iterable de floats
        Sucesión x_0, x_1, ..., x_N (longitud N+1).
    residual_seq : iterable de floats, opcional
        Sucesión f(x_0), ..., f(x_N) (o |g(x)-x| para punto fijo).
        Si es None, E_rond se rellena con NaN.

    Devuelve
    --------
    dict con cuatro arrays de la misma longitud que x_seq:
        {"abs": ..., "rel": ..., "rel2": ..., "rond": ...}

    La posición 0 de los tres primeros es NaN porque no existe x_{-1}.
    """
    x = np.asarray(list(x_seq), dtype=float)
    n = len(x)

    abs_e  = np.full(n, np.nan)
    rel_e  = np.full(n, np.nan)
    rel2_e = np.full(n, np.nan)

    for i in range(1, n):
        xi   = x[i]
        xim1 = x[i - 1]
        # Si una iteración produjo inf/nan, dejamos NaN y seguimos
        if not (np.isfinite(xi) and np.isfinite(xim1)):
            continue
        diff = abs(xi - xim1)
        abs_e[i] = diff
        if xi != 0.0:
            rel_e[i] = diff / abs(xi)
        if xim1 != 0.0:
            rel2_e[i] = diff / abs(xim1)

    if residual_seq is not None:
        r = np.asarray(list(residual_seq), dtype=float)
        if len(r) != n:
            # Tamaños distintos: recortamos sin lanzar error
            m = min(len(r), n)
            rond_e = np.full(n, np.nan)
            rond_e[:m] = np.abs(r[:m])
        else:
            rond_e = np.abs(r)
    else:
        rond_e = np.full(n, np.nan)

    return {
        "abs":  abs_e,
        "rel":  rel_e,
        "rel2": rel2_e,
        "rond": rond_e,
    }


def extract_sequences(method_name, table):
    """
    A partir del nombre del método y su tabla, extrae las dos sucesiones
    (x_n y residual) ya como listas de floats listas para compute_errors.

    Para Punto Fijo, el residual se calcula como |g(x_n) - x_n|.

    Lanza KeyError si el método no está en COLUMN_MAP, o ValueError si
    la tabla está vacía. No falla por celdas individuales con basura:
    esas celdas se reemplazan por NaN.
    """
    if method_name not in COLUMN_MAP:
        raise KeyError(
            f"Método '{method_name}' no está registrado en COLUMN_MAP."
        )
    if not isinstance(table, list) or len(table) == 0:
        raise ValueError("La tabla está vacía o no es una lista.")

    x_col, res_col = COLUMN_MAP[method_name]

    x_seq = []
    for row in table:
        try:
            x_seq.append(float(row[x_col]))
        except (TypeError, ValueError, IndexError):
            x_seq.append(float("nan"))

    if method_name == "Punto Fijo":
        # Fila de Punto Fijo: [iter, x_n, g(x_n), error]
        residual_seq = []
        for row in table:
            try:
                xn  = float(row[1])
                gxn = float(row[2])
                residual_seq.append(abs(gxn - xn))
            except (TypeError, ValueError, IndexError):
                residual_seq.append(float("nan"))
    elif res_col is None:
        residual_seq = None
    else:
        residual_seq = []
        for row in table:
            try:
                residual_seq.append(float(row[res_col]))
            except (TypeError, ValueError, IndexError):
                residual_seq.append(float("nan"))

    return x_seq, residual_seq


def select_error(errors_dict, error_type):
    """
    Devuelve el array correspondiente al error_type pedido.
    Si el tipo es desconocido, devuelve 'abs' como fallback seguro.
    """
    return errors_dict.get(error_type, errors_dict["abs"])