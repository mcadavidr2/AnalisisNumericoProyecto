import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
from tabulate import tabulate
import importlib
import inspect
import pandas as pd
import math

from Python.Vandermonde import comparar_metodos

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Diccionario de métodos con información completa
METHODS = {
    "Bisección": {
        "module": ("biseccion", "biseccion"),
        # Descripción corta, general
        "description": (
            "El método de bisección es un procedimiento numérico para encontrar una raíz real de "
            "una ecuación f(x) = 0 dentro de un intervalo [a, b] donde la función cambia de signo."
        ),
        # Para qué sirve
        "purpose": (
            "Se utiliza cuando se conoce un intervalo [a, b] tal que f(a) y f(b) tienen signos opuestos, "
            "lo que indica (bajo ciertas condiciones) la existencia de al menos una raíz en ese intervalo. "
            "Es un método muy seguro y sencillo, ideal para iniciar el estudio de métodos de búsqueda de raíces."
        ),
        # Cómo funciona
        "how_it_works": (
            "1. Se parte de un intervalo inicial [a, b] con f(a) y f(b) de signos opuestos.\n"
            "2. Se calcula el punto medio pm = (a + b) / 2.\n"
            "3. Se evalúa f(pm):\n"
            "   • Si f(a)·f(pm) < 0, la raíz está en [a, pm], entonces b = pm.\n"
            "   • Si f(pm)·f(b) < 0, la raíz está en [pm, b], entonces a = pm.\n"
            "4. Se repite el proceso reduciendo el tamaño del intervalo.\n"
            "5. El método se detiene cuando el error es menor que la tolerancia establecida o se alcanza "
            "el máximo número de iteraciones."
        ),
        # Datos requeridos (para mostrar en la interfaz)
        "required_inputs": [
            "Función f(x) continua en el intervalo [a, b] (por ejemplo: x**3 - x - 1)",
            "Límite inferior (a), donde f(a) y f(b) tengan signos opuestos",
            "Límite superior (b)",
            "Tolerancia (Tol), por ejemplo 1e-3 o 0.001",
            "Máximo iteraciones (it), por ejemplo 50 o 100"
        ],
        # Qué verá el usuario en tu GUI
        "ui_info": (
            "En la interfaz podrás ver:\n"
            "• Una tabla de iteraciones con los valores de a, b, el punto medio pm, f(pm) y el error.\n"
            "• La aproximación final de la raíz y el número total de iteraciones utilizadas.\n"
            "• Si utilizas la opción de graficar, verás la curva de f(x) y los puntos de aproximación "
            "mostrados sobre la gráfica, lo que ayuda a visualizar cómo el método se acerca a la raíz."
        ),
        # Ejemplo concreto
        "example": (
            "Ejemplo de uso:\n"
            "• f(x) = x**3 - x - 1\n"
            "• a = 1\n"
            "• b = 2\n"
            "• Tol = 0.01\n"
            "• it = 100\n\n"
            "En este caso, f(1) y f(2) tienen signos opuestos, por lo que el método de bisección buscará "
            "una raíz de f(x) = 0 dentro del intervalo [1, 2]."
        )
    },

    "Regla Falsa": {
        "module": ("regla_falsa", "false_position_method"),
        # Descripción corta, general
        "description": (
            "El método de regla falsa (o posición falsa) es un procedimiento numérico para encontrar una raíz real "
            "de la ecuación f(x) = 0 en un intervalo [a, b] donde la función cambia de signo, usando una aproximación "
            "lineal (una recta secante) entre los extremos del intervalo."
        ),
        # Para qué sirve
        "purpose": (
            "Se utiliza para localizar raíces reales cuando se conoce un intervalo [a, b] tal que f(a) y f(b) tienen "
            "signos opuestos. Suele converger más rápido que el método de bisección en muchos casos, manteniendo la "
            "seguridad de trabajar siempre en un intervalo donde existe una raíz."
        ),
        # Cómo funciona
        "how_it_works": (
            "1. Se parte de un intervalo inicial [a, b] con f(a) y f(b) de signos opuestos.\n"
            "2. Se traza la recta que une los puntos (a, f(a)) y (b, f(b)) y se calcula el punto de corte con el eje x:\n"
            "      xr = b - f(b) * (b - a) / (f(b) - f(a)).\n"
            "3. Se evalúa f(xr):\n"
            "   • Si f(a)·f(xr) < 0, la raíz está en [a, xr], entonces b = xr.\n"
            "   • Si f(xr)·f(b) < 0, la raíz está en [xr, b], entonces a = xr.\n"
            "4. Se repite el proceso actualizando siempre el extremo que conserva el cambio de signo.\n"
            "5. Se detiene cuando el error es menor que la tolerancia establecida o se alcanza "
            "el máximo número de iteraciones."
        ),
        # Datos requeridos
        "required_inputs": [
            "Función f(x) continua en el intervalo [a, b] (por ejemplo: x**2 - 2)",
            "Límite inferior (a), donde f(a) y f(b) tienen signos opuestos",
            "Límite superior (b)",
            "Tolerancia (Tol), por ejemplo 1e-3 o 0.001",
            "Máximo iteraciones (it), por ejemplo 50 o 100"
        ],
        # Qué verá el usuario en la GUI
        "ui_info": (
            "En la interfaz podrás ver:\n"
            "• Una tabla de iteraciones con los valores de a, b, xr (aproximación por regla falsa), f(xr) y el error.\n"
            "• La aproximación final de la raíz y el número total de iteraciones utilizadas.\n"
            "• Si utilizas la opción de graficar, verás la curva de f(x) y las aproximaciones xr sobre la gráfica, "
            "lo que permite visualizar cómo el método se acerca a la raíz usando rectas secantes al intervalo."
        ),
        # Ejemplo
        "example": (
            "Ejemplo de uso:\n"
            "• f(x) = x**2 - 2\n"
            "• a = 0\n"
            "• b = 2\n"
            "• Tol = 0.001\n"
            "• it = 50\n\n"
            "En este caso, f(0) y f(2) tienen signos opuestos, por lo que la regla falsa buscará una raíz de "
            "f(x) = 0 dentro del intervalo [0, 2], aproximando la raíz de √2."
        )
    },

    "Newton": {
        "module": ("newton", "newton_method"),
        # Descripción corta, general
        "description": (
            "El método de Newton (o Newton-Raphson) es un procedimiento iterativo para encontrar raíces de la ecuación "
            "f(x) = 0 utilizando información de la derivada de la función. "
            "En ESTA APLICACIÓN la derivada f'(x) se calcula automáticamente de forma numérica a partir de la f(x) "
            "que escribas, por lo que NO necesitas ingresar la derivada de manera explícita."
        ),
        # Para qué sirve
        "purpose": (
            "Se utiliza para obtener aproximaciones rápidas de raíces reales cuando se dispone de una buena estimación "
            "inicial y la función es derivable cerca de la raíz. Es uno de los métodos más rápidos en converger, "
            "aunque requiere más cuidado en la elección del valor inicial. "
            "En esta implementación basta con que escribas f(x); el programa se encarga de aproximar f'(x)."
        ),
        # Cómo funciona
        "how_it_works": (
            "1. Se elige un valor inicial x0 cercano a la raíz que se desea encontrar.\n"
            "2. En cada iteración se calcula la siguiente aproximación usando la fórmula:\n"
            "      x_{n+1} = x_n - f(x_n) / f'(x_n).\n"
            "3. Geométricamente, se toma la recta tangente a la curva y = f(x) en el punto (x_n, f(x_n)) y se busca "
            "el punto donde esa recta corta al eje x.\n"
            "4. El proceso se repite hasta que el cambio entre dos aproximaciones consecutivas sea menor que la "
            "tolerancia o se alcance el máximo número de iteraciones.\n"
            "5. Si la derivada f'(x_n) es muy pequeña o cero, el método puede fallar o volverse inestable.\n\n"
            "En esta aplicación, f'(x_n) NO se pide al usuario: se aproxima numéricamente a partir de la función "
            "f(x) que se ingresa en la caja de texto. De esta forma, no necesitas calcular derivadas a mano."
        ),
        # Datos requeridos
        "required_inputs": [
            "Función f(x) derivable cerca de la raíz (por ejemplo: x**2 - 2)",
            "Valor inicial (x0) cercano a la raíz buscada",
            "Tolerancia (Tol), por ejemplo 1e-6 o 1e-4",
            "Máximo iteraciones (it), por ejemplo 50 o 100"
        ],
        # Qué verá el usuario en la GUI
        "ui_info": (
            "En la interfaz podrás ver:\n"
            "• Una tabla de iteraciones con los valores de x_n, f(x_n), una aproximación numérica de f'(x_n) y el "
            "error en cada paso.\n"
            "• La aproximación final de la raíz obtenida, junto con el error final y el número de iteraciones usadas.\n"
            "• Si utilizas la opción de graficar, verás la función f(x) y los puntos x_n sobre la curva, lo que permite "
            "observar cómo el método se mueve siguiendo las tangentes hacia la raíz.\n\n"
            "IMPORTANTE: la derivada se calcula automáticamente; solo debes escribir f(x)."
        ),
        # Ejemplo
        "example": (
            "Ejemplo de uso:\n"
            "• f(x) = x**2 - 2\n"
            "• x0 = 1.5\n"
            "• Tol = 1e-5\n"
            "• it = 50\n\n"
            "En este caso, el método de Newton encontrará una aproximación de la raíz de f(x) = 0, que corresponde a √2, "
            "partiendo de la estimación inicial x0 = 1.5. La derivada f'(x) se aproxima automáticamente."
        )
    },

    "Secante": {
        "module": ("secante", "secante"),
        # Descripción corta, general
        "description": (
            "El método de la secante es un procedimiento iterativo para encontrar raíces de f(x) = 0 usando una "
            "aproximación de la derivada a partir de dos puntos, evitando calcular f'(x) de forma explícita."
        ),
        # Para qué sirve
        "purpose": (
            "Se utiliza cuando se desea la rapidez de un método tipo Newton pero no se quiere o no se puede calcular "
            "la derivada de la función. Solo requiere evaluar f(x) en cada iteración y partir de dos valores iniciales."
        ),
        # Cómo funciona
        "how_it_works": (
            "1. Se eligen dos valores iniciales x0 y x1, preferiblemente cercanos a la raíz buscada.\n"
            "2. Se aproxima la derivada con la pendiente de la recta secante entre (x0, f(x0)) y (x1, f(x1)):\n"
            "      f'(x) ≈ [f(x1) - f(x0)] / (x1 - x0).\n"
            "3. Se calcula la siguiente aproximación:\n"
            "      x_{n+1} = x_n - f(x_n) * (x_n - x_{n-1}) / (f(x_n) - f(x_{n-1})).\n"
            "4. Se repite el proceso, usando siempre los dos últimos valores para construir la nueva secante.\n"
            "5. El método se detiene cuando el error entre dos aproximaciones consecutivas es menor que la tolerancia "
            "o se alcanza el máximo número de iteraciones."
        ),
        # Datos requeridos
        "required_inputs": [
            "Función f(x) (por ejemplo: x**3 - 2)",
            "Primer valor inicial (x0), cercano a la raíz",
            "Segundo valor inicial (x1), cercano a la raíz y distinto de x0",
            "Tolerancia (Tol), por ejemplo 1e-5 o 1e-4",
            "Máximo iteraciones (it), por ejemplo 50 o 100"
        ],
        # Qué verá el usuario en la GUI
        "ui_info": (
            "En la interfaz podrás ver:\n"
            "• Una tabla de iteraciones con los valores de x_{n-1}, x_n, f(x_n) y el error.\n"
            "• La aproximación final de la raíz y el número de iteraciones necesarias.\n"
            "• Si utilizas la opción de graficar, verás la función f(x) y las aproximaciones sucesivas de la secante, "
            "lo que muestra cómo las rectas secantes van acercándose a la raíz."
        ),
        # Ejemplo
        "example": (
            "Ejemplo de uso:\n"
            "• f(x) = x**3 - 2\n"
            "• x0 = 1\n"
            "• x1 = 2\n"
            "• Tol = 1e-5\n"
            "• it = 50\n\n"
            "En este caso, el método de la secante buscará una raíz de f(x) = 0 (es decir, la raíz cúbica de 2) "
            "usando como puntos iniciales x0 = 1 y x1 = 2."
        )
    },

    "Punto Fijo": {
        "module": ("puntofijo", "fixed_point_iteration"),
        # Descripción corta, general
        "description": (
            "El método de punto fijo es un procedimiento iterativo que busca un valor x tal que x = g(x), "
            "reformulando el problema original f(x) = 0 en términos de una función g(x)."
        ),
        # Para qué sirve
        "purpose": (
            "Se utiliza para encontrar raíces de una ecuación f(x) = 0 reescribiéndola como x = g(x). "
            "Si se elige una función g(x) adecuada que cumpla ciertas condiciones, la iteración converge al punto fijo, "
            "que corresponde a la raíz buscada."
        ),
        # Cómo funciona
        "how_it_works": (
            "1. Se reescribe el problema f(x) = 0 en la forma x = g(x).\n"
            "2. Se elige un valor inicial x0.\n"
            "3. En cada iteración se calcula:\n"
            "      x_{n+1} = g(x_n).\n"
            "4. Se repite el proceso hasta que la diferencia |x_{n+1} - x_n| sea menor que la tolerancia establecida "
            "o se alcance el máximo número de iteraciones.\n"
            "5. La convergencia depende de la elección de g(x). Si |g'(x)| < 1 en una vecindad de la solución, "
            "el método suele converger."
        ),
        # Datos requeridos
        "required_inputs": [
            "Función g(x) tal que la solución de interés cumpla x = g(x) (por ejemplo: g(x) = (x + 2/x) / 2)",
            "Valor inicial (x0) cercano al punto fijo",
            "Tolerancia (Tol), por ejemplo 1e-5 o 1e-4",
            "Máximo iteraciones (it), por ejemplo 50 o 100"
        ],
        # Qué verá el usuario en la GUI
        "ui_info": (
            "En la interfaz podrás ver:\n"
            "• Una tabla de iteraciones con los valores de x_n, g(x_n) y el error entre aproximaciones.\n"
            "• El valor aproximado del punto fijo (que corresponde a la raíz) y el número de iteraciones.\n"
            "• Si utilizas la opción de graficar para métodos de raíces, podrás comparar visualmente las aproximaciones "
            "x_n sobre la función asociada."
        ),
        # Ejemplo
        "example": (
            "Ejemplo de uso:\n"
            "Supongamos que queremos aproximar la raíz positiva de la ecuación x^2 - 2 = 0.\n"
            "Podemos escribirla como x = g(x) = (x + 2/x) / 2.\n"
            "• g(x) = (x + 2/x) / 2\n"
            "• x0 = 1\n"
            "• Tol = 1e-5\n"
            "• it = 50\n\n"
            "Aplicando el método de punto fijo, la sucesión x_{n+1} = g(x_n) se aproxima a √2."
        )
    },

    "Raíces Múltiples": {
        "module": ("raices_m", "multiple_roots"),
        # Descripción corta, general
        "description": (
            "El método de raíces múltiples es una variante del método de Newton diseñada para encontrar raíces de "
            "multiplicidad mayor que 1, utilizando información de la primera y la segunda derivada de la función. "
            "En ESTA APLICACIÓN tanto f'(x) como f''(x) se calculan automáticamente de forma numérica a partir de f(x), "
            "por lo que NO necesitas escribir las derivadas."
        ),
        # Para qué sirve
        "purpose": (
            "Se utiliza cuando la función tiene raíces múltiples (por ejemplo, cuando f(x) = (x - r)^m con m > 1). "
            "En estos casos, el método de Newton estándar puede converger muy lentamente o comportarse mal, "
            "mientras que el método de raíces múltiples mejora la convergencia utilizando f(x), f'(x) y f''(x). "
            "En esta implementación, solo debes ingresar f(x); el programa aproxima f'(x) y f''(x) automáticamente."
        ),
        # Cómo funciona
        "how_it_works": (
            "1. Se elige un valor inicial x0 cercano a la raíz múltiple que se desea encontrar.\n"
            "2. En cada iteración se usa la fórmula (una versión modificada de Newton):\n"
            "      x_{n+1} = x_n - [f(x_n)·f'(x_n)] / ([f'(x_n)]^2 - f(x_n)·f''(x_n)).\n"
            "3. Esta fórmula tiene en cuenta la multiplicidad de la raíz al incorporar f'(x) y f''(x).\n"
            "4. Se repite el proceso hasta que el cambio entre dos aproximaciones consecutivas sea menor que la "
            "tolerancia o se alcance el máximo número de iteraciones.\n"
            "5. Es importante que la función sea suficientemente suave (derivable dos veces) en torno a la raíz.\n\n"
            "En esta aplicación NO se te pide escribir f'(x) ni f''(x): ambas derivadas se aproximan numéricamente "
            "a partir de la función f(x) que ingresas, cumpliendo el requisito de ayudar al usuario con el cálculo "
            "de derivadas."
        ),
        # Datos requeridos
        "required_inputs": [
            "Función f(x) con una raíz de multiplicidad mayor que 1 (por ejemplo: (x - 1)**2)",
            "Valor inicial (x0) cercano a la raíz múltiple",
            "Tolerancia (Tol), por ejemplo 1e-6 o 1e-4",
            "Máximo iteraciones (it), por ejemplo 50 o 100"
        ],
        # Qué verá el usuario en la GUI
        "ui_info": (
            "En la interfaz podrás ver:\n"
            "• Una tabla de iteraciones con los valores de x_n, f(x_n), una aproximación numérica de f'(x_n), "
            "otra de f''(x_n) y el error.\n"
            "• La aproximación final de la raíz múltiple y el número de iteraciones utilizadas.\n"
            "• Si utilizas la opción de graficar, podrás ver la función f(x) y cómo las aproximaciones x_n se "
            "acercan a la raíz donde la curva 'toca' el eje x con multiplicidad mayor que 1.\n\n"
            "IMPORTANTE: las derivadas f'(x) y f''(x) se calculan automáticamente; tú solo escribes f(x)."
        ),
        # Ejemplo
        "example": (
            "Ejemplo de uso:\n"
            "• f(x) = (x - 1)**2\n"
            "• x0 = 0.5\n"
            "• Tol = 1e-6\n"
            "• it = 50\n\n"
            "En este caso, la función tiene una raíz múltiple en x = 1. El método de raíces múltiples utilizará "
            "aproximaciones numéricas de f'(x) y f''(x) para aproximar eficientemente esa raíz, sin que tengas que "
            "calcular derivadas de forma manual."
        )
    },

    "Jacobi": {
        "module": ("jacobi", "jacobi"),
        # Descripción corta, general
        "description": (
            "El método iterativo de Jacobi es una técnica para resolver sistemas de ecuaciones lineales "
            "A·x = b, descomponiendo la matriz A y actualizando todas las componentes de x de forma simultánea "
            "en cada iteración."
        ),
        # Para qué sirve
        "purpose": (
            "Se utiliza para aproximar la solución de sistemas lineales de tamaño mediano o grande cuando "
            "la matriz A cumple ciertas condiciones de convergencia (por ejemplo, diagonal dominante) y se "
            "prefiere un método iterativo en lugar de factorizaciones directas como Gauss."
        ),
        # Cómo funciona
        "how_it_works": (
            "1. Se separa A en su parte diagonal D y el resto R = A - D.\n"
            "2. La ecuación A·x = b se reescribe como D·x = b - R·x.\n"
            "3. En la iteración k+1, cada componente x_i se actualiza usando solo los valores x_j^{(k)} de la iteración anterior:\n"
            "      x_i^{(k+1)} = (1 / a_ii) * (b_i - Σ_{j≠i} a_ij · x_j^{(k)}).\n"
            "4. Se repite el proceso hasta que la norma del cambio entre dos iteraciones consecutivas sea "
            "menor que la tolerancia o se alcance el máximo de iteraciones."
        ),
        # Datos requeridos
        "required_inputs": [
            "Matriz A (escrita como filas separadas por ';', por ejemplo: '4,1;2,3')",
            "Vector b (separado por comas, por ejemplo: '5,4')",
            "Vector inicial x0 (opcional, por defecto puede ser ceros; en la interfaz se escribe como '0,0')",
            "Tolerancia (Tol), por ejemplo 1e-4",
            "Máximo iteraciones (it), por ejemplo 50 o 100"
        ],
        # Qué verá el usuario en la GUI
        "ui_info": (
            "En la interfaz podrás ver:\n"
            "• Una tabla con las iteraciones del vector x^{(k)} y el error asociado a cada paso.\n"
            "• La aproximación final del vector solución x y el número de iteraciones usadas.\n"
            "• Mensajes de advertencia si el método no converge dentro del número de iteraciones indicado."
        ),
        # Ejemplo
        "example": (
            "Ejemplo de uso (sistema 2x2):\n"
            "Sistema: 3x + y = 5\n"
            "         x + 2y = 4\n\n"
            "En la interfaz puedes ingresar:\n"
            "• Matriz A = '3,1;1,2'\n"
            "• Vector b = '5,4'\n"
            "• x0 = '0,0'\n"
            "• Tol = 1e-4\n"
            "• it = 50\n\n"
            "El método de Jacobi aproximará la solución (x, y) que satisface el sistema."
        )
    },

    "Gauss-Seidel": {
        "module": ("gauss_seidel", "gauss_seidel_method"),
        # Descripción corta, general
        "description": (
            "El método de Gauss-Seidel es una mejora del método de Jacobi para resolver sistemas lineales A·x = b, "
            "donde las componentes de x se actualizan secuencialmente usando de inmediato los nuevos valores calculados."
        ),
        # Para qué sirve
        "purpose": (
            "Se utiliza para resolver sistemas lineales cuando se desea una convergencia más rápida que Jacobi "
            "y la matriz A cumple condiciones favorables (como diagonal dominante). Suele requerir menos iteraciones "
            "que Jacobi para alcanzar la misma precisión."
        ),
        # Cómo funciona
        "how_it_works": (
            "1. Se separa A en su parte estrictamente inferior L, diagonal D y estrictamente superior U.\n"
            "2. La ecuación A·x = b se reescribe de forma adecuada para usar los nuevos valores a medida que se calculan.\n"
            "3. En la iteración k+1, cada componente x_i se actualiza usando ya las x_j^{(k+1)} recién calculadas para j < i "
            "y las x_j^{(k)} antiguas para j > i:\n"
            "      x_i^{(k+1)} = (1 / a_ii) * (b_i - Σ_{j<i} a_ij · x_j^{(k+1)} - Σ_{j>i} a_ij · x_j^{(k)}).\n"
            "4. Se repite el proceso hasta que el error entre iteraciones sea menor que la tolerancia o se alcance "
            "el máximo número de iteraciones."
        ),
        # Datos requeridos
        "required_inputs": [
            "Matriz A (filas separadas por ';')",
            "Vector b (separado por comas)",
            "Vector inicial x0",
            "Tolerancia (Tol)",
            "Máximo iteraciones (it)"
        ],
        # Qué verá el usuario en la GUI
        "ui_info": (
            "En la interfaz podrás ver:\n"
            "• Una tabla con los vectores x^{(k)} aproximados en cada iteración y el error correspondiente.\n"
            "• La solución aproximada del sistema al finalizar.\n"
            "• En muchos casos, notarás que Gauss-Seidel necesita menos iteraciones que Jacobi para un mismo sistema."
        ),
        # Ejemplo
        "example": (
            "Ejemplo de uso:\n"
            "Sistema: 4x + y = 7\n"
            "         x + 3y = 8\n\n"
            "En la interfaz puedes ingresar:\n"
            "• Matriz A = '4,1;1,3'\n"
            "• Vector b = '7,8'\n"
            "• x0 = '0,0'\n"
            "• Tol = 1e-4\n"
            "• it = 50\n\n"
            "El método de Gauss-Seidel aproximará la solución del sistema usando los valores actualizados en cada paso."
        )
    },

    "SOR": {
        "module": ("SOR", "sor_method"),
        # Descripción corta, general
        "description": (
            "El método SOR (Successive Over-Relaxation) es una variante de Gauss-Seidel que introduce un factor de "
            "relajación w para acelerar (o en algunos casos estabilizar) la convergencia del método."
        ),
        # Para qué sirve
        "purpose": (
            "Se utiliza para resolver sistemas lineales A·x = b cuando se desea mejorar la velocidad de convergencia "
            "respecto a Gauss-Seidel. Con una buena elección de w (entre 0 y 2), se pueden reducir significativamente "
            "las iteraciones necesarias."
        ),
        # Cómo funciona
        "how_it_works": (
            "1. Parte de la idea de Gauss-Seidel pero corrige la actualización con un factor w.\n"
            "2. Para cada componente x_i, primero se calcula la actualización tipo Gauss-Seidel (x_i^{GS}).\n"
            "3. Luego se combina con el valor anterior x_i^{(k)} para obtener:\n"
            "      x_i^{(k+1)} = (1 - w) · x_i^{(k)} + w · x_i^{GS}.\n"
            "4. Si 0 < w < 1 se habla de sub-relajación (puede ayudar cuando el método tiende a oscilar).\n"
            "   Si 1 < w < 2 se habla de sobre-relajación (busca acelerar la convergencia).\n"
            "5. Se repite hasta que el error sea menor que la tolerancia o se alcance el máximo de iteraciones."
        ),
        # Datos requeridos
        "required_inputs": [
            "Matriz A",
            "Vector b",
            "Vector inicial x0",
            "Factor w (por ejemplo 1.1, 1.2, etc.)",
            "Tolerancia (Tol)",
            "Máximo iteraciones (it)"
        ],
        # Qué verá el usuario en la GUI
        "ui_info": (
            "En la interfaz podrás ver:\n"
            "• Una tabla con las iteraciones del vector x y el error asociado.\n"
            "• El valor final aproximado de la solución.\n"
            "• Podrás experimentar cambiando w para ver cómo afecta el número de iteraciones y la convergencia."
        ),
        # Ejemplo
        "example": (
            "Ejemplo de uso:\n"
            "Supongamos el mismo sistema resuelto con Gauss-Seidel, pero ahora usando SOR con w = 1.2.\n"
            "• Matriz A = '4,1;1,3'\n"
            "• Vector b = '7,8'\n"
            "• x0 = '0,0'\n"
            "• w = 1.2\n"
            "• Tol = 1e-4\n"
            "• it = 50\n\n"
            "La idea es comparar cuántas iteraciones requiere SOR frente a Gauss-Seidel para lograr la misma precisión."
        )
    },

    "Vandermonde": {
        "module": ("Vandermonde", "interpolacion_vandermonde"),
        # Descripción corta, general
        "description": (
            "La interpolación con matriz de Vandermonde construye un polinomio p(x) de grado n-1 que pasa exactamente "
            "por n puntos dados, resolviendo un sistema lineal con una matriz de tipo Vandermonde."
        ),
        # Para qué sirve
        "purpose": (
            "Se utiliza para encontrar el polinomio interpolante que ajusta exactamente un conjunto de puntos (x_i, y_i). "
            "Es útil para aproximar funciones conocidas solo en ciertos puntos, aunque no siempre es el método más "
            "estable para muchos puntos."
        ),
        # Cómo funciona
        "how_it_works": (
            "1. Dado un conjunto de n puntos (x_0, y_0), ..., (x_{n-1}, y_{n-1}), se construye la matriz de Vandermonde V:\n"
            "      V[i,j] = x_i^j  para i,j = 0,...,n-1.\n"
            "2. Se plantea el sistema V·a = y, donde a = (a_0, a_1, ..., a_{n-1}) son los coeficientes del polinomio.\n"
            "3. Se resuelve el sistema para obtener los coeficientes a_j.\n"
            "4. El polinomio interpolante es:\n"
            "      p(x) = a_0 + a_1 x + a_2 x^2 + ... + a_{n-1} x^{n-1}.\n"
            "5. Con este polinomio se pueden evaluar valores intermedios y graficar la curva interpolante."
        ),
        # Datos requeridos
        "required_inputs": [
            "Valores X (puntos x, separados por comas, por ejemplo: '1,2,3')",
            "Valores Y (puntos y, separados por comas, por ejemplo: '2,4,5')"
        ],
        # Qué verá el usuario en la GUI
        "ui_info": (
            "En la interfaz podrás ver:\n"
            "• La tabla de los puntos utilizados para la interpolación.\n"
            "• La matriz de Vandermonde y/o el polinomio resultante (dependiendo de la implementación).\n"
            "• Una opción para ver el polinomio completo en una ventana aparte.\n"
            "• Posiblemente la gráfica de los puntos originales y la curva del polinomio interpolante."
        ),
        # Ejemplo
        "example": (
            "Ejemplo de uso:\n"
            "Puntos: (1,2), (2,4), (3,5)\n\n"
            "En la interfaz:\n"
            "• X = '1,2,3'\n"
            "• Y = '2,4,5'\n\n"
            "El método construirá el polinomio p(x) que pasa exactamente por esos tres puntos, mostrando sus coeficientes."
        )
    },

    "Interpolación Newton": {
        "module": ("interpolacion_newton", "interpolacion_newton"),
        # Descripción corta, general
        "description": (
            "La interpolación de Newton usa diferencias divididas para construir un polinomio interpolante de forma "
            "progresiva, permitiendo agregar puntos con relativa facilidad."
        ),
        # Para qué sirve
        "purpose": (
            "Se utiliza para interpolar datos mediante un polinomio y es especialmente útil cuando se agregan "
            "nuevos puntos, ya que permite reutilizar las diferencias divididas ya calculadas."
        ),
        # Cómo funciona
        "how_it_works": (
            "1. Dado un conjunto de puntos (x_0, y_0), ..., (x_n, y_n), se construye la tabla de diferencias divididas.\n"
            "2. Las diferencias divididas f[x_i], f[x_i, x_{i+1}], f[x_i, x_{i+1}, x_{i+2}], ... se van calculando "
            "recursivamente.\n"
            "3. El polinomio de Newton tiene la forma:\n"
            "      P(x) = f[x_0] + f[x_0,x_1](x - x_0) + f[x_0,x_1,x_2](x - x_0)(x - x_1) + ...\n"
            "4. Una vez obtenidos los coeficientes (diferencias divididas), se puede evaluar P(x) en cualquier punto.\n"
            "5. La estructura incremental permite extender el polinomio cuando se añaden nuevos datos x_{n+1}, y_{n+1}."
        ),
        # Datos requeridos
        "required_inputs": [
            "Puntos x (separados por comas, por ejemplo: '1,2,3')",
            "Valores y (separados por comas, por ejemplo: '2,4,5')",
            "Punto a evaluar (un valor x donde se desea calcular P(x))"
        ],
        # Qué verá el usuario en la GUI
        "ui_info": (
            "En la interfaz podrás ver:\n"
            "• Una tabla con las diferencias divididas.\n"
            "• El polinomio de Newton (o sus coeficientes equivalentes).\n"
            "• El valor aproximado P(x_eval) para el punto que indiques.\n"
            "• Opcionalmente, la comparación con otros métodos de interpolación (mediante el botón de comparación)."
        ),
        # Ejemplo
        "example": (
            "Ejemplo de uso:\n"
            "Puntos: (1,2), (2,4), (3,5)\n"
            "Punto a evaluar: x = 2.5\n\n"
            "En la interfaz:\n"
            "• x = '1,2,3'\n"
            "• y = '2,4,5'\n"
            "• Punto a evaluar = '2.5'\n\n"
            "El método calculará el polinomio de Newton y te dará una aproximación de P(2.5)."
        )
    },

    "Spline Lineal": {
        "module": ("spline_lineal", "spline_lineal_con_polinomios"),
        # Descripción corta, general
        "description": (
            "La interpolación por spline lineal construye funciones lineales por tramos entre cada par de puntos "
            "consecutivos, uniendo los puntos dados mediante segmentos rectos."
        ),
        # Para qué sirve
        "purpose": (
            "Se utiliza para interpolar datos de forma simple y estable, sin introducir oscilaciones fuertes, "
            "siendo adecuada cuando se busca una aproximación continua pero no necesariamente suave en la derivada."
        ),
        # Cómo funciona
        "how_it_works": (
            "1. A partir de puntos ordenados (x_0, y_0), ..., (x_n, y_n), se define un polinomio lineal en cada intervalo [x_i, x_{i+1}].\n"
            "2. En cada tramo, el polinomio es de la forma:\n"
            "      S_i(x) = a_i x + b_i,\n"
            "   determinado por las condiciones S_i(x_i) = y_i y S_i(x_{i+1}) = y_{i+1}.\n"
            "3. Se obtienen así n funciones lineales que, juntas, forman el spline lineal.\n"
            "4. La función resultante es continua, pero su derivada presenta saltos en los nodos x_i."
        ),
        # Datos requeridos
        "required_inputs": [
            "Puntos x (ordenados, separados por comas)",
            "Valores y correspondientes a cada x (separados por comas)"
        ],
        # Qué verá el usuario en la GUI
        "ui_info": (
            "En la interfaz podrás ver:\n"
            "• Los tramos lineales calculados (los polinomios por intervalo) en una ventana de detalle.\n"
            "• La tabla con los puntos originales.\n"
            "• Opcionalmente, la gráfica de los puntos y las rectas que los unen.\n"
            "• La posibilidad de copiar los polinomios para usarlos en otro lado."
        ),
        # Ejemplo
        "example": (
            "Ejemplo de uso:\n"
            "Puntos: (0,0), (1,1), (2,0)\n\n"
            "En la interfaz:\n"
            "• x = '0,1,2'\n"
            "• y = '0,1,0'\n\n"
            "El método construirá dos tramos lineales: uno entre x=0 y x=1, y otro entre x=1 y x=2, "
            "que juntos forman el spline lineal."
        )
    },

    "Spline Cúbico": {
        "module": ("spline_cubico", "spline_cubico"),
        # Descripción corta, general
        "description": (
            "El spline cúbico interpolante construye polinomios cúbicos por tramos entre puntos consecutivos, "
            "garantizando continuidad en la función, en la primera derivada y en la segunda derivada."
        ),
        # Para qué sirve
        "purpose": (
            "Se utiliza cuando se desea una interpolación suave, sin esquinas, adecuada para aproximar curvas "
            "de forma visualmente agradable y con buenas propiedades numéricas."
        ),
        # Cómo funciona
        "how_it_works": (
            "1. A partir de puntos (x_0, y_0), ..., (x_n, y_n), se construyen n polinomios cúbicos S_i(x) en cada intervalo [x_i, x_{i+1}].\n"
            "2. Cada S_i(x) cumple:\n"
            "   • S_i(x_i) = y_i y S_i(x_{i+1}) = y_{i+1} (interpolación de datos).\n"
            "   • Continuidad de la primera y segunda derivada en los nodos internos.\n"
            "3. Estas condiciones llevan a un sistema lineal para las segundas derivadas (o coeficientes), que se resuelve.\n"
            "4. Dependiendo del tipo de spline (natural, con condiciones en los extremos, etc.), se imponen condiciones "
            "adicionales en los extremos, como S''(x_0) = 0 y S''(x_n) = 0 en el spline natural."
        ),
        # Datos requeridos
        "required_inputs": [
            "Puntos x (ordenados, separados por comas)",
            "Valores y (separados por comas)"
        ],
        # Qué verá el usuario en la GUI
        "ui_info": (
            "En la interfaz podrás ver:\n"
            "• Los polinomios cúbicos por cada tramo en una ventana de detalle.\n"
            "• La tabla de puntos (x_i, y_i).\n"
            "• La gráfica del spline cúbico superpuesta a los puntos originales, mostrando la suavidad de la curva."
        ),
        # Ejemplo
        "example": (
            "Ejemplo de uso:\n"
            "Puntos: (0,0), (1,2), (2,3), (3,2)\n\n"
            "En la interfaz:\n"
            "• x = '0,1,2,3'\n"
            "• y = '0,2,3,2'\n\n"
            "El método construirá los tramos cúbicos S_i(x) que conectan suavemente todos los puntos."
        )
    },

    "Interpolación Lagrange": {
        "module": ("interpolacion_lagrange", "interpolacion_lagrange"),
        # Descripción corta, general
        "description": (
            "La interpolación de Lagrange construye un polinomio interpolante usando combinaciones lineales de "
            "polinomios base L_i(x) que valen 1 en x_i y 0 en los demás nodos."
        ),
        # Para qué sirve
        "purpose": (
            "Se utiliza para encontrar el polinomio que pasa exactamente por un conjunto de puntos, sin necesidad "
            "de resolver sistemas lineales. Es didáctico y útil para problemas de tamaño moderado."
        ),
        # Cómo funciona
        "how_it_works": (
            "1. Dados n+1 puntos (x_0, y_0), ..., (x_n, y_n), se definen los polinomios base de Lagrange:\n"
            "      L_i(x) = Π_{j≠i} (x - x_j) / (x_i - x_j).\n"
            "2. El polinomio interpolante se construye como:\n"
            "      P(x) = Σ_{i=0}^n y_i · L_i(x).\n"
            "3. Por construcción, P(x_k) = y_k para cada k.\n"
            "4. Una vez definido P(x), se puede evaluar en cualquier punto para aproximar el valor de la función."
        ),
        # Datos requeridos
        "required_inputs": [
            "Puntos x (separados por comas)",
            "Valores y (separados por comas)"
        ],
        # Qué verá el usuario en la GUI
        "ui_info": (
            "En la interfaz podrás ver:\n"
            "• El polinomio de Lagrange o una representación equivalente.\n"
            "• Los puntos usados en la interpolación.\n"
            "• Una ventana opcional con el polinomio completo para copiarlo.\n"
            "• La gráfica del polinomio interpolante junto con los puntos de datos (si la implementación lo permite)."
        ),
        # Ejemplo
        "example": (
            "Ejemplo de uso:\n"
            "Puntos: (1,1), (2,4), (3,9)\n\n"
            "En la interfaz:\n"
            "• x = '1,2,3'\n"
            "• y = '1,4,9'\n\n"
            "El método construirá el polinomio de Lagrange que pasa por esos tres puntos, que en este caso "
            "se aproxima a la función x^2 en esos nodos."
        )
    }
}

# Métodos de raíces (Capítulo 1) para el informe comparativo
ROOT_METHODS = [
    "Bisección",
    "Regla Falsa",
    "Newton",
    "Secante",
    "Punto Fijo",
    "Raíces Múltiples"
]

# Traducción de parámetros comunes
SPANISH_PARAMS = {
    "x0": "Valor inicial (x0)",
    "x1": "Segundo valor inicial (x1)",
    "lower_bound": "Límite inferior (a)",
    "upper_bound": "Límite superior (b)",
    "tolerance": "Tolerancia",
    "max_iterations": "Máximo de iteraciones",
    "error_type": "Tipo de error (abs/rel)",
    "w": "Factor de relajación (w)",
    "A": "Matriz A (separar filas con ;)",
    "b": "Vector b (separar con ,)",
    "x_points": "Puntos x (separar con ,)",
    "y_points": "Puntos y (separar con ,)",
    "eval_point": "Punto a evaluar"
}

MODULE_PATH = "Python"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Análisis Numérico - Métodos Computacionales")
        self.geometry("800x700")
        self.configure(bg='#f0f0f0')

        self.style = ttk.Style()
        self.style.theme_use('clam')

        self.method_var = tk.StringVar()
        self.show_main_menu()
        # Historial de ejecuciones para el informe comparativo
        # Cada entrada: {"method", "root", "iterations", "final_error", "error_type"}
        self.run_history = []


    def show_main_menu(self):
        for widget in self.winfo_children():
            widget.destroy()

        main_frame = tk.Frame(self, bg='#f0f0f0')
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)

        title = tk.Label(main_frame, text="MÉTODOS NUMÉRICOS",
                         font=("Arial", 20, "bold"), bg='#f0f0f0', fg='#2c3e50')
        title.pack(pady=(0, 10))

        subtitle = tk.Label(main_frame, text="Seleccione el método a utilizar",
                            font=("Arial", 12), bg='#f0f0f0', fg='#7f8c8d')
        subtitle.pack(pady=(0, 20))

        button_frame = tk.Frame(main_frame, bg='#f0f0f0')
        button_frame.pack()

        categories = {
            "Ecuaciones No Lineales": ["Bisección", "Regla Falsa", "Newton", "Secante", "Punto Fijo", "Raíces Múltiples"],
            "Sistemas Lineales": ["Jacobi", "Gauss-Seidel", "SOR"],
            "Interpolación": ["Vandermonde", "Interpolación Newton", "Interpolación Lagrange", "Spline Lineal", "Spline Cúbico"]
        }

        for i, (category, methods) in enumerate(categories.items()):
            cat_label = tk.Label(button_frame, text=category,
                                 font=("Arial", 14, "bold"), bg='#f0f0f0', fg='#34495e')
            cat_label.grid(row=i * 10, column=0, columnspan=2, pady=(15, 5), sticky='w')

            for j, method in enumerate(methods):
                if method in METHODS:
                    btn = tk.Button(
                        button_frame,
                        text=method,
                        width=25, height=2,
                        font=("Arial", 10), bg='#3498db', fg='white',
                        activebackground='#2980b9', cursor='hand2',
                        command=lambda m=method: self.show_method_info(m)
                    )
                    btn.grid(row=i * 10 + 1 + j // 2, column=j % 2, padx=5, pady=2, sticky='ew')

        button_frame.grid_columnconfigure(0, weight=1)
        button_frame.grid_columnconfigure(1, weight=1)

    def show_method_info(self, method_name):
        # Limpiar ventana
        for widget in self.winfo_children():
            widget.destroy()

        main_frame = tk.Frame(self, bg='#f0f0f0')
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)

        # Título arriba
        title = tk.Label(
            main_frame,
            text=f"MÉTODO: {method_name.upper()}",
            font=("Arial", 16, "bold"),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        title.pack(pady=(0, 15))

        method_info = METHODS[method_name]

        # ====== CONTENEDOR SCROLLABLE (centro) ======
        content_container = tk.Frame(main_frame, bg='#f0f0f0')
        content_container.pack(fill='both', expand=True)

        # Canvas + Scrollbar dentro del contenedor
        canvas = tk.Canvas(content_container, bg='#f0f0f0', highlightthickness=0)
        scrollbar = ttk.Scrollbar(content_container, orient="vertical", command=canvas.yview)

        # Frame interno que va dentro del canvas
        scroll_frame = tk.Frame(canvas, bg='#f0f0f0')

        def on_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        scroll_frame.bind("<Configure>", on_configure)

        # Meter el frame dentro del canvas
        canvas.create_window((0, 0), window=scroll_frame, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)

        # Canvas a la izquierda, scrollbar a la derecha
        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        # ====== SECCIONES / APARTADOS ======

        # Lista de labels que deben ajustar su wraplength
        wrap_labels = []

        # 1. Descripción
        if "description" in method_info:
            desc_frame = tk.LabelFrame(
                scroll_frame, text="Descripción",
                font=("Arial", 12, "bold"),
                bg='#f0f0f0', fg='#2c3e50',
                padx=10, pady=10
            )
            desc_frame.pack(fill='x', pady=(0, 15))

            desc_label = tk.Label(
                desc_frame,
                text=method_info["description"],
                font=("Arial", 11),
                bg='#f0f0f0',
                justify='left',
                anchor='w'
            )
            desc_label.pack(anchor='w', fill='x')
            wrap_labels.append(desc_label)

        # 2. Para qué sirve
        if "purpose" in method_info:
            purpose_frame = tk.LabelFrame(
                scroll_frame, text="¿Para qué sirve?",
                font=("Arial", 12, "bold"),
                bg='#f0f0f0', fg='#2c3e50',
                padx=10, pady=10
            )
            purpose_frame.pack(fill='x', pady=(0, 15))

            purpose_label = tk.Label(
                purpose_frame,
                text=method_info["purpose"],
                font=("Arial", 11),
                bg='#f0f0f0',
                fg='#2c3e50',
                justify='left',
                anchor='w'
            )
            purpose_label.pack(anchor='w', fill='x')
            wrap_labels.append(purpose_label)

        # 3. Cómo funciona
        if "how_it_works" in method_info:
            how_frame = tk.LabelFrame(
                scroll_frame, text="¿Cómo funciona?",
                font=("Arial", 12, "bold"),
                bg='#f0f0f0', fg='#2c3e50',
                padx=10, pady=10
            )
            how_frame.pack(fill='x', pady=(0, 15))

            how_label = tk.Label(
                how_frame,
                text=method_info["how_it_works"],
                font=("Arial", 11),
                bg='#f0f0f0',
                fg='#2c3e50',
                justify='left',
                anchor='w'
            )
            how_label.pack(anchor='w', fill='x')
            wrap_labels.append(how_label)

        # 4. Datos requeridos
        req_frame = tk.LabelFrame(
            scroll_frame, text="Datos requeridos",
            font=("Arial", 12, "bold"),
            bg='#f0f0f0', fg='#2c3e50',
            padx=10, pady=10
        )
        req_frame.pack(fill='x', pady=(0, 15))

        for req in method_info.get("required_inputs", []):
            lbl = tk.Label(
                req_frame,
                text=f"• {req}",
                font=("Arial", 11),
                bg='#f0f0f0',
                fg='#2c3e50',
                justify='left',
                anchor='w'
            )
            lbl.pack(anchor='w', fill='x')
            wrap_labels.append(lbl)

        # 5. Qué verás en la interfaz
        if "ui_info" in method_info:
            ui_frame = tk.LabelFrame(
                scroll_frame, text="¿Qué verás en la interfaz?",
                font=("Arial", 12, "bold"),
                bg='#f0f0f0', fg='#2c3e50',
                padx=10, pady=10
            )
            ui_frame.pack(fill='x', pady=(0, 15))

            ui_label = tk.Label(
                ui_frame,
                text=method_info["ui_info"],
                font=("Arial", 11),
                bg='#f0f0f0',
                fg='#2c3e50',
                justify='left',
                anchor='w'
            )
            ui_label.pack(anchor='w', fill='x')
            wrap_labels.append(ui_label)

        # 6. Ejemplo
        if "example" in method_info:
            example_frame = tk.LabelFrame(
                scroll_frame, text="Ejemplo",
                font=("Arial", 12, "bold"),
                bg='#f0f0f0', fg='#2c3e50',
                padx=10, pady=10
            )
            example_frame.pack(fill='x', pady=(0, 20))

            example_label = tk.Label(
                example_frame,
                text=method_info["example"],
                font=("Arial", 10, "italic"),
                bg='#f0f0f0',
                fg='#7f8c8d',
                justify='left',
                anchor='w'
            )
            example_label.pack(anchor='w', fill='x')
            wrap_labels.append(example_label)

        # === ACTUALIZAR WRAPLENGTH SEGÚN EL ANCHO VISIBLE DEL CANVAS ===
        def _update_wrap_all(event, labels=wrap_labels):
            # event.width es el ancho visible del canvas
            wrap = max(event.width - 40, 200)   # margen a los lados
            for lab in labels:
                lab.config(wraplength=wrap)

        canvas.bind("<Configure>", _update_wrap_all)

        # ====== BOTONES FINALES, CENTRADOS ======
        button_frame = tk.Frame(main_frame, bg='#f0f0f0')
        button_frame.pack(pady=(10, 0))

        tk.Button(
            button_frame,
            text="Continuar",
            font=("Arial", 12, "bold"),
            bg='#27ae60',
            fg='white',
            padx=20,
            pady=5,
            cursor='hand2',
            command=lambda: self.load_method_form(method_name)
        ).pack(side='left', padx=10)

        tk.Button(
            button_frame,
            text="Volver",
            font=("Arial", 12),
            bg='#95a5a6',
            fg='white',
            padx=20,
            pady=5,
            cursor='hand2',
            command=self.show_main_menu
        ).pack(side='left', padx=10)


    def load_method_form(self, method_name):
        method_info = METHODS[method_name]["module"]
        package = METHODS[method_name].get("package", "")

        if isinstance(method_info, tuple):
            module_name, function_name = method_info
        else:
            module_name = function_name = method_info

        if package:
            full_module = f"Python.{package}.{module_name}"
        else:
            full_module = f"Python.{module_name}"

        try:
            module = importlib.import_module(full_module)
            func = getattr(module, function_name)
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar el método: {e}")
            return

        self.show_input_form(method_name, func)

    def show_input_form(self, method_name, func):
        for widget in self.winfo_children():
            widget.destroy()

        main_frame = tk.Frame(self, bg='#f0f0f0')
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)

        title = tk.Label(main_frame, text=f"{method_name} - Ingreso de Parámetros",
                         font=("Arial", 14, "bold"), bg='#f0f0f0', fg='#2c3e50')
        title.pack(pady=(0, 20))

        input_frame = tk.LabelFrame(main_frame, text="Parámetros", font=("Arial", 12, "bold"),
                                    bg='#f0f0f0', fg='#2c3e50', padx=15, pady=15)
        input_frame.pack(fill='x', pady=(0, 20))

        entries = {}

        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        use_f = params and params[0] == 'f'

        # Ahora también saltamos error_type porque lo manejamos con un combobox aparte
        SKIP_PARAMS = {'show_report', 'eval_grid', 'auto_compare', 'error_type'}
        params = [p for p in params if p not in SKIP_PARAMS]

        if use_f:
            tk.Label(input_frame, text="Función f(x) =", font=("Arial", 10, "bold"),
                     bg='#f0f0f0', fg='#2c3e50').grid(row=0, column=0, sticky='w', pady=5)
            f_entry = tk.Entry(input_frame, width=50, font=("Arial", 10))
            f_entry.grid(row=0, column=1, padx=(10, 0), pady=5, sticky='ew')

            tk.Label(input_frame, text="Ej: x**2 - 2, np.sin(x), x**3 - x - 1",
                     font=("Arial", 8), bg='#f0f0f0', fg='#7f8c8d').grid(row=1, column=1, sticky='w')

            params = params[1:]

        row = 2
        for param in params:
            label = SPANISH_PARAMS.get(param, param)
            tk.Label(input_frame, text=f"{label}:", font=("Arial", 10, "bold"),
                     bg='#f0f0f0', fg='#2c3e50').grid(row=row, column=0, sticky='w', pady=5)
            entry = tk.Entry(input_frame, width=50, font=("Arial", 10))
            entry.grid(row=row, column=1, padx=(10, 0), pady=5, sticky='ew')
            entries[param] = entry
            row += 1

        input_frame.grid_columnconfigure(1, weight=1)

        self.last_called_show_report = False
        self.auto_cmp_var = tk.BooleanVar(value=True)
        eval_grid_var = tk.StringVar(value='500')
        # Checkbox para que el usuario decida si quiere ejecutar el informe comparativo
        self.show_report_var = tk.BooleanVar(value=False)

        # 🔹 NUEVO: checkbox para comparar tipos de error en el Capítulo 1
        self.compare_error_types_var = tk.BooleanVar(value=False)

        # NUEVO: tipo de error seleccionado por el usuario
        error_type_var = tk.StringVar(value='rel')  # 'rel', 'abs' o 'cond'

        opts_frame = tk.Frame(main_frame, bg='#f0f0f0')
        opts_frame.pack(fill='x', pady=(0, 10))

        tk.Checkbutton(
            opts_frame,
            text='Comparación automática',
            variable=self.auto_cmp_var,
            bg='#f0f0f0'
        ).pack(side='left', padx=(0, 10))

        tk.Checkbutton(
            opts_frame,
            text='Mostrar informe',
            variable=self.show_report_var,
            bg='#f0f0f0'
        ).pack(side='left', padx=(10, 10))

        # checkbox “Comparar tipos de error (Cap. 1)”
        tk.Checkbutton(
            opts_frame,
            text='Comparar los tipos de error',
            variable=self.compare_error_types_var,
            bg='#f0f0f0'
        ).pack(side='left', padx=(10, 10))

        tk.Label(opts_frame, text='Eval grid:', bg='#f0f0f0').pack(side='left')
        tk.Entry(opts_frame, textvariable=eval_grid_var, width=6).pack(side='left', padx=(5, 0))

        tk.Label(opts_frame, text='   Tipo de error:', bg='#f0f0f0').pack(side='left', padx=(15, 0))
        error_combo = ttk.Combobox(
            opts_frame,
            textvariable=error_type_var,
            values=['rel', 'abs', 'cond'],  # relativo, absoluto, condición
            width=8,
            state='readonly'
        )
        error_combo.pack(side='left', padx=5)

        def execute():
            try:
                args = []

                # reset valores para graficar
                self.last_function_str = None
                self.last_a = None
                self.last_b = None

                if use_f:
                    f_str = f_entry.get()
                    if not f_str.strip():
                        messagebox.showerror("Error", "Debe ingresar una función f(x)")
                        return

                    self.last_function_str = f_str

                    f = lambda x: eval(f_str, {
                        "np": np, "x": x, "math": math,
                        "sin": np.sin, "cos": np.cos, "tan": np.tan,
                        "exp": np.exp, "log": np.log, "sqrt": np.sqrt
                    })
                    args.append(f)

                x_vals = None
                y_vals = None

                tol_value = None
                max_it_value = None

                for param in params:
                    val = entries[param].get().strip()
                    if not val:
                        messagebox.showerror("Error", f"Debe ingresar un valor para {SPANISH_PARAMS.get(param, param)}")
                        return

                    # guardar [a,b] para graficar en métodos de intervalo
                    if param == "lower_bound":
                        self.last_a = float(val)
                    elif param == "upper_bound":
                        self.last_b = float(val)

                    # NUEVO: tolerancia
                    if param == "tolerance":
                        try:
                            tol_value = float(val)
                            args.append(tol_value)
                        except ValueError:
                            messagebox.showerror("Error", "La tolerancia debe ser un número")
                            return

                    # máximo de iteraciones
                    elif param in ["max_iterations", "n_iter", "iteraciones"]:
                        try:
                            max_it_value = int(val)
                            args.append(max_it_value)
                        except ValueError:
                            messagebox.showerror("Error", f"{SPANISH_PARAMS.get(param, param)} debe ser un número entero")
                            return

                    elif param.lower() in ["x_points", "puntos x", "valoresx"]:
                        x_vals = [float(x.strip()) for x in val.split(',') if x.strip() != ""]
                    elif param.lower() in ["y_points", "valores y", "valoresy"]:
                        y_vals = [float(y.strip()) for y in val.split(',') if y.strip() != ""]
                    elif param == "A":
                        rows = val.split(';')
                        matrix = []
                        for row_ in rows:
                            matrix.append([float(x.strip()) for x in row_.split(',')])
                        args.append(np.array(matrix))
                    elif param == "b":
                        vector = [float(x.strip()) for x in val.split(',')]
                        args.append(np.array(vector))
                    else:
                        try:
                            args.append(float(val))
                        except ValueError:
                            args.append(val)

                if x_vals is not None and y_vals is not None:
                    self.ultimo_x = x_vals
                    self.ultimo_y = y_vals
                    args = [x_vals, y_vals] + args

                kwargs = {}
                try:
                    f_sig = inspect.signature(func)
                    if 'show_report' in f_sig.parameters:
                        try:
                            kwargs['show_report'] = bool(self.show_report_var.get())
                        except Exception:
                            kwargs['show_report'] = False
                        if 'eval_grid' in f_sig.parameters:
                            try:
                                kwargs['eval_grid'] = int(eval_grid_var.get())
                            except Exception:
                                kwargs['eval_grid'] = 500
                        if 'auto_compare' in f_sig.parameters:
                            kwargs['auto_compare'] = bool(self.auto_cmp_var.get())

                    # NUEVO: pasar tipo de error si el método lo permite
                    if 'error_type' in f_sig.parameters:
                        kwargs['error_type'] = error_type_var.get()
                except Exception:
                    pass

                # === EJECUCIÓN PRINCIPAL ===
                result = func(*args, **kwargs)
                self.last_called_show_report = bool(kwargs.get('show_report', False))

                # NUEVO: construir y guardar resumen solo para métodos de raíces
                if method_name in ROOT_METHODS:
                    summary = self.build_run_summary(method_name, result, error_type_var.get())
                    if summary:
                        self.run_history.append(summary)

                    # ⚠ AUTO-EJECUTAR los otros métodos de raíces para comparación
                    if self.auto_cmp_var.get() and use_f and self.last_a is not None and self.last_b is not None:
                        self.auto_run_other_root_methods(
                            current_method=method_name,
                            f_str=self.last_function_str,
                            a=self.last_a,
                            b=self.last_b,
                            tol=tol_value,
                            max_iter=max_it_value,
                            error_type=error_type_var.get(),
                            eval_grid=eval_grid_var.get()
                        )

                self.show_result(method_name, result)

                # NUEVO: si el usuario quiere comparación automática e hizo un método de raíces, mostrar informe
                if self.auto_cmp_var.get() and method_name in ROOT_METHODS:
                    self.show_comparison_report(error_type_var.get())

                # 🔹 NUEVO: Comparar TIPOS de error (abs/rel/cond) para ESTE método de raíces
                if method_name in ROOT_METHODS and self.compare_error_types_var.get():
                    error_runs = []
                    tipos = ['abs', 'rel', 'cond']

                    for et in tipos:
                        local_kwargs = kwargs.copy()
                        # Forzamos configuración silenciosa sin informes ni auto comparación
                        local_kwargs['show_report'] = False
                        local_kwargs['auto_compare'] = False
                        local_kwargs['error_type'] = et

                        try:
                            res_et = func(*args, **local_kwargs)
                            summary_et = self.build_run_summary(method_name, res_et, et)
                            if summary_et:
                                error_runs.append(summary_et)
                        except Exception:
                            continue

                    if error_runs:
                        self.show_error_type_report(method_name, error_runs)

            except Exception as e:
                messagebox.showerror("Error", f"Error en la ejecución: {str(e)}")

        button_frame = tk.Frame(main_frame, bg='#f0f0f0')
        button_frame.pack(pady=20)

        tk.Button(button_frame, text="Ejecutar Método", font=("Arial", 12, "bold"),
                  bg='#27ae60', fg='white', padx=20, pady=8, cursor='hand2',
                  command=execute).pack(side='left', padx=10)

        tk.Button(button_frame, text="Volver", font=("Arial", 12),
                  bg='#95a5a6', fg='white', padx=20, pady=8, cursor='hand2',
                  command=lambda: self.show_method_info(method_name)).pack(side='left')

    def show_result(self, method_name, result):
        for widget in self.winfo_children():
            widget.destroy()

        main_frame = tk.Frame(self, bg='#f0f0f0')
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)

        title = tk.Label(main_frame, text=f"RESULTADOS - {method_name.upper()}",
                         font=("Arial", 16, "bold"), bg='#f0f0f0', fg='#2c3e50')
        title.pack(pady=(0, 20))

        result_frame = tk.LabelFrame(main_frame, text="Resultados del Método",
                                     font=("Arial", 12, "bold"), bg='#f0f0f0', fg='#2c3e50')
        result_frame.pack(fill='both', expand=True, pady=(0, 20))

        if self.is_tabular_result(result):
            self.create_result_table(result_frame, result, method_name)
        else:
            text_widget = scrolledtext.ScrolledText(result_frame, height=20, width=80,
                                                    font=("Courier", 10), bg='white')
            text_widget.pack(fill='both', expand=True, padx=10, pady=10)
            text_widget.insert('1.0', str(result))
            text_widget.config(state='disabled')

        if method_name in ["Vandermonde", "spline_lineal", "spline_cubico",
                           "interpolacion_lagrange", "interpolacion_newton"] \
                and not getattr(self, 'last_called_show_report', False) \
                and (self.show_report_var.get() or self.auto_cmp_var.get()):
            if messagebox.askyesno("Comparar", "¿Desea comparar con otros métodos de interpolación?"):
                comparar_metodos(self.ultimo_x, self.ultimo_y)

        button_frame = tk.Frame(main_frame, bg='#f0f0f0')
        button_frame.pack()

        graficables = [
            "Bisección",
            "Regla Falsa",
            "Punto Fijo",
            "Newton",
            "Secante",
            "Raíces Múltiples"
        ]

        if method_name in graficables:
            tk.Button(
                button_frame,
                text="Graficar",
                font=("Arial", 12, "bold"),
                bg='#e67e22', fg='white',
                padx=20, pady=8, cursor='hand2',
                command=lambda m=method_name: self.plot_root_method(m, result)
            ).pack(side='left', padx=10)

        tk.Button(button_frame, text="Nuevo Cálculo", font=("Arial", 12, "bold"),
                  bg='#3498db', fg='white', padx=20, pady=8, cursor='hand2',
                  command=lambda: self.show_method_info(method_name)).pack(side='left', padx=10)

        tk.Button(button_frame, text="Menú Principal", font=("Arial", 12),
                  bg='#95a5a6', fg='white', padx=20, pady=8, cursor='hand2',
                  command=self.show_main_menu).pack(side='left')

    def is_tabular_result(self, result):
        if isinstance(result, tuple) and len(result) >= 2:
            last_element = result[-1]
            return (isinstance(last_element, list) and len(last_element) > 0) or \
                   (isinstance(last_element, dict) and len(last_element) > 0)
        if isinstance(result, dict) and len(result) > 0:
            return True
        return isinstance(result, list) and len(result) > 0

    def create_result_table(self, parent_frame, result, method_name):
        table_data = None
        header_text = None
        if isinstance(result, tuple):
            header_text = "\n".join([str(r) for r in result[:-1]]) if len(result) > 1 else None
            table_data = result[-1]
        elif isinstance(result, dict):
            table_data = result
        else:
            table_data = result

        if header_text:
            info_text = scrolledtext.ScrolledText(parent_frame, height=4, width=80,
                                                  font=("Courier", 10), bg='#ecf0f1')
            info_text.pack(fill='x', padx=10, pady=(10, 5))
            info_text.insert('1.0', header_text)
            info_text.config(state='disabled')

        table_frame = tk.Frame(parent_frame, bg='white')
        table_frame.pack(fill='both', expand=True, padx=10, pady=5)

        custom_headers = False  # <- para saber si usamos texto tal cual o el .title()

        if isinstance(table_data, dict):
            columns = ["Propiedad", "Valor"]
            rows_iter = [(k, table_data[k]) for k in table_data]

        elif isinstance(table_data, list) and len(table_data) > 0 and isinstance(table_data[0], dict):
            columns = list(table_data[0].keys())
            rows_iter = [tuple(row.get(col, '') for col in columns) for row in table_data]

        elif isinstance(table_data, list) and len(table_data) > 0 and isinstance(table_data[0], list):
            n_cols = len(table_data[0])

            # === CABECERAS PERSONALIZADAS POR MÉTODO ===
            if method_name == "Bisección" and n_cols == 8:
                columns = ["Iteración", "a", "f(a)", "pm", "f(pm)", "b", "f(b)", "Error Abs."]
                custom_headers = True

            elif method_name == "Regla Falsa" and n_cols == 8:
                # [Iter, A, F(A), B, F(B), Xr, F(Xr), Error]
                columns = ["Iteración", "a", "f(a)", "b", "f(b)", "xr", "f(xr)", "Error"]
                custom_headers = True

            elif method_name == "Newton" and n_cols == 5:
                # [Iter, x, f(x), f'(x), Error]
                columns = ["Iteración", "x_n", "f(x_n)", "f'(x_n)", "Error"]
                custom_headers = True

            elif method_name == "Secante" and n_cols == 5:
                # [Iter, x_{i-1}, x_i, f(x_i), Error]
                columns = ["Iteración", "x_{n-1}", "x_n", "f(x_n)", "Error"]
                custom_headers = True

            elif method_name == "Punto Fijo" and n_cols == 4:
                # [Iter, x, g(x), Error]
                columns = ["Iteración", "x_n", "g(x_n)", "Error"]
                custom_headers = True

            elif method_name == "Raíces Múltiples" and n_cols == 6:
                # [Iter, x, f(x), f'(x), f''(x), Error]
                columns = ["Iteración", "x_n", "f(x_n)", "f'(x_n)", "f''(x_n)", "Error"]
                custom_headers = True

            else:
                # Caso genérico
                columns = [f"Col_{i+1}" for i in range(n_cols)]

            rows_iter = table_data

        else:
            columns = ["Elemento"]
            rows_iter = [(str(r),) for r in table_data] if isinstance(table_data, list) else [(str(table_data),)]

        tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=15)

        for col in columns:
            if custom_headers:
                heading_text = col          # usamos el texto tal cual lo definimos arriba
            else:
                heading_text = col.replace('_', ' ').title()
            tree.heading(col, text=heading_text)
            tree.column(col, width=120, anchor='center')

        for row in rows_iter:
            if isinstance(row, dict):
                values = [str(row.get(col, '')) for col in columns]
            elif isinstance(row, (list, tuple)):
                values = [str(val) for val in row]
            else:
                values = [str(row)]
            tree.insert('', 'end', values=values)

        v_scrollbar = ttk.Scrollbar(table_frame, orient='vertical', command=tree.yview)
        h_scrollbar = ttk.Scrollbar(table_frame, orient='horizontal', command=tree.xview)
        tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

        tree.grid(row=0, column=0, sticky='nsew')
        v_scrollbar.grid(row=0, column=1, sticky='ns')
        h_scrollbar.grid(row=1, column=0, sticky='ew')

        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)

        summary_frame = tk.Frame(parent_frame, bg='#f0f0f0')
        summary_frame.pack(fill='x', padx=10, pady=5)

        if isinstance(table_data, dict):
            tk.Label(summary_frame, text=f"Propiedades: {len(table_data)}",
                     font=("Arial", 10, "bold"), bg='#f0f0f0', fg='#2c3e50').pack(side='left')
        elif isinstance(table_data, list):
            tk.Label(summary_frame, text=f"Filas: {len(table_data)}",
                     font=("Arial", 10, "bold"), bg='#f0f0f0', fg='#2c3e50').pack(side='left')
        else:
            tk.Label(summary_frame, text="Resultado",
                     font=("Arial", 10, "bold"), bg='#f0f0f0', fg='#2c3e50').pack(side='left')

        poly_keys = ['polinomio_str', 'polinomios_por_tramo', 'polinomios_base']
        has_poly = False
        poly_text = None
        if isinstance(table_data, dict):
            for k in poly_keys:
                if k in table_data and table_data[k]:
                    has_poly = True
                    if k in ['polinomios_por_tramo', 'polinomios_base']:
                        entries = table_data[k]
                        if isinstance(entries, (list, tuple)):
                            poly_text = "\n\n".join(str(e) for e in entries)
                            break
                    else:
                        poly_text = str(table_data[k])
                        break

        if has_poly and poly_text:
            btn_frame = tk.Frame(parent_frame, bg='#f0f0f0')
            btn_frame.pack(fill='x', padx=10, pady=(0, 10))
            tk.Button(btn_frame, text="Ver polinomio(s) completos", font=("Arial", 10, "bold"),
                      bg='#8e44ad', fg='white', padx=10, pady=4, cursor='hand2',
                      command=lambda txt=poly_text: self.show_polynomials_modal(txt)).pack(side='right')
            
    def show_polynomials_modal(self, pol_text):
        modal = tk.Toplevel(self)
        modal.title("Polinomio(s) completos")
        modal.geometry("700x500")

        txt = scrolledtext.ScrolledText(modal, wrap='none', font=("Courier", 10))
        txt.pack(fill='both', expand=True, padx=10, pady=10)
        txt.insert('1.0', pol_text)
        txt.config(state='disabled')

        def copy_all():
            self.clipboard_clear()
            self.clipboard_append(pol_text)
            messagebox.showinfo("Copiado", "Polinomio(s) copiado(s) al portapapeles.")

        btn_frame = tk.Frame(modal)
        btn_frame.pack(fill='x', padx=10, pady=(0, 10))
        tk.Button(btn_frame, text="Copiar todo", command=copy_all,
                  bg='#3498db', fg='white').pack(side='right')
    
    def build_run_summary(self, method_name, result, error_type):
        """
        Extrae un resumen de la ejecución de un método de raíces:
        raíz aproximada, número de iteraciones y error final.
        Supone que el último elemento de la tupla 'result' es la tabla de iteraciones.
        """

        table_data = result[-1] if isinstance(result, tuple) else result

        if not isinstance(table_data, list) or not table_data:
            return None

        last_row = table_data[-1]
        if not isinstance(last_row, (list, tuple)):
            return None

        n_cols = len(last_row)

        # Formatos esperados de tablas (si coinciden, mejor):
        # Bisección:        [Iter, A, F(A), Pm, F(Pm), B, F(B), Error]
        # Regla Falsa:      [Iter, A, F(A), B, F(B), Xr, F(Xr), Error]
        # Newton:           [Iter, x, f(x), f'(x), Error]
        # Secante:          [Iter, x_{i-1}, x_i, f(x_i), Error]
        # Punto Fijo:       [Iter, x, g(x), Error]
        # Raíces Múltiples: [Iter, x, f(x), f'(x), f''(x), Error]

        x_col_map = {
            "Bisección": 3,        # Pm
            "Regla Falsa": 5,      # Xr
            "Newton": 1,           # x
            "Secante": 2,          # x_i
            "Punto Fijo": 1,       # x
            "Raíces Múltiples": 1  # x
        }

        err_col_map = {
            "Bisección": 7,        # Error Abs.
            "Regla Falsa": 7,      # Error
            "Newton": 4,           # Error
            "Secante": 4,          # Error
            "Punto Fijo": 3,       # Error
            "Raíces Múltiples": 5  # Error
        }

        # Valores por defecto (plan B) si algo falla:
        # - raíz ≈ columna 1 (la segunda)
        # - error ≈ última columna
        default_x_col = 1 if n_cols > 1 else 0
        default_err_col = n_cols - 1

        # Elegir índices
        x_col = x_col_map.get(method_name, default_x_col)
        err_col = err_col_map.get(method_name, default_err_col)

        # Asegurar que están dentro de rango; si no, usar plan B
        if x_col >= n_cols:
            x_col = default_x_col
        if err_col >= n_cols:
            err_col = default_err_col

        try:
            approx_root = float(last_row[x_col])
        except Exception:
            # último intento: tomar la segunda columna que haya
            try:
                approx_root = float(last_row[default_x_col])
            except Exception:
                return None

        try:
            final_error = abs(float(last_row[err_col]))
        except Exception:
            # si falla, probamos con la última columna
            try:
                final_error = abs(float(last_row[default_err_col]))
            except Exception:
                return None

        iterations = len(table_data)

        return {
            "method": method_name,
            "root": approx_root,
            "iterations": iterations,
            "final_error": final_error,
            "error_type": error_type
        }

    def auto_run_other_root_methods(self, current_method, f_str, a, b, tol, max_iter, error_type, eval_grid):
        """
        Ejecuta automáticamente los demás métodos de raíces con la misma función f(x),
        el mismo intervalo [a, b] y la misma tolerancia / iteraciones, sin mostrar sus tablas,
        solo para llenar el informe comparativo.
        """

        # Valores por defecto si algo no vino definido
        if tol is None:
            tol = 1e-4
        if max_iter is None:
            max_iter = 50

        # Definir f(x) a partir de la cadena f_str
        def f(x):
            return eval(f_str, {
                "np": np, "x": x, "math": math,
                "sin": np.sin, "cos": np.cos, "tan": np.tan,
                "exp": np.exp, "log": np.log, "sqrt": np.sqrt
            })

        for method_name in ROOT_METHODS:
            if method_name == current_method:
                continue  # ya lo ejecutó el usuario

            method_info = METHODS[method_name]["module"]
            package = METHODS[method_name].get("package", "")

            if isinstance(method_info, tuple):
                module_name, function_name = method_info
            else:
                module_name = function_name = method_info

            if package:
                full_module = f"Python.{package}.{module_name}"
            else:
                full_module = f"Python.{module_name}"

            try:
                module = importlib.import_module(full_module)
                func = getattr(module, function_name)
            except Exception:
                # Si no se puede importar, se salta este método
                continue

            try:
                sig = inspect.signature(func)
            except Exception:
                continue

            params = list(sig.parameters.keys())
            SKIP_PARAMS = {'show_report', 'eval_grid', 'auto_compare', 'error_type'}
            core_params = [p for p in params if p not in SKIP_PARAMS]

            args = []
            kwargs = {}

            # Si el primer parámetro es f, lo agregamos
            idx = 0
            if core_params and core_params[0] == 'f':
                args.append(f)
                idx = 1

            # Construir argumentos según nombre de parámetro
            for p in core_params[idx:]:
                if p in ('lower_bound', 'a'):
                    args.append(a)
                elif p in ('upper_bound', 'b'):
                    args.append(b)
                elif p in ('x0', 'x_inicial', 'x_ini'):
                    x0 = (a + b) / 2.0
                    args.append(x0)
                elif p in ('x1', 'x_inicial2', 'x1_ini'):
                    x1 = b
                    args.append(x1)
                elif p == 'tolerance':
                    args.append(tol)
                elif p in ('max_iterations', 'n_iter', 'iteraciones'):
                    args.append(max_iter)
                else:
                    # Fallback: usar el punto medio del intervalo
                    args.append((a + b) / 2.0)

            # Parámetros opcionales por kwargs
            if 'error_type' in sig.parameters:
                kwargs['error_type'] = error_type
            if 'show_report' in sig.parameters:
                kwargs['show_report'] = False  # no queremos más ventanas
            if 'eval_grid' in sig.parameters:
                try:
                    kwargs['eval_grid'] = int(eval_grid)
                except Exception:
                    kwargs['eval_grid'] = 500
            if 'auto_compare' in sig.parameters:
                kwargs['auto_compare'] = False

            # Ejecutar el método y guardar resumen
            try:
                result = func(*args, **kwargs)
                summary = self.build_run_summary(method_name, result, error_type)
                if summary:
                    self.run_history.append(summary)
            except Exception:
                # Si algo falla en este método, simplemente no se añade
                continue

    
    def show_comparison_report(self, error_type):
        """
        Muestra un informe comparativo entre todos los métodos de raíces que se han ejecutado
        en esta sesión con el mismo tipo de error (rel/abs/cond).
        Identifica y resalta cuál fue el mejor método.
        """

        runs = [r for r in self.run_history if r["error_type"] == error_type]

        if not runs:
            messagebox.showinfo(
                "Informe",
                f"No hay ejecuciones registradas con tipo de error '{error_type}'."
            )
            return

        # Mejor método: menor error final; si empatan, menos iteraciones
        best = min(runs, key=lambda r: (r["final_error"], r["iterations"]))

        win = tk.Toplevel(self)
        win.title(f"Informe comparativo - Error: {error_type}")
        win.geometry("700x400")
        win.configure(bg='#f0f0f0')

        title = tk.Label(
            win,
            text=f"INFORME COMPARATIVO (error: {error_type})",
            font=("Arial", 14, "bold"),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        title.pack(pady=(10, 10))

        frame = tk.Frame(win, bg='#f0f0f0')
        frame.pack(fill='both', expand=True, padx=10, pady=10)

        columns = ("Método", "Raíz aprox.", "Iteraciones", "Error final")

        tree = ttk.Treeview(frame, columns=columns, show='headings', height=10)
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=150, anchor='center')

        for r in runs:
            vals = (
                r["method"],
                f"{r['root']:.6g}",
                r["iterations"],
                f"{r['final_error']:.3e}"
            )
            item_id = tree.insert('', 'end', values=vals)
            if r is best:
                tree.item(item_id, tags=('best',))

        tree.tag_configure('best', background='#d5f5e3')

        vscroll = ttk.Scrollbar(frame, orient='vertical', command=tree.yview)
        tree.configure(yscrollcommand=vscroll.set)

        tree.grid(row=0, column=0, sticky='nsew')
        vscroll.grid(row=0, column=1, sticky='ns')

        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        info_frame = tk.Frame(win, bg='#f0f0f0')
        info_frame.pack(fill='x', padx=10, pady=(0, 10))

        msg = (
            f"Mejor método (según error '{error_type}'):\n"
            f"- {best['method']} con error final ≈ {best['final_error']:.3e} "
            f"en {best['iterations']} iteraciones."
        )

        tk.Label(
            info_frame,
            text=msg,
            font=("Arial", 10, "bold"),
            bg='#f0f0f0',
            fg='#2c3e50',
            justify='left'
        ).pack(anchor='w')

        tk.Button(
            win,
            text="Cerrar",
            font=("Arial", 10, "bold"),
            bg='#95a5a6',
            fg='white',
            padx=15,
            pady=5,
            cursor='hand2',
            command=win.destroy
        ).pack(pady=(0, 10))

    # 🔹 NUEVO: informe comparando TIPOS de error para UN método
    def show_error_type_report(self, method_name, error_runs):
        """
        Muestra un informe comparando los tipos de error (abs, rel, cond)
        para UN solo método de raíces.
        """
        # Elegir el mejor: menor error final, luego menos iteraciones
        best = min(error_runs, key=lambda r: (r["final_error"], r["iterations"]))

        win = tk.Toplevel(self)
        win.title(f"Comparación de tipos de error - {method_name}")
        win.geometry("700x400")
        win.configure(bg='#f0f0f0')

        title = tk.Label(
            win,
            text=f"COMPARACIÓN DE TIPOS DE ERROR - {method_name.upper()}",
            font=("Arial", 14, "bold"),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        title.pack(pady=(10, 10))

        frame = tk.Frame(win, bg='#f0f0f0')
        frame.pack(fill='both', expand=True, padx=10, pady=10)

        columns = ("Tipo de error", "Raíz aprox.", "Iteraciones", "Error final")

        tree = ttk.Treeview(frame, columns=columns, show='headings', height=10)
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=150, anchor='center')

        for r in error_runs:
            vals = (
                r["error_type"],
                f"{r['root']:.6g}",
                r["iterations"],
                f"{r['final_error']:.3e}"
            )
            item_id = tree.insert('', 'end', values=vals)
            if r is best:
                tree.item(item_id, tags=('best',))

        tree.tag_configure('best', background='#d6eaf8')

        vscroll = ttk.Scrollbar(frame, orient='vertical', command=tree.yview)
        tree.configure(yscrollcommand=vscroll.set)

        tree.grid(row=0, column=0, sticky='nsew')
        vscroll.grid(row=0, column=1, sticky='ns')

        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        info_frame = tk.Frame(win, bg='#f0f0f0')
        info_frame.pack(fill='x', padx=10, pady=(0, 10))

        msg = (
            f"Mejor tipo de error para {method_name}:\n"
            f"- '{best['error_type']}' con error final ≈ {best['final_error']:.3e} "
            f"en {best['iterations']} iteraciones."
        )

        tk.Label(
            info_frame,
            text=msg,
            font=("Arial", 10, "bold"),
            bg='#f0f0f0',
            fg='#2c3e50',
            justify='left'
        ).pack(anchor='w')

        tk.Button(
            win,
            text="Cerrar",
            font=("Arial", 10, "bold"),
            bg='#95a5a6',
            fg='white',
            padx=15,
            pady=5,
            cursor='hand2',
            command=win.destroy
        ).pack(pady=(0, 10))

    def plot_root_method(self, method_name, result):
        """Grafica f(x) y la sucesión de aproximaciones x_n para varios métodos de raíces."""

        if not hasattr(self, 'last_function_str') or self.last_function_str is None:
            messagebox.showerror("Error", "No se encontró la función para graficar.")
            return

        func_str = self.last_function_str

        def f(x):
            return eval(func_str, {
                "np": np, "x": x, "math": math,
                "sin": np.sin, "cos": np.cos, "tan": np.tan,
                "exp": np.exp, "log": np.log, "sqrt": np.sqrt
            })

        table_data = result[-1] if isinstance(result, tuple) else result
        if not isinstance(table_data, list) or len(table_data) == 0:
            messagebox.showerror("Error", "No hay datos de iteraciones para graficar.")
            return

        x_col_map = {
            "Bisección": 3,
            "Regla Falsa": 5,
            "Newton": 1,
            "Secante": 2,
            "Punto Fijo": 1,
            "Raíces Múltiples": 1
        }

        if method_name not in x_col_map:
            messagebox.showerror("Error", f"No se ha configurado graficación para el método: {method_name}")
            return

        x_col = x_col_map[method_name]

        try:
            # CONVERSIÓN A FLOAT AQUÍ
            x_aprox = [float(row[x_col]) for row in table_data]
        except Exception:
            messagebox.showerror("Error", "No se pudo extraer la columna de aproximaciones x_n.\nRevisa el formato de la tabla.")
            return

        # Intervalo para graficar
        if method_name in ["Bisección", "Regla Falsa"] and \
        hasattr(self, 'last_a') and hasattr(self, 'last_b') and \
        self.last_a is not None and self.last_b is not None:
            a = self.last_a
            b = self.last_b
        else:
            xmin = min(x_aprox)
            xmax = max(x_aprox)
            if xmin == xmax:
                a = xmin - 1
                b = xmax + 1
            else:
                margen = (xmax - xmin) * 0.4
                a = xmin - margen
                b = xmax + margen

        xs = np.linspace(a, b, 400)
        ys = [f(x) for x in xs]

        win = tk.Toplevel(self)
        win.title(f"Gráfica - Método de {method_name}")
        win.geometry("700x500")

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axhline(0, color='black', linewidth=0.8)
        ax.plot(xs, ys, label='f(x)')

        try:
            ax.scatter(x_aprox, [f(x) for x in x_aprox],
                    color='red', marker='o', label='Aproximaciones x_n')
        except Exception:
            pass

        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.set_title(f'Método de {method_name}')
        ax.grid(True)
        ax.legend()

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

if __name__ == "__main__":
    app = App()
    app.mainloop()
