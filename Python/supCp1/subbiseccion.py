from tabulate import tabulate
import numpy as np

def biseccion(f, lower_bound, upper_bound, tolerance, max_iterations):
    """
    Encuentra una raíz de la función f en el intervalo [lower_bound, upper_bound] utilizando el método de bisección.

    Args:
        f (function): Función a evaluar.
        lower_bound (float): Límite inferior del intervalo.
        upper_bound (float): Límite superior del intervalo.
        tolerance (float): Tolerancia deseada.
        max_iterations (int): Número máximo de iteraciones.

    Returns:
        list: Tabla de iteraciones con columnas [Iteración, a, f(a), pm, f(pm), b, f(b), Error Abs.]
    """

    f_a = f(lower_bound)
    f_b = f(upper_bound)

    if f_a * f_b >= 0:
        raise ValueError("f(lower_bound) y f(upper_bound) deben tener signos opuestos")

    matriz = []
    midpoint = (lower_bound + upper_bound) / 2
    f_midpoint = f(midpoint)
    error = abs(upper_bound - lower_bound)
    iteration_count = 0

    matriz.append([iteration_count, lower_bound, f_a, midpoint, f_midpoint, upper_bound, f_b, error])

    while error > tolerance and iteration_count < max_iterations:
        if f_a * f_midpoint < 0:
            upper_bound = midpoint
            f_b = f(upper_bound)
        else:
            lower_bound = midpoint
            f_a = f(lower_bound)

        previous_midpoint = midpoint
        midpoint = (lower_bound + upper_bound) / 2
        f_midpoint = f(midpoint)
        error = abs(midpoint - previous_midpoint)
        iteration_count += 1

        matriz.append([iteration_count, lower_bound, f_a, midpoint, f_midpoint, upper_bound, f_b, error])

    return matriz

if __name__ == '__main__':
    try:
        function_str = input("Ingrese la función f(x) como una expresión en Python (por ejemplo, x**3 - 4*x + 1): ")
        lower_bound = float(input("Ingresa el extremo izquierdo a: "))
        upper_bound = float(input("Ingresa el extremo derecho b: "))
        tolerance = float(input("Ingresa la tolerancia: "))
        max_iterations = int(input("Ingresa el número máximo de iteraciones: "))

        # Define la función f usando eval() y una función lambda
        f = lambda x: eval(function_str)

        matriz = biseccion(f, lower_bound, upper_bound, tolerance, max_iterations)

        print(tabulate(matriz, headers=["Iteración", "a", "f(a)", "pm", "f(pm)", "b", "f(b)", "Error Abs."], tablefmt="fancy_grid"))
        print(f"\nRaíz aproximada encontrada después de {len(matriz)-1} iteraciones.")

    except Exception as e:
        print(f"Error: Entrada no válida. Compruebe la definición de la función y las entradas numéricas. {e}")
