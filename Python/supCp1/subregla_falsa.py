from tabulate import tabulate
import numpy as np

def false_position_method(f, lower_bound, upper_bound, tolerance, max_iterations):
    """
    Encuentra la raíz de una función f usando el método de posición falsa.

    Argumentos:
        f (función): La función para encontrar la raíz
        lower_bound (float): El límite inferior del intervalo.
        upper_bound (float): El límite superior del intervalo.
        tolerancia (float): La tolerancia para la raíz.
        max_iterations (int): El número máximo de iteraciones.

    Devuelve:
        tupla: Una tupla que contiene los valores finales de lower_bound y upper_bound, el número de iteraciones y la matriz de iteraciones.
    """
    f_lower_bound = f(lower_bound)
    f_upper_bound = f(upper_bound)

    if f_lower_bound * f_upper_bound > 0:
        print("Error: f(lower_bound) y f(upper_bound) deben tener signos opuestos.")
        return lower_bound, upper_bound, 0, []

    denominator = f_upper_bound - f_lower_bound
    if denominator == 0:
        print("Error: División por 0")
        return lower_bound, upper_bound, 0, []

    midpoint = (f_upper_bound * lower_bound - f_lower_bound * upper_bound) / denominator
    f_midpoint = f(midpoint)
    error = 1000
    iteration_count = 0
    iteration_data = []

    while error > tolerance and iteration_count < max_iterations:
        if f_upper_bound * f_midpoint < 0:
            lower_bound = midpoint
            f_lower_bound = f_midpoint
        else:
            upper_bound = midpoint
            f_upper_bound = f_midpoint

        previous_midpoint = midpoint
        denominator = f_upper_bound - f_lower_bound
        if denominator == 0:
            print("Error: División por 0")
            break
        midpoint = (f_upper_bound * lower_bound - f_lower_bound * upper_bound) / denominator
        f_midpoint = f(midpoint)
        error = abs(midpoint - previous_midpoint)
        iteration_count = iteration_count + 1

        iteration_data.append([iteration_count, lower_bound, f_lower_bound, midpoint, f_midpoint, upper_bound, f_upper_bound, error])

    return lower_bound, upper_bound, iteration_count, iteration_data


if __name__ == '__main__':
    function_str = input("Ingrese la función f(x) (e.g., x**3 - 7.51*x**2 + 18.4239*x - 14.8331): ")
    lower_bound = float(input("Ingresa el límite inferior: "))
    upper_bound = float(input("Ingresa el límite superior: "))
    tolerance = float(input("Ingresa la tolerancia: "))
    max_iterations = int(input("Ingresa el número máximo de iteraciones: "))

    def f(x):
        return eval(function_str)

    f_lower_bound = f(lower_bound)
    f_upper_bound = f(upper_bound)
    if f_lower_bound * f_upper_bound > 0:
        print("Error: f(lower_bound) y f(upper_bound) deben tener signos opuestos.")
        exit()

    lower_bound, upper_bound, iteration_count, iteration_data = false_position_method(f, lower_bound, upper_bound, tolerance, max_iterations)

    if iteration_count > 0:
        print(tabulate(iteration_data, headers=["Iter", "lower_bound", "f(lower_bound)", "midpoint", "f(midpoint)", "upper_bound", "f(upper_bound)", "Error"], tablefmt="fancy_grid"))
        print(f"\nRaiz encontrada en el intervalo [{lower_bound}, {upper_bound}] despues de {iteration_count} iteraciones.")
    else:
        print("\nNo se ha encontrado ninguna raíz dentro del intervalo y la tolerancia dados.")