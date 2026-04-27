import math
import sympy as sp
import re

def conversion(expr):
    expr = re.sub(r'(?<!\w)e', 'E', expr)
    expr = re.sub(r'\bln\b', 'log', expr)
    expr = re.sub(r'\bsin\b', 'sin', expr)
    expr = re.sub(r'\bcos\b', 'cos', expr)
    sympy_expr = sp.sympify(expr)
    converted_expr = str(sympy_expr).replace('E', 'math.exp(1)')
    converted_expr = converted_expr.replace('exp(', 'math.exp(')
    converted_expr = converted_expr.replace('log(', 'math.log(')
    converted_expr = converted_expr.replace('sin(', 'math.sin(')
    converted_expr = converted_expr.replace('cos(', 'math.cos(')
    return converted_expr

def secante(function_str, x0, x1, tolerance, max_iterations):
    f_str = conversion(function_str)

    def evaluate_expression(x):
        return eval(f_str)

    previous_x = x0
    current_x = x1
    previous_f = evaluate_expression(previous_x)
    current_f = evaluate_expression(current_x)
    iteration_number = 0
    results_matrix = []
    absolute_error = "-"
    relative_error = "-"

    results_matrix.append([iteration_number, previous_x, previous_f, absolute_error, relative_error])
    iteration_number = 1
    results_matrix.append([iteration_number, current_x, current_f, absolute_error, relative_error])

    while iteration_number <= max_iterations:
        denominator = current_f - previous_f
        if denominator == 0:
            raise ZeroDivisionError("División por cero en el denominador de la fórmula secante.")

        next_x = current_x - current_f * (current_x - previous_x) / denominator
        next_f = evaluate_expression(next_x)

        absolute_error = abs(next_x - current_x)
        relative_error = abs(next_x - current_x) / abs(next_x) if next_x != 0 else float('inf')

        results_matrix.append([iteration_number + 1, next_x, next_f, absolute_error, relative_error])

        if absolute_error < tolerance:
            break

        previous_x = current_x
        current_x = next_x
        previous_f = current_f
        current_f = next_f

        iteration_number += 1

    return results_matrix