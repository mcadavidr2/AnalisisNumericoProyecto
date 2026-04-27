function InteractiveNumericalMethods()
    clc;
    disp('=== SISTEMA INTERACTIVO DE MÉTODOS NUMÉRICOS ===');
    
    while true
        fprintf('\nMENÚ PRINCIPAL:\n');
        fprintf('1. Solución de sistemas lineales\n');
        fprintf('2. Interpolación\n');
        fprintf('3. Aproximación por splines\n');
        fprintf('4. Solución de ecuaciones no lineales\n');
        fprintf('5. Factorización LU\n');
        fprintf('6. Eliminación Gaussiana\n');
        fprintf('7. Integración Numérica\n'); % Nueva opción
        fprintf('8. Salir\n');
        
        opcion = input('Seleccione una opción: ');
        
        switch opcion
            case 1
                menuSistemasLineales();
            case 2
                menuInterpolacion();
            case 3
                menuSplines();
            case 4
                menuEcuacionesNoLineales();
            case 5
                menuLU();
            case 6
                menuGauss();
            case 7
                menuIntegracionNumerica(); % Llamada al nuevo menú
            case 8
                disp('Saliendo del sistema...');
                return;
            otherwise
                disp('Opción no válida. Intente nuevamente.');
        end
    end
end

function menuSistemasLineales()
    clc;
    fprintf('\nMÉTODOS PARA SISTEMAS LINEALES:\n');
    fprintf('1. Jacobi (matricial)\n');
    fprintf('2. Gauss-Seidel (matricial)\n');
    fprintf('3. Jacobi (componente por componente)\n');
    fprintf('4. Gauss-Seidel (componente por componente)\n');
    fprintf('5. SOR (sobre-relajación)\n');
    fprintf('6. Regresar\n');
    
    opcion = input('Seleccione un método: ');
    if opcion == 6, return; end
    
    A = input('Ingrese la matriz A (ej. [1 2; 3 4]): ');
    b = input('Ingrese el vector b (ej. [5; 6]): ');
    x0 = input('Ingrese el vector inicial x0 (ej. [0; 0]): ');
    Tol = input('Ingrese la tolerancia (ej. 1e-6): ');
    niter = input('Ingrese el número máximo de iteraciones: ');
    
    if size(A, 1) ~= size(A, 2)
        disp('Error: La matriz A debe ser cuadrada.');
        return;
    end
    if length(b) ~= size(A, 1)
        disp('Error: El vector b debe tener la misma cantidad de filas que A.');
        return;
    end
    
    if opcion == 5
        w = input('Ingrese el parámetro de relajación w (0 < w < 2): ');
        [E, s, resultTable] = SOR(x0, A, b, Tol, niter, w);
        disp('Tabla de iteraciones (Iteración | Error):');
        disp(array2table(resultTable, 'VariableNames', {'Iteracion', 'Error'}));
    else
        met = opcion - 1;
        if opcion <= 2
            [E, s] = MatJacobiSeid(x0, A, b, Tol, niter, mod(met,2));
        else
            [E, s] = Iterativos(x0, A, b, Tol, niter, mod(met,2));
        end
    end
    
    figure;
    plot(E);
    title('Convergencia del método');
    xlabel('Iteración');
    ylabel('Error');
    grid on;
end

function menuInterpolacion()
    clc;
    fprintf('\nMÉTODOS DE INTERPOLACIÓN:\n');
    fprintf('1. Newton (diferencias divididas)\n');
    fprintf('2. Lagrange\n');
    fprintf('3. Vandermonde\n');
    fprintf('4. Regresar\n');
    
    opcion = input('Seleccione un método: ');
    if opcion == 4, return; end
    
    x = input('Ingrese el vector x (ej. [1; 2; 3]): ');
    y = input('Ingrese el vector y (ej. [4; 5; 6]): ');
    
    switch opcion
        case 1
            Tabla = Newtonint(x, y);
            disp('Tabla de diferencias divididas:');
            disp(Tabla);
        case 2
            pol = Lagrange(x, y);
            disp('Coeficientes del polinomio (de mayor a menor grado):');
            disp(pol);
        case 3
            n = length(x);
            A = zeros(n, n);
            for i = 1:n
                A(:,i) = x.^(n-i);
            end
            a = A\y;
            disp('Coeficientes del polinomio (de mayor a menor grado):');
            disp(a);
            
            xpol = min(x):0.01:max(x);
            p = polyval(a, xpol);
            figure;
            plot(x, y, 'r*', xpol, p, 'b-');
            title('Interpolación por Vandermonde');
            grid on;
    end
end

function menuSplines()
    clc;
    disp('Gráfica de Splines Cuadráticos');
    x = input('Ingrese el vector x (ej. [1; 2; 3; 4]): ');
    y = input('Ingrese el vector y (ej. [2; 3; 5; 4]): ');
    % Aquí deberías calcular la tabla de coeficientes de los splines
    Tabla = calcularTablaSplines(x, y); % Debes implementar esta función
    E = []; % Si tienes errores de iteración, calcula aquí o elimina si no aplica

    assignin('base', 'Tabla', Tabla);
    assignin('base', 'x', x);
    assignin('base', 'y', y);
    assignin('base', 'E', E);

    disp('Ejecutando graficación de splines...');
    run('graficaspline.m');
end

function menuEcuacionesNoLineales()
    clc;
    fprintf('\nMÉTODOS PARA ECUACIONES NO LINEALES:\n');
    fprintf('1. Bisección\n');
    fprintf('2. Newton\n');
    fprintf('3. Regresar\n');
    
    opcion = input('Seleccione un método: ');
    if opcion == 3, return; end
    
    switch opcion
        case 1
            f = input('Ingrese la función como cadena (ej. ''x^2 - 2''): ', 's');
            xi = input('Ingrese el límite inferior del intervalo: ');
            xs = input('Ingrese el límite superior del intervalo: ');
            Tol = input('Ingrese la tolerancia: ');
            niter = input('Ingrese el número máximo de iteraciones: ');
            
            syms x;
            f_sym = str2sym(f);
            
            [s, E, fm] = Biseccion(xi, xs, Tol, niter, f);
            disp('Raíz encontrada:');
            disp(s);
            disp('Errores por iteración:');
            disp(E);
            
            figure;
            subplot(2,1,1);
            fplot(f_sym, [xi xs]);
            hold on;
            plot(s, subs(f_sym,s), 'ro');
            title('Función y raíz encontrada');
            grid on;
            
            subplot(2,1,2);
            plot(E);
            title('Convergencia del método');
            xlabel('Iteración');
            ylabel('Error');
            grid on;
            
        case 2
            f = input('Ingrese la función f(x) (ej. "sin(2*x)-(x/3)^3+0.1"): ', 's');
            x0 = input('Ingrese el valor inicial x0: ');
            Tol = input('Ingrese la tolerancia: ');
            niter = input('Ingrese el número máximo de iteraciones: ');
            
            [n, xn, fm, dfm, E] = newton(x0, Tol, niter, f);
            
            figure;
            subplot(2,1,1);
            fplot(str2sym(f), [xn-2 xn+2]);
            hold on;
            plot(xn, fm(end), 'ro');
            title('Método de Newton');
            grid on;
            
            subplot(2,1,2);
            plot(E);
            title('Convergencia');
            xlabel('Iteración');
            ylabel('Error');
            grid on;
    end
end

function menuLU()
    clc;
    fprintf('\nFACTORIZACIÓN LU:\n');
    fprintf('1. Sin pivoteo\n');
    fprintf('2. Con pivoteo parcial\n');
    fprintf('3. Regresar\n');
    
    opcion = input('Seleccione un método: ');
    if opcion == 3, return; end
    
    A = input('Ingrese la matriz A: ');
    b = input('Ingrese el vector b: ');
    n = size(A,1);
    
    [x, L, U] = LU(A, b, n, opcion-1);
    
    disp('Solución x:');
    disp(x);
    disp('Matriz L:');
    disp(L);
    disp('Matriz U:');
    disp(U);
    
    figure;
    subplot(1,3,1);
    spy(A);
    title('Matriz A');
    subplot(1,3,2);
    spy(L);
    title('Matriz L');
    subplot(1,3,3);
    spy(U);
    title('Matriz U');
end

function menuGauss()
    clc;
    fprintf('\nELIMINACIÓN GAUSSIANA:\n');
    fprintf('1. Sin pivoteo\n');
    fprintf('2. Pivoteo parcial\n');
    fprintf('3. Pivoteo total\n');
    fprintf('4. Regresar\n');
    
    opcion = input('Seleccione un método: ');
    if opcion == 4, return; end
    
    A = input('Ingrese la matriz A: ');
    b = input('Ingrese el vector b: ');
    n = size(A,1);
    
    if opcion == 3
        [x, mark] = GaussPiv(A, b, n, 2);
        disp('Reordenamiento de variables (mark):');
        disp(mark);
    else
        x = GaussPiv(A, b, n, opcion-1);
    end
    
    disp('Solución x:');
    disp(x);
    
    figure;
    plot(1:n, x, '-o');
    title('Solución del sistema');
    xlabel('Variable');
    ylabel('Valor');
    grid on;
end

function menuIntegracionNumerica()
    clc;
    fprintf('\nMÉTODOS DE INTEGRACIÓN NUMÉRICA:\n');
    fprintf('1. Trapecio\n');
    fprintf('2. Simpson (1/3 compuesto)\n'); % Nueva opción
    fprintf('3. Regresar\n');
    
    opcion = input('Seleccione un método: ');
    if opcion == 3, return; end
    
    f = input('Ingrese la función a integrar como cadena (ej. ''sin(x)''): ', 's');
    a = input('Ingrese el límite inferior de integración: ');
    b = input('Ingrese el límite superior de integración: ');
    
    syms x;
    f_sym = str2sym(f);
    f_num = matlabFunction(f_sym); % Convertir a función numérica
    
    switch opcion
        case 1
            % Método del Trapecio
            n = input('Ingrese el número de subintervalos: ');
            x_vals = linspace(a, b, n+1);
            y_vals = f_num(x_vals);
            I = trapz(x_vals, y_vals);
            disp('Integral aproximada (Trapecio):');
            disp(I);
            
        case 2
            % Método de Simpson (1/3 compuesto)
            n = input('Ingrese el número de subintervalos (debe ser par): ');
            if mod(n, 2) ~= 0
                disp('Error: El número de subintervalos debe ser par para el método de Simpson.');
                return;
            end
            I = Simpson(a, b, n); % Llamada al nuevo archivo Simpson
            disp('Integral aproximada (Simpson 1/3 compuesto):');
            disp(I);
    end
    
    % Graficar la función
    figure;
    fplot(f_sym, [a b]);
    title('Función a integrar');
    grid on;
end