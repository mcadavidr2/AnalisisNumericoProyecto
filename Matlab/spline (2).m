% Spline.m
% Spline: Calcula los coeficientes de los polinomios de interpolación de
% grado d (1, 2, 3) para el conjunto de n datos (x,y), mediante el método spline.
% Uso:
%   Tabla = Spline(x, y, d)
% Salida:
%   Tabla: matriz (n-1) x (d+1) con los coeficientes por tramo.
%          Para d=1: [a b] con p_i(x) = a*x + b   en [x_i, x_{i+1}]
%          Para d=2: [a b c] con p_i(x) = a*x^2 + b*x + c
%          Para d=3: [a b c d] con p_i(x) = a*x^3 + b*x^2 + c*x + d
%
% Nota:
% - Implementa las mismas ecuaciones de tu referencia:
%   d=1: interpola extremos de cada tramo.
%   d=2: además continuidad de 1ª derivada y condición de borde S''(x0)=0.
%   d=3: continuidad de 1ª y 2ª derivada, más condiciones naturales S''(x0)=S''(xn)=0.

function [Tabla] = Spline(x, y, d)
    x = x(:); y = y(:);
    n = length(x);

    if length(y) ~= n
        error('x e y deben tener la misma longitud.');
    end
    if any(diff(x) <= 0)
        error('x debe ser estrictamente creciente.');
    end
    if d ~= 1 && d ~= 2 && d ~= 3
        error('El grado d debe ser 1, 2 o 3.');
    end

    A = zeros((d+1)*(n-1));
    b = zeros((d+1)*(n-1), 1);

    cua = x.^2;
    cub = x.^3;

    % ---------- Lineal ----------
    if d == 1
        c = 1; h = 1;
        % Interpolación en x_i -> p_i(x_i) = y_i
        for i = 1:n-1
            A(i, c)   = x(i);
            A(i, c+1) = 1;
            b(i)      = y(i);
            c = c + 2; h = h + 1;
        end
        % Interpolación en x_{i+1} -> p_i(x_{i+1}) = y_{i+1}
        c = 1;
        for i = 2:n
            A(h, c)   = x(i);
            A(h, c+1) = 1;
            b(h)      = y(i);
            c = c + 2; h = h + 1;
        end

    % ---------- Cuadrático ----------
    elseif d == 2
        c = 1; h = 1;
        % Interpolación en x_i
        for i = 1:n-1
            A(i, c)   = cua(i);
            A(i, c+1) = x(i);
            A(i, c+2) = 1;
            b(i)      = y(i);
            c = c + 3; h = h + 1;
        end
        % Interpolación en x_{i+1}
        c = 1;
        for i = 2:n
            A(h, c)   = cua(i);
            A(h, c+1) = x(i);
            A(h, c+2) = 1;
            b(h)      = y(i);
            c = c + 3; h = h + 1;
        end
        % Continuidad de 1ª derivada en nodos internos
        c = 1;
        for i = 2:n-1
            % p_i'(x_i) = p_{i+1}'(x_i)
            A(h, c)     = 2*x(i);   % deriv a_i*x^2 -> 2*a_i*x
            A(h, c+1)   = 1;        % deriv b_i*x   -> b_i
            A(h, c+3)   = -2*x(i);
            A(h, c+4)   = -1;
            b(h)        = 0;
            c = c + 3; h = h + 1;
        end
        % Condición de borde: S''(x_0) = 0  -> 2*a_1 = 0
        A(h, 1) = 2;
        b(h)    = 0;

    % ---------- Cúbico (natural) ----------
    else % d == 3
        c = 1; h = 1;
        % Interpolación en x_i
        for i = 1:n-1
            A(i, c)   = cub(i);
            A(i, c+1) = cua(i);
            A(i, c+2) = x(i);
            A(i, c+3) = 1;
            b(i)      = y(i);
            c = c + 4; h = h + 1;
        end
        % Interpolación en x_{i+1}
        c = 1;
        for i = 2:n
            A(h, c)   = cub(i);
            A(h, c+1) = cua(i);
            A(h, c+2) = x(i);
            A(h, c+3) = 1;
            b(h)      = y(i);
            c = c + 4; h = h + 1;
        end
        % Continuidad de 1ª derivada en nodos internos
        c = 1;
        for i = 2:n-1
            % p_i'(x_i) = p_{i+1}'(x_i)
            A(h, c)     = 3*cua(i);
            A(h, c+1)   = 2*x(i);
            A(h, c+2)   = 1;
            A(h, c+4)   = -3*cua(i);
            A(h, c+5)   = -2*x(i);
            A(h, c+6)   = -1;
            b(h)        = 0;
            c = c + 4; h = h + 1;
        end
        % Continuidad de 2ª derivada en nodos internos
        c = 1;
        for i = 2:n-1
            % p_i''(x_i) = p_{i+1}''(x_i)
            A(h, c)     = 6*x(i);
            A(h, c+1)   = 2;
            A(h, c+4)   = -6*x(i);
            A(h, c+5)   = -2;
            b(h)        = 0;
            c = c + 4; h = h + 1;
        end
        % Condiciones naturales: S''(x_0)=0, S''(x_n)=0
        % En el primer tramo p_1''(x_0) = 6*a_1*x0 + 2*b_1 = 0
        A(h, 1) = 6*x(1);  A(h, 2) = 2;  b(h) = 0; h = h + 1;
        % En el último tramo p_{n-1}''(x_n) = 6*a_{n-1}*x_n + 2*b_{n-1} = 0
        % Índices del último bloque: c ya se incrementó por cada tramo; aquí usamos:
        % bloque final inicia en (4*(n-2)+1)
        c_end = 4*(n-2) + 1;
        A(h, c_end)   = 6*x(end);
        A(h, c_end+1) = 2;
        b(h)          = 0;
    end

    % Resolver (mejor que inv(A)*b)
    val = A \ b;

    % Reorganizar coeficientes por tramo
    Tabla = reshape(val, d+1, n-1).';
end
