%Newton: se ingresa el valor inicial (x0), la tolerancia del error (Tol), el máximo número de iteraciones (niter) y la función f como string

function [n, xn, fm, dfm, E, resultTable] = newton(x0, Tol, niter, fstr)
    syms x
    f = str2sym(fstr);           % Convertir string a función simbólica
    df = diff(f);                % Derivada simbólica
    c = 0;
    fm(c + 1) = double(subs(f, x0));
    fe = fm(c + 1);
    dfm = double(subs(df, x0));
    dfe = dfm;
    E(c + 1) = Tol + 1;
    error = E(c + 1);
    xn = x0;
    resultTable = []; % Inicializar tabla de resultados

    while error > Tol && dfe ~= 0 && c < niter
        xn = x0 - fe / dfe;
        fm(c + 2) = double(subs(f, xn));
        fe = fm(c + 2);
        dfm = double(subs(df, xn));
        dfe = dfm;
        E(c + 2) = abs((xn - x0) / xn); % Función de error
        error = E(c + 2);
        x0 = xn;
        c = c + 1;

        % Agregar resultados a la tabla
        resultTable = [resultTable; c, xn, fe, error];
    end

    n = c;
    if fe == 0
        fprintf('%f es raíz de f(x)\n', x0);
    elseif error < Tol
        fprintf('%f es una aproximación de una raíz de f(x) con una tolerancia = %f\n', x0, Tol);
    elseif dfe == 0
        fprintf('%f es una posible raíz múltiple de f(x)\n', x0);
    else
        fprintf('Fracasó en %d iteraciones\n', niter);
    end
end