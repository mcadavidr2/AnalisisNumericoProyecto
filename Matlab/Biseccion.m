%Bisección: se ingresa el valor inicial y final del intervalo (xi, xs), la tolerancia del error (Tol), el máximo número de iteraciones (niter) y la función f como string

function [s, E, fm] = Biseccion(xi, xs, Tol, niter, fstr)
    syms x
    f = str2sym(fstr); % Convertir string a función simbólica
    fi = double(subs(f, xi));
    fs = double(subs(f, xs));
    if fi == 0
        s = xi;
        E = 0;
        fprintf('%f es raiz de f(x)\n', xi)
    elseif fs == 0
        s = xs;
        E = 0;
        fprintf('%f es raiz de f(x)\n', xs)
    elseif fs * fi < 0
        c = 0;
        xm = (xi + xs) / 2;
        fm(c + 1) = double(subs(f, xm));
        fe = fm(c + 1);
        E(c + 1) = Tol + 1;
        error = E(c + 1);
        while error > Tol && fe ~= 0 && c < niter
            if fi * fe < 0
                xs = xm;
                fs = double(subs(f, xs));
            else
                xi = xm;
                fi = double(subs(f, xi));
            end
            xa = xm;
            xm = (xi + xs) / 2;
            fm(c + 2) = double(subs(f, xm));
            fe = fm(c + 2);
            E(c + 2) = abs(xm - xa);
            error = E(c + 2);
            c = c + 1;
        end
        if fe == 0
            s = xm;
            fprintf('%f es raiz de f(x)\n', xm)
        elseif error < Tol
            s = xm;
            fprintf('%f es una aproximación de una raiz de f(x) con una tolerancia= %f\n', xm, Tol)
        else
            s = xm;
            fprintf('Fracasó en %f iteraciones\n', niter)
        end
    else
        fprintf('El intervalo es inadecuado\n')
    end
end