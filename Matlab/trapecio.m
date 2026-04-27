%trapecio: Calcula la integral definida en el intervalo [a,b] para la
%función f, por el método del trapecio compuesto, con n puntos.
function [val] = trapecio(a, b, n, fstr)
    syms x
    f = str2sym(fstr);
    h = abs(b - a) / n;
    f0 = double(subs(f, a));
    fn = double(subs(f, b));
    sum = f0 + fn;
    for i = 1:n-1
        fi = double(subs(f, a + i * h));
        sum = sum + 2 * fi;
    end
    val = sum * h / 2;
end