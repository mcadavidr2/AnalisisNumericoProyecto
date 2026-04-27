%simpson13: Calcula la integral definida en el intervalo [a,b] para la
%función f, por el método de simpson 1/3 compuesto, con n puntos, con n par.
function [val] = simpson13(a, b, n, fstr)
    syms x
    f = str2sym(fstr); % Recibe la función como string
    h = abs(b - a) / n;
    f0 = double(subs(f, a));
    fn = double(subs(f, b));
    sum = f0 + fn;
    for i = 1:n-1
        fi = double(subs(f, a + i * h));
        if mod(i, 2) == 0
            sum = sum + 2 * fi;
        else
            sum = sum + 4 * fi;
        end
    end
    val = sum * h / 3;
end