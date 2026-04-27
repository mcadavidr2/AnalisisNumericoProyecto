%SOR: Calcula la solución del sistema
%Ax=b con base en una condición inicial x0,mediante el método Gauss Seidel (relajado), depende del valor de w 
%entre (0,2)

function [E, s, resultTable] = SOR(x0, A, b, Tol, niter, w)
    c = 0;
    error = Tol + 1;
    D = diag(diag(A));
    L = -tril(A, -1);
    U = -triu(A, +1);
    resultTable = [];
    
    while error > Tol && c < niter
        T = inv(D - w * L) * ((1 - w) * D + w * U);
        C = w * inv(D - w * L) * b;
        x1 = T * x0 + C;
        E(c + 1) = norm(x1 - x0, inf);
        error = E(c + 1);
        x0 = x1;
        c = c + 1;
        resultTable = [resultTable; c, error];
    end
    
    if error < Tol
        s = x0;
        fprintf('Es una aproximación de la solución del sistema con una tolerancia = %f\n', Tol);
    else
        s = x0;
        fprintf('Fracasó en %d iteraciones\n', niter);
    end
end