%Iterativos: Calcula la solución del sistema Ax=b, con una condición
%inicial dada x0, y una tolerancia dada Tol, así como el número de
%iteraciones máximas deseadas niter; dependiendo el método
%elegido: Jacobi o Gauss Seidel; se elige 0,1  respectivamente en met.

function [E,s] = Iterativos(x0,A,b,Tol,niter,met)
    c=0;
    error=Tol+1;
    while error>Tol && c<niter
        x1=NewJacobiSeid(x0,A,b,met);
        E(c+1)=norm(x1-x0,'inf');
        error=E(c+1);
        x0=x1;
        c=c+1;
    end
    if error < Tol
        s=x0;
        n=c;
        s
        fprintf('es una aproximación de la solución del sistmea con una tolerancia= %f',Tol)
    else 
        s=x0;
        n=c;
        fprintf('Fracasó en %f iteraciones',niter) 
    end
end