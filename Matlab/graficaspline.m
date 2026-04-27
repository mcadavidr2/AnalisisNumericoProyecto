clc 
close all

if ~exist('Tabla', 'var') || ~exist('x', 'var') || ~exist('y', 'var')
    error('Faltan variables necesarias: Tabla, x o y no están definidas en el workspace.');
end

xpol0=1:0.001:2;
xpol1=2:0.001:3;
xpol2=3:0.001:4;

% p0=2*xpol0+28;
% p1=4*xpol1+24;
% p2=1*xpol2+33;

% p0=Tabla(1,1)*xpol0.^3+Tabla(1,2)*xpol0.^2+Tabla(1,3)*xpol0+Tabla(1,4);
% p1=Tabla(2,1)*xpol1.^3+Tabla(2,2)*xpol1.^2+Tabla(2,3)*xpol1+Tabla(2,4);
% p2=Tabla(3,1)*xpol2.^3+Tabla(3,2)*xpol2.^2+Tabla(3,3)*xpol2+Tabla(3,4);

p0=Tabla(1,1)*xpol0.^2+Tabla(1,2)*xpol0+Tabla(1,3);
p1=Tabla(2,1)*xpol1.^2+Tabla(2,2)*xpol1+Tabla(2,3);
p2=Tabla(3,1)*xpol2.^2+Tabla(3,2)*xpol2+Tabla(3,3);

plot(x,y,'r*')
hold on
grid on
plot(xpol0,p0,'b',xpol1,p1,'g',xpol2,p2,'m')

figure;
plot(E, '-o');
title('Convergencia del método');
xlabel('Iteración');
ylabel('Error');
grid on;