% Puntos de interpolación originales
x_puntos = [1, 2, 3, 4];
y_puntos = [-15, 11, 39, 11];

% Coeficientes de diferencias divididas
coeficientes = [-15.0, 26.0, 1.0, -9.6667];

% Generar valores de x para la gráfica
x_vals = linspace(0, 5, 100);
y_vals = zeros(size(x_vals));

% Evaluar el polinomio de Newton manualmente
for i = 1:length(x_vals)
    x = x_vals(i);
    term1 = coeficientes(1);
    term2 = coeficientes(2) * (x - x_puntos(1));
    term3 = coeficientes(3) * (x - x_puntos(1)) * (x - x_puntos(2));
    term4 = coeficientes(4) * (x - x_puntos(1)) * (x - x_puntos(2)) * (x - x_puntos(3));
    y_vals(i) = term1 + term2 + term3 + term4;
end

% Graficar
figure;
plot(x_vals, y_vals, 'b-', 'LineWidth', 1.5); % Polinomio
hold on;
plot(x_puntos, y_puntos, 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r'); % Puntos
title('Polinomio de Newton (Exacto)');
xlabel('x');
ylabel('P(x)');
legend('Polinomio interpolante', 'Puntos originales', 'Location', 'northwest');
grid on;
xlim([0, 5]);
ylim([-20, 50]);

% Verificación en los puntos originales
y_check = zeros(size(x_puntos));
for i = 1:length(x_puntos)
    x = x_puntos(i);
    term1 = coeficientes(1);
    term2 = coeficientes(2) * (x - x_puntos(1));
    term3 = coeficientes(3) * (x - x_puntos(1)) * (x - x_puntos(2));
    term4 = coeficientes(4) * (x - x_puntos(1)) * (x - x_puntos(2)) * (x - x_puntos(3));
    y_check(i) = term1 + term2 + term3 + term4;
end
disp('Valores del polinomio en los puntos originales:');
disp(y_check);