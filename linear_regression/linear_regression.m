plot(x,y,'o')x = load('ex2x.dat');y = load('ex2y.dat');m = length(y);a = [0 0];x = [ones(m, 1), x];% Gradient descent% a = a - (c/m) * sum( h - y )* x% Reduction factorc = 0.07;% Total number of iterationslimit = 1500;for i = 1:limit   % Linear regression  % h = a0 + a1 * x  h = a * x';  % Gradient descent  % a = a - (c/m) * sum( (h - y) * x )  a = a - (c/m) * (h' - y)' * xendfor