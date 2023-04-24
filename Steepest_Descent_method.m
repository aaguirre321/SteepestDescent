%% Steepest Descent Method

%% Information and set up

g = @(x) ((x(1))^2 + x(2) - 11)^2 + (x(1)+ (x(2))^2 - 7)^2;
grad_g = @(x) [4*(x(1))^3 + 4*x(1)*x(2) - 42*x(1)+2*(x(2))^2-14;
               4*(x(2))^3 + 4*x(1)*x(2) - 26*x(2)+2*(x(1))^2-22];

x = [-1.0;
      1.0];
    
tol = 1e-7;                 % tolerance
max_iter = 100;              % max number of iterations

%% Steepest Descent's Method
i = 1;                      % iteration count
fprintf('i\tx\t\ty\t\terror\t\tg(x,y)\n');          % for display
fprintf('%d\t%.9f\t%.9f\t%d\t\t%.9f\n',0,x(1),x(2),0,g(x))
while( i <= max_iter) 
   
    d= -grad_g(x);
    
    alpha = 1;
    s = 0.95;
    t = 0.45;
    
    while(g(x + alpha*d) > g(x) - alpha*t*(norm(grad_g(x)))^2)
        alpha = alpha * s;
    end
    
    z = alpha * d;
    x = x + z;
  
    
    % check stopping condition
    inf_error = max(abs(z));
    
    % display information
    fprintf('%d\t%.9f\t%.9f\t%.9f\t%.9f\n',i,x(1),x(2),inf_error,g(x));  
    
    if(inf_error < tol)
        break;
    end
    
    % increase iteration count
    i = i + 1;
    
    
end

%% Display Information
if( i <= max_iter )         % successful
    fprintf('\nSteepest Descent Method approximated the solution x = (%.9f, %.9f) after %d iterations.\n\n',x(1),x(2),i);
else                        % not successful 
    fprintf('\nSteepest Descent Method did not converge within the tolerance in %d iterations.\n\n',max_iter)
end