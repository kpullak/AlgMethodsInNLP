% Input of the returns of different stocks over a period of 10 years
stockA_returns = [0.0063 0.0015 0.01861 0.0356 0.1011 0.0911 0.0981 0.1009 0.0670 0.1819];
stockB_returns = [0.0066 0.00762 -0.0248 -0.0551 0.0112 0.0019 0.7891 0.0912 0.0781 0.0911];
stockC_returns = [0.0107 -0.0684 0.02876 0.0320 0.1181 -0.01123 -0.00121 0.01231 0.0791 0.0812];
stockD_returns = [0.0234 -0.0753 0.08761 0.0315 0.1039 -0.1009 -0.0121 0.0978 0.0782 0.1012];

stock_returns = [stockA_returns; stockB_returns; stockC_returns; stockD_returns];

% geometricMean of the returns of the stock of a company over years
geoMean = zeros(4,1);
for i = 1:4
    geoMean(i) = findGeoMean(stock_returns(i,:));
end

% arithmeticMean of the returns of the stock of a company over years
arithmeticMean = zeros(4,1);
for i = 1:4
    arithmeticMean(i) = mean(stock_returns(i,:));
end

% here we will be evaluating the excess Returns matrix (which is: x-mean)
excessReturnsA = findExcessReturns(stockA_returns, arithmeticMean(1));
excessReturnsB = findExcessReturns(stockB_returns, arithmeticMean(2));
excessReturnsC = findExcessReturns(stockC_returns, arithmeticMean(3));
excessReturnsD = findExcessReturns(stockD_returns, arithmeticMean(4));

excessReturns = [excessReturnsA; excessReturnsB; excessReturnsC; excessReturnsD];

% here we will be evaluating the variance-covariance matrix
varianceCovarianceMatrix = excessReturns*excessReturns';
varianceCovarianceMatrix = 0.1*varianceCovarianceMatrix; % Divinding the elements of the variance-covariance matrix with 10 the number of observations

syms x1 x2 x3 x4 mu % these four variables represent the weights associated to the portfolio

weights = [x1; x2; x3; x4]; % this is a matrix which holds the portfolio weights

% formulating the functions for the returns and risk
portfolioReturn = transpose(weights)*geoMean;
portfolioVariance = 0.5*transpose(weights)*varianceCovarianceMatrix*weights;

%defined the objective functions:
f1 = -portfolioReturn;
f2 = portfolioVariance;

% alpha is the weight which tells us the relative importance of
% risk/reward we are targeting at 
% (eg; some one might think that they have to give 80% relative importance
% to risk over returns; someone might think they have to give 50-50
% importance to both risk and returns).

for alphas = 0:.1:1
    
    alphaFirst = alphas;
    alphaSecond = 1-alphas;
    
    X = ['Alpha1 is -> ', num2str(alphaFirst),' Alpha2 is -> ', num2str(alphaSecond)];
    disp(X);

    % constraints: x1+x2+x3+x4 = 1
    % generating the penalty function which would be the input to the exterior penalty method
    objectiveFunction = alphaFirst*f1 + alphaSecond*f2 + mu*(x1+x2+x3+x4-1)^2 + 10^(-1)*mu*(x1^2 + x2^2 + x3^2 + x4^2); % + 10^(-3)*mu*(x1^2 + x2^2 + x3^2 + x4^2)
    x = [x1; x2; x3; x4];
    
    x0 = [-0.10;-1.30;-0.8;-0.25];
    n = 1;
    epsilon = 10^-6;
    x_old = x0;
    
    mu_value = 10; % Initialization value for mu
    T = table;    
    while mu_value < 10^6
        
        mu_value = 10^n;
        objectiveFunction = subs(objectiveFunction, mu, mu_value);
        grad_f = gradient(objectiveFunction);
        hessian_f = hessian(objectiveFunction,[x1,x2,x3,x4]);
        grad_f_value = evaluateFunction(grad_f,x_old);
        hessian_f_value = evaluateFunction(hessian_f,x_old);
        x_new = x_old - hessian_f_value\grad_f_value; % calculation of x1
        
        while (~findSmallEnough(x_old,x_new, 10^-8))
            asd = 1;
            while (norm(hessian_f_value\grad_f_value) > 10^-6)
                x_old = x_new;
                grad_f_value = evaluateFunction(grad_f,x_old);
                hessian_f_value = evaluateFunction(hessian_f,x_old);
                x_new = x_old - hessian_f_value\grad_f_value;
            end
        end
        
%         grad_f_value = evaluateFunction(grad_f,x_old);
%         hessian_f_value = evaluateFunction(hessian_f,x_old);
%         x_new = x_old - hessian_f_value\grad_f_value; 
        n = n + 1; 
        x_t = table(n, mu_value, double(vpa(x_new(1),2)), ...
                double(vpa(x_new(2),2)), double(vpa(x_new(3),2)), double(vpa(x_new(4),2)));
        T = [T; x_t];
    end

    T.Properties.VariableNames = {'Iterations' 'Mu_Value' 'x_1' 'x_2' 'x_3' 'x_4'}; %  'Constraint_Equals_1'
    disp(T);
end

% this function is used to find the geometric mean of the rate of return
% for a particular stock over the years
function geometricMean = findGeoMean(stockA_returns)
    geometricMean = 1;
    for n = 1 : length(stockA_returns)
       geometricMean = geometricMean*(stockA_returns(n)+1);
    end
    geometricMean = geometricMean^(1/length(stockA_returns)) - 1;
end

% this function is used to find the excess returns matrix which is
% basically the value of (return - geometricMean of return)
function excessReturns = findExcessReturns(stockA_returns, geoMeanA)
    excessReturns = zeros(1,10);
    for n = 1 : length(stockA_returns)
        excessReturns(n) = stockA_returns(n) - geoMeanA;
    end
end

function smallEnough = findSmallEnough(x_old,x_new, epsilon)

    smallEnough = true;
    for i = 1:4
        if (abs((x_new(i)-x_old(i)))) < epsilon %(x_old(i))
            smallEnough = smallEnough*true;
        else 
            smallEnough = false;
        end
    end
end

function f = evaluateFunction(objectiveFunction,x)
    syms x1 x2 x3 x4
    f = subs(objectiveFunction,{x1,x2,x3,x4},{x(1),x(2),x(3),x(4)});
end