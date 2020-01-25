% Input of the returns of different stocks over a period of 10 years
stockA_returns = [0.0063 0.0015 0.01861 0.0356 0.1011 0.0911 0.0981 0.1009 0.0670 0.1819];
stockB_returns = [0.0066 0.00762 -0.0248 -0.0551 0.0112 0.0019 0.7891 0.0912 0.0781 0.0911];
stockC_returns = [0.0107 -0.0684 0.02876 0.0320 0.1181 -0.01123 -0.00121 0.01231 0.0791 0.0812];
stockD_returns = [0.0234 -0.0753 0.08761 0.0315 0.1039 -0.1009 -0.0121 0.0978 0.0782 0.1012];

tic;
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

stdDevA = std(stockA_returns);
stdDevB = std(stockB_returns);
stdDevC = std(stockC_returns);
stdDevD = std(stockD_returns);

excessReturnsA = findExcessReturns(stockA_returns, arithmeticMean(1));
excessReturnsB = findExcessReturns(stockB_returns, arithmeticMean(2));
excessReturnsC = findExcessReturns(stockC_returns, arithmeticMean(3));
excessReturnsD = findExcessReturns(stockD_returns, arithmeticMean(4));

excessReturns = [excessReturnsA; excessReturnsB; excessReturnsC; excessReturnsD];

varianceCovarianceMatrix = excessReturns*excessReturns';
varianceCovarianceMatrix = 0.1*varianceCovarianceMatrix; % Divinding the elements of the variance-covariance matrix with 10 the number of observations

syms x1 x2 x3 x4 mu % these four variables represent the weights associated to the portfolio

weights = [x1; x2; x3; x4]; % this is a matrix which holds the portfolio weights

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
    objectiveFunction = alphaFirst*f1 + alphaSecond*f2 + mu*(x1+x2+x3+x4-1)^2 + 0.001*mu*(x1^2 + x2^2 + x3^2 + x4^2);

    x0 = [0.80;0.70;0.30;0.30];
    n = 1;
    epsilon = 10^-6;
    x_new = x0;
    x_old = x0;

    mu_value = 10; % Initialization value for mu
    T = table;
    while mu_value < 10^8
        
        mu_value = 10^n;
        for counter = 1:100
            if counter ~= 1 && (findSmallEnough(x_old,x_new,epsilon))
                break;
            end

            objectiveFunction = subs(objectiveFunction, mu, mu_value);
            grad_f = gradient(objectiveFunction);
            dk = -subs(grad_f,{x1,x2,x3,x4},{x_old(1),x_old(2),x_old(3),x_old(4)});

            alpha = 10^-4;
            neta = 0.9;
            lambda = 1/5;

            x_new = x_old + lambda*dk;

            f_new = subs(objectiveFunction,{x1,x2,x3,x4},{x_new(1),x_new(2),x_new(3),x_new(4)});
            g_new = subs(grad_f,{x1,x2,x3,x4},{x_new(1),x_new(2),x_new(3),x_new(4)});

            f_old = subs(objectiveFunction,{x1,x2,x3,x4},{x_old(1),x_old(2),x_old(3),x_old(4)});
            g_old = subs(grad_f,{x1,x2,x3,x4},{x_old(1),x_old(2),x_old(3),x_old(4)});

            while (f_new - f_old) > (alpha*lambda*dk'*g_old) || (dk'*g_new) < (neta*dk'*g_old)
                lambda = lambda/5;
                x_new = x_old + lambda*dk;
                f_new = subs(objectiveFunction,{x1,x2,x3,x4},{x_new(1),x_new(2),x_new(3),x_new(4)});
                g_new = subs(grad_f,{x1,x2,x3,x4},{x_new(1),x_new(2),x_new(3),x_new(4)});
            end
            
            x_new = x_new + lambda*dk; % --> this is a redundant step as we already have the value we want
            x_old = x_new;
            
            x_new(1) = vpa(x_new(1),6);
            x_new(2) = vpa(x_new(2),6);
            x_new(3) = vpa(x_new(3),6);
            x_new(4) = vpa(x_new(4),6);
            f_new = subs(objectiveFunction,{x1,x2,x3,x4},{x_new(1), x_new(2), x_new(3), x_new(4)});
            x_t = table(n, mu_value, double(vpa(x_new(1),6)),double(vpa(x_new(2),6)), ...
                    double(vpa(x_new(3),6)),double(vpa(x_new(4),6))); % ,sum
            T = [T; x_t];
        end
        n = n + 1;
    end

    T.Properties.VariableNames = {'Iterations' 'Mu_Value' 'x_1' 'x_2' 'x_3' 'x_4'}; %  'Constraint_Equals_1'
    disp(T);
    toc;
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
        if (abs((x_new(i)-x_old(i))/(x_old(i)))) < epsilon
            smallEnough = smallEnough*true;
        else 
            smallEnough = false;
        end
    end
end