T = readtable('506.xlsx')
symbol = T.Properties.VariableNames(1:end);
dailyReturn = T;
p = Portfolio('AssetList',symbol,'RiskFreeRate',0.01/252);
p = estimateAssetMoments(p, dailyReturn);
p = setDefaultConstraints(p); 
w1 = estimateMaxSharpeRatio(p);

[risk1, ret1] = estimatePortMoments(p, w1)

f = figure;
tabgp = uitabgroup(f); % Define tab group
tab1 = uitab(tabgp,'Title','Efficient Frontier Plot'); % Create tab
ax = axes('Parent', tab1);
% Extract asset moments from portfolio and store in m and cov
[m, cov] = getAssetMoments(p); 
scatter(ax,sqrt(diag(cov)), m,'oc','filled'); % Plot mean and s.d.
xlabel('Risk')
ylabel('Expected Return')
text(sqrt(diag(cov))+0.0003,m,symbol,'FontSize',7); % Label ticker names

hold on;
[risk2, ret2]  = plotFrontier(p,10);
plot(risk1,ret1,'p','markers',15,'MarkerEdgeColor','k',...
                'MarkerFaceColor','y');
hold off