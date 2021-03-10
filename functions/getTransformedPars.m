function [posteriormean,posteriorstd,P] = getTransformedPars(ep_mean,ep_cov,numPars,paramtransformfun)

%% sample from transformed posterior and project to original parameter space


X = randn(10000, numPars);

X = bsxfun(@plus, X * chol(ep_cov), ep_mean);
P = paramtransformfun(X);

posteriormean = mean(P, 2); %#ok<NASGU>
posteriorstd=std(P,0,2);




end

