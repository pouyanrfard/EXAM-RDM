function [mu_trunc] = calcuateTruncatedMean_group(mu,sigma,a)
%CALCUATETRUNCATEDMEANS Summary of this function goes here
%   Detailed explanation goes here

mu_trunc=nan(size(mu));

for i=1:length(mu)

    alpha=(a-mu(i))/sigma(i);
    Z=(1-normcdf(alpha,0,1));

    posDensity=1-normcdf(0,mu(i),sigma(i));
       
    mu_trunc(i)=(mu(i)+sigma(i)*(normpdf(alpha,0,1)/Z))*posDensity;
    

end

end

