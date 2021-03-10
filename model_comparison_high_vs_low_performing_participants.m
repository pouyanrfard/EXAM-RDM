% The script "model_comparison_high_vs_low_performing_participants.m" conducts 
% the model comparison used in "Fard et al. (2021), Spatiotemporal 
% Modeling of Response Consistency with Incoherent Motion Stimuli in 
% Perceptual Decision Making, In submission".
% The script first loads all the data from computational models fitted to
% the behavioral data for each of high- and low-performing participants. 
% This may include parameters related to the posterior distributions 
% (mean and standard deviation for the posterior distribution) estimated for 
% each participant in each coherence level and the model marginal likelihood. 
% Afterwards, the script uses Bayesian model comparison (using VBA toolbox) 
% to provide the metrics for explanatory power of each computational model
% (DDM and EXaM) to explain the behavioral data in each coherence level. 
% The resulting model evidence variables (protected exceedance probability
% and model frequency) are visualized to compare the explanatory power of
% the models. In addition, the script computes the mean and standard 
% deviations of estimated posterior parameter distributions and conducts
% a t-test to provide significant differences between the estimated
% parameter values of EXaM between high- and low-performing participants. 

%dependencies: 
% 1. VBA Toolbox: The VBA toolbox must be installed and added to MATLAB 
% path to conduct the Bayesian model comparison please refer to:
% https://github.com/MBB-team/VBA-toolbox
% 2. The model fits: the model fits are prepared using the library 
% pyEPABC % (http://github.com/sbitzer/pyEPABC) in a % separate Python 
% script. The outcomes of model fits are stored in the % folder 
% "model_fit_data". 

%Hint: Please make sure that this script is placed next to the folder
%"model_fit_data", "behavioral_data", "analysis_data", "experiment_design", 
% and "functions".
%The script may take a few minutes to run depending on your system
%specifications.
%
% Author: Pouyan Rafieifard (January 2021)

clear all;
close all;
clc;
addpath('functions');

%% Load the data from the high- and low-performing participants

load('analysis_data//subject_ranking.mat','bestSubjsNo','bestSubjs','worstSubjsNo','worstSubjs');

%% Intialization of the variables used in the analysis

% Nubmer of participants
numSubs=length(bestSubjsNo);
                      
% Number of experimental conditions
numConds=1; %only zero coherence level

% Stimulus feature modes (0 for DDM [without stimulus information], 12 for
% EXaM [with complete stimulus information])
Apers=[0,12];
% Number of stimulus feature information
numApers=length(Apers);

% Number of model fitting runs
numRuns=5;
% Number of trials per coherence levels
numTrials=200;
% Number of model parameters
numPars=9;

%A matrix containing the model evidence of the fitted models for each
%participant, coherence level, stimulus feature type, and model fitting run
logML_set_high=nan(numSubs,numConds,numApers,numRuns); %for high-performing participants
logML_set_low=nan(numSubs,numConds,numApers,numRuns); %for low-performing participants

%A matrix containing the posterior means of the fitted models for each
%participant, model fitting run, stimulus feature type, and parameter in
%the zero % coherence level
posMeans_zero_high=zeros(numSubs,numRuns,numApers,numPars+1);%for high-performing participants
posMeans_zero_low=zeros(numSubs,numRuns,numApers,numPars+1);%for low-performing participants
%A matrix containing the posterior standard deviations of the fitted models 
% for each participant, model fitting run, stimulus feature type, and 
% parameter in the zero % coherence level
posStds_zero_high=zeros(numSubs,numRuns,numApers,numPars+1);%for high-performing participants
posStds_zero_low=zeros(numSubs,numRuns,numApers,numPars+1);%for low-performing participants

%A matrix containing the posterior covariance matrix of the fitted models 
% for each participant, model fitting run, stimulus feature type, and 
% parameter in the zero % coherence level
posCovs_zero_high=zeros(numSubs,numRuns,numApers,numPars+1,numPars+1);%for high-performing participants
posCovs_zero_low=zeros(numSubs,numRuns,numApers,numPars+1,numPars+1);%for low-performing participants

%% Loading the model fitting data into related data objects
% In a loop over participants model fitting data is loaded for each
% coherence level and model fitting run (5 runs in total) and for both
% fitted models (DDM and EXaM)


for subIdx=1:numSubs
    subIdx
    for condIdx=1:numConds
           for runIdx=1:numRuns        
                                                  
                    %%%%%%%%%%%%%%%%%%
                    % HIGH-PERFORMING PARTICIPANTS
                    % Load the model fitting data for the EXaM 
                    %%%%%%%%%%%%%%%%%%
                    aperIdx=2;
    
                    %load the model fitting data for 0% coherence level
                    if(condIdx==1)
                        load(strcat('model_fit_data//fitResults_sub_',num2str(bestSubjsNo(subIdx)),...
                        '_cond_',num2str(condIdx),'_aper_12_run_',num2str(runIdx),'_dc_cond_final_16.mat'));
                            
                        %assign the model fitting variables including
                        %posterior mean, posterior standard deviation and
                        %posterior covarianc matrix derived from EP-ABC
                        %process
                        posMeans_zero_high(subIdx,runIdx,aperIdx,:)=ep_mean;
                        posStds_zero_high(subIdx,runIdx,aperIdx,:)=diag(ep_cov);
                        posCovs_zero_high(subIdx,runIdx,aperIdx,:,:)=ep_cov;
                        
                    end
                                        
                    %assign the model evidence for the combination of
                    %partipant, coherence level, feature type, and fitting
                    %run
                    logML_set_high(subIdx,condIdx,aperIdx,runIdx)=ep_logml;

                    %%%%%%%%%%%%%%%%%%
                    % % HIGH-PERFORMING PARTICIPANTS
                    % Load the model fitting data for the DDM 
                    %%%%%%%%%%%%%%%%%%
                    aperIdx=1;
                    
                    %load the model fitting data for 0% coherence level
                    if(condIdx==1)
                        load(strcat('model_fit_data//fitResults_sub_',num2str(bestSubjsNo(subIdx)),...
                        '_cond_',num2str(condIdx),'_aper_0_run_',num2str(runIdx),'_rw_cond_final_16_3.mat'));
                        
                        %assign the model fitting variables including
                        %posterior mean, posterior standard deviation and
                        %posterior covarianc matrix derived from EP-ABC
                        %process
                        posMeans_zero_high(subIdx,runIdx,aperIdx,:)=ep_mean;
                        posStds_zero_high(subIdx,runIdx,aperIdx,:)=diag(ep_cov);
                        posCovs_zero_high(subIdx,runIdx,aperIdx,:,:)=ep_cov;
                    end
                    
                    %assign the model evidence for the combination of
                    %partipant, coherence level, feature type, and fitting
                    %run
                    logML_set_high(subIdx,condIdx,aperIdx,runIdx)=ep_logml;
                   
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    
                    
                                                  
                    %%%%%%%%%%%%%%%%%%
                    % LOW-PERFORMING PARTICIPANTS
                    % Load the model fitting data for the EXaM 
                    %%%%%%%%%%%%%%%%%%
                    aperIdx=2;
    
                    %load the model fitting data for 0% coherence level
                    if(condIdx==1)
                        load(strcat('model_fit_data//fitResults_sub_',num2str(worstSubjsNo(subIdx)),...
                        '_cond_',num2str(condIdx),'_aper_12_run_',num2str(runIdx),'_dc_cond_final_16.mat'));
                            
                        %assign the model fitting variables including
                        %posterior mean, posterior standard deviation and
                        %posterior covarianc matrix derived from EP-ABC
                        %process
                        posMeans_zero_low(subIdx,runIdx,aperIdx,:)=ep_mean;
                        posStds_zero_low(subIdx,runIdx,aperIdx,:)=diag(ep_cov);
                        posCovs_zero_low(subIdx,runIdx,aperIdx,:,:)=ep_cov;
                        
                    end
                                        
                    %assign the model evidence for the combination of
                    %partipant, coherence level, feature type, and fitting
                    %run
                    logML_set_low(subIdx,condIdx,aperIdx,runIdx)=ep_logml;

                    %%%%%%%%%%%%%%%%%%
                    % % LOW-PERFORMING PARTICIPANTS
                    % Load the model fitting data for the DDM 
                    %%%%%%%%%%%%%%%%%%
                    aperIdx=1;
                    
                    %load the model fitting data for 0% coherence level
                    if(condIdx==1)
                        load(strcat('model_fit_data//fitResults_sub_',num2str(worstSubjsNo(subIdx)),...
                        '_cond_',num2str(condIdx),'_aper_0_run_',num2str(runIdx),'_rw_cond_final_16_3.mat'));
                        
                        %assign the model fitting variables including
                        %posterior mean, posterior standard deviation and
                        %posterior covarianc matrix derived from EP-ABC
                        %process
                        posMeans_zero_low(subIdx,runIdx,aperIdx,:)=ep_mean;
                        posStds_zero_low(subIdx,runIdx,aperIdx,:)=diag(ep_cov);
                        posCovs_zero_low(subIdx,runIdx,aperIdx,:,:)=ep_cov;
                    end
                    
                    %assign the model evidence for the combination of
                    %partipant, coherence level, feature type, and fitting
                    %run
                    logML_set_low(subIdx,condIdx,aperIdx,runIdx)=ep_logml;
                   
                    
                    
           end
    end
end

%% Calculate the average model evidence and paramter estimates accross runs

%calculate the model evidence accross five runs
logML_set_mean_high=mean(logML_set_high,4);%for high-performing participants
logML_set_mean_low=mean(logML_set_low,4);%for low-performing participants

%posterior parameter estimates for zero coherence level
%calculate the posterior parameter means accross five runs 
posMeans_zero_high=squeeze(mean(posMeans_zero_high,2));%for high-performing participants
posMeans_zero_low=squeeze(mean(posMeans_zero_low,2));%for low-performing participants
%calculate the posterior parameter standard deviations accross five runs
posStds_zero_high=squeeze(mean(posStds_zero_high,2));%for high-performing participants
posStds_zero_low=squeeze(mean(posStds_zero_low,2));%for low-performing participants
%calculate the posterior parameter covariance matrix accross five runs
posCovs_zero_high=squeeze(mean(posCovs_zero_high,2));%for high-performing participants
posCovs_zero_low=squeeze(mean(posCovs_zero_low,2));%for low-performing participants

%% Bayesian model selection, Exeedance probability and Model frequency
%2-way model comparison between DDM and EXaM

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%HIGH-PERFORMING PARTICIPANTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
numApers=2;
L=nan(numApers,numSubs,numConds);
%Create the model evidence matrix using the model evidences from DDM and
%ExaM
L(:,:,1)=[logML_set_high(:,1,1)';logML_set_high(:,1,2)']; 

%Initialize the matrix for protected exceedance probability
EP_2way_high=nan(numConds,numApers);%for high-performing participants
%Initialize the matrix for model frequency mean
MF_mean_2way_high=nan(numConds,numApers);%for high-performing participants
%Initialize the matrix for model frequency standard deviation
MF_std_2way_high=nan(numConds,numApers);%for high-performing participants

%internal variables for the VBA toolbox
options_4way.DisplayWin = 0;

%conduct the Bayesian model comparison using VBA Toolbox
%NOTE: VBA Toolbox must be installed and added to Matlab path
post = cell(numConds, 1);
ou = cell(numConds, 1);
for d = 1 : numConds
    [post{d, 1}, ou{d, 1}] = VBA_groupBMC(L(:, :, d), options_4way);
end

%intialize the variables for model comparison
[K, ~, ~] = size(L);
hp1 = nan(K, 1);
hp2 = nan(K, 1);

%Computing the final model comparison variables including exceedance
%probability and model frequency
for d = 1 : numConds
    out = ou{d};
    posterior = post{d};
    % calculate protected exceedance probability
    out.pep = out.ep.*(1-out.bor) + (K^-1)*out.bor;
    
    EP_2way_high(d,:)=out.pep;
    MF_mean_2way_high(d,:)=out.Ef;
    MF_std_2way_high(d,:)=sqrt([out.Vf(1),out.Vf(2)]);    %
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%LOW-PERFORMING PARTICIPANTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
numApers=2;
L=nan(numApers,numSubs,numConds);
%Create the model evidence matrix using the model evidences from DDM and
%ExaM
L(:,:,1)=[logML_set_low(:,1,1)';logML_set_low(:,1,2)']; 

%Initialize the matrix for protected exceedance probability
EP_2way_low=nan(numConds,numApers);%for low-performing participants
%Initialize the matrix for model frequency mean
MF_mean_2way_low=nan(numConds,numApers);%for low-performing participants
%Initialize the matrix for model frequency standard deviation
MF_std_2way_low=nan(numConds,numApers);%for low-performing participants

%internal variables for the VBA toolbox
options_4way.DisplayWin = 0;

%conduct the Bayesian model comparison using VBA Toolbox
%NOTE: VBA Toolbox must be installed and added to Matlab path
post = cell(numConds, 1);
ou = cell(numConds, 1);
for d = 1 : numConds
    [post{d, 1}, ou{d, 1}] = VBA_groupBMC(L(:, :, d), options_4way);
end

%intialize the variables for model comparison
[K, ~, ~] = size(L);
hp1 = nan(K, 1);
hp2 = nan(K, 1);

%Computing the final model comparison variables including exceedance
%probability and model frequency
for d = 1 : numConds
    out = ou{d};
    posterior = post{d};
    % calculate protected exceedance probability
    out.pep = out.ep.*(1-out.bor) + (K^-1)*out.bor;
    
    EP_2way_low(d,:)=out.pep;
    MF_mean_2way_low(d,:)=out.Ef;
    MF_std_2way_low(d,:)=sqrt([out.Vf(1),out.Vf(2)]);    %
end


%% Visualize the model comparison results accross all coherence levels for 
% all participants (Figure 4, and Figure S1)

figure,
subplot(2,2,1)

bar(EP_2way_high)
hold on
plot([0 10],[0.95 0.95],'r--','LineWidth',2);
ylabel('Exceedance Probability','FontWeight','bold');
xlim([0 3]);
ylim([0 1]);
set(gca,'XTickLabel',{'DDM','EXaM'});
set(gca,'FontWeight','bold');
title('High-performing participants, Exceedance Probability');
box off


subplot(2,2,2)
hold on
ylabel('Model Frequency','FontWeight','bold');
bar(MF_mean_2way_high)
plot([0 10],[0.5 0.5],'r--','LineWidth',2);
title('High-performing participants, Model Frequency');
xlim([0 3]);
ylim([0 1]);
set(gca,'XTick',1:2);
set(gca,'XTickLabel',{'DDM','EXaM'});
set(gca,'FontWeight','bold');

subplot(2,2,3)

bar(EP_2way_low)
hold on
plot([0 10],[0.95 0.95],'r--','LineWidth',2);
ylabel('Exceedance Probability','FontWeight','bold');
xlim([0 3]);
ylim([0 1]);
set(gca,'XTickLabel',{'DDM','EXaM'});
set(gca,'FontWeight','bold');
title('Low-performing participants, Exceedance Probability');
box off


subplot(2,2,4)
hold on
ylabel('Model Frequency','FontWeight','bold');
bar(MF_mean_2way_low)
plot([0 10],[0.5 0.5],'r--','LineWidth',2);
title('Low-performing participants, Model Frequency');
xlim([0 3]);
ylim([0 1]);
set(gca,'XTick',1:2);
set(gca,'XTickLabel',{'DDM','EXaM'});
set(gca,'FontWeight','bold');

%% Calculate the posterior means for the parameters for zero coherence level

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%HIGH-PERFORMING PARTICIPANTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Initialize the matrix for transformed posterior mean parameters
posMeansTrans_zero_high=zeros(numSubs,numApers,10);
%Initialize the matrix for transformed posterior standard deviation parameters
posStdsTrans_zero_high=zeros(numSubs,numApers,10);

%number of parameters
numPars=10;

%Name of the parameters
parnames = {'scale_0', 'scale_1','scalestd', 'bound', 'bias', 'biaststd','ndtmean', 'ndtspread', 'lapseprob','lapsetoprob'}

% defines transformation over parameters which allow to use Gaussian priors
% the function maps from Gaussian-distributed parameters to the original
% parameter values, it gets [nsamples, np]=size(Ptransformed) as input, but
% returns a matrix [np, nsamples] = size(P)
paramtransformfun = @(Ptrans) [...
Ptrans(:, 1), ...
Ptrans(:, 2), ...
exp(Ptrans(:, 3)),...
exp(Ptrans(:, 4)), ...
Ptrans(:, 5), ...
exp(Ptrans(:, 6)),... 
exp(Ptrans(:, 7)),... 
exp(Ptrans(:, 8)),... 
normcdf(Ptrans(:, 9)),...
normcdf(Ptrans(:, 10))]';

%Calculate the posteriormean based on the transformation provided 
for aperIdx=1:numApers
  
        for subIdx=1:numSubs
            
            ep_mean1=squeeze(posMeans_zero_high(subIdx,aperIdx,:));
            ep_cov1=squeeze(posCovs_zero_high(subIdx,aperIdx,:,:));
            
            %calculate the transofmred parameter estimates for each participant
            [posteriormean,posteriorstd,P] = getTransformedPars(ep_mean1',ep_cov1,...
                numPars,paramtransformfun);

            posMeansTrans_zero_high(subIdx,aperIdx,:)=posteriormean;
            posStdsTrans_zero_high(subIdx,aperIdx,:)=posteriorstd;

        end
 end

%Calculate the means of the censored distribution for the scale parameters
for aperIdx=1:2
    for parIdx=1:2
        [mu_trunc_i_j] = calcuateTruncatedMean_group(posMeansTrans_zero_high(:,aperIdx,parIdx),posStdsTrans_zero_high(:,aperIdx,parIdx),0);
        posMeansTrans_zero_high(:,aperIdx,parIdx)=mu_trunc_i_j;
   end
end

%Calculate the means and stadndard deviations over parameter posteriormeans
posMean_mean_zero_high=squeeze(mean(posMeansTrans_zero_high,1));
posMean_std_zero_high=squeeze(std(posMeansTrans_zero_high,0,1))/sqrt(numSubs);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%LOW-PERFORMING PARTICIPANTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Initialize the matrix for transformed posterior mean parameters
posMeansTrans_zero_low=zeros(numSubs,numApers,10);
%Initialize the matrix for transformed posterior standard deviation parameters
posStdsTrans_zero_low=zeros(numSubs,numApers,10);

%number of parameters
numPars=10;

%Name of the parameters
parnames = {'scale_0', 'scale_1','scalestd', 'bound', 'bias', 'biaststd','ndtmean', 'ndtspread', 'lapseprob','lapsetoprob'}

% defines transformation over parameters which allow to use Gaussian priors
% the function maps from Gaussian-distributed parameters to the original
% parameter values, it gets [nsamples, np]=size(Ptransformed) as input, but
% returns a matrix [np, nsamples] = size(P)
paramtransformfun = @(Ptrans) [...
Ptrans(:, 1), ...
Ptrans(:, 2), ...
exp(Ptrans(:, 3)),...
exp(Ptrans(:, 4)), ...
Ptrans(:, 5), ...
exp(Ptrans(:, 6)),... 
exp(Ptrans(:, 7)),... 
exp(Ptrans(:, 8)),... 
normcdf(Ptrans(:, 9)),...
normcdf(Ptrans(:, 10))]';

%Calculate the posteriormean based on the transformation provided 
for aperIdx=1:numApers
  
        for subIdx=1:numSubs
            
            ep_mean1=squeeze(posMeans_zero_low(subIdx,aperIdx,:));
            ep_cov1=squeeze(posCovs_zero_low(subIdx,aperIdx,:,:));
            
            %calculate the transofmred parameter estimates for each participant
            [posteriormean,posteriorstd,P] = getTransformedPars(ep_mean1',ep_cov1,...
                numPars,paramtransformfun);

            posMeansTrans_zero_low(subIdx,aperIdx,:)=posteriormean;
            posStdsTrans_zero_low(subIdx,aperIdx,:)=posteriorstd;

        end
 end

%Calculate the means of the censored distribution for the scale parameters
for aperIdx=1:2
    for parIdx=1:2
        [mu_trunc_i_j] = calcuateTruncatedMean_group(posMeansTrans_zero_low(:,aperIdx,parIdx),posStdsTrans_zero_low(:,aperIdx,parIdx),0);
        posMeansTrans_zero_low(:,aperIdx,parIdx)=mu_trunc_i_j;
   end
end

%Calculate the means and stadndard errors over parameter posteriormeans
posMean_mean_zero_low=squeeze(mean(posMeansTrans_zero_low,1));
posMean_std_zero_low=squeeze(std(posMeansTrans_zero_low,0,1))/sqrt(numSubs);


%% Perform ttest between posterior parameters from EXaM in 0% coherence level 
% for high- and low-performing participants

%Initialize the matrix for p-values (t-test comparison
pVals_zero=zeros(1,10);

condIdx=1; %zero coherence
aperIdx=2; %EXaM

for parIdx=1:numPars

    [h,p]=ttest(posMeansTrans_zero_high(:,aperIdx,parIdx),posMeansTrans_zero_low(:,aperIdx,parIdx));

    pVals_zero(parIdx)=p;

end

%% Create a table for the posterior parameter means accross zero coherence level
% (Table 1 and Table S1 for zero coherence level)

numPars_zero=10; %number of parameters
numUse=2; %number of models
%intialize the table cell structure 
parsTable_zero_4=cell(numPars+2,numUse+1);
%initialize the cell headers
parsTable_zero_4{1,1}='Parameters';
parsTable_zero_4{1,2}='EXaM (high-performing participants)';parsTable_zero_4{1,3}='EXaM (low-performing participants)';
parsTable_zero_4{2,1}='Coherence Levels';
parsTable_zero_4{2,2}='0';parsTable_zero_4{2,3}='0';

%write the parameter names inside the table cell
parnames={'scale_0', 'scale_1', 'scalestd', 'bound', 'bias', 'biaststd','ndtmean', 'ndtspread', 'lapseprob','lapsetoprob'};
for parIdx=1:numPars_zero
     parsTable_zero_4{parIdx+2,1}=parnames{parIdx};
end  

%write the parameter values inside the table
for parIdx=1:numPars_zero
   
        parsTable_zero_4{parIdx+2,2}=...
            sprintf('%.3f (%.3f)',posMean_mean_zero_high(2,parIdx),...
                                  posMean_std_zero_high(2,parIdx));

        if(pVals_zero(parIdx)<0.05 && pVals_zero(parIdx)>0.01) 
             parsTable_zero_4{parIdx+2,3}=...
             sprintf('%.3f* (%.3f)',posMean_mean_zero_low(2,parIdx),...
                                  posMean_std_zero_low(2,parIdx));
        elseif(pVals_zero(parIdx)<0.01)
             parsTable_zero_4{parIdx+2,3}=...
             sprintf('%.3f** (%.3f)',posMean_mean_zero_low(2,parIdx),...
                                  posMean_std_zero_low(2,parIdx));
       elseif(pVals_zero(parIdx)>0.05)
            parsTable_zero_4{parIdx+2,3}=...
             sprintf('%.3f (%.3f)',posMean_mean_zero_low(2,parIdx),...
                                  posMean_std_zero_low(2,parIdx));
        end

end

%display the table in the matlab command window
parsTable_zero_4
%wite the parameter table inside a Excel file
xlswrite('tables\\posterior_parameter_table_0_coherence_high_vs_low_performing.csv', parsTable_zero_4)

%% Save the analysis data
 save('analysis_data//model_comparison_high_vs_low_performing.mat');