% The script "model_comparison_all_participants.m" conducts the model
% comparison used in "Fard et al. (2021), Spatiotemporal 
% Modeling of Response Consistency with Incoherent Motion Stimuli in 
% Perceptual Decision Making, In submission".
% The script basically uses enriches the behavioral data for each
% participants with the dot counts related to each trial. This is done in
% preparation of the data that is used to fit the exact input models to the
% behavioral data.

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

%% Intialization of the variables used in the analysis

% The participant numbers who passed the training phase (see Methods in the
% manuscript for more details)
Subjects=[62,63,64,65,66,67,71,74,76,77,80,82,84,85,86,90,91,92,93,94,95,96,...
                          97,100,101,102,104,105,107,109,110,111,112,114,119,120,121,123,124,126,127,128,129,130];
% Nubmer of participants
numSubs=length(Subjects);
                      
% Condictions in the behavioral experiment, 1: 0% coherence, 2: 10%
% coherence, 3: 25% coherecen, 4: 35% coherence
Conds=[1,2,3,4];
% Number of experimental conditions
numConds=length(Conds);

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
logML_set=nan(numSubs,numConds,numApers,numRuns);

%A matrix containing the posterior means of the fitted models for each
%participant, model fitting run, stimulus feature type, and parameter in
%the zero % coherence level
posMeans_zero=zeros(numSubs,numRuns,numApers,numPars+1);
%A matrix containing the posterior standard deviations of the fitted models 
% for each participant, model fitting run, stimulus feature type, and 
% parameter in the zero % coherence level
posStds_zero=zeros(numSubs,numRuns,numApers,numPars+1);
%A matrix containing the posterior covariance matrix of the fitted models 
% for each participant, model fitting run, stimulus feature type, and 
% parameter in the zero % coherence level
posCovs_zero=zeros(numSubs,numRuns,numApers,numPars+1,numPars+1);

%A matrix containing the posterior means of the fitted models for each
%participant, model fitting run, stimulus feature type, and parameter in
%the all non-zero coherence levels
posMeans=zeros(numSubs,numConds,numApers,numRuns,numPars);
%A matrix containing the posterior standard deviations of the fitted models 
% for each participant, model fitting run, stimulus feature type, and 
% parameter in all non-zero coherence levels
posStds=zeros(numSubs,numConds,numApers,numRuns,numPars);
%A matrix containing the posterior standard deviations of the fitted models 
% for each participant, model fitting run, stimulus feature type, and 
% parameter in all non-zero coherence levels
posCovs=zeros(numSubs,numConds,numApers,numRuns,numPars,numPars);

%% Loading the model fitting data into related data objects
% In a loop over participants model fitting data is loaded for each
% coherence level and model fitting run (5 runs in total) and for both
% fitted models (DDM and EXaM)


for subIdx=1:numSubs
    subIdx
    for condIdx=1:numConds
           for runIdx=1:numRuns        
                subjectNo=Subjects(subIdx);
                    
                
                    %%%%%%%%%%%%%%%%%%
                    % Load the model fitting data for the EXaM
                    %%%%%%%%%%%%%%%%%%
                    aperIdx=2;
    
                    %load the model fitting data for 0% coherence level
                    if(condIdx==1)
                        load(strcat('model_fit_data//fitResults_sub_',num2str(Subjects(subIdx)),...
                        '_cond_',num2str(condIdx),'_aper_12_run_',num2str(runIdx),'_dc_cond_final_16.mat'));
                            
                        %assign the model fitting variables including
                        %posterior mean, posterior standard deviation and
                        %posterior covarianc matrix derived from EP-ABC
                        %process
                        posMeans_zero(subIdx,runIdx,aperIdx,:)=ep_mean;
                        posStds_zero(subIdx,runIdx,aperIdx,:)=diag(ep_cov);
                        posCovs_zero(subIdx,runIdx,aperIdx,:,:)=ep_cov;
                        
                    %load the model fitting data for 10%,25% and 35%
                    %coherence levels                    
                    else
                         load(strcat('model_fit_data//fitResults_sub_',num2str(Subjects(subIdx)),...
                        '_cond_',num2str(condIdx),'_aper_12_run_',num2str(runIdx),'_dc_final_16.mat'));

                            posMeans(subIdx,condIdx,aperIdx,runIdx,:)=ep_mean;
                            posStds(subIdx,condIdx,aperIdx,runIdx,:)=diag(ep_cov);
                            posCovs(subIdx,condIdx,aperIdx,runIdx,:,:)=ep_cov;    
                    end
                    
                                        
                    %assign the model evidence for the combination of
                    %partipant, coherence level, feature type, and fitting
                    %run
                    logML_set(subIdx,condIdx,aperIdx,runIdx)=ep_logml;

                    %%%%%%%%%%%%%%%%%%
                    % Load the model fitting data for the DDM
                    %%%%%%%%%%%%%%%%%%
                    aperIdx=1;
                    
                    %load the model fitting data for 0% coherence level
                    if(condIdx==1)
                        load(strcat('model_fit_data//fitResults_sub_',num2str(Subjects(subIdx)),...
                        '_cond_',num2str(condIdx),'_aper_0_run_',num2str(runIdx),'_rw_cond_final_16_3.mat'));
                        
                        %assign the model fitting variables including
                        %posterior mean, posterior standard deviation and
                        %posterior covarianc matrix derived from EP-ABC
                        %process
                        posMeans_zero(subIdx,runIdx,aperIdx,:)=ep_mean;
                        posStds_zero(subIdx,runIdx,aperIdx,:)=diag(ep_cov);
                        posCovs_zero(subIdx,runIdx,aperIdx,:,:)=ep_cov;
                                                               
                    %load the model fitting data for 10%,25% and 35%
                    %coherence levels 
                    else
                        load(strcat('model_fit_data//fitResults_sub_',num2str(Subjects(subIdx)),...
                        '_cond_',num2str(condIdx),'_aper_0_run_',num2str(runIdx),'_rw_final_14.mat'));
                    
                        posMeans(subIdx,condIdx,aperIdx,runIdx,:)=ep_mean;
                        posStds(subIdx,condIdx,aperIdx,runIdx,:)=diag(ep_cov);
                        posCovs(subIdx,condIdx,aperIdx,runIdx,:,:)=ep_cov;

                    end
                    
                    %assign the model evidence for the combination of
                    %partipant, coherence level, feature type, and fitting
                    %run
                    logML_set(subIdx,condIdx,aperIdx,runIdx)=ep_logml;
    
                    
           end
    end
end

%% Calculate the average model evidence and paramter estimates accross runs

%calculate the model evidence accross five runs
logML_set_mean=mean(logML_set,4);

%posterior parameter estimates for non-zero coherence levels
%calculate the posterior parameter means accross five runs 
posMeans=squeeze(mean(posMeans,4));
%calculate the posterior parameter standard deviations accross five runs
posStds=squeeze(mean(posStds,4));
%calculate the posterior parameter covariance matrix accross five runs
posCovs=squeeze(mean(posCovs,4));

%posterior parameter estimates for zero coherence level
%calculate the posterior parameter means accross five runs 
posMeans_zero=squeeze(mean(posMeans_zero,2));
%calculate the posterior parameter standard deviations accross five runs
posStds_zero=squeeze(mean(posStds_zero,2));
%calculate the posterior parameter covariance matrix accross five runs
posCovs_zero=squeeze(mean(posCovs_zero,2));


%% Bayesian model selection, Exeedance probability and Model frequency
%2-way model comparison between DDM and EXaM
numApers=2;
L=nan(numApers,numSubs,numConds);
%Create the model evidence matrix using the model evidences from DDM and
%ExaM
L(:,:,1)=[logML_set(:,1,1)';logML_set(:,1,2)']; 
L(:,:,2)=[logML_set(:,2,1)';logML_set(:,2,2)'];
L(:,:,3)=[logML_set(:,3,1)';logML_set(:,3,2)'];
L(:,:,4)=[logML_set(:,4,1)';logML_set(:,4,2)'];

%Initialize the matrix for protected exceedance probability
EP_2way=nan(numConds,numApers);
%Initialize the matrix for model frequency mean
MF_mean_2way=nan(numConds,numApers);
%Initialize the matrix for model frequency standard deviation
MF_std_2way=nan(numConds,numApers);

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
    
    EP_2way(d,:)=out.pep;
    MF_mean_2way(d,:)=out.Ef;
    MF_std_2way(d,:)=sqrt([out.Vf(1),out.Vf(2)]);    %
end


%% Visualize the model comparison results accross all coherence levels for 
% all participants (Figure 4, and Figure S1)

figure,
subplot(2,1,1)

bar(EP_2way)
hold on
plot([0 10],[0.95 0.95],'r--','LineWidth',2);
ylabel('Exceedance Probability','FontWeight','bold');
 xlim([0 5]);
set(gca,'XTickLabel',[]);
set(gca,'FontWeight','bold');
title('Exceedance probability for all coherence levels and all participants');
box off


subplot(2,1,2)
hold on
ylabel('Model Frequency','FontWeight','bold');
bar(MF_mean_2way)
plot([0 10],[0.5 0.5],'r--','LineWidth',2);
title('Model frequency for all coherence levels and all participants');
xlim([0 6]);
set(gca,'XTickLabel',{'0%','10%','25%','35%'});
xlabel('Coherence Level','FontWeight','bold');
set(gca,'FontWeight','bold');
legend('DDM','EXaM');
set(gca,'XTick',1:4);
 xlim([0 5]);


%% Calculate the posterior means for the parameters for non-zero coherence levels
 
%Initialize the matrix for transformed posterior mean parameters
posMeansTrans=zeros(numSubs,numConds,numApers,numPars);
%Initialize the matrix for transformed posterior standard deviation parameters
posStdsTrans=zeros(numSubs,numConds,numApers,numPars);
%Initialize the matrix for p-values (t-test comparison)
pVals=zeros(numConds,numPars);

%Name of the parameters
parnames = {'scale', 'scalestd', 'bound', 'bias', 'biaststd','ndtmean', 'ndtspread', 'lapseprob','lapsetoprob'};

% Defines transformation over parameters which allow to use Gaussian priors
% the function maps from Gaussian-distributed parameters to the original
% parameter values, it gets [nsamples, np]=size(Ptransformed) as input, but
% returns a matrix [np, nsamples] = size(P)
paramtransformfun = @(Ptrans) [...
Ptrans(:, 1), ...
exp(Ptrans(:, 2)),...
exp(Ptrans(:, 3)), ...
Ptrans(:, 4), ...
exp(Ptrans(:, 5)),... 
exp(Ptrans(:, 6)),... 
exp(Ptrans(:, 7)),... 
normcdf(Ptrans(:, 8)),...
normcdf(Ptrans(:, 9))]';
                                                                     
%calculate the posteriormean based on the transformation provided 
for aperIdx=1:numApers
    for condIdx=1:numConds
        
        %skip the parameter transformation for 0% coherence level
        if(condIdx==1)
            continue
        end
            
        %calculate the transofmred parameter estimates for each participant
        for subIdx=1:numSubs
            
            ep_mean1=squeeze(posMeans(subIdx,condIdx,aperIdx,:));
            ep_cov1=squeeze(posCovs(subIdx,condIdx,aperIdx,:,:));
            
            [posteriormean,posteriorstd,P] = getTransformedPars(ep_mean1',ep_cov1,...
                numPars,paramtransformfun);

     
            posMeansTrans(subIdx,condIdx,aperIdx,:)=posteriormean;
            posStdsTrans(subIdx,condIdx,aperIdx,:)=posteriorstd;

        end
    end
    
end

%calculate the posterior mean and standard deviation
posMean_mean=squeeze(mean(posMeansTrans,1));
posMean_std=squeeze(std(posMeansTrans,0,1))/sqrt(numSubs);

%perform ttest between posterior parameters from the DDM and the EXaM in
%non-zero coherence levels
for condIdx=1:numConds
    for parIdx=1:numPars
        
        [h,p]=ttest(posMeansTrans(:,condIdx,1,parIdx),posMeansTrans(:,condIdx,2,parIdx));
            
        pVals(condIdx,parIdx)=p;
        
    end
end

%% Create a table for the posterior parameter means accross non-zero 
% coherence levels (Table S1 for non-zero coherence levels)

numPars=9; %number of parameters
numUse=2; %number of models

%intialize the table cell structure 
parsTable=cell(numPars+2,(numConds-1)*numUse+1);
%initialize the cell headers
parsTable{1,1}='Parameters';
parsTable{1,2}='DDM';parsTable{1,3}='DDM';parsTable{1,4}='DDM';
parsTable{1,5}='EXaM';parsTable{1,6}='EXaM';parsTable{1,7}='EXaM';
parsTable{2,1}='Coherence Levels';
parsTable{2,2}='10';parsTable{2,3}='25';parsTable{2,4}='35';
parsTable{2,5}='10';parsTable{2,6}='25';parsTable{2,7}='35';

%write the parameter names inside the table cell
parnames={'scale_0', 'scalestd', 'bound', 'bias', 'biaststd','ndtmean', 'ndtspread', 'lapseprob','lapsetoprob'};
% 
for parIdx=1:numPars
     parsTable{parIdx+2,1}=parnames{parIdx};
end    

%write the parameters inside the table
for parIdx=3:numPars+2
    for aperIdx=1:numUse
        for condIdx=2:numConds
            %used for significance stars for the ExaM 
            if(aperIdx==1)
                parsTable{parIdx,condIdx}=...
                    sprintf('%.2f (%.3f)',posMean_mean(condIdx,aperIdx,parIdx-2),...
                                          posMean_std(condIdx,aperIdx,parIdx-2));
            elseif(aperIdx==2)
                if(pVals(condIdx,parIdx-2)<0.05 && pVals(condIdx,parIdx-2)>0.01) 
                     parsTable{parIdx,(numUse-1)*4+condIdx-1}=...
                     sprintf('%.2f* (%.3f)',posMean_mean(condIdx,aperIdx,parIdx-2),...
                                          posMean_std(condIdx,aperIdx,parIdx-2));
                elseif(pVals(condIdx,parIdx-2)<0.01)
                     parsTable{parIdx,(numUse-1)*4+condIdx-1}=...
                     sprintf('%.2f** (%.3f)',posMean_mean(condIdx,aperIdx,parIdx-2),...
                                          posMean_std(condIdx,aperIdx,parIdx-2));
                elseif(pVals(condIdx,parIdx-2)>0.05)
                    parsTable{parIdx,(numUse-1)*4+condIdx-1}=...
                    sprintf('%.2f (%.3f)',posMean_mean(condIdx,aperIdx,parIdx-2),...
                      posMean_std(condIdx,aperIdx,parIdx-2));
                end
            end
        end
    end
end


%display the table in the matlab command window
parsTable
%write the parameters table cell in a CSV file
xlswrite('tables\\posterior_parameter_table_without_0_coherence_Table_S1.csv', parsTable)

%% Calculate the posterior means for the parameters for zero coherence level

%Initialize the matrix for transformed posterior mean parameters
posMeansTrans_zero=zeros(numSubs,numApers,10);
%Initialize the matrix for transformed posterior standard deviation parameters
posStdsTrans_zero=zeros(numSubs,numApers,10);
%Initialize the matrix for p-values (t-test comparison
pVals_zero=zeros(1,10);

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
            
            ep_mean1=squeeze(posMeans_zero(subIdx,aperIdx,:));
            ep_cov1=squeeze(posCovs_zero(subIdx,aperIdx,:,:));
            
            %calculate the transofmred parameter estimates for each participant
            [posteriormean,posteriorstd,P] = getTransformedPars(ep_mean1',ep_cov1,...
                numPars,paramtransformfun);

            posMeansTrans_zero(subIdx,aperIdx,:)=posteriormean;
            posStdsTrans_zero(subIdx,aperIdx,:)=posteriorstd;

        end
 end

%Calculate the means of the censored distribution for the scale parameters
for aperIdx=1:2
    for parIdx=1:2
        [mu_trunc_i_j] = calcuateTruncatedMean_group(posMeansTrans_zero(:,aperIdx,parIdx),posStdsTrans_zero(:,aperIdx,parIdx),0);
        posMeansTrans_zero(:,aperIdx,parIdx)=mu_trunc_i_j;
   end
end

%Calculate the means and stadndard errors over parameter posteriormeans
posMean_mean_zero=squeeze(mean(posMeansTrans_zero,1));
posMean_std_zero=squeeze(std(posMeansTrans_zero,0,1))/sqrt(numSubs);


%perform ttest between posterior parameters from the DDM and the EXaM in
%zero coherence level
for condIdx=1:numConds
    for parIdx=1:numPars
        
        [h,p]=ttest(posMeansTrans_zero(:,1,parIdx),posMeansTrans_zero(:,2,parIdx));
            
        pVals_zero(parIdx)=p;
        
    end
end

%% Create a table for the posterior parameter means accross zero coherence level
% (Table 1 and Table S1 for zero coherence level)

numPars_zero=10; %number of parameters
numUse=2; %number of models
%intialize the table cell structure 
parsTable_zero_4=cell(numPars+2,numUse+1);
%initialize the cell headers
parsTable_zero_4{1,1}='Parameters';
parsTable_zero_4{1,2}='DDM';parsTable_zero_4{1,3}='EXaM';
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
            sprintf('%.2f (%.3f)',posMean_mean_zero(1,parIdx),...
                                  posMean_std_zero(1,parIdx));

        if(pVals_zero(parIdx)<0.05 && pVals_zero(parIdx)>0.01) 
             parsTable_zero_4{parIdx+2,3}=...
             sprintf('%.2f* (%.3f)',posMean_mean_zero(2,parIdx),...
                                  posMean_std_zero(2,parIdx));
        elseif(pVals_zero(parIdx)<0.01)
             parsTable_zero_4{parIdx+2,3}=...
             sprintf('%.3f** (%.3f)',posMean_mean_zero(2,parIdx),...
                                  posMean_std_zero(2,parIdx));
       elseif(pVals_zero(parIdx)>0.05)
            parsTable_zero_4{parIdx+2,3}=...
             sprintf('%.2f (%.3f)',posMean_mean_zero(2,parIdx),...
                                  posMean_std_zero(2,parIdx));
        end

end

%display the table in the matlab command window
parsTable_zero_4
%wite the parameter table inside a Excel file
xlswrite('tables\\posterior_parameter_table_0_coherence_Table_1.csv', parsTable_zero_4)

%% Save the analysis data
 save('analysis_data//model_comparison_all_participants.mat');