% The script "test_prepare_fit_data.m" creates the data used to fit the
% computational models to the behavioral data used used in "Fard et al. (2021),
% Spatiotemporal Modeling of Response Consistency with Incoherent Motion 
% Stimuli in Perceptual Decision Making, In submission".
% The script first loads all the data from computational models fitted to
% the behavioral data for each participant. This may include parameters 
% related to the posterior distributions (mean and standard deviation for
% the posterior distribution) estimated for each participant in each 
% coherence level and the model marginal likelihood. Afterwards, the script
% uses Bayesian model comparison (using VBA toolbox) 
% to provide the metrics for explanatory power of each computational model
% (DDM and EXaM) to explain the behavioral data in each coherence level. 
% The resulting model evidence variables (protected exceedance probability
% and model frequency) are visualized to compare the explanatory power of
% the models. In addition, the script computes the mean and standard 
% deviations of estimated posterior parameter distributions and conducts
% a t-test to provide significant differences between the estimated
% parameter values between the DDM and EXaM. 

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

%% Intialization of the variables used in the analysis

condTrials=[240 240 160 160]; %number of trial per cond
numTrials=800; %number of trials
numConds=4; %numbeer of coherence levels

% load the experimental design data
load('..//experiment_design//stimulus_features_base.mat','designMatrix_base','zeroDotFeatures_dc_base');

%set of dot count features
zeroDotFeatures_dc_set=cell(1,numConds);
%comptue the standard deviation of dot counts per coherence level
zeroDotFeatures_std_dc_cond=nan(1,numConds);

%cell matrix containing the design matrix for each coherence level
designMatrix_cond=cell(1,numConds);
%identifier for trials of each coherence level
trs_cond=cell(1,numConds);
%average dot counts for each coherence level
meanFeatures=zeros(1,4);

%% find the average dot count values per coherence level
for condIdx=1:numConds
    
    condTrs=find(designMatrix_base(:,1)==condIdx);
    
    designMatrix_cond{condIdx}=designMatrix_base(condTrs,:);
    trs_cond{condIdx}=condTrs;
    
    zeorDotFeatures_dc_cond=cell(1,condTrials(condIdx));
    zeorDotFeatures_dc_cond_norm=cell(1,condTrials(condIdx));
    
    for trIdx=1:condTrials(condIdx)
        zeorDotFeatures_dc_cond{trIdx}=zeroDotFeatures_dc_base{condTrs(trIdx)};
    end
    
    %%calculate the standard deviation accross all repititions
    condTrs_right=find(designMatrix_base(condTrs,2)==0);
    condTrs_left=find(designMatrix_base(condTrs,2)==180);
    
    zeorDotFeatures_dc_cond_right=[];
    zeorDotFeatures_dc_cond_left=[];
    
    %shift each feature to its mean
    for i=1:length(condTrs_right)
        zeorDotFeatures_dc_cond_right=[zeorDotFeatures_dc_cond_right,...
            zeorDotFeatures_dc_cond{condTrs_right(i)}-mean(zeorDotFeatures_dc_cond{condTrs_right(i)})];
        %figure,plot(zeroDotFeatures_dc_cond_right{i})
    end
        
    for i=1:length(condTrs_left)
        zeorDotFeatures_dc_cond_left=[zeorDotFeatures_dc_cond_left,...
            zeorDotFeatures_dc_cond{condTrs_left(i)}-mean(zeorDotFeatures_dc_cond{condTrs_left(i)})];
    end
    
    
    zeroDotFeatures_std_dc_cond(condIdx)=std([zeorDotFeatures_dc_cond_right,zeorDotFeatures_dc_cond_left]);
    
    figure,plot([zeorDotFeatures_dc_cond_right,zeorDotFeatures_dc_cond_left])
    
    for trIdx=1:condTrials(condIdx)
        zeorDotFeatures_dc_cond_norm{trIdx}=zeorDotFeatures_dc_cond{trIdx};%/zeroDotFeatures_std_dc_cond(condIdx);
    end
%     
    zeroDotFeatures_dc_set{condIdx}=zeorDotFeatures_dc_cond_norm;
    
end

%% compare the average dot count values per ach coherence level and direction

dirMeanFeatures=zeros(2,4);
dirSEFeatures=zeros(2,4);

for condIdx=1:4
    condTrs=find(designMatrix_base(:,1)==condIdx);
    condTrs_right=find(designMatrix_base(condTrs,2)==0);
    condTrs_left=find(designMatrix_base(condTrs,2)==180);
    
    dirMeanFeatures(1,condIdx)=mean(cat(2,zeroDotFeatures_dc_set{condIdx}{condTrs_right}));
    dirSEFeatures(1,condIdx)=std(cat(2,zeroDotFeatures_dc_set{condIdx}{condTrs_right}))/sqrt(length(condTrs_right));
    
    dirMeanFeatures(2,condIdx)=mean(cat(2,zeroDotFeatures_dc_set{condIdx}{condTrs_left}));
    dirSEFeatures(2,condIdx)=std(cat(2,zeroDotFeatures_dc_set{condIdx}{condTrs_left}))/sqrt(length(condTrs_left));
    if(condIdx==1)
        meanFeatures_zero=mean(cat(2,zeroDotFeatures_dc_set{condIdx}{:}));
        seFetures_zero=std(cat(2,zeroDotFeatures_dc_set{condIdx}{:}))/sqrt(length(condTrs));
    end
end

dirMeans=[meanFeatures_zero,dirMeanFeatures(1,2),dirMeanFeatures(1,3),dirMeanFeatures(1,4),...
          dirMeanFeatures(2,2),dirMeanFeatures(2,3),dirMeanFeatures(2,4)];
dirSEs=[seFetures_zero,dirSEFeatures(1,2),dirSEFeatures(1,3),dirSEFeatures(1,4),...
        dirSEFeatures(2,2),dirSEFeatures(2,3),dirSEFeatures(2,4)];

figure,hold on
bar(dirMeans)
zeroDotFeatures_std_dc_cond=zeroDotFeatures_std_dc_cond./sqrt([240 240 160 160]);
errorbar(1:7,dirMeans,dirSEs,'r.');
ylabel('Dot counts');
xlabel('Coherence/direction');
set(gca,'XTick',1:7);
set(gca,'XTickLabel',{'0% Both Directions','10% Right','25% Right','35% Right',...
                    '10% Left','25% Left','35% Left'});

                
%% enrich the behavioral data with the dot counts and save them to respective .mat files

subjNos=[62 63 64 65 66 67 71 74 76 77 80 82 84 85 86 90 91 92 93 94 95 96,...
     97 100 101 102 104 105 107 109 110 111 112 114 119 120 121 123 124 126 127 128 129 130];
choicesIn=[-1,1];

numSubs=length(subjNos);

designMatrix_set=cell(1,4);

for condIdx=2:4

    designMatrix_stim_cond_base=designMatrix_cond{condIdx};
    zeroDotFeatures_cond_base=zeroDotFeatures_dc_set{condIdx};
    
    for subIdx=1:numSubs
    
        subjectNo=subjNos(subIdx);
        numTrialsCond=condTrials(condIdx);

        choices=nan(1,numTrialsCond);
        rts=nan(1,numTrialsCond);
        seeds=nan(1,numTrialsCond);
        trueA=nan(1,numTrialsCond);
        features=cell(1,numTrialsCond);
        averageFeatures=cell(1,numTrialsCond);
        
        loadStrGen=strcat('..\\behavioral_data\\data_subj_',num2str(subjectNo),'_genExp','.mat');
        load(loadStrGen,'subjectNo','designMatrix')

        designMatrixSubj=designMatrix;

        %extract only zero coherence trials
        condTr=find(designMatrixSubj(:,1)==condIdx);
        condDesignMatrix=designMatrixSubj(condTr,:);


        for trialNo=1:numTrialsCond

            trialNo

            trialIdx=condTr(trialNo);

            %load the experimental data     
            load(strcat('..\\behavioral_data\\data_subj_',num2str(subjectNo),'_trial_',num2str(trialIdx),'.mat'))
            

            if(resp>0 && RT<=2)
                choices(trialNo)=choicesIn(resp);
                rts(trialNo)=RT;
            else
                choices(trialNo)=0;
                rts(trialNo)=5;
            end

            trueA(trialNo)=trialDirection;
            seeds(trialNo)=rseed;
            
            %% find the features associated with the trial
            
            idc=find(designMatrix_stim_cond_base(:,3)==condDesignMatrix(trialNo,3));
            
            featureIdx=-1;
            
            for idx=1:length(idc)
               if(designMatrix_stim_cond_base(idc(idx),2)==condDesignMatrix(trialNo,2))
                    featureIdx=idc(idx);
               end   
            end
                       
            designMatrix_stim_cond_base(featureIdx,:)==condDesignMatrix(trialNo,:)
            features{trialNo}=zeroDotFeatures_cond_base{featureIdx};
            
        end

        avgFeature_r=mean(cat(2,features{find(condDesignMatrix(:,2)==0)}));
        avgFeature_l=mean(cat(2,features{find(condDesignMatrix(:,2)==180)}));
        
        avgFeature=mean(abs([avgFeature_r,avgFeature_l]));
        
        for trialNo=1:numTrialsCond
            
            if(condDesignMatrix(trialNo,2)==0)
                tmpFeatures=ones(size(features{trialNo}));
            else
                tmpFeatures=-1*ones(size(features{trialNo}));
            end
            averageFeatures{trialNo}=tmpFeatures;
            
        end
        
        trueAlt=condDesignMatrix(:,2);
        trueAlt(trueAlt==0)=1;
        trueAlt(trueAlt==180)=-1;
                
%            
        save(strcat('..\\test_model_fit_data\\fitData_sub_',num2str(subjectNo),'_cond_',num2str(condIdx),'_dc_norm_final_2.mat'),...
            'choices','rts','trueA','seeds','features','averageFeatures','trueAlt','avgFeature');

    end
end



