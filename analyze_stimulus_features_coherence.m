% The script "analyze_stimulus_features_coherence.m" conducts the 
% stimulus feature analysis used in "Fard et al. (2021), Spatiotemporal Modeling 
% of Response Consistency with Incoherent Motion Stimuli in Perceptual 
% Decision Making, In submission".
% The script runs a simulation in which the effect of coherence level in 
% on average stimulus features generated (by computing dot counts) is 
% evaludated. 

%Hint: Please make sure that this script is placed next to the folder
%"model_fit_data", "behavioral_data", "analysis_data", "experiment_design", 
% and "functions". The successful running of the script is depedent on the
% data from behavioral analysis, subject ranking, model fitting, and model
% comparison results. 

%The script may take a few minutes to run depending on your system
%specifications.
%
% Author: Pouyan Rafieifard (January 2021)

clear all;
close all;
clc;
addpath('functions');

%% Loading the neccessary data from the experiment design and stimulus features for each trial of the experiment

load('experiment_design\\stimulus_features_base.mat','designMatrix_base','zeroDotFeatures_dc_base');

%% Intialization and definition of the variables used in the analysis

%number of trials per coherence level (0%,10%,25%,35%)
condTrials=[240 240 160 160]; 
%total number of trials
numTrials=800;
%total number of experimental conditions (coherence levels)
numConds=4;

%a cell array containing all the stimulus features (dot counts)
zeroDotFeatures_dc_set=cell(1,numConds);
%a cell array containing all the design parameters of the experiment per
%coherence level
designMatrix_cond=cell(1,numConds);
%a cell array containing the trial numbers for each coherence level
trs_cond=cell(1,numConds);
%a matrix containing the standard deviation of dot counts per coherence
%level
zeroDotFeatures_std_dc_cond=nan(1,numConds);

%% Run through the experiemental design and the respective trial coherence levels
% and prepare the datasets for each coherence level containing the dot
% counts, nomralized dot counts, and variability of dot counts of each trial

for condIdx=1:numConds
    
    condTrs=find(designMatrix_base(:,1)==condIdx);
    
    designMatrix_cond{condIdx}=designMatrix_base(condTrs,:);
    trs_cond{condIdx}=condTrs;
    
    
    zeorDotFeatures_dc_cond=cell(1,condTrials(condIdx));
    zeorDotFeatures_dc_cond_norm=cell(1,condTrials(condIdx));
    
    for trIdx=1:condTrials(condIdx)
        zeorDotFeatures_dc_cond{trIdx}=zeroDotFeatures_dc_base{condTrs(trIdx)};
    end
    
    zeroDotFeatures_set_tmp=cat(2,zeorDotFeatures_dc_cond{:});
    zeroDotFeatures_std_dc_cond(condIdx)=std(abs(zeroDotFeatures_set_tmp));
        
    
    for trIdx=1:condTrials(condIdx)
        zeorDotFeatures_dc_cond_norm{trIdx}=zeorDotFeatures_dc_cond{trIdx}/zeroDotFeatures_std_dc_cond(condIdx);
    end
%     
    zeroDotFeatures_dc_set{condIdx}=zeorDotFeatures_dc_cond_norm;
    
end

%% Calculate the average dot count values separated for each coherence level
% and the trials that the target alternative is right or left
% the outcomes of this analysis is used to visualize the average dot count 
% values per coherence level and target alternative e.g. 25% coherence and
% rightward trials

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

%average dot counts values for each coherence level and direction
dirMeans=[meanFeatures_zero,dirMeanFeatures(1,2),dirMeanFeatures(1,3),dirMeanFeatures(1,4),...
          dirMeanFeatures(2,2),dirMeanFeatures(2,3),dirMeanFeatures(2,4)];
%standard errors of dot counts for each coherence level and direction
dirSEs=[seFetures_zero,dirSEFeatures(1,2),dirSEFeatures(1,3),dirSEFeatures(1,4),...
        dirSEFeatures(2,2),dirSEFeatures(2,3),dirSEFeatures(2,4)];

%visualize the results analysis of coherence level and direction effect on
%stimulus feature values
figure,hold on
bar(dirMeans)
zeroDotFeatures_std_dc_cond=zeroDotFeatures_std_dc_cond./sqrt([240 240 160 160]);
errorbar(1:7,dirMeans,dirSEs,'r.');
ylabel('Dot counts');
xlabel('Coherence/direction');
set(gca,'XTick',1:7);
set(gca,'XTickLabel',{'0% Both Directions','10% Right','25% Right','35% Right',...
                    '10% Left','25% Left','35% Left'});
                

%% Save the analysis data

%create dot counts one example trial for Figure 8
zeroDotFeatures_dc=zeroDotFeatures_dc_base{12};
%save the analysis data 
save('analysis_data\\dot_counts_analysis_results.mat','zeroDotFeatures_dc','dirMeans','dirSEs');