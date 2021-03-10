% The script "behavioral_analysis.m" conducts the behavioral analysis used
% in multiple figures in the the study "Fard et al. (2021), Spatiotemporal 
% Modeling of Response Consistency with Incoherent Motion Stimuli in 
% Perceptual Decision Making, In submission".
% The script, After removing the timed out trials, first loads the 
% behavioral data from the experiment into the matrices that can be used to
% compute variables such as reaction time and accuracy and response
% consistency. Afterwards the scripts computes the average behavioral 
% measures like "average accuracy" and "average median reaction time" (Ref.
% Fig. 1 in the manuscript). Furthermore, The analysis computues the
% measure reponse consistency (see Methods in the manuscript) as the
% fraction of right-ward responses per parcitipcant and stimulus type
% corrected by the general bias of the participant for right response
% accross all trials. The binomial test (via sampling) is used to determine
% the whether a response consistency value for a particular participant and
% sitmulus type deviates enough from the general bias of the participant to
% qualify to for a significantly consistent response (see Fig.2 andMethods 
% in the manuscript for more details). Finally, the average response
% consitency accross all participants per stimulus types is computed and
% compared to cluster the stimulus types (see Fig. 3).
%
%Hint: Please make sure that this script is placed next to the folder
%"behavioral_data", "analysis_data", "experiment_design", and "functions".
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
subjNos=[62 63 64 65 66 67 71 74 76 77 80 82 84 85 86 90 91 92 93 94 95 96,...
    97 100 101 102 104 105 107 109 110 111 112 114 119 120 121 123 124 126 127 128 129 130];

%number of participants
numSub=length(subjNos);
%number of trials
numTrials=800;
%number of stimulus types
numSeed=12;

%accuracy values accross coherence levels
acc_res=zeros(numSub,4);
%reaction time quantiles accross coherence levels
rt_qq_res=zeros(numSub,4,5);
%number of timed-out trials accross coherence levels
numTo=zeros(numSub,4);

%fraction of left-ward responses for each participant
fractions_left_res=zeros(1,numSub);
%fraction of right-ward responses for each participant
fractions_right_res=zeros(1,numSub);

%fraction of left-ward responses per participant and stimulus type
fractions_frozen_res=zeros(numSub,12);
%frequency of left-ward responses per participant and stimulus type
freq_frozen_res_left=zeros(numSub,12);
%frequency of right-ward responses per participant and stimulus type
freq_frozen_res_right=zeros(numSub,12);
%standard error of fractions of responses per participant and stimulus type
fractions_se_frozen_res=zeros(numSub,12);

%average reaction time for zero coherence trials per participant and
%stimulus type
mean_rt_frozen_zero_res=zeros(numSub,12);
%standard error reaction time for zero coherence trials per participant and
%stimulus type
se_rt_frozen_zero_res=zeros(numSub,12);

%a cell matrix containing reaction time values per participant and stimulus
%types
rtSeed_set=cell(numSub,numSeed);
%a cell matrix containing response values per participant and stimulus
%types
respSeed_set=cell(numSub,numSeed);
%a cell matrix containing the error responses (higher coherence levels)
condErrors=cell(numSub,4);

%a matrix containing trial-level responses per participant
resp_set=nan(numSub,numTrials);
%a matrix containing trial-level reaction times per participant
RT_set=nan(numSub,numTrials);
%a matrix containing trial-level correct alternative per participant
cond_set=nan(numSub,numTrials);

%% Loading the design matrix of the experiment and identify the stimulus types

load('experiment_design\\stimulus_types_random_seeds.mat','zeroSeeds');
zeroSeedsOrig=zeroSeeds;


%% Loading the behavioral data into related data objects
% In a loop over participants the experiment design data and behavioral 
% data from each participant is loaded into relevant such as reaction time
% and responses and correct alternative for each participant and trial.
% Afterwards, the general bias of the participant over all trials and the
% overall accuracy and reaction times (per coherence level) are computed.
% Finally, the behavioral measures on the level of participant and 
% stimulus type are stored into the related matrices.

for subIdx=1:numSub
    
    sprintf('loading the subject data for %d-th partcianpt', subIdx);    
    
    subjectNo=subjNos(subIdx);

    %loading the randomized design matrix for each participant
    loadStrGen=strcat('behavioral_data\\data_subj_',num2str(subjectNo),'_genExp','.mat');
    load(loadStrGen,'subjectNo','designMatrix')

    coh_set=[0 0.1 0.25 0.35];
    condTrials=[240 240 160 160];

    designMatrix(:,2)=(360-designMatrix(:,2))/180;
    designMatrix(:,1)=coh_set(designMatrix(:,1));
    
    %loading the responses and reaction times for each participant and
    %trials from the behavioral data-set
    for trialNo=1:numTrials

        loadStr=strcat('behavioral_data\\data_subj_',num2str(subjectNo),'_trial_',num2str(trialNo),'.mat');
        load(loadStr);

        resp_set(subIdx,trialNo)=resp;
        RT_set(subIdx,trialNo)=RT;
        cond_set(subIdx,trialNo)=designMatrix(trialNo,1);    
    end
    
    %computing the general bias of the participants 
    fractions_left_res(subIdx)=length(find(resp_set(subIdx,:)==1))/(length(find(resp_set(subIdx,:)==1))+length(find(resp_set(subIdx,:)==2)));
    fractions_right_res(subIdx)=length(find(resp_set(subIdx,:)==2))/(length(find(resp_set(subIdx,:)==1))+length(find(resp_set(subIdx,:)==2)));

    designMatrix=[designMatrix,squeeze(resp_set(subIdx,:)'),squeeze(RT_set(subIdx,:)')];

    qq=[0.1 0.3 0.5 0.7 0.9];

    cohColumn=designMatrix(:,1);

    condResults=cell(1,4);

    for cond=1:4

        condResults{cond}=designMatrix(cohColumn==coh_set(cond),:);
        condErrors_tmp=[];

        for tr=1:size(condResults{cond},1)

            if(cond==1)
                if(condResults{cond}(tr,4)==2)
                    acc_res(subIdx,cond)=acc_res(subIdx,cond)+1;
                end
            else
                if(condResults{cond}(tr,2)==condResults{cond}(tr,4))
                    acc_res(subIdx,cond)=acc_res(subIdx,cond)+1;
                else
                    condErrors_tmp=[condErrors_tmp,condResults{cond}(tr,5)];
                end
            end

        end
        
        %computing the behavioral measure per coherence level
        acc_res(subIdx,cond)=acc_res(subIdx,cond)/condTrials(cond);
        rt_qq_res(subIdx,cond,:)=quantile(condResults{cond}(:,5),qq);
        numTo(subIdx,cond)=sum(condResults{cond}(:,4)==0);
        condErrors{subIdx,cond}=condErrors_tmp;
    
        %computing the behavioral measures per stimulus type
        if(cond==1)

            seedColumn=condResults{cond}(:,3);
            condSeeds=zeroSeedsOrig;

            for seed=1:12

                condResultsSeed=condResults{cond}(seedColumn==condSeeds(seed),:);
                respCondResultsSeed=condResultsSeed(:,4);
                RTCondResultsSeed=condResultsSeed(:,5);

                to_idx=find(RTCondResultsSeed>2);
                RTCondResultsSeed(to_idx)=[];
                respCondResultsSeed(to_idx)=[];

                fractions_frozen_res(subIdx,seed)=length(find(respCondResultsSeed==1))/length(respCondResultsSeed);
                fractions_se_frozen_res(subIdx,seed)=std(respCondResultsSeed)/sqrt(length(respCondResultsSeed));

                freq_frozen_res_left(subIdx,seed)=length(find(respCondResultsSeed==1));
                freq_frozen_res_right(subIdx,seed)=length(find(respCondResultsSeed==2));

                mean_rt_frozen_zero_res(subIdx,seed)=mean(RTCondResultsSeed);
                se_rt_frozen_zero_res(subIdx,seed)=std(RTCondResultsSeed)/sqrt(length(RTCondResultsSeed));

                rtSeed_set{subIdx,seed}=RTCondResultsSeed;
                respSeed_set{subIdx,seed}=respCondResultsSeed;            

            end
        end
    end
end

%% calculation and visualization of basic behavioral measures (Fig 1)

%compute the median RT
med_rt_res=rt_qq_res(:,:,3);
%compute the average median RT accross participants
mean_med_rt_res=mean(med_rt_res,1);
%compute the standard deviation of the median RT accross participants
se_med_rt_res=std(med_rt_res,0,1);
%compute the average accuracy accorss participants
mean_acc_res=mean(acc_res,1);
%compute the standard deviation of accuracy accross participants
se_acc_res=std(acc_res,0,1);%/sqrt(numSub);

%visualize the results (Fig 1)
figure
subplot(2,1,1)
hold on
plot(mean_acc_res)
errorbar(mean_acc_res,se_acc_res,'r.')
set(gca,'XTick',1:4);
set(gca,'XTickLabel',{'0%','10%','25%','35%'});
ylabel('Accuracy','FontWeight','bold')
ylim([0 1.1]);
set(gca,'FontWeight','bold')

subplot(2,1,2)
hold on
bar(mean_med_rt_res)
errorbar(mean_med_rt_res,se_med_rt_res,'r.')
set(gca,'XTick',1:4);
set(gca,'XTickLabel',{'0%','10%','25%','35%'});
xlabel('Coherence levels','FontWeight','bold')
ylabel('Median RT(s)','FontWeight','bold')
set(gca,'FontWeight','bold')

%% Calculation of response consistency measure

%computing fraction of rightward responses per participant and stimulus type 
frac_res=freq_frozen_res_right./(freq_frozen_res_left+freq_frozen_res_right);
%correcting the response cosisntency by subtracting the response bias of
%participant for the rightward responses
frac_res=frac_res-repmat(fractions_right_res',1,12);


%% Determining of significance of consistent responses per participant and stimulus type
% In this section we implement a statistical test (binomial test) to
% determine whether a set of responses (per participant and stimulus type)
% is consistent towards right or left alternative by comparing the frequency 
% of responeses per participant and stimulus type with the general bias of
% the participant for right alternative accross all trials. In this
% implementation, we create a binomial distribution for frequncy of 
% right-ward centered response around the general bias of each participant.
% We will visualize the binomial distribution for each participant
% seperately. The p-values of a consistent response is deteremined by
% computing the proportion of the distribution that are more extreme than
% the current frequency (two-tailed bionmial test, see Methods in the
% manuscript). Afterwards, we apply the multiple-comparison correction on
% the computed p-values. Finally, we will determine the consistent
% responses per participant and stimulus type by using the p<0.05
% threshold.


binomAlt=zeros(numSub,12);
numSamples=2000000;
chunkSize=20;

figure,hold on

subBiasDist_set=cell(1,numSub);
bootStrapMean_set=zeros(1,numSub);

%conducting the two-tailed bionmial test (via sampling) for the resposne of
%each participant per stimulus type
for subIdx=1:numSub
    subBiasRight=fractions_right_res(subIdx);
    subSamples=rand(1,numSamples);
    subBiasDist=zeros(1,(numSamples/chunkSize));
    
    for chunk=1:chunkSize:numSamples-chunkSize+1
       tmpChunk=subSamples(chunk:chunk+chunkSize-1);
       subBiasDist((chunk-1)/20+1)=length(find(tmpChunk<subBiasRight));  
    end
    
    subBiasDist_set{subIdx}=subBiasDist;
    
    subplot(ceil(numSub/2),2,subIdx),hold on
    title(strcat('Bootstrapping H0 Dist for subject',num2str(subIdx)));
    hist(subBiasDist,max(subBiasDist)-min(subBiasDist))
    plot([mean(subBiasDist) mean(subBiasDist)],[0 10^4],'r--','LineWidth',2);
    biasRightFreq=subBiasRight*20;
    plot([biasRightFreq biasRightFreq],[0 10^4],'g--','LineWidth',2);
    if(subIdx>10)
        xlabel('Frequency of right-ward responses');
    end
    if(mod(subIdx,2)==1)
        ylabel('Counts');
    end
    if(subIdx<3)
        legend('H0 dist','H0 Dist Median ','Subject Bias');
    end 
    
    bootstrapMean=mean(subBiasDist);
    bootStrapMean_set(subIdx)=bootstrapMean;
    
    %implementing the two-tailed binomial test via sampling
    bootstrapMean=10;
    for seedIdx=1:12
       meanDiff=freq_frozen_res_right(subIdx,seedIdx)-bootstrapMean;
       if(meanDiff>=0)
            otherVal=bootstrapMean-abs(meanDiff);
            lowerProp=length(find(subBiasDist>freq_frozen_res_right(subIdx,seedIdx)))/length(subBiasDist);
            higherProp=length(find(subBiasDist<otherVal))/length(subBiasDist);
       else
            otherVal=bootstrapMean+abs(meanDiff);
            lowerProp=length(find(subBiasDist<freq_frozen_res_right(subIdx,seedIdx)))/length(subBiasDist);
            higherProp=length(find(subBiasDist>otherVal))/length(subBiasDist);

       end
       
       %p-value of the bionmial test
       binomAlt(subIdx,seedIdx)=(lowerProp+higherProp); 
        
    end
    
end

%applying multiple-comparison correction on the p-values generated by the
%bionomial test
[h1, crit_p1, adj_ci_cvrg1, adj_p1]=fdr_bh(binomAlt,.05,'pdep','yes');

%finding the consistent responses of participants per each stimulus type
[rSig5,cSig5]=find(adj_p1<0.05);
subjCons=rSig5;
stimulusCons=cSig5;

%visualization of the response consisntency map
figure
colormap('jet');
hold on
imagesc(frac_res)
plot(cSig5,rSig5,'o','LineWidth',2,'Color',[1 1 1]);
axis tight
h=colorbar
ylabel(h, 'Response Consistency','FontWeight','bold')
ylabel('Participants','FontWeight','bold')
xlabel('Stimulus Types','FontWeight','bold');
set(gca,'YTick',1:numSub,'YTickLabel',1:numSub);
set(gca,'XTick',1:12,'XTickLabel',1:12);
set(gca,'FontWeight','bold')
grid on
title('Response consistency map')

%% Clustering of the stimulus types according to average response consistency (Fig.3)
% here we calculate the average response consistency accross partcipants
% for each stimulus type and use the mean value of this measure as a
% threshold to visaully cluster the stimulus types into two groups. Note that
% the clustering methods that is used in the paper is k-means clustering
% (see Methods for more details) and here we compute the average response
% consistency accross participants per stimulus type as an input to k-means
% algorithm (Fig 3).

%compute the average response consistency per stimulus type
mean_frac_res=abs(mean(frac_res,1));
%sort the response consistency values
[frac_res_sorted,sorted_idx]=sort(mean_frac_res,'descend');

%visualize the clustering of the stimulus types
figure,hold on

meanCons=mean(mean_frac_res);
medianCons=median(mean_frac_res);

plot([0 13],[meanCons,meanCons],'r--','LineWidth',1);
plot([0 13],[medianCons,medianCons],'g--','LineWidth',1);

plot(frac_res_sorted,'o-','LineWidth',2)
set(gca,'XTick',1:12,'XtickLabel',sorted_idx);
ylabel('Average RC accross all subjects');
xlabel('Stimulus type');
legend('mean','median','Mean consistency across one stimulus type')

%determing the Cluster 1 stimulus types (in accordance with the results
%from the k-means clustering, see the Python code)
zeroSeeds_conditions=mean_frac_res>meanCons;

%% Save the analysis data
save('analysis_data\\behavioral_analysis_results.mat','zeroSeeds','zeroSeeds_conditions','respSeed_set','rtSeed_set','frac_res','acc_res','med_rt_res',...
    'stimulusCons','subjCons','zeroSeeds_conditions',...
     'mean_med_rt_res','se_med_rt_res','mean_acc_res','se_acc_res','fractions_right_res',...
     'bootStrapMean_set','subBiasDist_set');
    