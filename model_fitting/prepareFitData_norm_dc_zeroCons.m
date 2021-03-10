clear all; close all; clc;

condTrials=[240 240 160 160]; %number of trial per cond
numTrials=800;
numConds=1;

load('stimulusFeatures_base.mat','designMatrix_base','zeroDotFeatures_dc_base');
load('zeroSeedsFinal_2_2.mat','zeroSeeds','zeroSeeds_conditions');


zeroDotFeatures_dc_set=cell(1,numConds);
zeroDotFeatures_std_dc_cond=nan(1,numConds);

designMatrix_cond=cell(1,numConds);
trs_cond=cell(1,numConds);

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
    zeroDotFeatures_std_dc_cond(condIdx)=std(zeroDotFeatures_set_tmp);
    
    for trIdx=1:condTrials(condIdx)
        zeorDotFeatures_dc_cond_norm{trIdx}=zeorDotFeatures_dc_cond{trIdx};%/zeroDotFeatures_std_dc_cond(condIdx);
    end
    
    zeroDotFeatures_dc_set{condIdx}=zeorDotFeatures_dc_cond_norm;
    
end

%% put all subject data into three matrices
% 

subjNos=[62 63 64 65 66 67 71 74 76 77 80 82 84 85 86 90 91 92 93 94 95 96,...
     97 100 101 102 104 105 107 109 110 111 112 114 119 120 121 123 124 126 127 128 129 130];



choicesIn=[-1,1];

numSubs=length(subjNos);

for condIdx=1:numConds

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
        conditions=nan(1,numTrialsCond);
        averageFeatures=cell(1,numTrialsCond);
        
        loadStrGen=strcat('expData2\\data_subj_',num2str(subjectNo),'_genExp','.mat');
        load(loadStrGen,'subjectNo','designMatrix')

        designMatrixSubj=designMatrix;

        %extract only zero coherence trials
        condTr=find(designMatrixSubj(:,1)==condIdx);
        condDesignMatrix=designMatrixSubj(condTr,:);
        

        for trialNo=1:numTrialsCond

            trialNo

            trialIdx=condTr(trialNo);

            %load the experimental data     
            load(strcat('expData2\\data_subj_',num2str(subjectNo),'_trial_',num2str(trialIdx),'.mat'))
            

            if(resp>0 && RT<=2)
                choices(trialNo)=choicesIn(resp);
                rts(trialNo)=RT;
            else
                choices(trialNo)=0;
                rts(trialNo)=5;
            end

            trueA(trialNo)=trialDirection;
            seeds(trialNo)=rseed;
            
            %% find the condition associated with that trial
            
            trSeed=condDesignMatrix(trialNo,3);
            
            conditions(trialNo)=zeroSeeds_conditions(find(zeroSeeds==trSeed));
            
            
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

        avgFeature=1;%mean(cat(2,features{:}));
        
        for trialNo=1:numTrialsCond
            
            tmpFeatures=avgFeature*ones(size(features{trialNo}));
            
            averageFeatures{trialNo}=tmpFeatures;
            
        end
            
        
        save(strcat('new model\\fitData\\fitData_sub_',num2str(subjectNo),'_cond_',num2str(condIdx),'_dc_norm_final_2.mat'),...
            'choices','rts','trueA','seeds','features','conditions','averageFeatures');

    end
end



