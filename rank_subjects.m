clear all;
close all;
clc;
addpath('functions');


load('analysis_data\\behavioral_analysis_results.mat','frac_res','acc_res','med_rt_res');


%%
subj=[62,63,64,65,66,67,71,74,76,77,80,82,84,85,86,90,91,92,93,94,95,96,...
      97,100,101,102,104,105,107,109,110,111,112,114,119,120,121,123,124,126,127,128,129,130];

acc_res_total=(acc_res(:,2)*240+acc_res(:,3)*160+acc_res(:,4)*160)/640;
med_rt_res_total=(med_rt_res(:,2)*240+med_rt_res(:,3)*160+med_rt_res(:,4)*160)/640;
  
%%

[cons_sorted,sorted_cons_idx]=sort(mean(frac_res,2),'descend');
[acc_sorted,sorted_acc_idx]=sort(acc_res_total,'descend');


%% 

bestSubjs=[sorted_acc_idx(1:11)];
worstSubjs=[sorted_acc_idx(end-10:end)];
otherSubjs=sorted_acc_idx(12:end-11);

bestSubjsNo=subj(bestSubjs);
worstSubjsNo=subj(worstSubjs);
otherSubjsNo=subj(otherSubjs);


%%
acc_res_means=[mean(acc_res(worstSubjs,:))',mean(acc_res(bestSubjs,:))'];
acc_res_stds=[std(acc_res(worstSubjs,:),0,1)',std(acc_res(bestSubjs,:),0,1)'];

med_rt_res_means=[mean(med_rt_res(worstSubjs,:))',mean(med_rt_res(bestSubjs,:))'];
med_rt_res_stds=[std(med_rt_res(worstSubjs,:),0,1)',std(med_rt_res(bestSubjs,:),0,1)'];

pVals_acc=nan(1,4);
pVals_rt=nan(1,4);

for i=1:4
    [h,p]=ttest(acc_res(bestSubjs,i),acc_res(worstSubjs,i));
    pVals_acc(i)=p;

    [h,p]=ttest(med_rt_res(bestSubjs,i),med_rt_res(worstSubjs,i));
    pVals_rt(i)=p;

end

%%

save('analysis_data//subject_ranking.mat','bestSubjsNo','bestSubjs','worstSubjsNo','worstSubjs','otherSubjsNo','otherSubjs',...
'acc_res_total','acc_res_means','acc_res_stds','med_rt_res_stds','med_rt_res_means','pVals_acc','pVals_rt','frac_res');


