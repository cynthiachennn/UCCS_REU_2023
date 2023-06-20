subjList = {'A_405', 'A_408', 'B_110', 'B_309', 'B_311', 'B_316', 'C_204', 'C_429', 'E_321', 'E_415', 'E_429',...
    'F_027', 'F_209', 'F_210', 'G_413', 'G_428', 'H_804', 'I_719', 'I_723'};

for i=1:19
    addpath('C:\Users\Owner\Documents\MATLAB\eeglab_current\eeglab2023.0\plugins\PrepPipeline0.56.0\reporting')
    subj = cell2mat(subjList(i));
    EEG = pop_loadset('filename',strcat(subj, '_19.set'),'filepath',strcat('C:\\Users\\Owner\\Documents\\MATLAB\\REU_data\\5F_EEG_data\\', subj, '\\'));
    EEG = pop_prepPipeline(EEG);
    EEG = pop_saveset(EEG, 'filename',strcat(subj, '_PREP.set'),'filepath',strcat('C:\\Users\\Owner\\Documents\\MATLAB\\REU_data\\5F_EEG_data\\', subj, '\\'));

end