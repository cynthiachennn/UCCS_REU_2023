reformatting_200
reformatting_1000
% hopefully my stuff is formatted correctly yikes

subjList = {'A_405', 'A_408', 'B_110', 'B_309', 'B_311', 'B_316', 'C_204', 'C_429', 'E_321', 'E_415', 'E_429',...
    'F_027', 'F_209', 'F_210', 'G_413', 'G_428', 'H_804', 'I_719', 'I_723'};

for i=1:19
    subj = cell2mat(subjList(i));
    % "preprocessing"
    % one subject at a time:
    EEG = pop_loadset('filename',strcat(subj, '_19_noref.set'),'filepath',strcat('C:\\Users\\Owner\\Documents\\MATLAB\\REU_data\\5F_EEG_data\\preprocessing\\'));
    chanlocs = EEG.chanlocs;
    
    %filter data
    %filter data
    EEG = pop_eegfiltnew(EEG, 'locutoff', 0.5);
    
    % remove bad channels and bad portions of data
    EEG = pop_clean_rawdata(EEG, 'FlatlineCriterion',4,'ChannelCriterion',0.85,'LineNoiseCriterion',4,'Highpass','off',...
        'BurstCriterion',20,'WindowCriterion',0.25,'BurstRejection','on','Distance','Euclidian','WindowCriterionTolerances',[-Inf 7] );
    
    %common average referencing ??
    
    % Run ICA and IC Label
    EEG = pop_runica(EEG, 'icatype', 'picard', 'maxiter',500);
    EEG = pop_iclabel(EEG, 'default');
    EEG = pop_icflag(EEG, [NaN NaN;0.9 1;0.9 1;NaN NaN;NaN NaN;NaN NaN;NaN NaN]);
    EEG = pop_subcomp( EEG, [], 0);
    
    %Interpolate removed channels
    EEG = pop_interp(EEG, chanlocs);
    
    EEG = pop_saveset(EEG, 'filename',strcat(subj, '_clean_noref.set'),'filepath',strcat('C:\\Users\\Owner\\Documents\\MATLAB\\REU_data\\5F_EEG_data\\preprocessing\\'));

    % epoching but apparently i can just do it in mne ?
    % [ALLEEG EEG CURRENTSET ALLCOM] = eeglab;
    % EEG = pop_loadset('filename', strcat(subj, '_clean.set'),'filepath','C:\\Users\\Owner\\Documents\\MATLAB\\REU data\\5F EEG data\\');
    % [ALLEEG, EEG, CURRENTSET] = eeg_store( ALLEEG, EEG, 0 );
    % EEG = pop_epoch( EEG, {  '1'  '2'  '3'  '4'  '5'  }, [0         2], 'newname', 'epochs', 'epochinfo', 'yes');
    % [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 2,'gui','off');
    % EEG = pop_saveset(EEG, 'filename',strcat(subj, '_epoched.set'),'filepath',strcat('C:\\Users\\Owner\\Documents\\MATLAB\\REU_data\\5F_EEG_data\\', subj, '\\'));

end