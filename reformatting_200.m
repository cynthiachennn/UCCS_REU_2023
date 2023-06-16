fileList = {'5F-A-405.mat', '5F-B-110.mat', '5F-B-316.mat', '5F-C-204.mat' '5F-F-027.mat', '5F-F-209.mat'};
subjList = {'A_405', 'B_110', 'B_316', 'C_204', 'F_027', 'F_209'};
% sizes: {718600, 724600, 718800, 722200, 736600, 718400}
for i=1:6
    fileName = cell2mat(fileList(i));
    subj = cell2mat(subjList(i));

    % reformatting
    % load data
    subj_o = load(strcat('og_data\', fileName)).o;
    subj_data = subj_o.data';
    subj_marker = subj_o.marker';
    [ALLEEG EEG CURRENTSET ALLCOM] = eeglab;
    
    EEG = pop_importdata('dataformat','array','nbchan',22,'data','subj_data','srate',200,'pnts',size(subj_data,2),'xmin',0);
    [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 0,'setname','data','gui','off'); 
    EEG = pop_importdata('dataformat','array','nbchan',1,'data','subj_marker','srate',200,'pnts',size(subj_marker,2),'xmin',0);
    [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1,'setname','events','gui','off'); 
    EEG = pop_chanevent(EEG, 1,'edge','leading','edgelen',0,'delchan','off'); %getting events
    [ALLEEG EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);
    pop_expevents(EEG, strcat('C:\Users\Owner\Documents\MATLAB\REU_data\5F_EEG_data\', subj, '\events.txt'), 'samples');
    [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 2,'retrieve',1,'study',0); 
    EEG = pop_importevent( EEG, 'event',strcat('C:\\Users\\Owner\\Documents\\MATLAB\\REU_data\\5F_EEG_data\\', subj, '\\events.txt'),'fields',{'number','latency','type','urevent'},'skipline',1,'timeunit',NaN);
    EEG.chanlocs = struct('labels', { 'Fp1' 'Fp2' 'F3' 'F4' 'C3' 'C4' 'P3' 'P4' 'O1' 'O2' 'A1' 'A2' 'F7' 'F8' 'T7' 'T8' 'P7' 'P8' 'Fz' 'Cz' 'Pz' 'X3'});
    EEG = pop_chanedit(EEG, 'load', '22ch.loc'); %add locations
    EEG = pop_saveset( EEG, 'filename',strcat(subj,'.set'),'filepath',strcat('C:\\Users\\Owner\\Documents\\MATLAB\\REU_data\\5F_EEG_data\\', subj, '\\'));
    [ALLEEG EEG] = eeg_store(ALLEEG, EEG, CURRENTSET); %this saves basic 22 ch w/ events and locs 200 hz file 

    %load data
    if ~exist('pop_loadset'), eeglab; end
    EEG = pop_loadset('filename', strcat(subj, '\', subj, '.set'));
    
    %remove xtra channels
    [ALLEEG, EEG, CURRENTSET] = eeg_store( ALLEEG, EEG, 0 );
    %removeChans = {'X3'}; 
    EEG = pop_select( EEG, 'rmchannel', {'X3'}); %REMOVE EVENT CHANNEL WITHOUT CREATING EVENTS 
    [ALLEEG EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);
    EEG = pop_reref( EEG, [20 21] ); %rereference/get rid of mastoid for now
    
    %new, reformatted set! :D
    [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1,'setname','reref','gui','off'); %make a new set out of it i guessss
    EEG = pop_saveset( EEG, 'filename',strcat(subj, '_19.set'),'filepath',strcat('C:\\Users\\Owner\\Documents\\MATLAB\\REU_data\\5F_EEG_data\\', subj, '\\'));

end
% 
% function reformatting(fileName, subj)
%     % FOR ONE SUBJECT!
%     % get info
%     subj_o = load(fileName).o;
%     subj_data = subj_o.data';
%     subj_marker = subj_o.marker';
%     [ALLEEG EEG CURRENTSET ALLCOM] = eeglab;
% 
%     EEG = pop_importdata('dataformat','array','nbchan',22,'data','subj_data','srate',200,'pnts',size(subj_data,2),'xmin',0);
%     [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 0,'setname','data','gui','off'); 
%     EEG = pop_importdata('dataformat','array','nbchan',1,'data','subj_marker','srate',200,'pnts',size(subj_marker,2),'xmin',0);
%     [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1,'setname','events','gui','off'); 
%     EEG = pop_chanevent(EEG, 1,'edge','leading','edgelen',0,'delchan','off'); %getting events
%     [ALLEEG EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);
%     pop_expevents(EEG, strcat('C:\Users\Owner\Documents\MATLAB\REU data\5F EEG data\events', subj, '.txt'), 'samples');
%     [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 2,'retrieve',1,'study',0); 
%     EEG = pop_importevent( EEG, 'event',strcat('C:\\Users\\Owner\\Documents\\MATLAB\\REU data\\5F EEG data\\events', subj, '.txt'),'fields',{'number','latency','type','urevent'},'skipline',1,'timeunit',NaN);
%     EEG.chanlocs = struct('labels', { 'Fp1' 'Fp2' 'F3' 'F4' 'C3' 'C4' 'P3' 'P4' 'O1' 'O2' 'A1' 'A2' 'F7' 'F8' 'T7' 'T8' 'P7' 'P8' 'Fz' 'Cz' 'Pz' 'X3'});
%     EEG = pop_chanedit(EEG, 'load', '22ch.loc'); %add locations
%     EEG = pop_saveset( EEG, 'filename',strcat(subj,'.set'),'filepath','C:\\Users\\Owner\\Documents\\MATLAB\\REU data\\5F EEG data\\');
%     [ALLEEG EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);
% 
%     fileName = fullfile(strcat(subj, '.set')); %set file
% 
%     %load data
%     if ~exist('pop_loadset'), eeglab; end
%     EEG = pop_loadset('filename', fileName);
% 
%     %remove xtra channels
%     [ALLEEG, EEG, CURRENTSET] = eeg_store( ALLEEG, EEG, 0 );
%     %removeChans = {'X3'}; 
%     EEG = pop_select( EEG, 'rmchannel', {'X3'}); %REMOVE EVENT CHANNEL WITHOUT CREATING EVENTS 
%     [ALLEEG EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);
%     EEG = pop_reref( EEG, [20 21] ); %rereference/get rid of mastoid for now
% 
%     %new, reformatted set! :D
%     [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1,'setname','reref','gui','off'); %make a new set out of it i guessss
%     EEG = pop_saveset( EEG, 'filename',strcat(subj, '_19.set'),'filepath','C:\\Users\\Owner\\Documents\\MATLAB\\REU data\\5F EEG data\\');
% end
% 
% function preprocessing(subj)
%     %one subject at a time:
%     EEG = pop_loadset('filename',strcat(subj, '_19.set'),'filepath','C:\\Users\\Owner\\Documents\\MATLAB\\REU data\\5F EEG data\\');
%     chanlocs = EEG.chanlocs;
% 
%     %filter data
%     EEG = pop_eegfiltnew(EEG, 'locutoff', 0.5);
% 
%     % remove bad channels and bad portions of data
%     EEG = pop_clean_rawdata(EEG, 'FlatlineCriterion',4,'ChannelCriterion',0.85,'LineNoiseCriterion',4,'Highpass','off',...
%         'BurstCriterion',20,'WindowCriterion',0.25,'BurstRejection','on','Distance','Euclidian','WindowCriterionTolerances',[-Inf 7] );
% 
%     %common average referencing ??
% 
%     % Run ICA and IC Label
%     EEG = pop_runica(EEG, 'icatype', 'picard', 'maxiter',500);
%     EEG = pop_iclabel(EEG, 'default');
%     EEG = pop_icflag(EEG, [NaN NaN;0.9 1;0.9 1;NaN NaN;NaN NaN;NaN NaN;NaN NaN]);
%     EEG = pop_subcomp( EEG, [], 0);
% 
%     %Interpolate removed channels
%     EEG = pop_interp(EEG, chanlocs);
% 
%     EEG = pop_saveset(EEG, 'filename',strcat(subj, '_clean.set'),'filepath','C:\\Users\\Owner\\Documents\\MATLAB\\REU data\\5F EEG data\\');
% end 
% 
% function epoch(subj)
%     [ALLEEG EEG CURRENTSET ALLCOM] = eeglab;
%     EEG = pop_loadset('filename', strcat(subj, '_clean.set'),'filepath','C:\\Users\\Owner\\Documents\\MATLAB\\REU data\\5F EEG data\\');
%     [ALLEEG, EEG, CURRENTSET] = eeg_store( ALLEEG, EEG, 0 );
%     EEG = pop_epoch( EEG, {  '1'  '2'  '3'  '4'  '5'  }, [0         1.5], 'newname', 'epochs', 'epochinfo', 'yes');
%     [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 2,'gui','off');
%     EEG = pop_saveset(EEG, 'filename',strcat(subj, '_epoched.set'),'filepath','C:\\Users\\Owner\\Documents\\MATLAB\\REU data\\5F EEG data\\');
% end
