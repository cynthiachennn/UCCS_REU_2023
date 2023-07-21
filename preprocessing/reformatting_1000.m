fileList = {'A_408_HFREQ.mat', 'B_309_HFREQ.mat', 'B_311_HFREQ.mat', 'C_429_HFREQ.mat',...
    'E_321_HFREQ.mat', 'E_415_HFREQ.mat', 'E_429_HFREQ.mat', 'F_210_HFREQ.mat',...
'G_413_HFREQ.mat', 'G_428_HFREQ.mat', 'H_804_HFREQ.mat', 'I_719_HFREQ.mat', 'I_723_HFREQ.mat'};
subjList = {'A_408', 'B_309', 'B_311', 'C_429', 'E_321', 'E_415', 'E_429', 'F_210', 'G_413', 'G_428', 'H_804', 'I_719', 'I_723'};
for i=1:13
    fileName = cell2mat(fileList(i));
    subj = cell2mat(subjList(i));

    % reformatting
    % load data
    subj_o = load(strcat('raw_data\', fileName)).o;
    subj_data = subj_o.data';
    subj_marker = subj_o.marker';
    [ALLEEG EEG CURRENTSET ALLCOM] = eeglab;
    
    EEG = pop_importdata('dataformat','array','nbchan',22,'data','subj_data','srate',1000,'pnts',size(subj_data,2),'xmin',0);
    [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 0,'setname','data','gui','off'); 
    EEG = pop_importdata('dataformat','array','nbchan',1,'data','subj_marker','srate',1000,'pnts',size(subj_marker,2),'xmin',0);
    [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1,'setname','events','gui','off'); 
    EEG = pop_chanevent(EEG, 1,'edge','leading','edgelen',0,'delchan','off'); %getting events
    [ALLEEG EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);
    pop_expevents(EEG, strcat('C:\Users\Owner\Documents\MATLAB\REU_data\5F_EEG_data\preprocessing\', subj, '_events.txt'), 'samples');
    [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 2,'retrieve',1,'study',0); 
    EEG = pop_importevent( EEG, 'event',strcat('C:\\Users\\Owner\\Documents\\MATLAB\\REU_data\\5F_EEG_data\\preprocessing\\', subj, '_events.txt'),'fields',{'number','latency','type','urevent'},'skipline',1,'timeunit',NaN);
    EEG.chanlocs = struct('labels', { 'Fp1' 'Fp2' 'F3' 'F4' 'C3' 'C4' 'P3' 'P4' 'O1' 'O2' 'A1' 'A2' 'F7' 'F8' 'T7' 'T8' 'P7' 'P8' 'Fz' 'Cz' 'Pz' 'X3'});
    EEG = pop_chanedit(EEG, 'load', '22ch.loc'); %add locations
    EEG = pop_resample( EEG, 200); % resample as 200 Hz
    % EEG = pop_saveset( EEG, 'filename',strcat(subj,'.set'),'filepath',strcat('C:\\Users\\Owner\\Documents\\MATLAB\\REU_data\\5F_EEG_data\\preprocessing', subj, '\\'));
    % [ALLEEG EEG] = eeg_store(ALLEEG, EEG, CURRENTSET); %this saves basic 22 ch w/ events and locs 200 hz file 

    %load data
    % if ~exist('pop_loadset'), eeglab; end
    % EEG = pop_loadset('filename', strcat(subj, '\', subj, '.set'));
    
    %remove xtra channels
    [ALLEEG, EEG, CURRENTSET] = eeg_store( ALLEEG, EEG, 0 );
    %removeChans = {'X3'}; 
    EEG = pop_select( EEG, 'rmchannel', {'X3', 'A1', 'A2'}); %REMOVE EVENT CHANNEL WITHOUT CREATING EVENTS 
    [ALLEEG EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);
    % EEG = pop_reref( EEG, [20 21] ); %rereference/get rid of mastoid for now
    
    %new, reformatted set! :D
    [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1,'setname','reref','gui','off'); %make a new set out of it i guessss
    EEG = pop_saveset( EEG, 'filename',strcat(subj, '_19_noref.set'),'filepath',strcat('C:\\Users\\Owner\\Documents\\MATLAB\\REU_data\\5F_EEG_data\\preprocessing\\'));

end