path_to_data = '/home/guillaume/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/HDCellData/';

file = List2Cell(fullfile(path_to_data,'datasets_PosData.list')); 

x = [];
y = [];

for i = 1:size(file)
    dset = file(i)               
    data_dir = fullfile(path_to_data, char(dset));
    cd(data_dir);
    %binSize = 0.100; %in seconds
    load('Analysis/BehavEpochs.mat','wakeEp');
    load('Analysis/SpikeData.mat', 'S', 'shank');
    load('Analysis/HDCells.mat'); 
    load('Analysis/GeneralInfo.mat', 'shankStructure'); 
    [~,fbasename,~] = fileparts(pwd);
    %[X,Y,~,wstruct] = LoadPosition_Wrapper(fbasename);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     pos_data_dir = fullfile(path_to_data, 'PosData', char(dset), '/', strcat(fbasename, '_Move.position'));    
%     tmp = dlmread(pos_data_dir);
%     time = tmp(:,1);
%     X = tsd(time,tmp(:,2));
%     Y = tsd(time,tmp(:,3));
%     ang = tsd(time,tmp(:,4));
%     epoch_data_dir = fullfile(path_to_data, 'PosData', char(dset), '/', strcat(fbasename, '_Move.epoch'));    
%     ep = dlmread(epoch_data_dir);
%     ep = intervalSet(ep(:,1),ep(:,2));
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    hdC = hdCellStats(:,end)==1;    
    %Who were the hd cells recorded in the thalamus?
    thIx = hdC & ismember(shank,shankStructure{'thalamus'});
    %thIx = ismember(shank,shankStructure{'thalamus'});
    %ordering their prefered direction
    [~,prefAngThIx] = sort(hdCellStats(thIx,1));
    %Who were the hd cells recorded in the postsub?
    poIx = hdC & ismember(shank,shankStructure{'postsub'});
    %poIx = ismember(shank,shankStructure{'postsub'});
    %ordering their prefered direction
    [~,prefAngPoIx] = sort(hdCellStats(poIx,1));
    %Restrict exploration to times were the head-direction was correctly
    %detected (you need to detect the blue and red leds, sometimes one of  the
    %two is just not visible)
    %wakeEp  = intersect(wakeEp,angGoodEp);
    
    %Restrict all data to wake (i.e. exploration)
    S       = Restrict(S,wakeEp);
    
%     ang     = Restrict(ang,wakeEp);
%     X       = Restrict(X,wakeEp);
%     Y       = Restrict(Y,wakeEp);
%     linSpd  = Restrict(linSpd,wakeEp);
    %reinitialize indices (there may be hd cells that were not in the thalamus
    %nor in 0the postub. Well, actually it's not possible knowing the structure
    %of this dataset, but you never know)
     %hdC     = thIx | poIx;
     thIx    = thIx(hdC);
     poIx    = poIx(hdC);
    %and restrict spike data to hd cells
     S = S(hdC);
     
     Sadn = S(thIx);
     Spos = S(poIx);
     
         
    cd(path_to_data);    
    data_to_save = struct('adn',  Sadn, 'pos', Spos);
    
    
    tmp = strsplit(char(dset), '/');    
    save(strcat('/home/guillaume/Prediction_xgb_head_direction/python/data/spike_timing/wake/spike_timing.', char(tmp(2)), '.mat'), '-struct', 'data_to_save');
end
    