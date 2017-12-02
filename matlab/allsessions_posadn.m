path_to_data = '/home/guillaume/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/HDCellData/';

file = List2Cell(fullfile(path_to_data,'datasets_PosData.list')); 

x = [];
y = [];

for i = 1:size(file)
    dset = 'Mouse28/Mouse28-140313'
    %dset = file(i)               
    data_dir = fullfile(path_to_data, char(dset));
    cd(data_dir);
    binSize = 0.005; %in seconds
    load('Analysis/BehavEpochs.mat','wakeEp');
    load('Analysis/SpikeData.mat', 'S', 'shank');
    load('Analysis/HDCells.mat'); 
    load('Analysis/GeneralInfo.mat', 'shankStructure'); 
    [~,fbasename,~] = fileparts(pwd);
    %[X,Y,~,wstruct] = LoadPosition_Wrapper(fbasename);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    pos_data_dir = fullfile(path_to_data, 'PosData', char(dset), '/', strcat(fbasename, '_Move.position'));    
    tmp = dlmread(pos_data_dir);
    time = tmp(:,1);
    X = tsd(time,tmp(:,2));
    Y = tsd(time,tmp(:,3));
    ang = tsd(time,tmp(:,4));
    epoch_data_dir = fullfile(path_to_data, 'PosData', char(dset), '/', strcat(fbasename, '_Move.epoch'));    
    ep = dlmread(epoch_data_dir);
    ep = intervalSet(ep(:,1),ep(:,2));
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    hdC = hdCellStats(:,end)==1;    
    %Who were the hd cells recorded in the thalamus?
    thIx = hdC & ismember(shank,shankStructure{'thalamus'});
%     thIx = ismember(shank,shankStructure{'thalamus'});
    %ordering their prefered direction
    [~,prefAngThIx] = sort(hdCellStats(thIx,1));
    %Who were the hd cells recorded in the postsub?
    poIx = hdC & ismember(shank,shankStructure{'postsub'});
%     poIx = ismember(shank,shankStructure{'postsub'});
    %ordering their prefered direction
    [~,prefAngPoIx] = sort(hdCellStats(poIx,1));
    %Restrict exploration to times were the head-direction was correctly
    %detected (you need to detect the blue and red leds, sometimes one of  the
    %two is just not visible)
    wakeEp  = intersect(wakeEp,angGoodEp);
    
    %Restrict all data to wake (i.e. exploration)
    S       = Restrict(S,wakeEp);
     ang     = Restrict(ang,wakeEp);
     X       = Restrict(X,wakeEp);
     Y       = Restrict(Y,wakeEp);
%      linSpd  = Restrict(linSpd,wakeEp);
    %reinitialize indices (there may be hd cells that were not in the thalamus
    %nor in 0the postub. Well, actually it's not possible knowing the structure
    %of this dataset, but you never know)
     hdC     = thIx | poIx;
     thIx    = thIx(hdC);
     poIx    = poIx(hdC);
    %and restrict spike data to hd cells
     S = S(hdC);
    %Bin it!
    Q       = MakeQfromS(S,binSize);
    Q       = Restrict(Q,ep);
    %And give some data
    dQ      = Data(Q);
    dQadn   = dQ(:,thIx);
    dQpos   = dQ(:,poIx);
    smWd = 2.^(0:8);
%     dQadn   = gaussFilt(dQadn,5,0); 
%     dQpos   = gaussFilt(dQpos,5,0);
   
    %Note to regress spike Data to position, you need to get the same timestamps for the two measures. Easy:
    Xq = Restrict(X,Q);
    Yq = Restrict(Y,Q);
    Aq = Restrict(ang,Q);

    cd(path_to_data);    
    data_to_save = struct('X',  Data(Xq), 'Y',  Data(Yq), 'Ang', Data(Aq), 'ADn', dQadn, 'Pos', dQpos);
    
    [m,n] = size(dQadn);
    x = [x n];    
    [m,n] = size(dQpos);
    y = [y n];
    
    tmp = strsplit(char(dset), '/');    
    Sth = struct(S(thIx));
    Spos = struct(S(poIx));
    Sbis =struct('ADn', Sth, 'Pos', Spos);
    Qbis = MakeQfromS(S, 0.001);
    Qbis = Restrict(Qbis, ep);
    dQbis      = Data(Qbis);
    dQadn_bis   = dQbis(:,thIx);
    dQpos_bis   = dQbis(:,poIx);    
    data_to_save_bis = struct('ADn', dQadn_bis, 'Pos', dQpos_bis);
    %save(strcat('/home/guillaume/Prediction_xgb_head_direction/python/data/spikes_binned.', char(tmp(2)), '.mat'), '-struct', 'data_to_save_bis');
    %save(strcat('/home/guillaume/Prediction_xgb_head_direction/python/data/spikes.', char(tmp(2)), '.mat'), '-struct', 'Sbis');
    save(strcat('/home/guillaume/Prediction_xgb_head_direction/python/data/sessions_nosmoothing_5ms/wake/boosted_tree.', char(tmp(2)), '.mat'), '-struct', 'data_to_save');
end    