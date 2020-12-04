%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This code is matlab version for MIL tracker, which was proposed by Boris
% Babenko (see http://vision.ucsd.edu/~bbabenko/project_miltrack.shtml for 
% more details). I rewrote it for understand it easily. The copyright
% belongs to Boris. Please use it for adacemic purpose. And it is not 
% guaranteed to be safe, thus you should use this code at own risk 
%
% For bug report, please mail to whluo.china@gmail.com
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clc
clear

rand('state',sum(100*clock));    % Shuffle the pack!
randn('state',sum(100*clock));   % Shuffle the pack!

InputDir='.\david\imgs\img';
OutputDir='.\result\';

num_frames = 462;
inistate = [122 58 75 97];%coordinate of the top left corner, width an height
states = zeros(num_frames,4);
states(1,:) = inistate;

%set parameters
trparams.posradtrain = 4;
trparams.negnumtrain = 65;
trparams.posmaxtrain = 1000000;

trparams.srchwinsz = 25;
trparams.initpostrainrad = 3;
trparams.initnegnumtrain = 65;
trparams.states = inistate;

%parameters of strong classfier 
clfparams.numsel = 50;
clfparams.numfeat = 250;
clfparams.lrate = 0.85;  %learning rate
clfparams.selectedftr = zeros(50,5);

%sep feature parameters
ftrparams.width = 75;
ftrparams.height = 97;
ftrparams.minnumrect = 2;
ftrparams.maxnumrect = 6;

ftrparams.weight = 0;
ftrparams.numrects = 0;
ftrparams.rsums = 0;
ftrparams.maxsum = 0;

ftrs = generateharrftrs(ftrparams);

for FrameNum=1:num_frames
    FrameNum
    I0 = imread(sprintf('%s%05d%s',InputDir,FrameNum,'.png'));
    
    c = 0;
    [m n c] = size(I0);
    if c==3
        I1 = I0;
        I0=rgb2gray(I0);    
    else
        I1 = repmat(I0,[1,1,3]);%for drawing result
    end
    %%%%%%%%initialization 
    if FrameNum == 1
        [clfle] = iniclfweakle(clfparams); 
        possamples = sampleimage(trparams,states(FrameNum,:),1,I0);
        negsamples = sampleimage(trparams,states(FrameNum,:),0,I0);
        [posftrval,negftrval]=computeharr(possamples,negsamples,ftrs,I0);
        [clf] = updateclfweakle(posftrval,negftrval,clfle,clfparams,0); 
        [posres,negres] = getlearnres(clf,posftrval,negftrval);
        selectors = getselectors(posres,negres,clfparams.numsel);
    %%%%%%%%%%run time 
    else
        detectx = sampleimage(trparams,states(FrameNum-1,:),4,I0);
        prob = classify(detectx,selectors,clf,I0,ftrs);
        [maxprob,idx] = max(prob);
        states(FrameNum,:) = detectx(idx,:);
        newim = draw_rect_to_im(I1,states(FrameNum,:));
        figure(1)
        imshow(newim);
        hold on
        imwrite(newim,sprintf('%s_%04d.jpg',[OutputDir 'TrackResult'],FrameNum));
   
        tracknegsamples = sampleimage(trparams,states(FrameNum,:),2,I0);
        trackpossamples = sampleimage(trparams,states(FrameNum,:),3,I0);
        [trackposftrval,tracknegftrval]=computeharr(trackpossamples,tracknegsamples,ftrs,I0);
        [clf]=updateclfweakle(trackposftrval,tracknegftrval,clf,clfparams,1); 
        [trackposres,tracknegres] = getlearnres(clf,trackposftrval,tracknegftrval);
        selectors = getselectors(trackposres,tracknegres,clfparams.numsel);
        clear detectx
        clear prob
        clear idx
        clear tracknegsamples
        clear trackpossamples
        clear trackposftrval
        clear tracknegftrval
        clear trackposres
        clear tracknegres
   end 
end