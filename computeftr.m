function [val]=computeftr(ftr,img,state)
%   ---------------------------------> x
%   |
%   |
%   |
%   |
%   |
%   |
%   |
%   \/
%   y
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%to calculate harr-like features. Note that I did not use integral 
%image for computing. You can speed it up by using integral image
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
num = size(ftr.rects,1);

orix = state(1,1);
oriy = state(1,2);
width = state(1,3);
height = state(1,4);

values = zeros(num,1);


for i = 1:num
    lux = orix + ftr.rects(i,1);
    luy = oriy + ftr.rects(i,2);
    ftrwidth = ftr.rects(i,3);
    ftrheight = ftr.rects(i,4);
    values(i,1) = sum(sum(img(luy:luy+ftrheight,lux:lux+ftrwidth)));
end

val = sum(values.*ftr.weights,1);
    