% normalization function

function [f_norm]= featureNorm(a)

v= a(:);
     s = (v-min(v))/(max(v)-min(v));
%        s= mapminmax (v,0,1);
      f_norm = reshape(s,size(a,1),size(a,2),...
                         size(a,3));