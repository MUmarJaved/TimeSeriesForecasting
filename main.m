% Initialization
 close all;clc;clear all;

%% Data Acquisition  
 load('processed_data.mat')
 load('features.mat')
 %% System Design
   M=7;  % future >> present+ahead
   L=24;   %<<<< past L-1 behind 
   Data=RecData(Data1_cond,M,L,Features);
   split=0.8; % Divide data into train and test
   
 %% Standardized Training Data
  [TrainSX,TrainSY,Mu,C]=TrainData(Data,split,M,Features);
  cut=0.9; % Divide train into train and validation
   len=size(TrainSX);
   cut_len=round(cut*len(1));
   TrainSXX=TrainSX(1:cut_len,:);
   TrainSYY=TrainSY(1:cut_len,:);
   
 %% Standarized Validation Data
   TrainSVX=TrainSX((cut_len+1):end,:);
   TrainSVY=TrainSY((cut_len+1):end,:);
 
  %% Standardized X and Unstandard Y Test Data
   [TestSX,TestY] = TestData(Data,split,M,Mu,C,Features);
 
 %% store mu, c, TestSX, TestY, TrainSXX, TrainSYY, TrainSVX, TrainSVY 
 len1=size(TestSX);
 len2=size(TestY);
 len=len1(2)+len2(2);
 C=C((len-Features*M+1):len);
 Mu=Mu((len-Features*M+1):len);
% 
%   Data_gen=zeros(size(Data0));
%  for i=2:length(Data)-1
%      k=0;
%      for j=2:i
%     k=Data0+Data_new(j,:) ;
%      end
%      Data_gen=[Data_gen k];
%  end 
%       Data_gen=Data_gen(2:end,:);
%        plot(Data_new(:,1));hold on; plot(Data(:,1));hold on; plot(Data_gen(:,1));
%  writematrix(Data,'Data.csv')
%  K_new=corr(Data_new)
%  K=corr(Data)
 M=Features*M;

 
 writematrix(M,'M.csv')
 
  writematrix(L,'L.csv')
  
 writematrix(Mu,'Mu.csv')
 writematrix(C,'C.csv')
 
 writematrix(TestSX,'TestSX.csv')
 writematrix(TestY,'TestY.csv')
   
 TrainSXX=[TrainSXX;TrainSVX];
 TrainSYY=[TrainSYY;TrainSVY];
 
 writematrix(TrainSXX,'TrainSXX.csv')
 writematrix(TrainSYY,'TrainSYY.csv')
     
 writematrix(TrainSVX,'TrainSVX.csv')
 writematrix(TrainSVY ,'TrainSVY.csv')
 
 
 