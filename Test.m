warning off;
clear all;
close all;
clc;

% IMAGE ACQUISITION
% Module 0

ImageName=uigetfile('*.jpg;*.png','Pick an Image');
Image=imread(ImageName); %Read the Image into MATLAB
figure,imshow(Image);
title('Input Image');

% Intensity Image 
[r c d]=size(Image); %Calculate Row Column Dimention (Layers) of the Image

if d>2
    Image2=rgb2gray(Image); %If input image is RGB, then convert it into Gray format to calculate the intensity of the input image
else
    Image2=Image;
end

% Module 1
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% IMAGE DENOISING OR RESTORATION

% Adaptive Anisotropic  Filter
Iterations = 5; %Number of Iterations to be denoised
DiffusionPara = 30; %Normalization factor
Image2 = double(Image2); %Double Precision ... to improve the MATLAB Memory
AnistoDiffuse = Image2;

% Center pixel distances.
X_Distance = 1; %Spacing between X and Y Directional coordinates
Y_Distance = 1; %Spacing between X and Y Directional coordinates
dd = sqrt(2); %Scaling Factor 

% 2D convolution masks - finite differences - Impulse Response coefficients
% to do convolutional filter
Mask1 = [0 1 0; 0 -1 0; 0 0 0];
Mask2 = [0 0 0; 0 -1 0; 0 1 0];
Mask3 = [0 0 0; 0 -1 1; 0 0 0];
Mask4 = [0 0 0; 1 -1 0; 0 0 0];
Mask5 = [0 0 1; 0 -1 0; 0 0 0];
Mask6 = [0 0 0; 0 -1 0; 0 0 1];
Mask7 = [0 0 0; 0 -1 0; 1 0 0];
Mask8 = [1 0 0; 0 -1 0; 0 0 0];

% Anisotropic diffusion.
for t = 1:Iterations

        % performs multidimensional filtering using convolution
        Filter1 = imfilter(AnistoDiffuse,Mask1,'conv');
        Filter2 = imfilter(AnistoDiffuse,Mask2,'conv');   
        Filter3 = imfilter(AnistoDiffuse,Mask4,'conv');
        Filter4 = imfilter(AnistoDiffuse,Mask3,'conv');   
        Filter5 = imfilter(AnistoDiffuse,Mask5,'conv');
        Filter6 = imfilter(AnistoDiffuse,Mask6,'conv');   
        Filter7 = imfilter(AnistoDiffuse,Mask7,'conv');
        Filter8 = imfilter(AnistoDiffuse,Mask8,'conv'); 
% %         2D Convolution Filter ends here
% Reciprocity theorem
        
        Diffusion1 = 1./(1 + (Filter1/DiffusionPara).^2);
        Diffusion2 = 1./(1 + (Filter2/DiffusionPara).^2);
        Diffusion3 = 1./(1 + (Filter3/DiffusionPara).^2);
        Diffusion4 = 1./(1 + (Filter4/DiffusionPara).^2);
        Diffusion5 = 1./(1 + (Filter5/DiffusionPara).^2);
        Diffusion6 = 1./(1 + (Filter6/DiffusionPara).^2);
        Diffusion7 = 1./(1 + (Filter7/DiffusionPara).^2);
        Diffusion8 = 1./(1 + (Filter8/DiffusionPara).^2);
        
        % Discrete Partial Differential Function to Reconstruct the Image
        AnistoDiffuse = AnistoDiffuse + 0.1429*((1/(Y_Distance^2))*Diffusion1.*Filter1 + (1/(Y_Distance^2))*Diffusion2.*Filter2 +(1/(X_Distance^2))*Diffusion3.*Filter3 + (1/(X_Distance^2))*Diffusion4.*Filter4 + ...
                  (1/(dd^2))*Diffusion5.*Filter5 + (1/(dd^2))*Diffusion6.*Filter6 +(1/(dd^2))*Diffusion7.*Filter7 + (1/(dd^2))*Diffusion8.*Filter8 );           
end

AnistoDiffuse=uint8(ceil(AnistoDiffuse));
figure,imshow(AnistoDiffuse);
title('Filtered Image using Anisotropic Diffusion Filter');
xx=imsubtract(uint8(Image2),uint8(AnistoDiffuse));
figure,imshow(xx*10,[])
title('Noise Content on the Image before denoising');
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
x=uint8(Image2);
p=uint8(AnistoDiffuse);
si=size(x);
m=si(1);
n=si(2);
x=double(x);
p=double(p);
mse=0;
for i=1:m
    for j=1:n
    mse=mse+(x(i,j)-p(i,j))^2;
    end
end
mse=mse/(m*n);
psn=10*log10((255^2)/mse);
disp('Peak Signal to Noise Ratio using Anisotropic Diffusion Filter ...');
disp(psn);
disp('Mean Square Error using Anisotropic Diffusion Filter ...');
disp(mse);

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% Module 2

% IMAGE ENHANCEMENT

% Intensity Thresholding
MediumThresholding=0.5; %Medium Thresholding limit .5
LowerThresholding=0.008; %Lower Thresholding limit 0
UpperThresholding=0.992; %Upper thresholding limit 1
AMF=AnistoDiffuse;
[R C D]=size(AMF); %If D=3 color  Image, If D=1 Gray  image

if D==3 %Condition for Image
    ColorImageUpperThreshold=0.04; %Bandwidth of the image 0.05 (+)
    ColorImageLowerThreshold=-0.04;%Bandwidth of the image -0.05 (-)
%     National Television System Committee ... Which contains Luminance
    NTSC=rgb2ntsc(AMF); %Standard color format (National Television Standard Color)
    MeanAdjust=ColorImageUpperThreshold-mean(mean(NTSC(:,:,2)));%Medium Intensity Layer
    NTSC(:,:,2)=NTSC(:,:,2)+MeanAdjust*(0.596-NTSC(:,:,2));
    MeanAdjust=ColorImageLowerThreshold-mean(mean(NTSC(:,:,3))); %Blue Layer
    NTSC(:,:,3)=NTSC(:,:,3)+MeanAdjust*(0.523-NTSC(:,:,3));
else
    NTSC=double(AMF)./255; %All the image class is uint8 (unsigned Integer 8) 2^8=256.. there is linear variation
%     from 0 to 255) ... InputImage should be converted from uint8 to double
% '.'Scalar Product
end
% Mean Adjustment on the  Input Image 
MeanAdjust=MediumThresholding-mean(mean(NTSC(:,:,1)));
NTSC(:,:,1)=NTSC(:,:,1)+MeanAdjust*(1-NTSC(:,:,1)); %Mean adjustment formula
if D==3
    NTSC=ntsc2rgb(NTSC);
end
AMF=NTSC.*255; %uint8

%--------------------calculation of Minima and Maxima of the Mean adjusted InputImage ----------------------
for k=1:D
    Sort=sort(reshape(AMF(:,:,k),R*C,1)); %Convert the MxN matrix into column Matrix (Mx1)
    Minima(k)=Sort(ceil(LowerThresholding*R*C)); %Calculate the Minima
    Maxima(k)=Sort(ceil(UpperThresholding*R*C));%Calculate the maxima
end
%----------------------------------------------------------------------
if D==3
    Minima=rgb2ntsc(Minima);
    Maxima=rgb2ntsc(Maxima);
end
%----------------------------------------------------------------------
AMF=(AMF-Minima(1))/(Maxima(1)-Minima(1));%Ganzolez book 'Fundamentals of Digital Image Processing'
Enhancement=uint8(AMF.*255);
figure,imshow(Enhancement);
title('Contrast Enhanced InputImage');