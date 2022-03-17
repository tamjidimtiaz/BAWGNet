clc
clear all
close all

location = 'F:\nucleus data\stage1_train\00ae65c1c6631ae6f2be1a449902976e6eb8483bf6b0740d00530220832c6d3e\masks\*.png';
% img = imread('F:\nucleus data\MonuSeg\Training\GroundTruth\TCGA-G9-6363-01Z-00-DX1_bin_mask.png');
ds = imageDatastore(location);
count = 0;

rec_image = zeros(256,320);
while hasdata(ds)

img = read(ds);
[B,L] = bwboundaries(img,'noholes');
% image = createMask(B{1,1});
% %Image raed, ensure that image_bw is the binary image
image = im2bw(img);
% %Extration largest 1 blob only
% for i=1:length(B) 
object1=bwareafilt(image,1);
% Find the rows and colums of object1
[rows_obj1,cols_obj1]=find(object1);
% Find second object
object2=image & imcomplement(object1);
B = cell2mat( B );
% Now use the rows_obj1 & rows_obj1 as you want
% Also rows_obj2 & cols_obj2
matrix_A1=[rows_obj1 cols_obj1];

dist = zeros(length(B),1);
for i=1:length(matrix_A1)
    for j=1:length(B)
        dist(j) = pdist([matrix_A1(i,:);B(j,:)],'euclidean');
    end
    rec_image(matrix_A1(i,1),matrix_A1(i,2)) = max(dist)-min(dist);
end

count = count+1
end


imshow(rec_image,[]);
