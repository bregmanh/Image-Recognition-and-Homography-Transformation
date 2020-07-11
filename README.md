# Image Recognition and Stitching

This  is one of my assignments for the smart Structure Technology Class at the University of Waterloo.  I learned about RANSAC (estimate parameters of a mathematical model from a set of observed data that contains outliers) and subsequently was able to use feature recognition to measure distances on an image as well as perform image stitching. 

## Part 1: Theory Practice

**What is the Laplacian of Gaussian (LoG)? When do we use LoG?**

Laplacian filters are derivative filters used to find areas of rapid change (edges) in images. Since derivative filters are very sensitive to noise, it is common to smooth the image (e.g., using a Gaussian filter) before applying the Laplacian. This two-step process is call the Laplacian of Gaussian (LoG) operation (Source: academic.mu.edu).

LOG filters are circularly symmetric and are used to detect "blobs" in 2D. "Blobs" at different sizes of the image can be detected by convolving the image with a LOG filter of different scales.

**What is a difference of the Guassian (DoG)? When do we use and what is DoG advantageous compared to LoG?**

The Difference of Gaussian is a filter that identifies edges. The DOG filter is similar to the [LOG](http://www.roborealm.com/help/LOG.php) filter in that it is a two stage edge detection process.

The DOG performs edge detection by performing a Gaussian blur on an image at a specified sigma or standard deviation. The resulting image is a blurred version of the source image. The module then performs another blur with a sharper theta that blurs the image less than previously. The final image is then calculated by replacing each pixel with the difference between the two blurred images and detecting when the values cross zero, i.e. negative becomes positive and vice versa. The resulting zero crossings will be focused at edges or areas of pixels that have some variation in their surrounding neighborhood (Source: Roborealm).

These Difference of Gaussian is approximately equivalent to the Laplacian of Gaussian. However, it is a computationally less intensive process with a simple subtraction (fast and efficient).



## Part 2: RANSAC (Line Fitting)

**a) fitting a line through an elipse**

```matlab
clc;close all;
load('prob3_ellipse.mat','x','y');
syms xx yy; %for plotting the line
x_data=x;
y_data=y;

N=50; %number of trials

coeff=zeros(6,1);

valid_pts_final=zeros(N,size(x_data,2));

valid=zeros(1,N); %keeping track of number of valid points

threshold = 0.0001;
for ii=1:N 
pt_idx = randperm(size(x_data,2),5);
x_sub = x_data(pt_idx);
y_sub = y_data(pt_idx);

coeff=null([x_sub(1)^2, x_sub(1)*y_sub(1), y_sub(1)^2, x_sub(1), y_sub(1), 1;
    x_sub(2)^2, x_sub(2)*y_sub(2), y_sub(2)^2, x_sub(2), y_sub(2), 1;
    x_sub(3)^2, x_sub(3)*y_sub(3), y_sub(3)^2, x_sub(3), y_sub(3), 1;
    x_sub(4)^2, x_sub(4)*y_sub(4), y_sub(4)^2, x_sub(4), y_sub(4), 1;
     x_sub(5)^2, x_sub(5)*y_sub(5), y_sub(5)^2, x_sub(5), y_sub(5), 1]);
a=coeff(1);
b=coeff(2);
c=coeff(3);
d=coeff(4);
e=coeff(5);
f=coeff(6);

y_pos=y_positive(x_data,a,b,c,d,e,f);
y_neg=y_negative(x_data,a,b,c,d,e,f);

dist_pt=zeros(1,size(y_data,2));
%calculating the distance between the y values
for jj=1:size(y_data,2)
    dist_pt(jj)=min(abs(y_data(jj)-y_pos(jj)),abs(y_data(jj)-y_neg(jj)));
end
valid_pt = dist_pt<threshold;

aa(ii)=a;
bb(ii)=b;
cc(ii)=c;
dd(ii)=d;
ee(ii)=e;
ff(ii)=f;
valid(ii)=sum(valid_pt,2);
end

maxval=max(valid);
indx=max(find(valid==maxval));
a=aa(indx);
b=bb(indx);
c=cc(indx);
d=dd(indx);
e=ee(indx);
f=ff(indx);

ellipseplot=a*xx^2+b*xx*yy+c*yy^2+d*xx+e*yy+f;

figure(4);
plot(x_data, y_data, 'ob', 'linewidth', 1.5);hold on;
plot(x_sub, y_sub, 'or', 'linewidth', 3);
fimplicit(ellipseplot,'--g','LineWidth',3);
legend({'data','selected','hyposis line','valid points'},'location','northwest','FontSize',10)
axis tight;grid on; hold off
xlabel('\bf X-axis');
ylabel('\bf Y-axis');
set(gca,'FontSize',10,'linewidth',2,'fontweight','bold');

function y_pos=y_positive(x_data,a,b,c,d,e,f)
for ii=1:size(x_data,2)
    y_pos(ii)=(-(b*x_data(ii)+e)+sqrt((b*x_data(ii)+e)^2-4*c*(a*x_data(ii)^2+a*x_data(ii)+f)))/(2*c*(a*x_data(ii)^2+d*x_data(ii)+f));
end
end
function y_neg=y_negative(x_data,a,b,c,d,e,f)
for ii=1:size(x_data,2)
    y_neg(ii)=(-(b*x_data(ii)+e)-sqrt((b*x_data(ii)+e)^2-4*c*(a*x_data(ii)^2+a*x_data(ii)+f)))/(2*c*(a*x_data(ii)^2+d*x_data(ii)+f));
end
end
```

![elipse](https://github.com/bregmanh/Image-Recognition-and-Homography-Transformation/blob/master/docs/elipse.png?raw=true)

**b) polynomial line fitting**

```matlab
load('prob3_polynomial.mat','x','y');
syms xx yy;

N=20; %number of trials


valid_pts_final=zeros(N,size(x,2));

valid=zeros(1,N); %keeping track of number of valid points

threshold = 1;
for ii=1:N 
pt_idx = randperm(size(x,2),5);
xs = x(pt_idx);
ys = y(pt_idx);

A=[xs(1)^4 xs(1)^3 xs(1)^2 xs(1) 1;
    xs(2)^4 xs(2)^3 xs(2)^2 xs(2) 1;
    xs(3)^4 xs(3)^3 xs(3)^2 xs(3) 1;
    xs(4)^4 xs(4)^3 xs(4)^2 xs(4) 1;
    xs(5)^4 xs(5)^3 xs(5)^2 xs(5) 1];

coeff=inv(A)*[ys(1);ys(2);ys(3);ys(4);ys(5)];

a=coeff(1);
b=coeff(2);
c=coeff(3);
d=coeff(4);
e=coeff(5);

y_poly=a*x.^4+b.*x.^3+c*x.^2+d*x+e;
dist_pt=zeros(1,size(y,2));
for jj=1:size(y,2)
    dist_pt(jj)=abs(y(jj)-y_poly(jj));
end
valid_pt = dist_pt<threshold;
aa(ii)=a;
bb(ii)=b;
cc(ii)=c;
dd(ii)=d;
ee(ii)=e;
valid(ii)=sum(valid_pt,2);
end

maxval=max(valid);
indx=find(valid==maxval);
a=aa(indx);
b=bb(indx);
c=cc(indx);
d=dd(indx);
e=ee(indx);

yy=a*xx^4+b*xx^3+c*xx^2+d*xx+e;

figure(4);
plot(x, y, 'ob', 'linewidth', 1.5);hold on;
plot(xs, ys, 'or', 'linewidth', 3);
fplot(xx, yy);

legend({'data','selected','hyposis line','valid points'},'location','northwest','FontSize',10)
axis tight;grid on; hold off
xlabel('\bf X-axis');
ylabel('\bf Y-axis');
set(gca,'FontSize',10,'linewidth',2,'fontweight','bold');
```

![polynomial](https://github.com/bregmanh/Image-Recognition-and-Homography-Transformation/blob/master/docs/polynomial.png?raw=true)

**c) Plane fitting **

```matlab
close all;
load('prob3_plane.mat','x','y','z');
numx=size(x,1)*size(x,2);
x=reshape(x,1,numx);
y=reshape(y,1,numx);
z=reshape(z,1,numx);

syms xx yy zz;

N=20; %number of trials


valid_pts_final=zeros(N,size(x,2));

valid=zeros(1,N); %keeping track of number of valid points

threshold = 0.000005;
for ii=1:N 
pt_idx = randperm(size(x,2),3);
xs = x(pt_idx);
ys = y(pt_idx);
zs=z(pt_idx);

A=[xs(1) ys(1) zs(1) 1;
    xs(2) ys(2) zs(2) 1;
    xs(3) ys(3) zs(3) 1];

coeff=null(A);

a=coeff(1);
b=coeff(2);
c=coeff(3);
d=coeff(4);

dist_pt=zeros(1,size(y,2));
for jj=1:size(y,2)
    dist_pt(jj)=abs(a*x(jj)+b*y(jj)+c*z(jj)+d)/sqrt(a^2+b^2+c^2);
end
valid_pt = dist_pt<threshold;
aa(ii)=a;
bb(ii)=b;
cc(ii)=c;
dd(ii)=d;

valid(ii)=sum(valid_pt,2);
end

maxval=max(valid);
indx=find(valid==maxval);
a=max(aa(indx));
b=max(bb(indx));
c=max(cc(indx));
d=max(dd(indx));

y_plane=(-a*xx-c*zz-d)/b;

plot3(x,y,z,'b');

hold on;
plot3(xs, ys,zs, 'or', 'linewidth', 3);
ezsurf(xx,y_plane,zz,'k');


legend({'data','selected','hyposis line','valid points'},'location','northwest','FontSize',10)
axis tight;grid on; hold off
xlabel('\bf X-axis');
ylabel('\bf Y-axis');
set(gca,'FontSize',10,'linewidth',2,'fontweight','bold');
```

![plane](https://github.com/bregmanh/Image-Recognition-and-Homography-Transformation/blob/master/docs/plane.png?raw=true)

## Part 3: Measuring Distances on an Image

**a)**

*Note: The H function is shown in b)

```matlab
img = imread('IMG_0086.JPG');
img=imresize(img,[1000, NaN]); %resize to have 1000 rows
imgBW=single(rgb2gray(img));

cover = imread('cover.JPG');
cover=imresize(cover,[1000, NaN]); %resize to have 1000 rows
coverBW=single(rgb2gray(cover));

%scaling
scX=24/size(cover,2); %cm/pixel
scY=31.5/size(cover,1); %cm/pixel

[fcover, dcover] = vl_sift(coverBW) ; %cover features
[fimg, dimg] = vl_sift(imgBW) ; %image features

[matches, scores] = vl_ubcmatch(dcover, dimg) ;

nm=size(matches,2);

pt_cover=fcover(1:2,matches(1,:));
pt_img=fimg(1:2,matches(2,:));

N=1000;  %number of trails
valid=zeros(1,N); %keeping track of number of valid points
dist_threshold = 2;
H_use=[];
for ii=1:N
index=randperm(nm,4);
imgpic=[pt_img(:,index)];
coverpic=[pt_cover(:,index)];

H=ComputeH(imgpic', coverpic');

pt_ieh=inv(H)*[pt_cover;ones(1,nm)];
tmp=pt_ieh./pt_ieh(3,:);
pt_ie=tmp(1:2,:);
a=pt_ie-pt_img;
dist=sqrt(sum(a.^2,1));

valid_pt = dist<dist_threshold;
valid(ii)=sum(valid_pt,2);
H_use=horzcat(H_use,H);
end

maxval=max(valid);
idx=find(valid==maxval);
H_final=H_use(:,(((idx-1)*3+1):((idx-1)*3+3)));

figure(1); imshow(img);
p = drawpolygon('LineWidth',5,'Color','red');
points=p.Position;
point1=H_final*[points(1,1);points(1,2);1];
point2=H_final*[points(2,1);points(2,2);1];

point1_cord=[point1(1)/point1(3);point1(2)/point1(3)];
point2_cord=[point2(1)/point2(3);point2(2)/point2(3)];
delta_x=point1_cord(1)-point2_cord(1);
delta_y=point1_cord(2)-point2_cord(2);

sc=mean([scX,scY]);

distance_4a=sc*sqrt(delta_x^2+delta_y^2)
```

![ruler](https://github.com/bregmanh/Image-Recognition-and-Homography-Transformation/blob/master/docs/ruler.png?raw=true)

![ruler-measure](https://github.com/bregmanh/Image-Recognition-and-Homography-Transformation/blob/master/docs/ruler-measure.png?raw=true)

**b)**

```matlab
img = imread('IMG_1.JPG');
img=imresize(img,[1000, NaN]); %resize to have 1000 rows
imgBW2=single(rgb2gray(img));

cover = imread('Vintage.JPG');
cover=imresize(cover,[1000, NaN]); %resize to have 1000 rows
coverBW2=single(rgb2gray(cover));

%scaling
scX=24/size(cover,2); %cm/pixel
scY=31.5/size(cover,1); %cm/pixel

[fcover, dcover] = vl_sift(coverBW2) ; %cover features
[fimg, dimg] = vl_sift(imgBW2) ; %image features

[matches, scores] = vl_ubcmatch(dcover, dimg) ;

nm=size(matches,2);

pt_cover=fcover(1:2,matches(1,:));
pt_img=fimg(1:2,matches(2,:));

N=1000;  %number of trails
valid=zeros(1,N); %keeping track of number of valid points
dist_threshold = 2;
H_use=[];
for ii=1:N
index=randperm(nm,4);
imgpic=[pt_img(:,index)];
coverpic=[pt_cover(:,index)];

H=ComputeH(imgpic', coverpic');

pt_ieh=inv(H)*[pt_cover;ones(1,nm)];
tmp=pt_ieh./pt_ieh(3,:);
pt_ie=tmp(1:2,:);
a=pt_ie-pt_img;
dist=sqrt(sum(a.^2,1));

valid_pt = dist<dist_threshold;
valid(ii)=sum(valid_pt,2);
H_use=horzcat(H_use,H);
end

maxval=max(valid);
idx=find(valid==maxval);
H_final=H_use(:,(((idx-1)*3+1):((idx-1)*3+3)));

figure(1); imshow(img);
p = drawpolygon('LineWidth',5,'Color','red');
points=p.Position;
point1=H_final*[points(1,1);points(1,2);1];
point2=H_final*[points(2,1);points(2,2);1];

point1_cord=[point1(1)/point1(3);point1(2)/point1(3)];
point2_cord=[point2(1)/point2(3);point2(2)/point2(3)];
delta_x=point1_cord(1)-point2_cord(1);
delta_y=point1_cord(2)-point2_cord(2);

sc=mean([scX,scY]);

distance_4b=sc*sqrt(delta_x^2+delta_y^2)


function H=ComputeH(imgpic, coverpic)
%image coords:
x1_img=imgpic(1,1);
x2_img=imgpic(2,1);
x3_img=imgpic(3,1);
x4_img=imgpic(4,1);
y1_img=imgpic(1,2);
y2_img=imgpic(2,2);
y3_img=imgpic(3,2);
y4_img=imgpic(4,2);

%real world coords:
x1_cover=coverpic(1,1);
x2_cover=coverpic(2,1);
x3_cover=coverpic(3,1);
x4_cover=coverpic(4,1);
y1_cover=coverpic(1,2);
y2_cover=coverpic(2,2);
y3_cover=coverpic(3,2);
y4_cover=coverpic(4,2);

A=[x1_img y1_img 1 0 0 0 -x1_img*x1_cover -x1_cover*y1_img -x1_cover;
    0 0 0 x1_img y1_img 1 -x1_img*y1_cover -y1_cover*y1_img -y1_cover;
    x2_img y2_img 1 0 0 0 -x2_img*x2_cover -x2_cover*y2_img -x2_cover;
    0 0 0 x2_img y2_img 1 -x2_img*y2_cover -y2_cover*y2_img -y2_cover;
    x3_img y3_img 1 0 0 0 -x3_img*x3_cover -x3_cover*y3_img -x3_cover;
    0 0 0 x3_img y3_img 1 -x3_img*y3_cover -y3_cover*y3_img -y3_cover;
    x4_img y4_img 1 0 0 0 -x4_img*x4_cover -x4_cover*y4_img -x4_cover;
    0 0 0 x4_img y4_img 1 -x4_img*y4_cover -y4_cover*y4_img -y4_cover];
B=null(A);
B=B(:,1);
H=reshape(B,3,3)';

end
```

![desk-measure](https://github.com/bregmanh/Image-Recognition-and-Homography-Transformation/blob/master/docs/desk-img.png?raw=true)

![desk-measure](https://github.com/bregmanh/Image-Recognition-and-Homography-Transformation/blob/master/docs/desk-measure.png?raw=true)

**c)**

Measurements can be slightly less accurate using the RANSAC feature recognition method than manually inputting coordinates of corners. That is because there may be inaccuracies in the feature matching process resulting in a slightly skewed measurement. However, with the threshold and number of trials that I have set, I managed to get accurate results.

## Image Detection and Recognition

```matlab
img = imread('IMG_0215.JPG');%this image can be changed to any of the other 17 images

img=imresize(img,[1000, NaN]); %resize to have 1000 rows
imgBW=single(rgb2gray(img));

cover1 = imread('cover1.png');
cover1=imresize(cover1,[1000, NaN]); %resize to have 1000 rows
coverBW1=single(rgb2gray(cover1));

cover2 = imread('cover2.jpg');
cover2=imresize(cover2,[1000, NaN]); %resize to have 1000 rows
coverBW2=single(rgb2gray(cover2));

cover3 = imread('cover3.jpg');
cover3=imresize(cover3,[1000, NaN]); %resize to have 1000 rows
coverBW3=single(rgb2gray(cover3));

cover4 = imread('cover4.jpg');
cover4=imresize(cover4,[1000, NaN]); %resize to have 1000 rows
coverBW4=single(rgb2gray(cover4));

%scaling is the same for all covers
scX1=24/size(cover1,2); %cm/pixel
scY1=32/size(cover1,1); %cm/pixel

scX2=21/size(cover1,2); %cm/pixel
scY2=32/size(cover1,1); %cm/pixel

scX3=21/size(cover1,2); %cm/pixel
scY3=31/size(cover1,1); %cm/pixel

scX4=23/size(cover1,2); %cm/pixel
scY4=31/size(cover1,1); %cm/pixel


[fcover1, dcover1] = vl_sift(coverBW1) ; %cover features
[fcover2, dcover2] = vl_sift(coverBW2) ; %cover features
[fcover3, dcover3] = vl_sift(coverBW3) ; %cover features
[fcover4, dcover4] = vl_sift(coverBW4) ; %cover features

[fimg, dimg] = vl_sift(imgBW) ; %image features

[matches1, scores1] = vl_ubcmatch(dcover1, dimg) ;
[matches2, scores2] = vl_ubcmatch(dcover2, dimg) ;
[matches3, scores3] = vl_ubcmatch(dcover3, dimg) ;
[matches4, scores4] = vl_ubcmatch(dcover4, dimg) ;

N=250; %number of trials
threshold=1;

points=question5_function(N,threshold,scX1,scY1,matches1,fcover1, fimg);
X=points(1,:);
Y=points(2,:);
figure(1); imshow(img); hold on;
plot(X,Y,'r','linewidth', 3);
plot([X(4),X(1)],[Y(4),Y(1)],'r','linewidth', 3);
text((X(4)+X(2))/2,(Y(1)+Y(3))/2,'Deep Learning','HorizontalAlignment','center','backgroundColor','red','Color','white','FontSize',6)

points=question5_function(N,threshold,scX2,scY2,matches2,fcover2, fimg);
X=points(1,:);
Y=points(2,:);
plot(X,Y,'r','linewidth', 3);
plot([X(4),X(1)],[Y(4),Y(1)],'r','linewidth', 3);
text((X(4)+X(2))/2,(Y(1)+Y(3))/2,'Image & Video Processing','HorizontalAlignment','center','backgroundColor','red','Color','white','FontSize',6)

points=question5_function(N,threshold,scX3,scY3,matches3,fcover3, fimg);
X=points(1,:);
Y=points(2,:);
plot(X,Y,'r','linewidth', 3);
plot([X(4),X(1)],[Y(4),Y(1)],'r','linewidth', 3);
text((X(4)+X(2))/2,(Y(1)+Y(3))/2,'Pattern Recognition','HorizontalAlignment','center','backgroundColor','red','Color','white','FontSize',6)

points=question5_function(N,threshold,scX4,scY4,matches4,fcover4, fimg);
X=points(1,:);
Y=points(2,:);
plot(X,Y,'r','linewidth', 3);
plot([X(4),X(1)],[Y(4),Y(1)],'r','linewidth', 3);
text((X(4)+X(2))/2,(Y(1)+Y(3))/2,'Modern Photogrammetry','HorizontalAlignment','center','backgroundColor','red','Color','white','FontSize',6)


function result=question5_function(N,threshold,scX,scY,matches, fcover, fimg)
nm=size(matches,2);
pt_cover=fcover(1:2,matches(1,:));
pt_img=fimg(1:2,matches(2,:));


valid=zeros(1,N); %keeping track of number of valid points
H_use=[];
%H matrix is same for all books (same plane)
for ii=1:N
index=randperm(nm,4);
imgpic=[pt_img(:,index)];
coverpic=[pt_cover(:,index)];

H=ComputeH(imgpic', coverpic');

pt_ieh=inv(H)*[pt_cover;ones(1,nm)];
tmp=pt_ieh./pt_ieh(3,:);
pt_ie=tmp(1:2,:);
a=pt_ie-pt_img;
dist=sqrt(sum(a.^2,1));

valid_pt = dist<threshold;
valid(ii)=sum(valid_pt,2);
H_use=horzcat(H_use,H);
end

maxval=max(valid);
idx=find(valid==maxval);
H_final=H_use(:,(((idx-1)*3+1):((idx-1)*3+3)));

%finding corners on the image:
%scaled coords of all covers:
covercoords=[0,31.5/scY,1;0,0,1;24/scX,0,1;24/scX,31.5/scY,1];
covercoords=covercoords';
%coords of all books in image
points=[];
for jj=1:4
pointsh=inv(H_final)*covercoords(:,jj);
tmp=pointsh./pointsh(3,:);
points=horzcat(points,tmp(1:2,:))
end
result=points
end


function H=ComputeH(imgpic, coverpic)
%image coords:
x1_img=imgpic(1,1);
x2_img=imgpic(2,1);
x3_img=imgpic(3,1);
x4_img=imgpic(4,1);
y1_img=imgpic(1,2);
y2_img=imgpic(2,2);
y3_img=imgpic(3,2);
y4_img=imgpic(4,2);

%real world coords:
x1_cover=coverpic(1,1);
x2_cover=coverpic(2,1);
x3_cover=coverpic(3,1);
x4_cover=coverpic(4,1);
y1_cover=coverpic(1,2);
y2_cover=coverpic(2,2);
y3_cover=coverpic(3,2);
y4_cover=coverpic(4,2);

A=[x1_img y1_img 1 0 0 0 -x1_img*x1_cover -x1_cover*y1_img -x1_cover;
    0 0 0 x1_img y1_img 1 -x1_img*y1_cover -y1_cover*y1_img -y1_cover;
    x2_img y2_img 1 0 0 0 -x2_img*x2_cover -x2_cover*y2_img -x2_cover;
    0 0 0 x2_img y2_img 1 -x2_img*y2_cover -y2_cover*y2_img -y2_cover;
    x3_img y3_img 1 0 0 0 -x3_img*x3_cover -x3_cover*y3_img -x3_cover;
    0 0 0 x3_img y3_img 1 -x3_img*y3_cover -y3_cover*y3_img -y3_cover;
    x4_img y4_img 1 0 0 0 -x4_img*x4_cover -x4_cover*y4_img -x4_cover;
    0 0 0 x4_img y4_img 1 -x4_img*y4_cover -y4_cover*y4_img -y4_cover];
B=null(A);
B=B(:,1);
H=reshape(B,3,3)';

end

```

![books-detection](https://github.com/bregmanh/Image-Recognition-and-Homography-Transformation/blob/master/docs/books-detection.png?raw=true)



## Image Stitching

```matlab
clc; clear; close all;
 
% Load images.
img1 = imresize(imread('building1.JPG'),[1000 NaN]);
img2 = imresize(imread('building2.JPG'),[1000 NaN]);
img3 = imresize(imread('building3.JPG'),[1000 NaN]);
img4 = imresize(imread('building4.JPG'),[1000 NaN]);
img5 = imresize(imread('building5.JPG'),[1000 NaN]);
numimg=5;
BuildingScene={img1,img2,img3,img4,img5};


I=BuildingScene{1};
grayImage = single(rgb2gray(I));
[fimg, dimg] = vl_sift(grayImage);
% Initialize variable to hold image sizes.
imageSize = zeros(numimg,2);
ImageSize(1,:)=size(grayImage);
 
 %This is the array of transformtation to get the nth image on the same
 %homography as the first image. Ie, each layer is the multiplication of each previous image
 arrayH(numimg) = projective2d(eye(3));
 
for n=2:numimg
    fimgprev=fimg;
    dimgprev=dimg;
    I=BuildingScene{n};
    grayImage = single(rgb2gray(I));
    % Save image size.
    imageSize(n,:) = size(grayImage);
   
    [fimg, dimg] = vl_sift(grayImage);
    [matches, scores] = vl_ubcmatch(dimgprev, dimg);
    nm=size(scores,2);
    pt_imgprev=fimgprev(1:2,matches(1,:));
    pt_img=fimg(1:2,matches(2,:));
    N=10000;
    threshold=10;
    
    valid=zeros(1,N); %keeping track of number of valid points
    H_use=[];
    %H matrix is same for all books (same plane)
    for ii=1:N
        index=randperm(nm,4);
        imgprevpic=[pt_imgprev(:,index)];
        imgpic=[pt_img(:,index)];
        
        H=ComputeH(imgpic', imgprevpic');
       
        pt_ieh=H*[pt_img;ones(1,nm)];
        tmp=pt_ieh./pt_ieh(3,:);
        pt_ie=tmp(1:2,:);
        a=pt_ie-pt_imgprev;
        dist=sqrt(sum(a.^2,1));
        
        valid_pt = dist<threshold;
        valid(ii)=sum(valid_pt,2);
        H_use=horzcat(H_use,H);
    end
    
    maxval=max(valid);
    idx=find(valid==maxval);
    H_final=H_use(:,(((idx-1)*3+1):((idx-1)*3+3)));
    arrayH(n) = H_final';
    % Compute T(n) * T(n-1) * ... * T(1)
    arrayH(n).T = arrayH(n).T * arrayH(n-1).T;
end

% Compute the output limits  for each transform
for i = 1:numimg
    [xlim(i,:), ylim(i,:)] = outputLimits(arrayH(i), [1 imageSize(i,2)], [1 imageSize(i,1)]);
end
 
avgXLim = mean(xlim, 2);
 
[~, idx] = sort(avgXLim);
 
centerIdx = floor((numimg+1)/2);
 
centerImageIdx = idx(centerIdx);
 
Hinv = invert(arrayH(centerImageIdx));
 
for i = 1:numimg
     arrayH(i).T=arrayH(i).T*Hinv.T;
end
  
for i=1:numimg
[xlim(i,:), ylim(i,:)] = outputLimits(arrayH(i), [1 imageSize(i,2)], [1 imageSize(i,1)]);
end
 
maxImageSize = max(imageSize);
 
% Find the minimum and maximum output limits
xMin = min([1; xlim(:)]);
xMax = max([maxImageSize(2); xlim(:)]);
 
yMin = min([1; ylim(:)]);
yMax = max([maxImageSize(1); ylim(:)]);
 
% Width and height of panorama.
width  = round(xMax - xMin);
height = round(yMax - yMin);
 
% Initialize the "empty" panorama.
panorama = zeros([height width 3], 'like', I);
 
blender = vision.AlphaBlender('Operation', 'Binary mask', ...
    'MaskSource', 'Input port');
 
% Create a 2-D spatial reference object defining the size of the panorama.
xLimits = [xMin xMax];
yLimits = [yMin yMax];
panoramaView = imref2d([height width], xLimits, yLimits);
 
% Create the panorama.
for i = [1 2 3 4 5]
    
    I = BuildingScene{i};
    
    % Transform I into the panorama.
    warpedImage = imwarp(I, arrayH(i), 'OutputView', panoramaView);
    
    % Generate a binary mask.
    mask = imwarp(true(size(I,1),size(I,2)), arrayH(i), 'OutputView', panoramaView);
    
    % Overlay the warpedImage onto the panorama.
    panorama = step(blender, panorama, warpedImage, mask);
    
    
end
 
figure
imshow(panorama)

function H=ComputeH(imgpic, coverpic)
%image coords:
x1_img=imgpic(1,1);
x2_img=imgpic(2,1);
x3_img=imgpic(3,1);
x4_img=imgpic(4,1);
y1_img=imgpic(1,2);
y2_img=imgpic(2,2);
y3_img=imgpic(3,2);
y4_img=imgpic(4,2);

%real world coords:
x1_cover=coverpic(1,1);
x2_cover=coverpic(2,1);
x3_cover=coverpic(3,1);
x4_cover=coverpic(4,1);
y1_cover=coverpic(1,2);
y2_cover=coverpic(2,2);
y3_cover=coverpic(3,2);
y4_cover=coverpic(4,2);

A=[x1_img y1_img 1 0 0 0 -x1_img*x1_cover -x1_cover*y1_img -x1_cover;
    0 0 0 x1_img y1_img 1 -x1_img*y1_cover -y1_cover*y1_img -y1_cover;
    x2_img y2_img 1 0 0 0 -x2_img*x2_cover -x2_cover*y2_img -x2_cover;
    0 0 0 x2_img y2_img 1 -x2_img*y2_cover -y2_cover*y2_img -y2_cover;
    x3_img y3_img 1 0 0 0 -x3_img*x3_cover -x3_cover*y3_img -x3_cover;
    0 0 0 x3_img y3_img 1 -x3_img*y3_cover -y3_cover*y3_img -y3_cover;
    x4_img y4_img 1 0 0 0 -x4_img*x4_cover -x4_cover*y4_img -x4_cover;
    0 0 0 x4_img y4_img 1 -x4_img*y4_cover -y4_cover*y4_img -y4_cover];
B=null(A);
B=B(:,1);
H=reshape(B,3,3)';

end

```

![building-panorama](https://github.com/bregmanh/Image-Recognition-and-Homography-Transformation/blob/master/docs/building-pano.png?raw=true)

Panorama made with images I took myself (quarantined at home):

![moose-panorama](https://github.com/bregmanh/Image-Recognition-and-Homography-Transformation/blob/master/docs/moose-pano.png?raw=true)

I used 5 images for the moose panorama. The reason why the quality is not perfect is mostly RANSAC not being able to calculate the perfect Homography matrix. If I used more iterations (larger N) and knew the exact threshold needed I could get a better quality panorama. 