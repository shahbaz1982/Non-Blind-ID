clear, clc, close all, format long
tic

%u_exact = double(imread('cameraman.tif'));
u_exact = double(imread('moon.tif'));

umax = max(max(u_exact));
u_exact = u_exact/umax;
N=size(u_exact,1); kernel=ke_kernel(N,300,10);

nxy =256; nx = nxy; ny = nxy; % Resize to reduce Problem

u_exact=imresize(u_exact,[nx nx]); kernel=imresize(kernel,[nx nx]);     
hx = 1 / nx; hy = 1 / ny; N=nx; hx2 = hx^2;
kernel=kernel/sum(kernel(:));
m2 = 2*nx; nd2 = nx / 2; kernele = zeros(m2, m2) ;
kernele(nd2+1:nx+nd2,nd2+1:nx+nd2) = kernel ; %extention kernel
k_hat = fft2(fftshift(kernele)) ; clear kernele

[z] = integral_op(u_exact,k_hat,nx,nx);  % Blur Only  PLUS NOISE if needed
Blur_psnr = psnr(z,u_exact)

beta =1;  alpha = 8e-9;  n = nx^2;  m = 2*nx*(nx-1);  nm = n + m;
[B]=computeB(nx);    u0 = zeros(nx,nx);  U = u0;
M=speye(n,n);
tol = 1e-8; maxit = 1000; 
% u = z;
fprintf('iter    Deblur_psnr\n')
fprintf('----    -----------\n')
figure;   imagesc(u_exact); colormap(gray);
figure;  imagesc(z); colormap(gray);

% ------- Parameters  ---------
c1 = 9.5e-7;  
c2 = 1e-6;
c3 = 1e-8;  
c4 = 1e-5;

d = 1.01; 
p = 2;
xeps = 0.00;
wgh = 1;

[b0] = integral_op(z,conj(k_hat),nx,nx); b0 = b0(:);

% u = z;%Initial guess
u = zeros(nx,nx);

TQ = zeros(nx,nx);

Px = zeros(nx,nx); 
Py = zeros(nx,nx);
Pz = zeros(nx,nx);

Nx = zeros(nx,nx); 
Ny = zeros(nx,nx);
Nz = zeros(nx,nx);

Mx = zeros(nx,nx); 
My = zeros(nx,nx);
Mz = zeros(nx,nx);

%Lagrange multipliers
lam_1 = zeros(nx,nx);

lamx_2 = zeros(nx,nx); 
lamy_2 = zeros(nx,nx);
lamz_2 = zeros(nx,nx);

lamx_3 = zeros(nx,nx); 
lamy_3 = zeros(nx,nx);
lamz_3 = zeros(nx,nx);

bete = 1e-8;
c_hat = fft2(kernel, nx, nx);
for k=1:3

[DerT1] = DerX(c2*Px+lamx_2,nx);
[DerT2] = DerY(c2*Py+lamy_2,nx);
[DerT3] = DerZ(c2*Pz+lamz_2,nx);

b = b0 - wgh*hx2*DerT1(:) - wgh*hx2*DerT2(:)- wgh*hx2*DerT3(:);

L = -B'*B;
     [U,flag,rr,iter,rv] = pcg(@(x)KKCH(nx,x,k_hat,L,alpha,c2),b,tol,maxit,[],[],u(:));

u = reshape(U,nx,nx); Upsnr = psnr(u,u_exact);

fprintf('%d      %11.9g\n',k,Upsnr)

[G1,G2] = GradU(u,nx);
TPx = G1 - lamx_2/c2 ; 
TPy = G2 - lamy_2/c2 ;
TPz = G2 - lamz_2/c2 ;
Px = max( 0 , 1 - ((c1 + lam_1)./(c2*abs(TPx))))*TPx;
Py = max( 0 , 1 - ((c1 + lam_1)./(c2*abs(TPy))))*TPy;
Pz = max( 0 , 1 - ((c1 + lam_1)./(c2*abs(TPz))))*TPz;

[DerNx] = DerX(Nx,nx);
[DerNy] = DerY(Ny,nx);
[DerNz] = DerZ(Nz,nx);
TQ = DerNx + DerNy + DerNz - lamx_3/c3;

[RHSx1] = DerX(c3*TQ+lamx_3,nx);
[RHSx21] = DerX(Nx,nx);
[RHSx22] = DerY(Ny,nx);
[RHSx2] = DerX(RHSx21+RHSx22,nx);
Nx = Mx - lamx_3/c4 - RHSx1/c4 + c3*RHSx2/c4;

[RHSy1] = DerY(c3*TQ+lamy_3,nx);
[RHSy2] = DerY(RHSx21+RHSx22,nx);
Ny = My - lamy_3/c4 - RHSy1/c4 + c3*RHSy2/c4;

Nz = Mz - lamz_3/c4;

Mx = Nx - lam_1/c1 ; 
My = Ny - lam_1/c1 ;
Mz = Nz - lam_1/c1 ;
if norm([Mx;My;Mz]) > 1
    Mx = Mx./norm([Mx;My;Mz]);
    My = My./norm([Mx;My;Mz]);
    Mz = Mz./norm([Mx;My;Mz]);
end
%Lagrange multipliers update
lam_1 = lam_1 + c1*((abs(Px)+abs(Py)+abs(Pz)) - (Px + Py + Pz));

lamx_2 = lamx_2 + c1*(abs(Px) - G1);
lamy_2 = lamy_2 + c1*(abs(Py) - G2);
lamz_2 = lamz_2 + c1*(abs(Pz) - G2);

lamx_3 = lamx_3 + c4*(Nx - Mx);
lamy_3 = lamy_3 + c4*(Ny - My);
lamz_3 = lamz_3 + c4*(Nz - Mz);

end
toc
figure;  imagesc(u); colormap(gray);

%--------------Functions--------------------------------
function K = ke_kernel(n, tau, radi);
if nargin<1,help ke_gen;return; end
if nargin<2, tau=200; end
if nargin<3, radi=4; end
K=zeros(n);
R=n/2; h=1/n; h2=h^2;
%RR=n^2/radi+1; 
RR=radi^2;

if radi>0 

for j=1:n
  for k=1:n
    v=(j-R)^2+(k-R)^2;
    if v <= RR,
      K(j,k)=exp(-v/4/tau^2);
    end;
  end;
end;
sw=sum(K(:));
K=K/sw; %*tau/pi;

else radi<0 
    range=R-2:R+2;
 K(range,range)=1/25;
end
end
  function [Ku] = integral_op(u,k_hat,nux,nuy)
  [nkx,nky] = size(k_hat);
  h=1/nkx;
  Ku = real(ifft2( ((fft2(u,nkx,nky)) .* k_hat)));
  if nargin == 4
    Ku = Ku(1:nux,1:nuy);
  end
  end
function [Der] = DerX(Dx,nx)

Px = zeros(nx);

m = nx-1;
for j=1:nx
    for i=1:nx
        if j<m
Px(i,j) = Dx(i,j)- Dx(i,j+1);
        else
        Px(i,j) = Dx(i,j);
        end
    end
end

Der = Px;
end
function [Der] = DerY(Dy,nx)

Py = zeros(nx);

m = nx-1;

for i=1:nx
    for j=1:nx
        if i<m
Py(i,j) = Dy(i,j) - Dy(i+1,j);
        else
        Py(i,j) = Dy(i,j);
        end
    end
end
Der = Py;
end
function [Der] = DerZ(Dy,nx)

Py = zeros(nx);

m = nx-1;

for i=1:nx
    for j=1:nx
        if i<m
Py(i,j) = Dy(i,j) - Dy(i+1,j);
        else
        Py(i,j) = Dy(i,j);
        end
    end
end
Der = Py;
end
function [G1,G2] = GradU(u,nx)

Px = zeros(nx);
Py = zeros(nx);

m = nx-1;
for j=1:nx
    for i=1:nx
        if j<m
Px(i,j) = u(i,j)- u(i,j+1);
        else
        Px(i,j) = u(i,j);
        end
    end
end
for i=1:nx
    for j=1:nx
        if i<m
Py(i,j) = u(i,j) - u(i+1,j);
        else
        Py(i,j) = u(i,j);
        end
    end
end
G1 = Px;
G2 = Py;
end
function [y] = KKCH(nx,x,k_hat,L,alpha,c)
x = reshape(x,nx,nx);
[y1] = integral_op(x,k_hat,nx,nx);
[y1] = integral_op(y1,conj(k_hat),nx,nx);
y = y1 + reshape(alpha * L * x(:),nx,nx);
hx2 = (1/nx)^2;
y = y(:) + c * hx2 * x(:);
end
function [B]=computeB(nx)
e = ones(nx,1);
E = spdiags([0*e -1*e e], -1:1, nx, nx);
E1 =E(1:nx-1,:);
 
M1=eye(nx,nx);
B1=kron(E1,M1);
 
E2 = eye(nx);
M2 = spdiags([0*e -1*e e], -1:1, nx-1, nx);
B2 = kron(E2,M2);
 
B = [B1;B2];
% L = B'*D*B;
end
function p = psnr(x,y)

d = mean( mean( (x(:)-y(:)).^2 ) );
m1 = max( abs(x(:)) );
m2 = max( abs(y(:)) );
m = max(m1,m2);

p = 10*log10( m^2/d );
end











