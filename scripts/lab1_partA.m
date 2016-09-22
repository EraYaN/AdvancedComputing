%%
size = 8;
sizeReal = 5;
matrix = zeros(size*size,1);
for i=0:size*size-1
    if mod(i,size) < sizeReal
        matrix(i+1) = i*1.3 + 1;
    else
        matrix(i+1) = 0.0;
    end
end

%%
matrix1_real = zeros(5);
matrix2_real = zeros(5);
matrix_out_real = zeros(5);
matrix1_in = zeros(25,1);
matrix2_in = zeros(25,1);
matrix_out = zeros(25,1);
size = 5;
for i=0:4
    for j=1:5
        matrix1_in(i*5+j) = j;
        matrix2_in(i*5+j) = j;
    end
end
for i=1:5
    for j=1:5
        matrix1_real(i,j) = j;
        matrix2_real(i,j) = j;
    end
end
matrix_out_real = matrix1_real * matrix2_real;
for cols=0:size-1
    for rowsOut=0:size-1
	    matrix_out(1 + cols + rowsOut*size) = 0.0;
	    for rowsIn=0:size-1
            matrix_out(1 + cols + rowsOut*size,1) = matrix_out(1 + cols + rowsOut*size,1) + matrix1_in(1 + rowsIn + rowsOut*size,1) * matrix2_in(1 + rowsIn*size + cols,1);
        end
    end
end

%% parse data
% calculate difference between SEQ and PAR
exeTimeDiff = SEQ - PAR;

%% create image
power_image_handle = imagesc(numThreads,sizeData,exeTimeDiff);

%% set axes
set(gca,'YDir','normal')
xlabel('Threads');
ylabel('Data Size');
%set(h,'EdgeColor','none');

%% set colormap
col = hot(256);
tmp = linspace(0,1,256)';
for n = 1:3, col(:,n) = interp1( 3+19*tmp, col(:,n), 20.^tmp, 'linear'); end
colormap(col);
% colormap('hot');
colorbar;

