%%
matrix1_real = zeros(5);
matrix2_real = zeros(5);
matrix_out_real = zeros(5);
matrix_out_real = matrix1_real * matrix2_real;
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
for cols=0:size
    for rowsOut=0:size
	    matrix_out(1 + cols + rowsOut*size) = 0.0;
	    for rowsIn=0:size
            matrix_out(1 + cols + rowsOut*size) = matrix_out(cols + rowsOut*size) + matrix1_in(rowsIn + rowsOut*size) * matrix2_in(rowsIn*size + cols);
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

