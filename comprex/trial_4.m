% this should input the "categorical" data
% such that each of the categories do not have any shared ids
% number them sequentially
input_file =  'comprexData/china_import/data_train.txt';
x = textread(input_file);
s = cputime;
op_file = 'intermediate_data';
% [cost CT] =  buildModelVar (x, op_file , 1);

buildModelVar (x, op_file , 1);
data_file = 'comprexData/china_import/test_data_c1.txt';

inp_3 = strcat('CT_',op_file,'.mat');
results_file = strcat('comprexData/china_import/','results_c1.txt');
z = computeCompressionScoresVar(data_file, inp_3);
e = cputime;
dlmwrite(results_file,z,'\t')
t = e - s;
disp('Time elapsed')
t
