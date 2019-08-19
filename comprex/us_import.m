% this should input the "categorical" data
% such that each of the categories do not have any shared ids
% number them sequentially

input_file =  'comprexData/us_import3/data_train.txt';
x = textread(input_file);
s = cputime;
op_file = 'intermediate_data_us_import3';
% [cost CT] =  buildModelVar (x, op_file , 0);
buildModelVar (x, op_file , 1);
e = cputime;
m_time = e-s;
s = cputime;
total_time = 0
for c = 2:3
for sample_id=1:1
data_file = strcat('comprexData/us_import3/test_data_c',num2str(c,'%d'),'_sample',num2str(sample_id,'%d'),'.txt')
inp_3 = strcat('CT_',op_file,'.mat');
s = cputime;
results_file = strcat('comprexData/us_import3/','results_test_c',num2str(c,'%d'),'_sample',num2str(sample_id,'%d'),'.txt')
z = computeCompressionScoresVar(data_file, inp_3);
e = cputime;
dlmwrite(results_file,z,'\t')
t = e - s;
disp('Time elapsed')
t
total_time = total_time + t;
end
end
disp('total_time')
total_time
m_time