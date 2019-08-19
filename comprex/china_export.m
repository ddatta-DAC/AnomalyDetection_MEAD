% this should input the "categorical" data
% such that each of the categories do not have any shared ids
% number them sequentially

input_file =  'comprexData/china_export/data_train.txt';
x = textread(input_file);
s = cputime;
op_file = 'intermediate_data_china_export';
% [cost CT] =  buildModelVar (x, op_file , 0);
buildModelVar (x, op_file , 0);
m_time = cputime-s
total_time = 0
s=cputime
for c = 1:1
for sample_id=1:1
data_file = strcat('comprexData/china_export/test_data_c',num2str(c,'%d'),'_sample',num2str(sample_id,'%d'),'.txt')
inp_3 = strcat('CT_',op_file,'.mat');

results_file = strcat('comprexData/china_export/','results_test_c',num2str(c,'%d'),'_sample',num2str(sample_id,'%d'),'.txt')
z = computeCompressionScoresVar(data_file, inp_3);
e = cputime;
dlmwrite(results_file,z,'\t')
t = e - s;
disp('Time elapsed')
t
total_time = total_time + t
end
end
disp('total_time')
total_time
m_time