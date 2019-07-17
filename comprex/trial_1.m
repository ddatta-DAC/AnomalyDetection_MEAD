% this should input the "categorical" data
% such that each of the categories do not have any shared ids
% number them sequentially
input_file =  'data/shuttle/shuttle2class.txt_10nmlbins_eps.txt'
x = textread(input_file);

op_file = 'intermediate_data'
% [cost CT] =  buildModelVar (x, op_file , 1);

buildModelVar (x, op_file , 1);
data_file = 'data/shuttle/shuttle2class.txt_10nmlbins_eps.txt';

inp_3 = strcat('CT_',op_file,'.mat');
z = computeCompressionScoresVar(data_file, inp_3);