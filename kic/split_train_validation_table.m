function [train_indices, validation_indices, train_data, validation_data] = split_train_validation_table(data, validation_ratio)
rng(1); %Assign a seed to the random number generator in order to ensure 
%that it produces the same random sequence all the time
%shuffled_indices = ceil(rand(length(data),1) * (length(data)));
shuffled_indices = randperm(height(data))
size(shuffled_indices);
validation_set_size = int16(height(data) * validation_ratio);
validation_indices = shuffled_indices(1:validation_set_size);
train_indices = shuffled_indices(validation_set_size+1:end);
train_data = data(train_indices, 1:end);
validation_data = data(validation_indices, 1:end);
end