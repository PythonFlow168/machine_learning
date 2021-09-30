mushroom = readtable('assignment_mushroom_csv');

%ctree = fitctree(kickstarter,'Funded')   %use the column name "funded" to indicate the target feature. 
%view(ctree,'mode','graph');  %plot the decision tree
%check https://uk.mathworks.com/help/stats/fitctree.html for more
%information on fitctree

validation_curve_data = [];
training_curve_data = [];
iteration = 1%set different validation ratios

for hold_out_ratio = 0.7:-0.1:0.3   %hold out ratio from 70% down to 30%. That is hold out 70% of data for validation to 30% of data
    %a table for future reference. You can use either the indices or the data
    %returned. 
    train_indices = [];
    test_indices = [];
    train_data = [];
    test_data = [];
    [train_indices, test_indices, train_data, test_data] = split_train_validation_table(mushroom,hold_out_ratio); %split the data using hold out validation
    
    %train_data_from_table = data_x(train_indices, 1:11);
    %train_labels_from_table = train_data;%(train_indices(1),1:11)
    train_data = mushroom(train_indices, 1:12); %store all the data from columns one to 12. 
    train_labels_from_table = mushroom(train_indices, 12);  %the fourteenth column is the target feature and so contains the label 
    
    test_data = mushroom(test_indices, 1:12); %store all the data from columns one to 12. 
    test_labels_from_table = mushroom(test_indices, 12); %the fourteenth column is the target feature and so contains the label 
    
  
    %{Experiment with various parameters. Remember the default values for Matlab
    %are
    %n - 1 for MaxNumSplits. n is the training sample size.
    %1 for MinLeafSize.
    %10 for MinParentSize.%}
    %rng(1); % For reproducibility
    %ctree = fitctree(data_x,species_labels_y,'MaxNumSplits',7,'CrossVal','on');
    %view(ctree.Trained{1},'mode','graph')
    %-----------------------------------------------------------------------

    %Experiment with different cross-valdiation methods
    rng(1); % For reproducibility
    %ctree = fitctree(train_data_from_table,train_labels_from_table,'MaxNumSplits',7, 'Holdout',0.2);
    
    ctree = fitctree(train_data,'class', 'MaxNumSplits',7) %Decision tree with maximum depth of 7
    view(ctree,'mode','graph'); %show treess. 

    %use the test data to validate your decision tree by prediciting the labels
    %based on input data
    predictLabels_test = predict(ctree,test_data); %use the trained tree model to predict
    predictLabels_training = predict(ctree,train_data);

    %store the true classes based on test_data. 
    trueLabels_test = test_labels_from_table(:,1);  
    
    %Construct confusion matrix using predictLabels and trueLabels
    %figure
    
    %cm = confusionchart(table2cell(trueLabels_test),predictLabels_test)
    %%uncomment to show confusion matrixes
    cm_2 = confusionmat(table2cell(trueLabels_test),predictLabels_test)% confusion matrix
    %calculate accuracies for training and test data
    TP = cm_2(1,1)
    FP = cm_2(1,2)
    FN = cm_2(2,1)
    TN = cm_2(2,2)
    Accuracy_test = (TP + TN) / (TP + TN + FP + FN);%test the accuracy
    
    trueLabels_training = train_labels_from_table(:,1);
    cm_2 = confusionmat(table2cell(trueLabels_training),predictLabels_training)
    TP = cm_2(1,1)
    FP = cm_2(1,2)
    FN = cm_2(2,1)
    TN = cm_2(2,2)
    Accuracy_train = (TP + TN) / (TP + TN + FP + FN);
    %[TP, FP, TN, FN] = calError(table2cell(trueLabels_test), predictLabels_test)
    
    %store data in array for plotting later 
    validation_curve_data(iteration) = Accuracy_test;  
    training_curve_data(iteration) = Accuracy_train;
 
    iteration = iteration + 1;
    
    %you can decide to use the random forest tree too.
    %Using B = TreeBagger(NumTrees,X,Y) to construct a random forest tree 
    %rf_tree = TreeBagger(5,data_x,species_labels_y)
    %figure
    %view(rf_tree.Trees{3},'mode','graph') %You can not plot all the forest, so you need to choose which 
    %tree to view in it. 
end

figure
plot(training_curve_data, 'r')
hold on
plot(validation_curve_data, 'b')
xlabel('iteration');
ylabel('Accuracy');
legend('training Accuracy','validation Accuracy')
