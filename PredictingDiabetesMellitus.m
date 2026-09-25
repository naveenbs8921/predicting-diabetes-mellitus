%[text] # Introduction
%[text] This challenge focuses on ensuring patient health through data. You are tasked to train a model that takes as input the patient record data and outputs a prediction of whether or not the patient has Diabetes Mellitus, which could inform treatment in the ICU. Learn more about and download the challenge dataset by registering for [this Kaggle competition](https://www.kaggle.com/competitions/widsdatathon2021/data) and downloading the training dataset. You may also choose to download the data dictionary, which contains additional information about the dataset.
%[text] This tutorial uses an example dataset to show you how to complete the steps of building your own regression model, so **make sure to replace this dataset with the challenge dataset before making your own models.** 
%[text] # Load & Prepare Data
%[text] This dataset presents an opportunity to learn about the data modelling and processing challenges a real-world data problem brings. This tutorial will show some basic methods to handle data challenges, but you can learn about more methods by going through this [Data Science Tutorial video series](https://www.mathworks.com/videos/series/data-science-tutorial.html).
%[text] ## Step 1: Load Data
%[text] Use the [readtable](https://www.mathworks.com/help/matlab/ref/readtable.html?) function to read the files and store it as tables. You can also import data using the MATLAB® [Import tool](https://www.mathworks.com/help/matlab/import_export/select-spreadsheet-data-interactively.html).
fullSet = readtable('exampleTrain.csv');
%[text] Once imported, preview the data to get a sense of what you're working with.
head(fullSet)
%%
%[text] ## Step 2: Clean Data
%[text] The biggest challenge with real datasets is that they are messy. Data transformation and modelling will be the key area to work on to avoid overfitting the problem.
%[text] Using the *summary* function, analyze the types of the predictors, some statistics about each predictor, and the number of missing values for each predictor column. This can help you decide how to clean your dataset.
summary(fullSet);
%[text] There are many different approaches to work with the missing values and predictor selection. We will go through one of the basic approaches, but you can also refer to this document to learn about other methods: [Clean Messy and Missing Data](https://www.mathworks.com/help/stats/find-missing-data.html).
%[text] *Note: This approach of data cleaning demonstrated is chosen arbitrarily to cut down number of predictor columns.*
%%
%[text] **Remove the character columns of the table**
%[text] Many machine learning algorithms only allow for numeric values as the input arguments, so you can remove variables of any non-numeric type. Those are imported as cell arrays, so check which columns are of type 'cell' and remove them.
columnDataTypes = varfun(@iscell, fullSet, OutputFormat="cell");
nonNumericCols = cell2mat(columnDataTypes);
nonNumColNames = fullSet.Properties.VariableNames(nonNumericCols);

fullSet = removevars(fullSet, nonNumColNames)
%%
%[text] **Remove extreme values from all the vitals predictors**
%[text] Some data points may be on the extreme low or high end of what you'd expect to see, which might cause unexpected behavior when this data is used to train a model. Use the Clean Outlier Data live task to remove these outlier data points. 
% Remove outliers %[task:83a5]
[cleanedSet,outlierIndices,outliersForPlot2,lo2,hi2] = rmoutliers(fullSet, ...
    "percentiles",[1 99]);

% Display results
figure
plot(fullSet.Var1,"SeriesIndex",6,"DisplayName","Input data")
hold on
plot(find(~outlierIndices),cleanedSet.Var1,"SeriesIndex",1,"LineWidth",1.5, ...
    "DisplayName","Cleaned data")

% Plot outliers
plot(find(outliersForPlot2(:,1)),fullSet.Var1(outliersForPlot2(:,1)),"x", ...
    "SeriesIndex",2,"DisplayName","Outliers")

% Plot data in rows where other variables contain outliers
mask = outlierIndices & ~outliersForPlot2(:,1);
plot(find(mask),fullSet.Var1(mask),"x","SeriesIndex","none", ...
    "DisplayName","Removed by other variables")

% Plot outlier thresholds
plot([xlim missing xlim],[lo2.Var1 lo2.Var1 missing hi2.Var1 hi2.Var1], ...
    "Color",[145 145 145]/255,"DisplayName","Outlier thresholds")

hold off
title("Number of outliers cleaned: " + nnz(outliersForPlot2(:,1)))
legend
ylabel("Var1")
clear outliersForPlot2 lo2 hi2 mask %[task:83a5]
%%
%[text] **Remove the rows which have at least 10 missing values**
%[text] The other assumption I made is the observations (rows) which have missing predictor values can be removed.
cleanedSet = rmmissing(cleanedSet,1,"MinNumMissing",10);
%%
%[text] **Move response variable to the end**
%[text] Last, move our label predictor *OutputVar* to the last column of the table because for some algorithms in MATLAB the last column is the default response variable.
cleanedSet = movevars(cleanedSet,'OutputVar')
%%
%[text] ## Step 3: Create Training Data
%[text] Once the data is clean, separate the output variable *OutputVar* from the dataset and create two separate tables. *XSet:* Predictor data, *YSet*: Class labels
XSet = removevars(cleanedSet,{'OutputVar'});
YSet = cleanedSet.OutputVar;
%%
%[text] ## Step 4: Create Test Data
%[text] Set aside a portion of the dataset to evaluate your trained models, the rest will be used for training. You can adjust how much you hold out for testing. 
c = cvpartition(height(XSet), 'Holdout', 0.2);

idxTrain = training(c);
XTrain = XSet(idxTrain,:);
YTrain = YSet(idxTrain);

idxTest = test(c);
XTest = XSet(idxTest,:);
YTest = YSet(idxTest);
%%
%[text] ## Step 5: Train a Model
%[text] In MATLAB you can train a classification model using two different methods:
%[text] 1. Using MATLAB machine learning algorithm functions
%[text] 2. Using the Classification Learner app \
%[text] ## Option 1: Using custom algorithms
%[text] A Binary classification problem can be approached using various algorithms like decision tress, svm, and logistic regression. Here I train using [fitclinear](https://www.mathworks.com/help/stats/fitclinear.html?) classification model. It trains the linear binary classification models with high dimensional predictor data.
%[text] Convert the table to a numeric matrix because *fitclinear* function takes only numeric matrix as an input argument.
XTrainMat = table2array(XTrain);
XTestMat = table2array(XTest);
%[text] Using Name-Value options, let's set the solver as sparsa (Sparse Reconstruction by Separable Approximation), which has default *lasso* regularization. Check out the [fitclinear](https://www.mathworks.com/help/stats/fitclinear.html?) document to learn more about all of the input argument options.
Mdl = fitclinear(XTrainMat,YTrain,'ObservationsIn','rows',...
    'Solver','sparsa');
%%
%[text] **Predict on the Test Set**
%[text] Once we have your model ready, you can perform predictions on your test set using predict function. It takes as input the fitted model and test data. The output is the predicted labels and scores. You can then evaluate how well the model performs by plotting a confusion matrix, which compares the predicted values to the actual values. Since this is just a dummy dataset, the performance is very poor, but when you start working with real data and testing different model configurations, you should start to see significant improvement. An ideal model would show two dark blue squares, one in the top-left and one in the bottom-right, to indicate that the model predicted correctly every time. 
[label,~] = predict(Mdl,XTestMat);
confusionchart(YTest, label);
%[text] This model appears to predict 0 every time, indicating we should make some adjustments to the data, machine learning algorithm, or training options. 
%%
%[text] ## Option 2: Using Classification Learner App
%[text] Second method of training the model is by using the Classification Learner app. It lets you interactively train, validate and tune classification model. Get started with it by following the steps below, and check out the [documentation](https://www.mathworks.com/help/stats/classification-learner-app.html) to learn more. 
%[text] - On the ***Apps*** tab, in the Machine Learning group, click ***Classification Learner***.
%[text] - Click ***New Session*** and select data (***XTrain***) from the workspace. Specify the response variable as 'From workspace' (***YTrain***).
%[text] - Select the validation method to avoid overfitting. You can either choose ***holdout validation*** or ***cross-validation,*** selecting the number of k-folds.
%[text] - Hit ***Start Session***
%[text] - On the ***Classification Learner tab***, in the ***Model Type*** section, select the algorithm to be trained (e.g. *logistic regression, All svm, All Quick-to-train).*
%[text] - You can use the ***Options*** section to customize your algorithms throuch PCA, Feature Selection, or other advanced options.
%[text] - Once all required options are selected, click ***Train***.
%[text] - The history window on the left displays the different models trained and their accuracy.
%[text] - Performance of the model on the validation data can be evaluated by ***Confusion Matrix*** plot or other options on the ***Plots and Results*** section of the toolstrip.
%[text] - To make predictions on the test set, export the model by selecting ***Export Model*** on *Classification Learner tab*. \
%%
%[text] **Predict on the Test Set**
%[text] The exported model is saved as *trainedModel* in the workspace. You can then predict labels and scores using *predictFcn*.
%[text] The *label* is the predicted labels on Test set. *Scores* are the scores of how confident the model is that the correct output is each class. You can then evaluate how well the model performs by plotting a confusion matrix, which compares the predicted values to the actual values.
[label,~] = trainedModel.predictFcn(XTest);
confusionchart(YTest, label);
%[text] This shows that the model is predicting 0 more often than it should. You should play around with other metrics and visualizations to showcase the quality of your model.
%%
%[text] # Additional Resources
%[text] 1. [Data Science Tutorial](https://www.mathworks.com/videos/series/data-science-tutorial.html)
%[text] 2. [Missing Data in MATLAB](https://www.mathworks.com/help/releases/R2019b/matlab/data_analysis/missing-data-in-matlab.html)
%[text] 3. [Supervised Learning Workflow and Algorithms](https://www.mathworks.com/help/releases/R2019b/stats/supervised-learning-machine-learning-workflow-and-algorithms.html)
%[text] 4. [Train Classification Models in Classification Learner App](https://www.mathworks.com/help/stats/train-classification-models-in-classification-learner-app.html)
%[text] 5. [Export Classification Model to Predict New Data](https://www.mathworks.com/help/stats/export-classification-model-for-use-with-new-data.html)
%[text] 6. [8 MATLAB Cheat Sheets for Data Science](https://www.mathworks.com/campaigns/offers/data-science-cheat-sheets.html) \

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"inline","rightPanelPercent":40}
%---
%[task:83a5]
%   data: {"appState":"{\"VersionSavedFrom\":7,\"MinCompatibleVersion\":1,\"InputDataDropDownValue\":\"fullSet\",\"InputDataTableVarDropDownValues\":[\"select variable\"],\"InputDataTableVarDropDownVisible\":true,\"SamplePointsDropDownValue\":\"default value\",\"SamplePointsTableVarDropDownValue\":\"select variable\",\"SamplePointsTableVarDropDownVisible\":false,\"SamplePointsVarClass\":\"\",\"SupportsVisualization\":true,\"InputDataTableVarNames\":[\".Var1\",\".Var2\",\".Var4\",\".Var5\",\".Var7\",\".Var8\",\".Var9\",\".Var10\",\".Var11\",\".Var13\",\".Var14\",\".Var15\",\".Var16\",\".Var17\",\".Var18\",\".Var19\",\".Var21\",\".Var22\",\".Var24\",\".Var25\",\".Var26\",\".Var27\",\".Var29\",\".Var30\",\".Var31\",\".Var32\",\".Var33\",\".Var34\",\".Var35\",\".Var36\",\".Var37\",\".Var38\",\".Var39\",\".Var40\",\".Var41\",\".Var42\",\".Var43\",\".Var44\",\".Var46\",\".Var47\",\".Var49\",\".Var50\",\".Var51\",\".Var52\",\".Var53\",\".Var54\",\".Var55\",\".Var56\",\".Var57\",\".Var58\",\".Var59\",\".Var60\",\".Var61\",\".Var62\",\".Var64\",\".Var66\",\".Var67\",\".Var68\",\".Var69\",\".Var71\",\".Var72\",\".Var75\",\".Var76\",\".Var79\",\".Var81\",\".Var82\",\".Var83\",\".Var84\",\".Var85\",\".Var86\",\".Var89\",\".Var90\",\".Var91\",\".Var92\",\".Var93\",\".Var94\",\".Var95\",\".Var96\",\".Var97\",\".Var98\",\".Var99\",\".Var100\",\".Var101\",\".Var102\",\".Var104\",\".Var105\",\".Var106\",\".Var108\",\".Var109\",\".Var110\",\".Var111\",\".Var113\",\".Var114\",\".Var115\",\".Var116\",\".Var118\",\".Var119\",\".Var122\",\".Var123\",\".Var124\",\".Var125\",\".Var126\",\".Var127\",\".Var129\",\".Var130\",\".Var131\",\".Var132\",\".Var133\",\".Var134\",\".Var135\",\".Var136\",\".Var137\",\".Var138\",\".Var139\",\".Var140\",\".Var141\",\".Var142\",\".Var144\",\".Var145\",\".Var148\",\".Var150\",\".Var151\",\".Var152\",\".Var153\",\".Var154\",\".Var155\",\".Var156\",\".Var157\",\".Var160\",\".Var161\",\".Var162\",\".Var163\",\".Var165\",\".Var167\",\".Var168\",\".Var169\",\".Var170\",\".Var171\",\".Var172\",\".Var173\",\".Var174\",\".Var179\",\".Var180\",\".OutputVar\"],\"TimetableDimName\":\".\",\"SupportedTableVars\":[\".Var1\",\".Var2\",\".Var4\",\".Var5\",\".Var7\",\".Var8\",\".Var9\",\".Var10\",\".Var11\",\".Var13\",\".Var14\",\".Var15\",\".Var16\",\".Var17\",\".Var18\",\".Var19\",\".Var21\",\".Var22\",\".Var24\",\".Var25\",\".Var26\",\".Var27\",\".Var29\",\".Var30\",\".Var31\",\".Var32\",\".Var33\",\".Var34\",\".Var35\",\".Var36\",\".Var37\",\".Var38\",\".Var39\",\".Var40\",\".Var41\",\".Var42\",\".Var43\",\".Var44\",\".Var46\",\".Var47\",\".Var49\",\".Var50\",\".Var51\",\".Var52\",\".Var53\",\".Var54\",\".Var55\",\".Var56\",\".Var57\",\".Var58\",\".Var59\",\".Var60\",\".Var61\",\".Var62\",\".Var64\",\".Var66\",\".Var67\",\".Var68\",\".Var69\",\".Var71\",\".Var72\",\".Var75\",\".Var76\",\".Var79\",\".Var81\",\".Var82\",\".Var83\",\".Var84\",\".Var85\",\".Var86\",\".Var89\",\".Var90\",\".Var91\",\".Var92\",\".Var93\",\".Var94\",\".Var95\",\".Var96\",\".Var97\",\".Var98\",\".Var99\",\".Var100\",\".Var101\",\".Var102\",\".Var104\",\".Var105\",\".Var106\",\".Var108\",\".Var109\",\".Var110\",\".Var111\",\".Var113\",\".Var114\",\".Var115\",\".Var116\",\".Var118\",\".Var119\",\".Var122\",\".Var123\",\".Var124\",\".Var125\",\".Var126\",\".Var127\",\".Var129\",\".Var130\",\".Var131\",\".Var132\",\".Var133\",\".Var134\",\".Var135\",\".Var136\",\".Var137\",\".Var138\",\".Var139\",\".Var140\",\".Var141\",\".Var142\",\".Var144\",\".Var145\",\".Var148\",\".Var150\",\".Var151\",\".Var152\",\".Var153\",\".Var154\",\".Var155\",\".Var156\",\".Var157\",\".Var160\",\".Var161\",\".Var162\",\".Var163\",\".Var165\",\".Var167\",\".Var168\",\".Var169\",\".Var170\",\".Var171\",\".Var172\",\".Var173\",\".Var174\",\".Var179\",\".Var180\",\".OutputVar\"],\"SamplePointsTableVarNames\":[],\"OutputTypeValue\":\"replace\",\"OutputTypeItems\":[\"Table with specified variables replaced\",\"Table with cleaned variables appended\"],\"OutputTypeItemsData\":[\"replace\",\"append\"],\"DataVarSelectionTypeDropDownItems\":[\"All supported variables\",\"Specified variables\"],\"DataVarSelectionTypeDropDownItemsData\":[\"all\",\"manual\"],\"DataVarSelectionTypeDropDownValue\":\"all\",\"TableVarPlotDropDownItems\":[\"Var1\",\"Var2\",\"Var4\",\"Var5\",\"Var7\",\"Var8\",\"Var9\",\"Var10\",\"Var11\",\"Var13\",\"Var14\",\"Var15\",\"Var16\",\"Var17\",\"Var18\",\"Var19\",\"Var21\",\"Var22\",\"Var24\",\"Var25\",\"Var26\",\"Var27\",\"Var29\",\"Var30\",\"Var31\",\"Var32\",\"Var33\",\"Var34\",\"Var35\",\"Var36\",\"Var37\",\"Var38\",\"Var39\",\"Var40\",\"Var41\",\"Var42\",\"Var43\",\"Var44\",\"Var46\",\"Var47\",\"Var49\",\"Var50\",\"Var51\",\"Var52\",\"Var53\",\"Var54\",\"Var55\",\"Var56\",\"Var57\",\"Var58\",\"Var59\",\"Var60\",\"Var61\",\"Var62\",\"Var64\",\"Var66\",\"Var67\",\"Var68\",\"Var69\",\"Var71\",\"Var72\",\"Var75\",\"Var76\",\"Var79\",\"Var81\",\"Var82\",\"Var83\",\"Var84\",\"Var85\",\"Var86\",\"Var89\",\"Var90\",\"Var91\",\"Var92\",\"Var93\",\"Var94\",\"Var95\",\"Var96\",\"Var97\",\"Var98\",\"Var99\",\"Var100\",\"Var101\",\"Var102\",\"Var104\",\"Var105\",\"Var106\",\"Var108\",\"Var109\",\"Var110\",\"Var111\",\"Var113\",\"Var114\",\"Var115\",\"Var116\",\"Var118\",\"Var119\",\"Var122\",\"Var123\",\"Var124\",\"Var125\",\"Var126\",\"Var127\",\"Var129\",\"Var130\",\"Var131\",\"Var132\",\"Var133\",\"Var134\",\"Var135\",\"Var136\",\"Var137\",\"Var138\",\"Var139\",\"Var140\",\"Var141\",\"Var142\",\"Var144\",\"Var145\",\"Var148\",\"Var150\",\"Var151\",\"Var152\",\"Var153\",\"Var154\",\"Var155\",\"Var156\",\"Var157\",\"Var160\",\"Var161\",\"Var162\",\"Var163\",\"Var165\",\"Var167\",\"Var168\",\"Var169\",\"Var170\",\"Var171\",\"Var172\",\"Var173\",\"Var174\",\"Var179\",\"Var180\",\"OutputVar\"],\"TableVarPlotDropDownValue\":\"Var1\",\"TableVarPlotDropDownDoSelectAll\":false,\"TableVarPlotDropDownHasSelectAll\":true,\"TableVarPlotDropDownDoSelectAllNumeric\":false,\"TableVarPlotDropDownHasSelectAllNumeric\":false,\"InputDataTableVarDropDownValue\":\".Var1\",\"OutputTypeDropDownItems\":[\"Table with specified variables replaced\"],\"OutputTypeDropDownItemsData\":[\"table\"],\"OutputTypeDropDownValue\":\"table\",\"TaskIsStale\":false,\"WindowTypeDropDownValue\":\"full\",\"WindowSizeSpinner1Value\":2,\"WindowSizeSpinner2Value\":1,\"WindowUnitDropDownValue\":\"days\",\"WindowUnitVisible\":false,\"FindMethodDropDownValue\":\"median\",\"CleanMethodDropDownValue\":\"remove\",\"FillMethodDropDownValue\":\"linear\",\"FillConstantSpinnerValue\":0,\"ThresholdSpinnerValue\":3,\"LowerPercentileSpinnerValue\":1,\"UpperPercentileSpinnerValue\":99,\"LowerRangeEditFieldValue\":\"-Inf\",\"UpperRangeEditFieldValue\":\"Inf\",\"OutlierLocationsWSDDValue\":\"select variable\",\"PlotFilledCheckBoxValue\":true,\"PlotCleanedDataCheckBoxValue\":true,\"PlotInputDataCheckBoxValue\":true,\"PlotOutliersCheckBoxValue\":true,\"PlotThresholdsCheckBoxValue\":true,\"PlotCenterCheckBoxValue\":false,\"PlotTypeDropDownValue\":\"line\",\"PlotOtherRemovedCheckBoxValue\":true,\"IsPercentiles\":true,\"IsRange\":false,\"IsLocFromWorkspace\":false,\"IsConvertToMissing\":false,\"DefaultCleanMethod\":\"fill\"}","autorun":"0","collapsed":"0","outputs":"newTable,outlierIndices","run":"section","taskClassDefFile":"matlab.internal.dataui.outlierDataCleaner","uniqueId":"matlab\/CleanOutlierData","variablesMap":"{\"newTable\":\"cleanedSet\",\"outlierIndices\":\"outlierIndices\",\"outliersForPlot\":\"outliersForPlot2\",\"lo\":\"lo2\",\"hi\":\"hi2\",\"mask\":\"mask\"}","view":"controls-only"}
%---
