% Data Exploration and Feature Importance Pipeline
% Run this script to analyze your training data and visualize which
% features are most important for predicting diabetes.

disp('Loading data for exploration...');
data = readtable('TrainingWiDS2021.csv');

% Remove ID columns that shouldn't be used for training
idCols = {'Var1', 'encounter_id', 'hospital_id', 'icu_id'};
colsToRemove = intersect(idCols, data.Properties.VariableNames);
if ~isempty(colsToRemove)
    data = removevars(data, colsToRemove);
end

%% 1. Missing Data Analysis
disp('Calculating missing data percentages...');
% Count missing values per column
numMissing = sum(ismissing(data));
missingPercent = (numMissing / height(data)) * 100;

% Find columns that have more than 0% missing data
missingColsIdx = missingPercent > 0;
missingNames = data.Properties.VariableNames(missingColsIdx);
missingVals = missingPercent(missingColsIdx);

% Sort to show the columns with the most missing data first
[sortedVals, sortIdx] = sort(missingVals, 'descend');
sortedNames = missingNames(sortIdx);

% Plot the top 20 columns with the highest percentage of missing data
figure;
if ~isempty(sortedVals)
    numToPlot = min(20, length(sortedVals));
    bar(sortedVals(1:numToPlot));
    xticks(1:numToPlot);
    xticklabels(sortedNames(1:numToPlot));
    xtickangle(45);
    ylabel('Percentage Missing (%)');
    title('Top 20 Features with Missing Data');
    saveas(gcf, 'MissingData.png');
else
    disp('No missing data found in this dataset!');
end

%% 2. Feature Importance using Random Forest
% To understand what drives the predictions, we can train a quick Random 
% Forest and ask it which features it relied on the most.

disp('Preparing data for Feature Importance Analysis...');
if ismember('diabetes_mellitus', data.Properties.VariableNames)
    Y = data.diabetes_mellitus;
    X = removevars(data, {'diabetes_mellitus'});
else
    error('diabetes_mellitus not found in the dataset.');
end

% Handle missing numeric data quickly with median (for analysis purposes)
varTypes = varfun(@class, X, 'OutputFormat', 'cell');
isNumericCol = strcmp(varTypes, 'double') | strcmp(varTypes, 'single');
isCellCol = strcmp(varTypes, 'cell') | strcmp(varTypes, 'string') | strcmp(varTypes, 'char');

numericData = X(:, isNumericCol);
numericData = fillmissing(numericData, 'constant', median(numericData.Variables, 'omitnan'));
X(:, isNumericCol) = numericData;

% Convert categorical columns
categoricalColNames = X.Properties.VariableNames(isCellCol);
for i = 1:length(categoricalColNames)
    X.(categoricalColNames{i}) = categorical(X.(categoricalColNames{i}));
end

disp('Training model to extract Feature Importances (this may take a moment)...');
% We use TreeBagger (Random Forest) and turn on OOBPredictorImportance
numTrees = 20; % Using fewer trees for speed during exploration
mdl = TreeBagger(numTrees, X, Y, 'OOBPredictorImportance', 'on', 'Method', 'classification');

% Extract and plot the predictor importances
imp = mdl.OOBPermutedPredictorDeltaError;

figure;
[sortedImp, impIdx] = sort(imp, 'descend');
% Plot top 20 most important features
topN = 20;
bar(sortedImp(1:topN));
xticks(1:topN);
xticklabels(X.Properties.VariableNames(impIdx(1:topN)));
xtickangle(45);
ylabel('Predictor Importance Estimate');
title('Top 20 Most Important Features for Predicting Diabetes');
saveas(gcf, 'FeatureImportance.png');

disp('Exploration complete! Check the generated figures.');
