% Improved Baseline for Predicting Diabetes Mellitus

% 1. Load Data
disp('Loading data...');
% Assuming 'TrainingWiDS2021.csv' is the dataset to be used. 
data = readtable('TrainingWiDS2021.csv');

% 2. Data Cleaning & Preprocessing
disp('Cleaning data...');

% Remove ID columns that shouldn't be used for training
idCols = {'Var1', 'encounter_id', 'hospital_id', 'icu_id'};
colsToRemove = intersect(idCols, data.Properties.VariableNames);
if ~isempty(colsToRemove)
    data = removevars(data, colsToRemove);
end

% Remove columns that are missing more than 50% of their data
disp('Dropping columns with >50% missing data...');
numMissing = sum(ismissing(data));
missingPercent = (numMissing / height(data)) * 100;
emptyCols = data.Properties.VariableNames(missingPercent > 50);
if ~isempty(emptyCols)
    data = removevars(data, emptyCols);
    fprintf('Dropped %d columns due to missing data.\n', length(emptyCols));
end

% Separate predictors (X) and response (Y) early on
if ismember('diabetes_mellitus', data.Properties.VariableNames)
    Y = data.diabetes_mellitus;
    X = removevars(data, {'diabetes_mellitus'});
else
    error('diabetes_mellitus not found in the dataset.');
end

% Identify numeric and non-numeric (categorical) columns
varTypes = varfun(@class, X, 'OutputFormat', 'cell');
isNumericCol = strcmp(varTypes, 'double') | strcmp(varTypes, 'single');
isCellCol = strcmp(varTypes, 'cell') | strcmp(varTypes, 'string') | strcmp(varTypes, 'char');

% Handle missing values in numeric columns (Imputation using Median)
% Instead of dropping rows, we fill missing numeric values with the median of that column
disp('Imputing missing numeric values...');
numericData = X(:, isNumericCol);
numericData = fillmissing(numericData, 'constant', median(numericData.Variables, 'omitnan'));
X(:, isNumericCol) = numericData;

% Handle categorical variables
% Instead of deleting them, we convert them to 'categorical' datatype.
% Many MATLAB machine learning algorithms (like Decision Trees) can handle categoricals directly.
disp('Processing categorical variables...');
categoricalColNames = X.Properties.VariableNames(isCellCol);
for i = 1:length(categoricalColNames)
    colName = categoricalColNames{i};
    % Convert to categorical, treating empty strings/missing as <undefined>
    X.(colName) = categorical(X.(colName));
    
    % Optional: Fill missing categories with a new category 'Unknown'
    % X.(colName) = addcats(X.(colName), 'Unknown');
    % X.(colName)(isundefined(X.(colName))) = 'Unknown';
end

% Recombine into final dataset
finalData = X;
finalData.diabetes_mellitus = Y; % Put response variable back for partitioning

% 3. Create Training and Test Data
disp('Splitting data into training and test sets...');
% 80% training, 20% testing
cv = cvpartition(height(finalData), 'Holdout', 0.2);

idxTrain = training(cv);
dataTrain = finalData(idxTrain, :);

idxTest = test(cv);
dataTest = finalData(idxTest, :);

% Separate X and Y for training and testing
XTrain = removevars(dataTrain, {'diabetes_mellitus'});
YTrain = dataTrain.diabetes_mellitus;

XTest = removevars(dataTest, {'diabetes_mellitus'});
YTest = dataTest.diabetes_mellitus;

% 4. Train a Robust Baseline Model (Random Forest / TreeBagger)
disp('Training Random Forest model...');
% fitcensemble is a powerful tool for classification
% We use 'Method', 'Bag' which creates a Random Forest.
% It natively handles categorical predictors.
numTrees = 50; % Start with 50 trees for speed; increase for better accuracy
mdl = fitcensemble(XTrain, YTrain, ...
    'Method', 'Bag', ...
    'NumLearningCycles', numTrees, ...
    'Learners', templateTree());

% 5. Evaluate the Model
disp('Evaluating model on test data...');
[predictedLabels, scores] = predict(mdl, XTest);

% Generate Confusion Matrix
figure;
confusionchart(YTest, predictedLabels);
title('Confusion Matrix - Random Forest Baseline');

% Calculate Accuracy
accuracy = sum(predictedLabels == YTest) / length(YTest);
fprintf('Model Accuracy: %.2f%%\n', accuracy * 100);

disp('Baseline complete! You can now tune hyperparameters or add feature engineering.');
