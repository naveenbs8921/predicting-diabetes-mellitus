% Final Pipeline for Predicting Diabetes Mellitus

% 0. Initialization
rng('shuffle'); % Ensure truly random train/test splits each run

% 1. Load Data
disp('Loading training data...');
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
disp('Imputing missing numeric values...');
numericData = X(:, isNumericCol);
% Store medians to use them on the unlabeled data later
numericMedians = median(numericData.Variables, 'omitnan');
numericData = fillmissing(numericData, 'constant', numericMedians);
X(:, isNumericCol) = numericData;

% Handle categorical variables
disp('Processing categorical variables...');
categoricalColNames = X.Properties.VariableNames(isCellCol);
for i = 1:length(categoricalColNames)
    colName = categoricalColNames{i};
    X.(colName) = categorical(X.(colName));
end

% Recombine into final dataset
finalData = X;
finalData.diabetes_mellitus = Y; 

% 3. Create Training and Test Data
disp('Splitting data into training and test sets...');
cv = cvpartition(height(finalData), 'Holdout', 0.2);

idxTrain = training(cv);
dataTrain = finalData(idxTrain, :);

idxTest = test(cv);
dataTest = finalData(idxTest, :);

XTrain = removevars(dataTrain, {'diabetes_mellitus'});
YTrain = dataTrain.diabetes_mellitus;

XTest = removevars(dataTest, {'diabetes_mellitus'});
YTest = dataTest.diabetes_mellitus;

% 4. Train a Medical Risk Model (Random Forest with Probabilities)
disp('Training Random Forest model...');
% We use 'Bag' to get true percentages (probabilities).
% We use 'Prior', 'uniform' so the model doesn't ignore the minority (diabetic) patients.
numTrees = 50; 
mdl = fitcensemble(XTrain, YTrain, ...
    'Method', 'Bag', ...
    'NumLearningCycles', numTrees, ...
    'Prior', 'uniform', ...
    'Learners', templateTree());

% 5. Evaluate the Model
disp('Evaluating model on test data...');
[predictedLabels, scores] = predict(mdl, XTest);

figure;
confusionchart(YTest, predictedLabels);
title('Confusion Matrix - Random Forest Baseline');
saveas(gcf, 'ConfusionMatrix_RF.png');

accuracy = sum(predictedLabels == YTest) / length(YTest);
fprintf('Model Accuracy: %.2f%%\n', accuracy * 100);

% 6. Generate Kaggle Submission
disp('Loading unlabeled data for Kaggle submission...');
if isfile('UnlabeledWiDS2021.csv')
    unlabeledDataRaw = readtable('UnlabeledWiDS2021.csv');
    encounter_ids = unlabeledDataRaw.encounter_id; % Save for submission file
    
    % Create a clean table for unlabeled data that exactly matches the columns in X
    unlabeledData = table();
    
    % Match columns exactly and apply imputation
    numericIdx = 1; % Counter for medians
    for i = 1:length(X.Properties.VariableNames)
        varName = X.Properties.VariableNames{i};
        
        % Extract data from unlabeled set
        if ismember(varName, unlabeledDataRaw.Properties.VariableNames)
            colData = unlabeledDataRaw.(varName);
        else
            colData = NaN(height(unlabeledDataRaw), 1); % Missing entirely
        end
        
        % Process according to what the model expects
        if isNumericCol(i)
            % If it's stored as text but should be numeric, convert it
            if ~isnumeric(colData)
                colData = str2double(string(colData));
            end
            % Impute using the specific median for this column from training data
            colData = fillmissing(colData, 'constant', numericMedians(numericIdx));
            numericIdx = numericIdx + 1;
        else
            % Categorical
            colData = categorical(colData);
        end
        
        unlabeledData.(varName) = colData;
    end
    
    % Predict
    disp('Predicting on unlabeled data...');
    [~, unlabeledScores] = predict(mdl, unlabeledData);
    
    % Get the exact probability (0.0 to 1.0) of having diabetes (Class 1)
    diabetes_prob = unlabeledScores(:, 2);
    
    % Round the probabilities to 3 decimal places so it isn't messy (e.g. 0.842 instead of 0.84239103)
    diabetes_prob = round(diabetes_prob, 3);
    
    % Create submission table
    submission = table(encounter_ids, diabetes_prob, 'VariableNames', {'encounter_id', 'diabetes_mellitus'});
    writetable(submission, 'Submission.csv');
    disp('Saved predictions to Submission.csv! You are ready to submit!');
else
    disp('Could not find UnlabeledWiDS2021.csv to generate submission.');
end
