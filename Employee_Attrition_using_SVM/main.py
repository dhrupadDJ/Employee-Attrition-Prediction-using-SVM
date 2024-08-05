from data_preprocessing.load_data import load_data
from data_preprocessing.preprocess_data import drop_columns, create_dummies, map_columns, split_data, scale_data
from models.logistic_regression import train_logistic_regression, predict as predict_lr
from models.svm_model import train_svm, predict as predict_svm
from evaluation.evaluation import metrics_score

# Load the data
file_path = 'HR_Employee_Attrition.xlsx'
df = load_data(file_path)

# Drop unnecessary columns
df = drop_columns(df, ['EmployeeNumber', 'Over18', 'StandardHours'])

# Create dummy variables
dummy_columns = ['BusinessTravel', 'Department', 'Education', 'EducationField', 'EnvironmentSatisfaction', 
                 'Gender', 'JobInvolvement', 'JobLevel', 'JobRole', 'MaritalStatus']
df = create_dummies(df, dummy_columns)

# Map categorical columns
mappings = {'OverTime': {'Yes': 1, 'No': 0}, 'Attrition': {'Yes': 1, 'No': 0}}
df = map_columns(df, mappings)

# Split the data
x_train, x_test, y_train, y_test = split_data(df, 'Attrition', stratify=df['Attrition'])

# Scale the data
x_train_scaled, x_test_scaled = scale_data(x_train, x_test)

# Train and evaluate Logistic Regression
lr_model = train_logistic_regression(x_train_scaled, y_train)
y_pred_train_lr = predict_lr(lr_model, x_train_scaled)
y_pred_test_lr = predict_lr(lr_model, x_test_scaled)

print("Logistic Regression - Train Data")
metrics_score(y_train, y_pred_train_lr)

print("Logistic Regression - Test Data")
metrics_score(y_test, y_pred_test_lr)

# Train and evaluate SVM with RBF kernel
svm_model_rbf = train_svm(x_train_scaled, y_train, kernel='rbf')
y_pred_train_svm_rbf = predict_svm(svm_model_rbf, x_train_scaled)
y_pred_test_svm_rbf = predict_svm(svm_model_rbf, x_test_scaled)

print("SVM (RBF Kernel) - Train Data")
metrics_score(y_train, y_pred_train_svm_rbf)

print("SVM (RBF Kernel) - Test Data")
metrics_score(y_test, y_pred_test_svm_rbf)
