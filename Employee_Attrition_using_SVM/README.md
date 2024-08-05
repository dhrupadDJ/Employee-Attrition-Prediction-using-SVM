
# Employee Attrition Prediction using SVM

## Project Overview
This project aims to predict employee attrition using machine learning models, specifically focusing on Logistic Regression and Support Vector Machine (SVM) with an RBF kernel. The analysis utilizes a dataset (`HR_Employee_Attrition.xlsx`), which includes various attributes related to employee performance, demographics, and job characteristics to predict whether an employee will leave the company.

## Getting Started

### Prerequisites
Before running the project, ensure you have Python installed along with these necessary libraries:
- pandas
- numpy
- scikit-learn
- openpyxl (for handling Excel files)

Install these packages using pip if they are not already installed:

```bash
pip install pandas numpy scikit-learn openpyxl
```

### Installation
To get a local copy up and running, follow these simple steps:

```bash
git clone https://github.com/dhrupadDJ/Employee-Attrition-Prediction-using-SVM
cd Employee_Attrition_using_SVM
```

### File Structure
- `data_preprocessing/`
  - Modules for data loading, preprocessing, and scaling.
- `models/`
  - Modules for training Logistic Regression and SVM models.
- `evaluation/`
  - Module for evaluating models using various metrics.
- `HR_Employee_Attrition.xlsx` - Dataset file.

### Running the Code
Execute the main script to run the model training and evaluation process:

```bash
python main.py
```

Replace `main.py` with the actual name of your script if different.

## Usage
This project can serve HR departments in predicting employee turnover, helping them to implement better retention strategies based on model insights. It can also be used by data scientists and students interested in understanding the application of logistic regression and SVM in real-world scenarios.

## Contributing
Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

