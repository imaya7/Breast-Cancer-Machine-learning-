The results the code should generate 

Dataset shape: (569, 30)
Feature names: ['mean radius' 'mean texture' 'mean perimeter' 'mean area'
 'mean smoothness' 'mean compactness' 'mean concavity'
 'mean concave points' 'mean symmetry' 'mean fractal dimension'
 'radius error' 'texture error' 'perimeter error' 'area error'
 'smoothness error' 'compactness error' 'concavity error'
 'concave points error' 'symmetry error' 'fractal dimension error'
 'worst radius' 'worst texture' 'worst perimeter' 'worst area'
 'worst smoothness' 'worst compactness' 'worst concavity'
 'worst concave points' 'worst symmetry' 'worst fractal dimension']
Target names: ['malignant' 'benign']
Class distribution: [212 357]

Logistic Regression Results:
Accuracy: 0.9737
Precision: 0.9722
Recall: 0.9859
F1 Score: 0.9790
ROC-AUC: 0.9974
Confusion Matrix:
[[41  2]
 [ 1 70]]

Classification Report:
              precision    recall  f1-score   support

   malignant       0.98      0.95      0.96        43
      benign       0.97      0.99      0.98        71

    accuracy                           0.97       114
   macro avg       0.97      0.97      0.97       114
weighted avg       0.97      0.97      0.97       114


Random Forest Results:
Accuracy: 0.9649
Precision: 0.9589
Recall: 0.9859
F1 Score: 0.9722
ROC-AUC: 0.9953
Confusion Matrix:
[[40  3]
 [ 1 70]]

Classification Report:
              precision    recall  f1-score   support

   malignant       0.98      0.93      0.95        43
      benign       0.96      0.99      0.97        71

    accuracy                           0.96       114
   macro avg       0.97      0.96      0.96       114
weighted avg       0.97      0.96      0.96       114


Decision Tree Results:
Accuracy: 0.9474
Precision: 0.9577
Recall: 0.9577
F1 Score: 0.9577
ROC-AUC: 0.9440
Confusion Matrix:
[[40  3]
 [ 3 68]]

Classification Report:
              precision    recall  f1-score   support

   malignant       0.93      0.93      0.93        43
      benign       0.96      0.96      0.96        71

    accuracy                           0.95       114
   macro avg       0.94      0.94      0.94       114
weighted avg       0.95      0.95      0.95       114


Best Models by Metric:
Best Overall Model: Logistic Regression (Score: 0.9816)
Best Accuracy: Logistic Regression (0.9737)
Best Precision: Logistic Regression (0.9722)
Best Recall: Logistic Regression (0.9859)
Best F1 Score: Logistic Regression (0.9790)
Best ROC-AUC: Logistic Regression (0.9974)

Feature Importance for Logistic Regression:

Top 10 Important Features (Logistic Regression):
                 feature  coefficient
21         worst texture     1.350606
10          radius error     1.268178
28        worst symmetry     1.208200
7    mean concave points     1.119804
26       worst concavity     0.943053
13            area error     0.907186
20          worst radius     0.879840
23            worst area     0.841846
6         mean concavity     0.801458
27  worst concave points     0.778217

  




        
