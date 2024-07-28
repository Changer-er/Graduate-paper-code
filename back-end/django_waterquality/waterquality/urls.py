from django.urls import path, include
from .views import (analyze_file, analyze_ada, analyze_knn, analyze_model_knn, analyze_smote,
                    analyze_gdbt, analyze_model_gdbt, analyze_svm, analyze_model_svm, analyze_ann, analyze_model_ann)

urlpatterns = [
    path('analyze', analyze_file, name='analyze_file'),
    path('ADASYN', analyze_ada, name='analyze_ada'),
    path('smote', analyze_smote, name='analyze_smote'),

    path('KNN', analyze_knn, name='analyze_knn'),
    path('model_knn', analyze_model_knn, name='analyze_model_knn'),

    path('GDBT', analyze_gdbt, name='analyze_gdbt'),
    path('model_gdbt', analyze_model_gdbt, name='analyze_model_gdbt'),

    path('SVM', analyze_svm, name='analyze_svm'),
    path('model_svm', analyze_model_svm, name='analyze_model_svm'),

    path('ANN', analyze_ann, name='analyze_svm'),
    path('model_ann', analyze_model_ann, name='analyze_model_svm'),
    # Other URLs...
]

