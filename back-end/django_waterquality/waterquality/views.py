# views.py
import base64
import io
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import ADASYN
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.datasets import make_classification
import json
from sklearn import neighbors, datasets
import matplotlib
from imblearn.over_sampling import BorderlineSMOTE, SMOTE
from collections import Counter
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score
import tensorflow as tf
matplotlib.use('Agg')


features = ['Temperature(Min)', 'Temperature(Max)', 'Dissolved Oxygen(mg/L)(Min)',
                'Dissolved Oxygen(mg/L)(Max)', 'pH(Min)', 'pH(Max)', 'Conductivity(?mhos/cm)(Min)',
                'Conductivity(?mhos/cm)(Max)', 'BOD(mg/L)(Min)', 'BOD(mg/L)(Max)',
                'Nitrate N+Nitrite N(mg/L)(Min)', 'Nitrate N+Nitrite N(mg/L)(Max)',
                'Fecal Coliform(MPN/100ml)(Min)', 'Fecal Coliform(MPN/100ml)(Max)',
                'Total Coliform(MPN/100ml)(Min)', 'Total Coliform(MPN/100ml)(Max)']


@csrf_exempt
def analyze_file(request):
    return handle_uploaded_file(request, data_processing_func)


@csrf_exempt
def analyze_ada(request):
    return handle_uploaded_file(request, adasyn_processing_func)


@csrf_exempt
def analyze_smote(request):
    return handle_uploaded_file(request, smote_processing_func)


@csrf_exempt
def analyze_knn(request):
    return handle_model(request, knn_train)


@csrf_exempt
def analyze_gdbt(request):
    return handle_model(request, gdbt_train)


@csrf_exempt
def analyze_svm(request):
    return handle_model(request, svm_train)


@csrf_exempt
def analyze_ann(request):
    return handle_model(request, ann_train)


def analyze_model_knn(request):
    return handle_model_analyze(request, knn_model_analyse)


@csrf_exempt
def analyze_model_gdbt(request):
    return handle_model_analyze(request, gdbt_model_analyse)


def analyze_model_svm(request):
    return handle_model_analyze(request, svm_model_analyse)


def analyze_model_ann(request):
    return handle_model_analyze(request, ann_model_analyse)


def handle_model_analyze(request, processing_func):
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']
        data = pd.read_csv(uploaded_file)
        dataframe = processing_func(data)
        json_data = dataframe.to_json(orient='records')
        return JsonResponse({'value': json_data})
    else:
        return JsonResponse({'error': 'File not provided'}, status=400)


def handle_model(request, processing_func):
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']
        data = pd.read_csv(uploaded_file)
        plot_base64, number, k, _, plot2_base64 = processing_func(data)
        return JsonResponse({'plot': plot_base64, 'row': number, 'value': k, 'report_plot': plot2_base64})
    else:
        return JsonResponse({'error': 'File not provided'}, status=400)


def handle_uploaded_file(request, processing_func):
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']
        data = pd.read_csv(uploaded_file)
        plot_base64, number, dataframe = processing_func(data)
        json_data = dataframe.to_json(orient='records')
        return JsonResponse({'plot': plot_base64, 'row': number, 'value': json_data})
    else:
        return JsonResponse({'error': 'File not provided'}, status=400)


def data_preprocessing(original_df):
    original_df = original_df.replace('-', np.nan)
    original_df = original_df.replace('\n4', '', regex=True)
    original_df = original_df.replace('\n', ' ', regex=True)
    original_df = original_df.astype(float)
    original_df.dropna(inplace=True)
    return original_df


def data_processing_func(data):
    df = data_preprocessing(data)
    df_wq = df.copy(deep=True)
    df_wq['water quality'] = df_wq.apply(condition_func, axis=1).astype(int)
    data_distribution = df_wq['water quality'].value_counts()
    return generate_plot(data_distribution), df_wq.shape[0], df_wq


def adasyn_processing_func(data):
    df_resampled = adasyn_processing(data)
    df_resampled['water quality'] = df_resampled.apply(condition_func, axis=1).astype(int)
    data_distribution = df_resampled['water quality'].value_counts()
    return generate_plot(data_distribution), df_resampled.shape[0], df_resampled


def smote_processing_func(data):
    df_resampled = smote_processing(data)
    df_resampled['water quality'] = df_resampled.apply(condition_func, axis=1).astype(int)
    data_distribution = df_resampled['water quality'].value_counts()
    return generate_plot(data_distribution), df_resampled.shape[0], df_resampled


def condition_func(row):
    condition = ((row['Temperature(Min)'] >= 20) & (row['Temperature(Max)'] <= 30) &
                (row['Dissolved Oxygen(mg/L)(Min)'] >= 4) & (row['Dissolved Oxygen(mg/L)(Max)'] <= 8) &
                (row['pH(Min)'] >= 6) & (row['pH(Max)'] <= 8) & (row['Conductivity(?mhos/cm)(Min)'] >= 150) &
                (row['Conductivity(?mhos/cm)(Max)'] <= 500) & (row['BOD(mg/L)(Max)'] <= 5)&
                (row['Nitrate N+Nitrite N(mg/L)(Max)'] <= 5.5) & (row['Fecal Coliform(MPN/100ml)(Max)'] <= 200) &
                (row['Total Coliform(MPN/100ml)(Max)'] <= 500))
    return condition


def generate_plot(data_distribution):
    # Plot data distribution
    colors = ['blue' if x == 0 else 'lightcoral' for x in data_distribution.index]
    plt.bar(data_distribution.index, data_distribution.values, color=colors)
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.title('Data Distribution')
    plt.xticks([0, 1], ['0', '1'])
    # Convert plot to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    return plot_base64


def adasyn_processing(data):
    original_df = data_preprocessing(data)
    df_wq = original_df.copy(deep=True)
    df_wq['water quality'] = df_wq.apply(condition_func, axis=1).astype(int)
    oversample = ADASYN(n_neighbors=3, random_state=40)
    x = df_wq[features]
    y = df_wq['water quality']
    x_resampled, y_resampled = oversample.fit_resample(x, y)
    x_df = pd.DataFrame(x_resampled, columns=features)
    y_df = pd.DataFrame(y_resampled, columns=['water quality'])
    df_resampled = pd.concat([x_df, y_df], axis=1)
    return df_resampled


def smote_processing(data):
    original_df = data_preprocessing(data)
    df_wq = original_df.copy(deep=True)
    df_wq['water quality'] = df_wq.apply(condition_func, axis=1).astype(int)
    smote = SMOTE(k_neighbors=3,random_state=42)
    x = df_wq[features]
    y = df_wq['water quality']
    x_resampled, y_resampled = smote.fit_resample(x, y)
    x_df = pd.DataFrame(x_resampled, columns=features)
    y_df = pd.DataFrame(y_resampled, columns=['water quality'])
    df_resampled = pd.concat([x_df, y_df], axis=1)
    return df_resampled


def knn_train(data):
    df_resampled = adasyn_processing(data)
    x = df_resampled[features]
    y = df_resampled['water quality']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    k_values = range(1, 20)
    accuracies = []
    for k in k_values:
        knn = neighbors.KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
    plt.plot(k_values, accuracies, marker='o')
    plt.xlabel('Number of Neighbors (K)')
    plt.ylabel('Accuracy')
    plt.title('KNN Performance')
    plt.xticks(k_values)
    plt.grid(True)
    # Convert plot to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    knn_best = neighbors.KNeighborsClassifier(k_values[accuracies.index(max(accuracies))])
    knn_best.fit(x, y)
    y_hat = knn_best.predict(x)
    # 解析分类报告字符串为 JSON 格式
    # 注意：需要设置为输出为字典形式，而不是字符串形式
    report = classification_report(y, y_hat, output_dict=True)
    # 将字典转换为 JSON 格式字符串
    plot2_base64 = report_plot(report)
    return plot_base64, round(max(accuracies), 3), k_values[accuracies.index(max(accuracies))], knn_best, plot2_base64


def gdbt_train(data):
    df_resampled = adasyn_processing(data)
    x = df_resampled[features]
    y = df_resampled['water quality']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    learning_rates = [0.01, 0.05, 0.1, 0.2, 0.5]
    accuracies = []
    for rate in learning_rates:
        # 初始化 GBDT 分类器模型
        gbc = GradientBoostingClassifier(learning_rate=rate, n_estimators=100, random_state=42)
        # 训练模型
        gbc.fit(x_train, y_train)
        y_pred = gbc.predict(x_test)
        # 在测试集上评估模型性能并记录准确率
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
    plt.plot(learning_rates, accuracies, marker='o')
    plt.xlabel('Learning Rate')
    plt.ylabel('Accuracy')
    plt.title('GBDT Performance')
    plt.xscale('log')  # 使用对数刻度以便更好地展示学习率范围
    plt.grid(True)
    # Convert plot to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    gdbt_best = GradientBoostingClassifier(learning_rate=learning_rates[accuracies.index(max(accuracies))], n_estimators=100, random_state=42)
    gdbt_best.fit(x, y)
    y_hat = gdbt_best.predict(x)
    # 解析分类报告字符串为 JSON 格式
    # 注意：需要设置为输出为字典形式，而不是字符串形式
    report = classification_report(y, y_hat, output_dict=True)
    # 将字典转换为 JSON 格式字符串
    plot2_base64 = report_plot(report)
    return (plot_base64, round(max(accuracies), 3), learning_rates[accuracies.index(max(accuracies))], gdbt_best,
            plot2_base64)


def svm_train(data):
    df_resampled = adasyn_processing(data)
    x = df_resampled[features]
    y = df_resampled['water quality']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    C_values = [0.1, 1, 10]  # 正则化参数范围
    accuracies = []
    for C in C_values:
        # 初始化 SVM 分类器模型
        print('success！！！！')
        svm_model = svm.SVC(C=C, kernel='linear', random_state=42)
        # 训练模型
        svm_model.fit(x_train, y_train)
        # 在测试集上评估模型性能并记录准确率
        y_pred = svm_model.predict(x_test)
        # 在测试集上评估模型性能并记录准确率
        accuracy = accuracy_score(y_test, y_pred)
        print('success？？？？')
        accuracies.append(accuracy)
    plt.plot(C_values, accuracies, marker='o')
    plt.xlabel('C (Regularization Parameter)')
    plt.ylabel('Accuracy')
    plt.title('SVM Performance')
    plt.xscale('log')  # 使用对数刻度以便更好地展示超参数范围
    plt.grid(True)
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    print('success')
    svm_best = SVC(C=C_values[accuracies.index(max(accuracies))], kernel='linear', random_state=42)
    svm_best.fit(x, y)
    y_hat = svm_best.predict(x)
    # 解析分类报告字符串为 JSON 格式
    # 注意：需要设置为输出为字典形式，而不是字符串形式
    report = classification_report(y, y_hat, output_dict=True)
    # 将字典转换为 JSON 格式字符串
    plot2_base64 = report_plot(report)
    return (plot_base64, round(max(accuracies), 3), C_values[accuracies.index(max(accuracies))], svm_best,
            plot2_base64)


def create_ann(learning_rate=0.01):
    model = keras.Sequential([
        layers.Dense(12, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.2)),
        layers.Dense(6, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.2)),
        layers.Dense(1, activation='sigmoid')
    ])
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def ann_train(data):
    df_resampled = adasyn_processing(data)
    x = df_resampled[features]
    y = df_resampled['water quality']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    learning_rate_values = [0.01, 0.1, 0.5]  # 学习率范围
    accuracies = []
    for learning_rate in learning_rate_values:
        # 创建和编译模型
        model = create_ann(learning_rate=learning_rate)
        # 训练模型
        model.fit(x_train, y_train, epochs=100)
        # 在测试集上评估模型性能并记录准确率
        y_pred = model.predict(x_test)
        Y_pred_classes = (y_pred > 0.5).astype("int32")
        accuracy = accuracy_score(y_test, Y_pred_classes)
        accuracies.append(accuracy)
    # 将一维列表转换为二维数组
    # 绘制热力图展示超参数和准确率的关系
    plt.plot(learning_rate_values, accuracies, marker='o')
    plt.xlabel('Learning Rate Index')
    plt.ylabel('Accuracy')
    plt.title('Accuracy performance')
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    ann_best = create_ann(learning_rate=learning_rate_values[accuracies.index(max(accuracies))])
    ann_best.fit(x, y, epochs=100)
    y_hat = ann_best.predict(x)
    Y_hat_classes = (y_hat > 0.5).astype("int32")
    # 解析分类报告字符串为 JSON 格式
    # 注意：需要设置为输出为字典形式，而不是字符串形式
    report = classification_report(y, Y_hat_classes, output_dict=True)
    # 将字典转换为 JSON 格式字符串
    plot2_base64 = report_plot(report)
    return (plot_base64, round(max(accuracies), 3), learning_rate_values[accuracies.index(max(accuracies))], ann_best,
            plot2_base64)


def knn_model_analyse(data):
    df_resampled = adasyn_processing(data)
    x = df_resampled[features]
    _, _, _, knn_best, _ = knn_train(data)
    y_pred = knn_best.predict(x)
    x_df = pd.DataFrame(x, columns=features)
    y_df = pd.DataFrame(y_pred, columns=['water quality'])
    df_resampled = pd.concat([x_df, y_df], axis=1)
    return df_resampled


def gdbt_model_analyse(data):
    df_resampled = adasyn_processing(data)
    x = df_resampled[features]
    _, _, _, gdbt_best, _ = gdbt_train(data)
    y_pred = gdbt_best.predict(x)
    x_df = pd.DataFrame(x, columns=features)
    y_df = pd.DataFrame(y_pred, columns=['water quality'])
    df_resampled = pd.concat([x_df, y_df], axis=1)
    return df_resampled


def svm_model_analyse(data):
    df_resampled = adasyn_processing(data)
    x = df_resampled[features]
    _, _, _, svm_best, _ = svm_train(data)
    y_pred = svm_best.predict(x)
    x_df = pd.DataFrame(x, columns=features)
    y_df = pd.DataFrame(y_pred, columns=['water quality'])
    df_resampled = pd.concat([x_df, y_df], axis=1)
    return df_resampled


def ann_model_analyse(data):
    df_resampled = adasyn_processing(data)
    x = df_resampled[features]
    _, _, _, ann_best, _ = ann_train(data)
    y_pred = ann_best.predict(x)
    y_pred_classes = (y_pred > 0.5).astype("int32")
    x_df = pd.DataFrame(x, columns=features)
    y_df = pd.DataFrame(y_pred_classes, columns=['water quality'])
    df_resampled = pd.concat([x_df, y_df], axis=1)
    return df_resampled


def report_plot(report):
    classes = list(report.keys())[:-3]  # 排除 'accuracy', 'macro avg', 'weighted avg'
    precision = [report[c]['precision'] for c in classes]
    recall = [report[c]['recall'] for c in classes]
    f1_score = [report[c]['f1-score'] for c in classes]
    print(precision)
    print(recall)
    print(f1_score)
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制条形图
    bar_width = 0.2
    index = np.arange(len(classes))
    rects1 = ax.bar(index - bar_width, precision, bar_width, label='Precision')
    rects2 = ax.bar(index, recall, bar_width, label='Recall')
    rects3 = ax.bar(index + bar_width, f1_score, bar_width, label='F1-score')

    # 添加标签、标题等
    ax.set_xlabel('Classes')
    ax.set_ylabel('Scores')
    ax.set_title('Classification Report')
    ax.set_xticks(index)
    ax.set_xticklabels(classes)
    ax.legend()
    # 显示图表
    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    return plot_base64





# def knn_func(data):
#     knn = knn_train()
#     original_df = data_preprocessing(data)
#     x = original_df[features]
#     y_pred = knn.predict(x)
#     return after_process(x, y_pred, features)
#
#
# def after_process(x, y_predict, features):
#     x_df = pd.DataFrame(x, columns=features)
#     y_df = pd.DataFrame(y_predict, columns=['water quality'])
#     result = pd.concat([x_df, y_df], axis=1)
#     json_result = dataframe_to_json(result)
#     return json_result


def dataframe_to_json(dataframe):
    # 转换 DataFrame 为 JSON 格式
    json_data = dataframe.to_json(orient='records')
    # 返回 JSON 数据至前端
    return JsonResponse(json_data, safe=False)


# def gdbt_train():
#     x, y = train_data_processing()
#     bdt = GradientBoostingClassifier(n_estimators=100)
#     bdt.fit(x, y)
#     return bdt
#
#
# def svm_train():
#     x, y = train_data_processing()
#     svm_model = svm.SVC(kernel='linear')
#     svm_model.fit(x, y)
#     return svm_model


# def gdbt_func(data):
#     gbdt = gdbt_train()
#     df_resampled, features = adasyn_processing(data)
#     x = df_resampled[features]
#     y = df_resampled['water quality']
#     y_predict = gbdt.predict(x)
#     return after_process(x, y_predict, features)
#
#
# def svm_func(data):
#     sv = svm_train()
#     df_resampled, features = adasyn_processing(data)
#     x = df_resampled[features]
#     y = df_resampled['water quality']
#     y_predict = sv.predict(x)
#     return after_process(x, y_predict, features)
