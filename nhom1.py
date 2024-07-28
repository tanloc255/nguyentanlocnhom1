import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense

# Tải tập tin CSV và cache dữ liệu
@st.cache_data
def load_data():
    return pd.read_csv('D:\\visualcode\\nhom\\test\\diabetes.csv')

data = load_data()

# Ứng dụng Streamlit
st.title('Dự Đoán Tiểu Đường Sử Dụng Các Mô Hình Machine Learning')

# Phần Hiển Thị Dữ Liệu
show_plots = st.sidebar.checkbox("Hiển Thị Biểu Đồ Dữ Liệu", value=True)

if show_plots:
    if st.sidebar.checkbox("Hiển Thị Tóm Tắt Dữ Liệu"):
        st.write("### Tóm Tắt Dữ Liệu")
        st.write(data.describe())

    if st.sidebar.checkbox("Hiển Thị Phân Phối Dữ Liệu"):
        st.write("### Phân Phối Các Đặc Trưng")
        fig, ax = plt.subplots(4, 2, figsize=(20, 20))
        sns.histplot(data.Pregnancies, bins=20, ax=ax[0,0], color="red", kde=True, line_kws={'linewidth': 2})
        sns.histplot(data.Glucose, bins=20, ax=ax[0,1], color="red", kde=True, line_kws={'linewidth': 2})
        sns.histplot(data.BloodPressure, bins=20, ax=ax[1,0], color="red", kde=True, line_kws={'linewidth': 2})
        sns.histplot(data.SkinThickness, bins=20, ax=ax[1,1], color="red", kde=True, line_kws={'linewidth': 2})
        sns.histplot(data.Insulin, bins=20, ax=ax[2,0], color="red", kde=True, line_kws={'linewidth': 2})
        sns.histplot(data.BMI, bins=20, ax=ax[2,1], color="red", kde=True, line_kws={'linewidth': 2})
        sns.histplot(data.DiabetesPedigreeFunction, bins=20, ax=ax[3,0], color="red", kde=True, line_kws={'linewidth': 2})
        sns.histplot(data.Age, bins=20, ax=ax[3,1], color="red", kde=True, line_kws={'linewidth': 2})
        plt.tight_layout()
        st.pyplot(fig)

    if st.sidebar.checkbox("Hiển Thị Biểu Đồ Cặp"):
        st.write("### Biểu Đồ Cặp")
        fig = sns.pairplot(data, hue='Outcome', palette='Set1')
        st.pyplot(fig)

    if st.sidebar.checkbox("Hiển Thị Bản Đồ Tương Quan"):
        st.write("### Bản Đồ Tương Quan")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax, fmt='.2f', linewidths=0.5)
        ax.set_title('Bản Đồ Tương Quan')
        st.pyplot(fig)


# Chia dữ liệu thành các đặc trưng và mục tiêu
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa các đặc trưng
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Khởi tạo các mô hình
models = {
    "k-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(),
    "Support Vector Machines": SVC(kernel='linear'),
    "Decision Trees": DecisionTreeClassifier(random_state=42),
    "Random Forests": RandomForestClassifier(n_estimators=100, random_state=42),
    "Neural Networks": Sequential([
        Dense(32, input_dim=X_train.shape[1], activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
}

# Huấn luyện các mô hình
@st.cache_data
def train_models():
    model_accuracies = {}
    for name, model in models.items():
        if name == "Neural Networks":
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=0)
        else:
            model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        if name == "Neural Networks":
            y_pred = (y_pred > 0.5).astype(int)
        
        accuracy = accuracy_score(y_test, y_pred)
        model_accuracies[name] = accuracy
    return models, model_accuracies

models, accuracies = train_models()

# Lựa chọn mô hình
model_choice = st.selectbox("Chọn Mô Hình", list(models.keys()))

# Form nhập liệu cho dữ liệu mới
with st.form(key='input_form'):
    st.write("### Nhập các thông tin sau:")
    pregnancies = st.number_input("Số lần mang thai", min_value=0, value=1, step=1)
    glucose = st.number_input("Lượng glucose", min_value=0, value=85)
    blood_pressure = st.number_input("Huyết áp", min_value=0, value=66)
    skin_thickness = st.number_input("Độ dày da", min_value=0, value=29)
    insulin = st.number_input("Lượng insulin", min_value=0, value=0)
    bmi = st.number_input("Chỉ số BMI", min_value=0.0, value=26.6)
    dpf = st.number_input("Hệ số chức năng gia đình tiểu đường", min_value=0.0, value=0.351)
    age = st.number_input("Tuổi", min_value=0, value=31, step=1)

    submit_button = st.form_submit_button(label='Dự Đoán')

    if submit_button:
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
        input_data = scaler.transform(input_data)

        model = models[model_choice]
        if model_choice == "Neural Networks":
            prediction = model.predict(input_data)
            prediction = (prediction > 0.5).astype(int)
        else:
            prediction = model.predict(input_data)
        
        st.write(f"Dự đoán kết quả tiểu đường là: {'Tiểu đường' if prediction[0] == 1 else 'Không tiểu đường'}")

# Hiển thị các chỉ số hiệu suất cho mô hình đã chọn
st.write(f"### Hiệu Suất {model_choice} Trên Dữ Liệu Kiểm Tra")

if model_choice == "Neural Networks":
    y_pred_prob_nn = model.predict(X_test)
    y_pred_nn = (y_pred_prob_nn > 0.5).astype(int)
    accuracy_nn = accuracy_score(y_test, y_pred_nn)
    conf_matrix_nn = confusion_matrix(y_test, y_pred_nn)
    class_report_nn = classification_report(y_test, y_pred_nn, output_dict=True)
    
    st.write(f"**Độ Chính Xác:** {accuracy_nn:.2f}")
    
    st.write("**Ma Trận Nhầm Lẫn:**")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf_matrix_nn, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                annot_kws={"size": 16}, linewidths=0.5)
    ax.set_xlabel('Dự Đoán', fontsize=14)
    ax.set_ylabel('Thực Tế', fontsize=14)
    ax.set_title('Ma Trận Nhầm Lẫn Mạng Nơ-ron', fontsize=16)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    st.pyplot(fig)
    
    st.write("**Báo Cáo Phân Loại:**")
    class_report_df = pd.DataFrame(class_report_nn).transpose()
    st.dataframe(class_report_df.style.background_gradient(cmap='Blues').format(precision=2))
else:
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    st.write(f"**Độ Chính Xác:** {accuracy:.2f}")
    
    st.write("**Ma Trận Nhầm Lẫn:**")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                annot_kws={"size": 16}, linewidths=0.5)
    ax.set_xlabel('Dự Đoán', fontsize=14)
    ax.set_ylabel('Thực Tế', fontsize=14)
    ax.set_title(f'Ma Trận Nhầm Lẫn {model_choice}', fontsize=16)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    st.pyplot(fig)

    st.write("**Báo Cáo Phân Loại:**")
    class_report_df = pd.DataFrame(class_report).transpose()
    st.dataframe(class_report_df.style.background_gradient(cmap='Blues').format(precision=2))

# So sánh độ chính xác giữa các mô hình
st.write("### So Sánh Độ Chính Xác Giữa Các Mô Hình")

# Vẽ biểu đồ so sánh độ chính xác
fig, ax = plt.subplots(figsize=(12, 6))
model_names = list(accuracies.keys())
accuracy_values = list(accuracies.values())
sns.barplot(x=model_names, y=accuracy_values, ax=ax, palette='viridis')
ax.set_xlabel('Mô Hình', fontsize=14)
ax.set_ylabel('Độ Chính Xác', fontsize=14)
ax.set_title('So Sánh Độ Chính Xác Giữa Các Mô Hình', fontsize=16)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
st.pyplot(fig)

# Kết luận mô hình tốt nhất
best_model = max(accuracies, key=accuracies.get)
best_accuracy = accuracies[best_model]

st.write("### Kết Luận")
st.write(f"Mô hình có độ chính xác cao nhất là **{best_model}** với độ chính xác là **{best_accuracy:.2f}**.")
st.write("Dựa trên độ chính xác, mô hình này là lựa chọn tốt nhất cho dự đoán bệnh tiểu đường trong tập dữ liệu này.")
