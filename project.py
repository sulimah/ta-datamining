import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score

st.write("""
# Project UAS DATA MINING
Oleh : Sulimah | 200411100054
""")

importData, preprocessing, modeling, implementation = st.tabs(["Import Data", "Prepocessing", "Modeling", "Implementation"])

def loadData():
    data = pd.read_csv("https://raw.githubusercontent.com/sulimah/ta-datamining/main/Breast_cancer_data.csv")
    return data

data = loadData()

with importData:
    st.write("""
    # Deskripsi
    Data yang digunakan adalah data breast cancer :
    https://www.kaggle.com/datasets/merishnasuwal/breast-cancer-prediction-dataset?resource=download
    """)

    st.markdown("---")
    st.write("# Import Data")    
    st.write(data)

with preprocessing:
    st.write("# Preprocessing")

    st.markdown("---")

    st.write("## Normalisasi")
    st.write("Melakukan Normalisasi pada semua fitur dan mengambil fitur yang memiliki tipe data numerik")
    data_baru = data.drop(columns=["diagnosis"])

    sebelum_dinormalisasi = ['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness']
    setelah_dinormalisasi = ['Norm_mean_radius', 'Norm_mean_texture', 'Norm_mean_perimeter', 'Norm_mean_area', 'Norm_mean_smoothness']

    normalisasi_fitur = data[sebelum_dinormalisasi]
    st.dataframe(normalisasi_fitur)

    scaler = MinMaxScaler()
    scaler.fit(normalisasi_fitur)
    fitur_ternormalisasi = scaler.transform(normalisasi_fitur)
    
    # save normalisasi
    joblib.dump(scaler, 'normal')

    fitur_ternormalisasi_df = pd.DataFrame(fitur_ternormalisasi, columns = setelah_dinormalisasi)

    st.markdown("---")
    st.write("## Data yang telah dinormalisasi")
    st.write("Fitur numerikal sudah dinormalisasi")
    st.dataframe(fitur_ternormalisasi)        
    st.markdown("---")
    
    data_sudah_normal = fitur_ternormalisasi_df

    st.write("Hasil data yang sudah dinormalisasi dan diencoding disatukan dalam satu frame")
    st.dataframe(data_sudah_normal)

with modeling:
    st.write("# Modeling")

    st.write("Sistem ini menggunakan 3 modeling yaitu KNN, Naive-Bayes, dan Decission Tree")
    knn_cekbox = st.checkbox("KNN")
    bayes_gaussian_cekbox = st.checkbox("Naive-Bayes Gaussian")
    decission3_cekbox = st.checkbox("Decission Tree")

    #=========================== Spliting data ======================================
    X = data_sudah_normal.iloc[:,0:5]
    Y = data.iloc[:,-1]

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.25, random_state=0)    

    #============================ Model =================================
    #===================== KNN =======================
    knn = KNeighborsClassifier()
    knn.fit(X_train, Y_train)
    y_predknn = knn.predict(X_test)
    knn_accuracy = round(100 * accuracy_score(Y_test, y_predknn), 2)

    #===================== Bayes Gaussian =============
    gaussian = GaussianNB()
    gaussian.fit(X_train,Y_train)
    y_pred_gaussian   =  gaussian.predict(X_test)
    gauss_accuracy  = round(100*accuracy_score(Y_test, y_pred_gaussian),2)
    gaussian_eval = classification_report(Y_test, y_pred_gaussian,output_dict = True)
    gaussian_eval_df = pd.DataFrame(gaussian_eval).transpose()

    #===================== Decission tree =============
    decission3  = DecisionTreeClassifier(criterion="gini")
    decission3.fit(X_train,Y_train)
    y_pred_decission3 = decission3.predict(X_test)
    decission3_accuracy = round(100*accuracy_score(Y_test, y_pred_decission3),2)
    decission3_eval = classification_report(Y_test, y_pred_decission3,output_dict = True)
    decission3_eval_df = pd.DataFrame(decission3_eval).transpose()

    st.markdown("---")

    #===================== Cek Box ====================
    if knn_cekbox:
        st.write("##### KNN")
        st.warning("Dengan menggunakan metode KNN didapatkan akurasi sebesar:")
        # st.warning(knn_accuracy)
        st.warning(f"Akurasi  =  {knn_accuracy}%")
        st.markdown("---")

    if bayes_gaussian_cekbox:
        st.write("##### Naive Bayes Gausssian")
        st.info("Dengan menggunakan metode Bayes Gaussian didapatkan hasil akurasi sebesar:")
        st.info(f"Akurasi = {gauss_accuracy}%")
        st.markdown("---")

    if decission3_cekbox:
        st.write("##### Decission Tree")
        st.success("Dengan menggunakan metode Decission tree didapatkan hasil akurasi sebesar:")
        st.success(f"Akurasi = {decission3_accuracy}%")

with implementation:
    st.write("# Implementation")
    st.write("##### Input fitur")
    name = st.text_input("Masukkan nama")
    radius = st.number_input("Masukkan radius")
    texture = st.number_input("Masukkan texture")
    perimeter = st.number_input("Masukkan perimeter")
    area = st.number_input("Masukkan area")
    smoothness = st.number_input("Masukkan smoothness")

    cek_hasil = st.button("Cek Prediksi")

    # knn = joblib.load("import/knn.joblib")
    # decission3 = joblib.load("import/decission3.joblib")
    # gaussian = joblib.load("import/gaussian.joblib")
    # scaler = joblib.load("import/scaler.joblib") 

    #============================ Mengambil akurasi tertinggi ===========================
    if knn_accuracy > gauss_accuracy and knn_accuracy > decission3_accuracy:
        use_model = knn
        metode = "KNN"
    elif gauss_accuracy > knn_accuracy and gauss_accuracy > decission3_accuracy:
        use_model = gaussian
        metode = "Naive-Bayes Gaussian"
    else:
        use_model = decission3
        metode = "Decission Tree"

    #============================ Normalisasi inputan =============================
    inputan = [[radius, texture, perimeter, area, smoothness]]
    inputan_norm = scaler.transform(inputan)
    # inputan
    # inputan_norm
    if cek_hasil:
        hasil_prediksi = use_model.predict(inputan_norm)[0]
        if hasil_prediksi == 1:
            st.success(f"{name} didiagnosis kanker payudara, berdasarkan metode {metode}")
        else:
            st.error(f"{name} tidak didiagnosis kanker payudara, berdasarkan metode {metode}")