# Laporan Proyek Machine Learning - Nabila Lailatanzila

## Domain Proyek
Permasalahan konsumsi energi pada berbagai tipe bangunan merupakan isu penting dalam konteks efisiensi energi dan keberlanjutan. Dengan meningkatnya jumlah bangunan dan perangkat listrik yang digunakan, pemahaman mengenai pola konsumsi energi menjadi penting bagi perencana kebijakan dan pengembang sistem otomatisasi energi. Menurut laporan dari International Energy Agency (IEA), sektor bangunan menyumbang hampir 30% dari total konsumsi energi global dan menjadi salah satu kontributor utama emisi karbon [IEA, 2023]. Oleh karena itu, memprediksi konsumsi energi sangat penting untuk meningkatkan efisiensi dan pengambilan keputusan berbasis data dalam manajemen bangunan.

Selain itu, studi oleh Ahmad et al. (2018) menegaskan bahwa penerapan teknik Machine Learning dalam prediksi energi sangat membantu dalam manajemen bangunan cerdas (smart building), mengurangi biaya energi, serta mendukung pembangunan berkelanjutan. Proyek ini bertujuan untuk membangun model prediksi Energy Consumption berdasarkan karakteristik bangunan dan lingkungan seperti luas bangunan, suhu, jumlah penghuni, dan jenis bangunan.

## Business Understanding

### Problem Statements
1. Ketidakpastian dalam memprediksi konsumsi energi bangunan menyebabkan perencanaan energi yang kurang efisien, sehingga berpotensi meningkatkan biaya operasional dan dampak lingkungan.
2. Pemahaman yang terbatas terhadap faktor-faktor utama yang mempengaruhi konsumsi energi membuat peluang untuk meningkatkan efisiensi energi belum dimanfaatkan secara maksimal.
3. Pemilihan algoritma machine learning yang tepat menjadi tantangan karena kompleksitas dan variasi data, yang berdampak pada akurasi dan keandalan model prediksi konsumsi energi.

### Goals
1. Mengembangkan model prediktif berbasis machine learning yang akurat untuk membantu dalam mengoptimalkan konsumsi energi berdasarkan karakteristik bangunan.
2. Menganalisis kontribusi masing-masing fitur terhadap target variabel menggunakan pendekatan regresi.
3. Membandingkan 2 algoritma berbeda yang digunakan dalam prediksi konsumsi energi .

### Solution Statements
1. Membangun dan melatih model baseline menggunakan algoritma `Linear Regression` untuk memprediksi penggunaan energi berdasarkan fitur-fitur bangunan. Model ini dipilih karena interpretabilitasnya tinggi, sehingga dapat digunakan untuk menganalisis kontribusi tiap fitur terhadap target `Energy Consumption`. Evaluasi dilakukan menggunakan metrik seperti **MAE**, **RMSE**, **MSE** dan **R²**.
2. Mengembangkan model kedua menggunakan algoritma `Gradient Boosting Regressor`. Gradient Boosting dipilih karena kemampuannya dalam menangani hubungan non-linear dan kompleks antar fitur. Performa model dibandingkan dengan baseline menggunakan metrik evaluasi yang sama.
3. Menginterpretasi hasil model dengan menampilkan feature importance dari kedua model (koefisien regresi dan feature importance tree-based model) untuk mengidentifikasi faktor paling berpengaruh terhadap konsumsi energi, sehingga hasilnya dapat digunakan dalam pengambilan keputusan efisiensi energi.

## Data Understanding
Dataset yang digunakan dalam proyek ini adalah [Energy Consumption Dataset ](https://www.kaggle.com/datasets/govindaramsriram/energy-consumption-dataset-linear-regression) yang didapat dari kaggle. Dataset ini dirancang untuk memprediksi konsumsi energi berdasarkan karakteristik bangunan dan faktor lingkungan.

Dataset ini memiliki 1000 baris dan 7 fitur di antaranya:
- `Square Footage`: Luas bangunan dalam satuan kaki persegi.
- `Number of Occupants`: Jumlah penghuni di dalam bangunan.
- `Building Type`: Kategori bangunan (misalnya: Residential, Commercial, Industrial).
- `Appliances Used`: Jumlah atau jenis peralatan listrik yang digunakan.
- `Average Temperature`: Suhu rata-rata lingkungan selama periode pengamatan.
- `Day of Week`: Hari dalam seminggu (Weekday/Weekend), yang memengaruhi pola penggunaan energi.
- `Energy Consumption` : Jumlah konsumsi energi

Dari hasil pengecekan, dapat diketahui bahwa dataset ini bebas dari missing value dan tidak terdapat data duplikat, sehingga tidak diperlukan proses imputasi atau pembersihan data terkait hal tersebut. Selain itu, analisis visual dengan boxplot dilakukan untuk mengidentifikasi potensi keberadaan outlier pada fitur-fitur numerik. Meskipun terdapat variasi nilai yang cukup signifikan pada beberapa fitur, distribusi data secara keseluruhan masih tergolong wajar dan tidak menunjukkan adanya outlier yang ekstrim sehingga tidak memerlukan penanganan khusus.


### Exploratory Data Analysis (EDA):
**1. Mengecek distribusi data numerik menggunakan histogram**

Sebagian besar fitur numerik dalam dataset memiliki distribusi yang cukup seimbang. Namun, terdapat variasi nilai yang cukup besar pada fitur seperti `Square Footage` dan `Energy Consumption`, yang menunjukkan adanya keragaman antar sampel.
    
**2. Menganalisis korelasi antara Square Footage dan Energy Consumption**

Menghasilkan korelasi positif yang berarti semakin luas suatu bangunan, maka energi yang digunakan juga akan semakin banyak. Meskipun ada variasi, titik-titik data cenderung naik dari kiri bawah ke kanan atas.
    
**3. Menganalisis korelasi antara Occupants dan Energy Consumption**

Tidak terlihat adanya korelasi linear yang jelas antara jumlah penghuni dan konsumsi energi. Titik-titik data tampak tersebar secara acak di seluruh rentang jumlah penghuni, tanpa menunjukkan pola peningkatan atau penurunan yang konsisten pada konsumsi energi seiring bertambahnya jumlah penghuni.
    
**4. Menganalisis korelasi antara Temperature dan dan Energy Consumption**

Tidak ada korelasi linear yang jelas antara suhu rata-rata dan konsumsi energi yang terlihat dari plot ini. Titik-titik data tampak tersebar secara acak di seluruh rentang suhu rata-rata (sekitar 10 hingga 35 derajat), tanpa menunjukkan pola peningkatan atau penurunan konsumsi energi yang konsisten seiring perubahan suhu.
    
**5. Menganalisis korelasi antara Appliance dan Energy Consumption**

Tidak terbentuk pola yang begitu linear. Namun, ada sedikit kecenderungan peningkatan konsumsi energi seiring bertambahnya jumlah perangkat, titik-titik data tampak sangat tersebar dan tidak membentuk pola yang ketat.

## Data Preparation
### Tahapan yang dilakukan:
**1. Seleksi Fitur Target**

Pemilihan fitur dilakukan berdasarkan analisis bisnis, di mana fokus analisis ditujukan pada penggunaan energi. Maka, dilakukan pemisahkan data input (x) yang akan digunakan model untuk belajar dari variabel yang ingin diprediksi (y), yaitu `Energy Consumption`.

**2. Split Data**

Membagi data menjadi **data traning (80%)** untuk melatih model dan **data testing (20%)** menggunakan `train_test_split` untuk mengevaluasi kinerja model pada data baru.

**3. Encoding** fitur kategorikal `Building Type` dan `Day of Week` menggunakan `OneHotEncoding`

Data kategori pada dataset ini cenderung memiliki variasi yang rendah atau sedikit. Ini memungkinkan untuk melakukan `OneHotEncoding` dalam representasi biner agar menghindari ordinalitas.

**4. Scaling** fitur numerik dengan `StandardScaler`

Fitur numerik diskalakan menggunakan `StandardScaler` untuk memastikan setiap fitur memiliki distribusi dengan **rata-rata 0** dan **standar deviasi 1**. Fitur kategorikal dibiarkan tanpa skala karena sudah dalam bentuk yang sesuai setelah encoding.

## Modeling
### Model 1: Linear Regression
Linear Regression adalah algoritma statistik yang memodelkan hubungan linear antara variabel input (fitur) dan output (target) dengan mencari garis terbaik yang meminimalkan perbedaan antara prediksi dan nilai asli. Algoritma ini menghasilkan koefisien untuk tiap fitur yang menunjukkan besarnya pengaruh fitur tersebut terhadap target, sehingga dapat digunakan untuk memprediksi nilai kontinu seperti konsumsi energi. **Kelebihannya** adalah model ini cepat dilatih dan cocok untuk data dengan hubungan linear antara fitur dan target. Namun, **kekurangannya** model ini kurang efektif jika data memiliki pola non-linear atau interaksi kompleks antar fitur.
 ```
model_lr = LinearRegression(fit_intercept=False)
model_lr.fit(train_final, y_train)

y_pred_lr = model_lr.predict(test_final)
 ```
 Menggunakan parameter `fit_intercept=False` yang berarti model tidak akan menghitung atau menggunakan intercept (bias). Dengan kata lain, garis regresi harus melewati titik nol pada sumbu y. Pilihan ini sesuai jika diasumsikan bahwa saat semua variabel fitur bernilai nol, nilai target juga harus nol. Namun, jika asumsi ini tidak terpenuhi, model mungkin tidak dapat menangkap pola data dengan baik.
 
 **Feature Importance Koefisien Regresi**
 
 Feature Importance Koefisien Regresi digunakan untuk mengetahui seberapa besar pengaruh masing-masing fitur (variabel input) terhadap hasil prediksi dalam model regresi linear atau logistik. Ini sangat berguna untuk interpretasi model, feature selection, dan pemahaman data.

Adapun hasilnya yaitu:

![fitur regresi](https://github.com/user-attachments/assets/6035f9ba-f4e2-4535-bbb2-0c1236d5b113)

Hari dalam minggu (Weekday dan Weekend) dan tipe bangunan paling berpengaruh besar terhadap konsumsi energi. Luas bangunan, jumlah penghuni, dan penggunaan peralatan juga meningkatkan konsumsi, sedangkan suhu rata-rata sedikit menurunkannya. Jadi, konsumsi energi dipengaruhi utama oleh waktu, jenis bangunan, dan aktivitas di dalamnya.
 

### Model 2: Gradient Boosting Regressor
**Gradient Boosting Regressor**  adalah algoritma ensemble yang membangun model prediksi secara bertahap dengan menggabungkan banyak model pohon keputusan (decision trees) yang lemah (weak learners). Setiap model berikutnya berfokus memperbaiki kesalahan dari model sebelumnya dengan meminimalkan fungsi loss menggunakan metode gradient descent. **Kelebihannya**, akurasi tinggi dan fleksibel untuk berbagai tipe data. **Kekurangannya**, proses pelatihan lebih lambat dan rentan overfitting jika parameter tidak diatur dengan baik.
 ```
model_gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
model_gb.fit(train_final, y_train)

y_pred_gb = model_gb.predict(test_final)
 ```
Model Gradient Boosting Regressor dengan `n_estimators=100` berarti model akan membangun 100 pohon keputusan bertingkat secara berurutan untuk memperbaiki kesalahan prediksi. Parameter `random_state=42` digunakan untuk memastikan hasil yang konsisten setiap kali model dijalankan, memudahkan reproduksi eksperimen. 

**Feature Importance pada Model Tree-Based**

Model berbasis pohon, seperti Gradient Boosting Regressor, secara alami menghitung **feature importance** berdasarkan kontribusi setiap fitur dalam membagi data untuk mengurangi kesalahan prediksi. Nilai importance mencerminkan seberapa sering dan seberapa efektif fitur digunakan dalam membangun pohon keputusan.
Semakin tinggi nilai importance, semakin besar peran fitur tersebut dalam mempengaruhi prediksi.

| No | Fitur                    | Importance |
|----|--------------------------|------------|
| 1  | Square Footage           | 0.607152   |
| 2  | Building Type_Residential| 0.103194   |
| 3  | Number of Occupants      | 0.101980   |
| 4  | Appliances Used          | 0.094511   |
| 5  | Building Type_Industrial | 0.091709   |

**Square Footage** merupakan fitur paling penting dengan skor > 0.60, artinya luas bangunan sangat dominan dalam menentukan konsumsi energi. Fitur-fitur lain seperti jumlah penghuni dan tipe bangunan juga berkontribusi, meskipun lebih kecil.

### Pemilihan Model Terbaik:
| i   | Actual Energy Consumption | Linear Regression | Gradient Boosting Regressor  |
|-----|---------------------------|-------------------|------------------------------|
| 521 | 4549.59                   | 4549.598244       | 4622.515458                  |
| 737 | 2842.91                   | 2842.901083       | 2843.991151                  |
| 740 | 5781.83                   | 5781.847788       | 5687.186715                  |
| 660 | 4773.54                   | 4773.550577       | 4825.540694                  |
| 411 | 3791.04                   | 3791.050217       | 3894.664899                  |

Dari data tersebut, **Linear Regression** memberikan prediksi yang sangat mendekati nilai aktual, misalnya pada **indeks ke-521** dan **737** hasil prediksi hampir sama persis. Sementara itu, **Gradient Boosting** juga mendekati, tapi kadang sedikit meleset, seperti di indeks **ke-521** dan **660**, di mana prediksi Gradient Boosting lebih jauh dari nilai aktual dibanding  Linear Regression. Ini menunjukkan Linear Regression model lebih stabil dan akurat pada data ini, sedangkan Gradient Boosting kurang presisi pada beberapa titik meskipun secara umum masih bagus.

Berdasarkan hasil evaluasi, **Linear Regression**  merupakan model terbaik untuk prediksi konsumsi energi pada data ini karena menghasilkan error **(MAE, MSE, RMSE)** yang sangat kecil dan nilai **R²** sempurna, menunjukkan prediksi yang sangat akurat dan sesuai dengan pola data yang bersifat linear. Sementara itu, meskipun **Gradient Boosting Regressor** memiliki nilai **R²** tinggi, error yang jauh lebih besar menandakan prediksi kurang presisi di beberapa titik, sehingga model ini kurang cocok tanpa tuning lebih lanjut. Oleh karena itu, **Linear Regression** lebih efisien dan tepat digunakan untuk kasus ini.

## Evaluation

### Metrik Evaluasi:
Metriks evaluasi yang digunakan ada 3 yaitu:
1. **MAE** 

   ```MAE = (1/n) * Σ |yᵢ - ŷᵢ| ```

Mengukur rata-rata selisih absolut antara nilai prediksi dan nilai aktual, memberikan gambaran seberapa besar kesalahan prediksi secara umum tanpa memperhatikan arah kesalahan.

2. **RMSE** 

```RMSE = sqrt(MSE) = sqrt((1/n) * Σ (yᵢ - ŷᵢ)²) ```

Merupakan akar kuadrat dari MSE. Metrik ini mengembalikan nilai kesalahan ke satuan asli dari target, sehingga lebih mudah diinterpretasikan daripada MSE, sambil tetap mempertahankan sensitivitas terhadap kesalahan besar.

3. **R² (R-squared Score)** 

```R² = 1 - (Σ (yᵢ - ŷᵢ)² / Σ (yᵢ - ȳ)²) ```

Mengukur proporsi variansi dalam data target yang dapat dijelaskan oleh fitur prediktor dalam model. Semakin tinggi nilai R² (maksimum 1), semakin baik model dalam menjelaskan pola data.

### Hasil Evaluasi:

| Metrik                         | Linear Regression | Gradient Boosting Regressor |
|-------------------------------|-------------------|------------------------------|
| MAE (Mean Absolute Error)     | 0.0114            | 74.7694                      |
| RMSE (Root Mean Squared Error)| 0.0137            | 94.0786                      |
| R² (R-squared Score)          | 1.0000            | 0.9891                       |

Hasil evaluasi `Linear Regression` menunjukkan performa yang sangat baik dengan **MAE** dan **RMSE** yang sangat kecil yaitu, **0.0114** dan **0.0137** serta **R² = 1**. Ini mengindikasikan model mampu memprediksi nilai penggunaan energi dengan akurasi tinggi dan kesalahan yang sangat minimal.

Sedangkan pada hasil evaluasi `Gradient Boosting Regressor` menunjukkan **MAE 74.77** dan **RMSE 94.08**, yang berarti rata-rata prediksi meleset cukup jauh dari nilai sebenarnya. Namun, dengan **R² sebesar 0.9891**, model mampu menjelaskan **98.91%** variansi data, jadi prediksi secara keseluruhan cukup akurat meski beberapa titik kurang presisi.

Berikut adalah jika dilihat perbandingan titik prediksinya:

![comparasi](https://github.com/user-attachments/assets/d9ba7c6b-4ce0-48ad-bb3a-8b21ac2564bc)


---

### Referensi:
IEA, [“Buildings – Global Energy Review 2023”](https://www.iea.org/reports/global-energy-review-2023/buildings)
T. Ahmad, H. Chen, ["Potential of three variant machine-learning models for forecasting district level medium-term and long-term energy demand in smart grid environment"](https://doi.org/10.1016/j.egyr.2018.01.008), Energy Reports, vol. 4, pp. 62–73, 2018.
