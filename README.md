# 🧠 Tugas Besar 2 IF3270 Pembelajaran Mesin - RNN, CNN, dan LSTM

---

## 📚 Deskripsi Repositori  

Repositori ini berisi implementasi **Forward Propagation** untuk **Convolutional Neural Network (CNN), Recurrent Neural Network (RNN), dan Long Short Term Memory (LSTM)** menggunakan Python. Projek ini dibuat untuk memenuhi Tugas Besar 2 mata kuliah IF3270 Pembelajaran Mesin.

**Fitur Utama:**  
✅ Implementasi **forward propagation CNN** <br/>
✅ Implementasi **forward propagation RNN** <br/>
✅ Implementasi **forward propagation LSTM**

---

## 🗂 Struktur Folder  

```
|   README.md
|
+---data
|   \---nusax
|           analyze_vocab.py
|           test.csv
|           train.csv
|           valid.csv
|
+---doc
|       laporan.txt
|
\---src
    +---cnn
    |   |   cnn.py
    |   |   layer.py
    |   |
    +---lstm
    |   |   dropout_layer.py
    |   |   embedding_layer.py
    |   |   lstm_layer.py
    |   |   lstm_model.py
    |   |   text_preprocessor.py
    |   |   __init__.py
    |   |
    +---notebooks
    |   +---cnn
    |   |       conv_layer_analysis.ipynb
    |   |       filter_amount_analysis.ipynb
    |   |       filter_size_analysis.ipynb
    |   |       keras_model_cifar10.h5
    |   |       pooling_analysis.ipynb
    |   |       scratch_keras_comparison.ipynb
    |   |
    |   +---lstm
    |   |       lstm_cells_analysis.ipynb
    |   |       lstm_directionality_analysis.ipynb
    |   |       lstm_layers_analysis.ipynb
    |   |       lstm_model_comparison.ipynb
    |   |
    |   \---rnn
    |           cell_analysis.ipynb
    |           directionality_analysis.ipynb
    |           layer_analysis.ipynb
    |           scratch_keras_comparison.ipynb
    |
    \---rnn
        |   keras.py
        |   layer.py
        |   model.py
        |   preprocessing.py
        |   train.py
        |   utils.py
        |
        \---output
                model_weights.npz
                training_results.json
```  

---

## 🚀 Cara Mulai  

### 📥 Instalasi  

1. **Clone repositori:**  

   ```bash
   git clone https://github.com/nicholasrs05/if3270-tubes-2.git
   cd if3270-tubes-2
   ```  

2. **Instal dependensi:**  

   ```bash
   pip install *
   ```  


### Run Program  

1. Buka notebook yang ingin dijalankan
2. Pada bagian atas, tekan 'Run All'

---

## 📊 Analisis yang Tersedia di `src/notebooks`  

1. CNN
- Analisis Pengaruh Jumlah Layer Konvolusi
- Analisis Pengaruh Banyak Filter per Layer Konvolusi
- Analisis Pengaruh Ukuran Filter per Layer Konvolusi
- Analisis Pengaruh Jenis Pooling Layer
- Perbandingan dengan CNN Scratch
2. RNN
- Analisis Pengaruh Jumlah Layer RNN
- Analisis Pengaruh Banyak Cell RNN per Layer
- Analisis Pengaruh Jenis Layer RNN berdasarkan Arah
- Perbandingan RNN Keras dengan Scratch Implementation
3. LSTM
- Analisis Pengaruh Jumlah Layer LSTM
- Analisis Pengaruh Banyak Cell LSTM per Layer
- Analisis Pengaruh Jenis Layer LSTM berdasarkan Arah
- Perbandingan LSTM Keras dengan Scratch Implementation

---

## 👥 Pembagian Tugas Kelompok  

| Nim Anggota | Kontribusi |  
|--------------|------------|  
| **13522122**     | Implementasi dan Analisis RNN |  
| **13522144**      | Implementasi dan Analisis CNN |  
| **13522150**     | Implementasi dan Analisis LSTM |  

---

## 📜 Referensi  

- [Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)  

---
