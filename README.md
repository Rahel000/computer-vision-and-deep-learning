# Transfer Learning: Klasifikasi Sendok vs Garpu

## Dataset
100 foto kamera HP (50 spoon, 50 fork) di folder `spoon/` dan `fork/`, metadata di `metadata.csv`.
Split acak (seed 42): 80 train, 20 validation. Resize 224x224, normalisasi ImageNet.

## Model
ResNet-50 pretrained ImageNet, classifier diganti menjadi 2 output.
Hasil di bawah berasal dari run ke-4 (output tersimpan di notebook).

| Metode | Val acc akhir | Val acc terbaik | Val loss akhir | Waktu training | Latensi inferensi |
|---|---|---|---|---|---|
| Feature extraction | 80% | 85% (epoch 5-8) | 0.465 | 7.2 s | XX ms/foto |
| Fine-tuning layer4 | 95% | 95% (epoch 2-4, 6-10) | 0.160 | 8.5 s | XX ms/foto |

## Analisis
Seed PyTorch tidak dikunci, sehingga angka tiap run berbeda. Karena itu analisis berfokus
pada pola yang muncul di beberapa kali run, bukan angka persis.

- **Akurasi:** pada run ini fine-tuning berakhir di 95% dan feature extraction di 80%
  (selisih 3 dari 20 foto). Pada run awal, keduanya sama-sama berakhir di 90%, jadi besar
  selisih akurasi berubah-ubah antar run dan belum cukup untuk disimpulkan signifikan.
- **Kecepatan belajar dan loss:** fine-tuning layer4 konsisten belajar lebih cepat. Val acc
  sudah 95% sejak epoch 2 dan val loss akhir 0.160, sedangkan feature extraction masih
  0.465 dan val loss-nya masih turun di epoch 10.
- **Overfitting:** pada fine-tuning, train acc mencapai 100% sejak epoch 5 dan val loss naik
  tipis setelah epoch 8 (0.142 menjadi 0.160), tetapi val acc tetap 95%, sehingga overfitting
  tergolong ringan pada run ini. Pada run lain gejalanya lebih jelas (val acc turun).
  Feature extraction memiliki jarak train-validation yang cukup besar (98,75% vs 80%),
  yang mungkin menandakan fitur ImageNet yang dibekukan kurang pas untuk tugas ini.
- **Waktu dan latensi:** selisih waktu training sekitar 1 detik dan arahnya berbalik antar
  run, sehingga tidak bermakna. Latensi inferensi kedua metode hampir sama karena arsitekturnya
  sama (ResNet-50).
- **Kesimpulan:** pada dataset sekecil ini, fine-tuning layer4 memberi loss lebih rendah dan
  belajar lebih cepat, namun berisiko overfitting sehingga sebaiknya dihentikan lebih awal.
  Feature extraction lebih sederhana tetapi pada percobaan ini hasilnya lebih rendah.

## Keterbatasan
Validation hanya 20 foto, sehingga 1 foto salah = 5%. Eksperimen dijalankan empat kali tanpa
mengunci seed, sehingga hasilnya bervariasi. Pengulangan dengan beberapa seed dan test set
terpisah akan memberi evaluasi yang lebih andal. Latensi diukur pada GPU Colab (XX).

## Cara menjalankan
Buka `transfer-learning/notebook.ipynb` di Google Colab, aktifkan GPU, jalankan semua cell.

![Feature extraction](grafik_feature_extraction.png)
![Perbandingan](grafik_perbandingan.png)
