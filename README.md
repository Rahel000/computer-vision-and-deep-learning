# Transfer Learning: Klasifikasi Sendok vs Garpu

## Dataset
100 foto (50 spoon, 50 fork) di folder `spoon/` dan `fork/`, metadata di `metadata.csv`.
Split acak (seed 42): 80 train, 20 validation. Resize 224x224, normalisasi ImageNet.

## Model
ResNet-50 pretrained ImageNet dengan metode **feature extraction**: seluruh bobot ResNet
dibekukan, hanya classifier baru (2 output: spoon/fork) yang dilatih.
Optimizer Adam (lr 0.001), batch size 16, 10 epoch.
Hasil di bawah berasal dari run ke-4 (output tersimpan di notebook).

| Metrik | Hasil ResNet-50 |
|---|---|
| Train acc akhir | 98,75% |
| Val acc akhir | 80% |
| Val acc terbaik | 85% (epoch 5-8) |
| Val loss akhir | 0.465 |
| Waktu training | 7.2 s |
| Latensi inferensi | 6.4 ms/foto |

## Analisis
Seed PyTorch tidak dikunci, sehingga angka tiap run berbeda. Karena itu analisis berfokus
pada pola, bukan angka persis.

- **Model belajar:** train acc naik dari 43,75% (epoch 1, setara tebakan acak) ke 98,75%,
  dan val loss turun terus dari 0.668 ke 0.465. Ini menandakan classifier baru berhasil
  memanfaatkan fitur ImageNet untuk membedakan sendok dan garpu.
- **Akurasi validation:** naik dari 65% ke puncak 85% (epoch 5-8), lalu 80% di dua epoch
  terakhir. Dengan 20 foto validation, penurunan itu setara 1 foto (5%) dan masih bisa
  dianggap fluktuasi. Pada run awal, val acc akhir mencapai 90%, jadi hasil akhirnya
  berkisar 80-90% antar run.
- **Jarak train-validation:** train acc (98,75%) jauh di atas val acc (80%). Ini bisa
  menandakan model lebih cocok dengan foto latihan daripada foto baru, tetapi karena
  val loss belum naik, belum ada bukti overfitting yang jelas. Jumlah data yang kecil
  membuat kesimpulan ini belum pasti.
- **Waktu dan latensi:** training 10 epoch hanya sekitar 7 detik karena yang dilatih hanya
  classifier dan GPU dipakai. Latensi inferensi adalah waktu menebak satu foto setelah
  model selesai dilatih.
- **Kesimpulan:** feature extraction dengan ResNet-50 pretrained sudah cukup untuk
  mengklasifikasi sendok dan garpu dengan akurasi sekitar 80-90% pada dataset kecil ini,
  dengan waktu training yang sangat singkat. Hasil bisa ditingkatkan dengan lebih banyak
  data atau fine-tuning sebagian layer.

## Keterbatasan
Validation hanya 20 foto, sehingga 1 foto salah = 5%. Eksperimen dijalankan empat kali tanpa
mengunci seed, sehingga hasilnya bervariasi. Tidak ada test set terpisah. Pengulangan dengan
beberapa seed dan test set akan memberi evaluasi yang lebih andal. Latensi diukur pada GPU
Colab (6.4 ms/foto).

## Cara menjalankan
Buka `transfer-learning/notebook.ipynb` di Google Colab, aktifkan GPU, jalankan semua cell.

![Feature extraction](grafik_feature_extraction.png)
