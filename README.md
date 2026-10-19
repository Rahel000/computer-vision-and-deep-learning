# Computer Vision and Deep Learning

Repositori ini berisi tugas dan praktik mata kuliah RET503 Computer Vision and Deep Learning.
Nama: Rahel Evanggelisa | NIM: 4222401022

## Daftar Proyek

| Proyek | Folder | Ringkasan |
|---|---|---|
| Transfer Learning | [`Transfer_Learning/`](Transfer_Learning/) | Eksperimen transfer learning (feature extraction) pada foto sendok dan garpu. Berisi notebook, 50 foto per kelas, `metadata.csv`, dan grafik hasil. |
| Deteksi Sendok dan Garpu (YOLOv8) | [`YOLO-Spoon_and_Fork/`](YOLO-Spoon_and_Fork/) | Detektor objek YOLOv8 untuk kelas Fork dan Spoon dengan dataset sendiri (dianotasi di Roboflow). Membandingkan transfer learning, pelatihan dari nol, dan tuning ukuran input. |

Penjelasan lengkap tiap proyek ada di `README.md` di dalam foldernya masing-masing.

## Struktur Repositori

```
computer-vision-and-deep-learning/
  README.md                 (dokumen ini)
  Transfer_Learning/
    README.md
    Transfer_Learning.ipynb
    fork/  spoon/           (foto)
    metadata.csv
    grafik_feature_extraction.png
  YOLO-Spoon_and_Fork/
    README.md
    latih_banding.py        (pelatihan dan evaluasi E1, E2, E3)
    gabung.py
    class_balance.csv
    dataset_v1/             (dataset format YOLOv8 dari Roboflow)
    hasil_praktik/          (grafik dan confusion matrix)
    hasil_praktik_4222401022.txt / .csv
```

## Alat yang Dipakai

Python, Ultralytics (YOLOv8), Roboflow (anotasi dan pembagian dataset), dan VS Code.
