import os, shutil

sumber = "dataset_raw"
tujuan = "dataset_bersih"
os.makedirs(tujuan, exist_ok=True)

for kelas in ["spoon", "fork"]:
    folder = os.path.join(sumber, kelas)
    for i, f in enumerate(sorted(os.listdir(folder)), start=1):
        ext = os.path.splitext(f)[1].lower()
        shutil.copy(os.path.join(folder, f), os.path.join(tujuan, f"{kelas}_{i:03d}{ext}"))

print("Selesai. Jumlah gambar:", len(os.listdir(tujuan)))