# Analisis fitur: prediksi benar vs salah (BRISC2025 test, EdgeNeXt xx_small)

Skrip: `scripts/analyze_error_features.py` · Data: `features.csv`, `summary.txt`
Sampel: 860 gambar tumor (848 benar, 12 salah). `no_tumor` tidak dianalisis karena tidak ada mask dan tidak ada kesalahan.

## 1. Secara statistik, gambar salah tidak berbeda dari yang benar
Ada 19 fitur yang diuji dengan Mann–Whitney U: ukuran, posisi, bentuk, kontras, dan heterogenitas tumor, serta ketajaman, noise, SNR, dan padding citra. Semua p > 0,38.
Catatan: n salah = 12, jadi uji ini hanya bisa mendeteksi efek besar.

## 2. Hanya 3 dari 12 kesalahan yang dijelaskan oleh kemiripan ciri dengan kelas prediksi
Profil median per kelas: area/kepala, jarak dari pusat kepala, dan kontras terhadap sekitar.

| kelas | area | jarak pusat | kontras |
|---|---|---|---|
| glioma | 0,029 | 0,46 | 14,8 |
| meningioma | 0,040 | 0,51 | 60,3 |
| pituitary | 0,011 | 0,20 | 52,0 |

Kasus yang cocok dengan hipotesis "mirip kelas lain":
- **00365, 00367 (meningioma → pituitary):** kecil (persentil 7 dan 13), di tengah (jarak 0,19–0,25), terang. Ini profil pituitary, sehingga **lokasi dan ukuran** menyesatkan.
- **00229 (glioma → meningioma):** kontras sangat tinggi (59,9, persentil 95) dan berbatas tegas. Ini profil meningioma, sehingga **penampakan tumor** menyesatkan.

## 3. Kasus khusus
- **00244 (glioma → meningioma, conf 0,54):** kontras negatif (−9,8), yaitu tumor lebih gelap dari sekitarnya (non-enhancing). Ini **kasus kontras rendah**, dan model ragu.
- **00974 (pituitary → meningioma, conf 1,00):** fiturnya sangat khas pituitary (jarak ke profil pituitary 0,44, terdekat dari semua kasus), tetapi model 100% yakin meningioma. Ini indikasi kuat model melihat sesuatu **di luar tumor**, atau labelnya perlu diverifikasi.
- **4 kasus meningioma → glioma:** fiturnya tidak mirip glioma. Secara visual, 00281 tampak seperti sekuens T2, 00533/00543 berbeda protokol. Perbedaan protokol ini tidak tertangkap oleh fitur yang diukur.

## Kesimpulan
Ukuran, lokasi, dan kualitas citra yang diukur **tidak menjelaskan sebagian besar kesalahan (9/12)**.
Penyebabnya lebih mungkin:
1. perbedaan sekuens atau protokol MRI;
2. model memperhatikan area di luar tumor (perlu verifikasi Grad-CAM vs mask);
3. label yang ambigu.
