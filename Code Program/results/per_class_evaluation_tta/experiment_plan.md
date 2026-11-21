# Eksperimen lanjutan yang disiapkan

Setiap eksperimen harus dijalankan terpisah dengan split train/validation dan checkpoint sendiri. Test set hanya dipakai sekali pada evaluasi akhir masing-masing eksperimen; jangan memilih konfigurasi berdasarkan test set. Kolom `not_run` sengaja dibiarkan kosong agar tidak ada angka fiktif. Metrik utama: macro F1-score dan balanced accuracy; recall per kelas tetap dilaporkan.

Urutan yang disarankan: class weight, focal loss, oversampling, augmentasi, CLAHE/kontras, lalu ukuran citra. Ubah satu faktor pada satu waktu dan simpan config, history, checkpoint, serta hasil test ke folder eksperimen yang berbeda.
