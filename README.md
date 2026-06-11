# Tree Image Search CBIR

He thong truy van anh cay theo noi dung anh, su dung vector dac trung 62 chieu:

- Mau sac: 24 chieu
- Hinh thai: 12 chieu
- Ket cau: 17 chieu
- Tan cay: 9 chieu

## Cai dat

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install gradio pillow rembg
```

## Build database

Lenh nay trich xuat dac trung 62 chieu tu dataset, fit Z-score normalizer va tao lai `vector_db.npz`, `normalizer.npz`.

```bash
python3 main.py --build --image_dir Raw_Tree_Dataset_Test
```

Ket qua mong doi:

```text
Records    : 1769
So chieu   : 62
Loai cay   : 20
```

## Chay ung dung Gradio

```bash
python3 app.py
```

Sau khi chay, mo trinh duyet tai:

```text
http://127.0.0.1:7860
```

## Truy van bang command line

```bash
python3 main.py --query Raw_Tree_Dataset_Test/10_Jacaranda_Tree/000001.jpg --k 5
```

## Danh gia

Danh gia nhanh tren mau nho:

```bash
python3 evaluate.py --k 5 --sample 200
```

Danh gia toan bo database:

```bash
python3 evaluate.py --k 5
```

## Demo truc quan dac trung

```bash
python3 demo.py Raw_Tree_Dataset_Test/10_Jacaranda_Tree/000001.jpg
```
