#!/usr/bin/env python3
"""
YOLOv11 Model Eğitimi
Roboflow veri seti klasöründeki README.roboflow.txt'ten
model ismini otomatik okur.
"""

from ultralytics import YOLO
import os
import re
import torch


def parse_model_name_from_readme(data_dir: str) -> str:
    """
    README.roboflow.txt dosyasının ilk satırından model ismini çeker.
    Örnek: "Car Detection - v4 Car Detection 4"
    Çıktı : "Car_Detection_v4_Car_Detection_4"
    """
    readme_path = os.path.join(data_dir, "README.roboflow.txt")

    if not os.path.exists(readme_path):
        print(f"[UYARI] README.roboflow.txt bulunamadı: {readme_path}")
        print("[UYARI] Varsayılan isim kullanılıyor: yolo11_model")
        return "yolo11_model"
    
    with open(readme_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # ilk 2 satırı al ve birleştir
    first_two_lines = " ".join(line.strip() for line in lines[:2])

    if not first_two_lines:
        print("[UYARI] README.roboflow.txt boş. Varsayılan isim kullanılıyor.")
        return "yolo11_model"

    # Harf, rakam ve boşluk dışındaki karakterleri temizle, boşlukları '_' yap
    clean_name = re.sub(r"[^\w\s]", "", first_two_lines)   # noktalama kaldır
    clean_name = re.sub(r"\s+", "_", clean_name.strip())  # boşluk → _

    print(f"[BİLGİ] README'den okunan isim : '{first_two_lines}'")
    print(f"[BİLGİ] Kullanılacak model adı : '{clean_name}'")
    return clean_name


def main():
    # ------------------------------------------------------------------ #
    # Cihaz seçimi
    # ------------------------------------------------------------------ #
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Kullanılan cihaz: {device}")

    # ------------------------------------------------------------------ #
    # Veri seti yolu  →  script'in bulunduğu klasöre göre hesaplanır
    # ------------------------------------------------------------------ #
    script_dir = os.path.dirname(os.path.abspath(__file__))  # V7/
    data_path  = os.path.join(script_dir, "model", "data.yaml")
    data_dir   = os.path.dirname(data_path)                  # V7/model/

    # ------------------------------------------------------------------ #
    # Model adını README.roboflow.txt'ten otomatik oku
    # ------------------------------------------------------------------ #
    model_name = parse_model_name_from_readme(data_dir)

    # ------------------------------------------------------------------ #
    # YOLOv11 modelini yükle
    # ------------------------------------------------------------------ #
    model = YOLO("yolo11n.pt")  # nano versiyonu

    # ------------------------------------------------------------------ #
    # Eğitim parametreleri
    # ------------------------------------------------------------------ #
    training_args = {
        "data"            : data_path,
        "epochs"          : 100,
        "imgsz"           : 640,
        "batch"           : 16,
        "device"          : device,
        "workers"         : 4,
        "patience"        : 20,
        "save"            : True,
        "save_period"     : 10,
        "cache"           : True,
        "project"         : "runs/train",
        "name"            : model_name,   # ← README'den gelen isim
        "exist_ok"        : True,
        "pretrained"      : True,
        "optimizer"       : "AdamW",
        "lr0"             : 0.01,
        "lrf"             : 0.01,
        "momentum"        : 0.937,
        "weight_decay"    : 0.0005,
        "warmup_epochs"   : 3,
        "warmup_momentum" : 0.8,
        "warmup_bias_lr"  : 0.1,
        "box"             : 7.5,
        "cls"             : 0.5,
        "dfl"             : 1.5,
        "label_smoothing" : 0.0,
        "nbs"             : 64,
        "hsv_h"           : 0.015,
        "hsv_s"           : 0.7,
        "hsv_v"           : 0.4,
        "degrees"         : 0.0,
        "translate"       : 0.1,
        "scale"           : 0.5,
        "shear"           : 0.0,
        "perspective"     : 0.0,
        "flipud"          : 0.0,
        "fliplr"          : 0.5,
        "mosaic"          : 1.0,
        "mixup"           : 0.0,
        "copy_paste"      : 0.0,
    }

    print("\nEğitim başlatılıyor...")
    print(f"  Veri seti   : {data_path}")
    print(f"  Model adı   : {model_name}")
    print(f"  Epoch sayısı: {training_args['epochs']}")
    print(f"  Batch size  : {training_args['batch']}")
    print(f"  Görüntü boyutu: {training_args['imgsz']}")

    try:
        results = model.train(**training_args)

        print("\nEğitim tamamlandı!")
        print(f"En iyi model : {results.save_dir}/weights/best.pt")
        print(f"Son model    : {results.save_dir}/weights/last.pt")

        print("\nModel değerlendiriliyor...")
        metrics = model.val(data=data_path)

        print(f"mAP50    : {metrics.box.map50:.4f}")
        print(f"mAP50-95 : {metrics.box.map:.4f}")

        return results

    except Exception as e:
        print(f"Eğitim sırasında hata oluştu: {e}")
        return None


if __name__ == "__main__":
    results = main()