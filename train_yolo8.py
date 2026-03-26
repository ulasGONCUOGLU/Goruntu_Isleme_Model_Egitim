#!/usr/bin/env python3
"""
YOLOv8 Model Eğitimi
CCTV Araç ve Motosiklet Tespiti için
"""

from ultralytics import YOLO
import os
import torch
import yaml

def find_data_yaml():
    """data.yaml dosyasını farklı konumlarda ara"""
    possible_paths = [
        'data.yaml',
        'dataset/data.yaml',
        'datasets/data.yaml',
        'model/data.yaml',
        'models/data.yaml',
        'yolo_data/data.yaml',
        './data.yaml',
        '../data.yaml',
        os.path.join(os.getcwd(), 'data.yaml'),
        os.path.join(os.path.dirname(__file__), 'data.yaml'),
        os.path.join(os.path.dirname(__file__), 'model/data.yaml'),
        os.path.join(os.path.dirname(__file__), 'dataset/data.yaml'),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ data.yaml bulundu: {path}")
            return path
    
    return None

def create_default_data_yaml(save_path='data.yaml'):
    """Varsayılan data.yaml dosyasını oluştur"""
    data_config = {
        'path': os.path.join(os.getcwd(), 'dataset'),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 5,
        'names': [
            'articulated_truck',
            'bus',
            'long_vehicle',
            'motorcycle',
            'short_vehicle'
        ]
    }
    
    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(data_config, f, allow_unicode=True, default_flow_style=False)
        print(f"✅ Varsayılan data.yaml oluşturuldu: {save_path}")
        print(f"   Sınıflar: {data_config['names']}")
        return save_path
    except Exception as e:
        print(f"❌ data.yaml oluşturulamadı: {e}")
        return None

def load_class_names(data_yaml_path):
    """data.yaml dosyasından sınıf isimlerini yükler"""
    try:
        with open(data_yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        names = data.get('names', [])
        nc = data.get('nc', len(names))
        
        if not names and nc > 0:
            names = [f'class_{i}' for i in range(nc)]
        
        return names, nc
    except Exception as e:
        print(f"⚠️ data.yaml okunurken hata: {e}")
        return None, 0

def check_dataset_structure(data_yaml_path):
    """Veri seti yapısını kontrol et"""
    try:
        with open(data_yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        base_path = data.get('path', '')
        train_path = data.get('train', '')
        val_path = data.get('val', '')
        
        # Tam yolları oluştur
        if base_path:
            train_full = os.path.join(base_path, train_path) if not os.path.isabs(train_path) else train_path
            val_full = os.path.join(base_path, val_path) if not os.path.isabs(val_path) else val_path
        else:
            train_full = train_path
            val_full = val_path
        
        print("\n📁 Veri seti kontrolü:")
        print(f"  - data.yaml konumu: {data_yaml_path}")
        print(f"  - Base path: {base_path}")
        print(f"  - Train path: {train_full}")
        print(f"  - Val path: {val_full}")
        
        # Klasörleri kontrol et
        train_images_exist = os.path.exists(train_full)
        val_images_exist = os.path.exists(val_full)
        
        # Label klasörlerini kontrol et
        train_label_path = train_full.replace('images', 'labels') if 'images' in train_full else os.path.join(os.path.dirname(train_full), 'labels')
        val_label_path = val_full.replace('images', 'labels') if 'images' in val_full else os.path.join(os.path.dirname(val_full), 'labels')
        
        if train_images_exist:
            print(f"  ✅ Train görüntü klasörü: {train_full}")
            train_images = [f for f in os.listdir(train_full) if f.endswith(('.jpg', '.jpeg', '.png'))]
            print(f"     - {len(train_images)} görüntü bulundu")
        else:
            print(f"  ❌ Train görüntü klasörü bulunamadı: {train_full}")
            
        if val_images_exist:
            print(f"  ✅ Val görüntü klasörü: {val_full}")
            val_images = [f for f in os.listdir(val_full) if f.endswith(('.jpg', '.jpeg', '.png'))]
            print(f"     - {len(val_images)} görüntü bulundu")
        else:
            print(f"  ❌ Val görüntü klasörü bulunamadı: {val_full}")
        
        if os.path.exists(train_label_path):
            print(f"  ✅ Train label klasörü: {train_label_path}")
        else:
            print(f"  ⚠️ Train label klasörü bulunamadı: {train_label_path}")
            
        if os.path.exists(val_label_path):
            print(f"  ✅ Val label klasörü: {val_label_path}")
        else:
            print(f"  ⚠️ Val label klasörü bulunamadı: {val_label_path}")
            
    except Exception as e:
        print(f"⚠️ Veri seti kontrolü yapılamadı: {e}")

def main():
    # GPU kullanılabilirliğini kontrol et
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"💻 Kullanılan cihaz: {device}")
    if torch.cuda.is_available():
        print(f"   GPU Modeli: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Belleği: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # data.yaml dosyasını bul
    data_path = find_data_yaml()
    
    # Eğer data.yaml bulunamadıysa, oluştur
    if not data_path:
        print("⚠️ data.yaml bulunamadı, varsayılan dosya oluşturuluyor...")
        data_path = create_default_data_yaml()
        
        if not data_path:
            print("❌ data.yaml oluşturulamadı! Lütfen veri seti konumunu kontrol edin.")
            return None
    
    # Sınıf isimlerini yükle
    class_names, num_classes = load_class_names(data_path)
    
    print("\n" + "="*60)
    print("🚀 YOLOv8 EĞİTİM BAŞLATILIYOR")
    print("="*60)
    print(f"📊 Model: YOLOv8 Nano (yolov8n.pt)")
    print(f"📁 Veri seti: {data_path}")
    
    if class_names:
        print(f"🎯 Sınıf sayısı: {num_classes}")
        print("   Sınıflar:")
        for idx, name in enumerate(class_names):
            print(f"     {idx}: {name}")
    else:
        print(f"🎯 Sınıf sayısı: {num_classes}")
    
    # Veri seti yapısını kontrol et
    check_dataset_structure(data_path)
    
    # Kullanıcıdan onay al
    print("\n" + "="*60)
    devam = input("Eğitimi başlatmak için 'e' tuşuna basın (çıkmak için 'h'): ")
    if devam.lower() != 'e':
        print("❌ Eğitim iptal edildi.")
        return None
    print("="*60 + "\n")
    
    # YOLOv8 modelini yükle
    try:
        # YOLOv8 nano modeli
        model = YOLO('yolov8n.pt')
        print("✅ YOLOv8 nano modeli yüklendi")
        
        # Model bilgilerini göster
        print(f"   Model tipi: {model.task}")
        print(f"   Model sınıfları: {model.model.names if hasattr(model, 'model') else 'Bilinmiyor'}")
        
    except Exception as e:
        print(f"❌ Model yüklenemedi: {e}")
        print("   Alternatif model deneniyor...")
        try:
            model = YOLO('yolov8s.pt')  # Small model
            print("✅ YOLOv8 small modeli yüklendi")
        except:
            print("❌ Model yüklenemedi! Lütfen 'pip install ultralytics' komutunu çalıştırın.")
            return None
    
    # Eğitim parametreleri - YOLOv8 için optimize edilmiş
    training_args = {
        'data': data_path,
        'epochs': 100,           # Eğitim epoch sayısı
        'imgsz': 640,           # Giriş görüntü boyutu
        'batch': 16,            # Batch size (GPU varsa 16, yoksa 8)
        'device': device,       # Kullanılacak cihaz
        'workers': 4,           # DataLoader worker sayısı
        'patience': 20,        # Early stopping patience
        'save': True,          # Model kaydet
        'save_period': 10,     # Her 10 epoch'ta bir kaydet
        'cache': True,         # Veri önbellekleme
        'project': 'runs/train', # Proje klasörü
        'name': 'cctv_vehicle_detection_yolov8', # Model adı
        'exist_ok': True,      # Mevcut klasörü kullan
        'pretrained': True,    # Önceden eğitilmiş ağırlıkları kullan
        'optimizer': 'AdamW',  # Optimizer
        'lr0': 0.01,           # Başlangıç öğrenme oranı
        'lrf': 0.01,           # Final öğrenme oranı
        'momentum': 0.937,     # SGD momentum
        'weight_decay': 0.0005, # Weight decay
        'warmup_epochs': 3,    # Warmup epoch sayısı
        'warmup_momentum': 0.8, # Warmup momentum
        'warmup_bias_lr': 0.1, # Warmup bias learning rate
        'box': 7.5,           # Box loss gain
        'cls': 0.5,           # Class loss gain
        'dfl': 1.5,           # DFL loss gain
        'label_smoothing': 0.0, # Label smoothing
        'nbs': 64,            # Nominal batch size
        # Veri augmentasyonu - CCTV için optimize
        'hsv_h': 0.015,       # HSV-Hue augmentation
        'hsv_s': 0.7,         # HSV-Saturation augmentation
        'hsv_v': 0.4,         # HSV-Value augmentation
        'degrees': 0.0,       # Rotation (CCTV için 0)
        'translate': 0.1,     # Translation
        'scale': 0.5,         # Scaling
        'shear': 0.0,         # Shear (CCTV için 0)
        'perspective': 0.0,   # Perspective (CCTV için 0)
        'flipud': 0.0,        # Vertical flip (CCTV için 0)
        'fliplr': 0.5,        # Horizontal flip
        'mosaic': 1.0,        # Mosaic augmentation
        'mixup': 0.0,         # Mixup augmentation
        'copy_paste': 0.0,    # Copy-paste augmentation
        'close_mosaic': 10,   # Mosaic kapatma epoch
    }
    
    # Batch size'ı CPU için optimize et
    if device == 'cpu':
        training_args['batch'] = 8
        training_args['workers'] = 2
        print("⚠️ CPU ile eğitim yapılıyor, batch size 8'e düşürüldü")
    
    print("\n📊 Eğitim Parametreleri:")
    print(f"   - Epoch: {training_args['epochs']}")
    print(f"   - Batch size: {training_args['batch']}")
    print(f"   - Görüntü boyutu: {training_args['imgsz']}")
    print(f"   - Optimizer: {training_args['optimizer']}")
    print(f"   - Öğrenme oranı: {training_args['lr0']}")
    print(f"   - Worker sayısı: {training_args['workers']}")
    
    try:
        # Modeli eğit
        print("\n🏋️  Eğitim başlıyor...")
        results = model.train(**training_args)
        
        print("\n" + "="*60)
        print("✅ EĞİTİM TAMAMLANDI!")
        print("="*60)
        print(f"📁 En iyi model: {results.save_dir}/weights/best.pt")
        print(f"📁 Son model: {results.save_dir}/weights/last.pt")
        
        # Model performansını değerlendir
        print("\n📊 Model değerlendiriliyor...")
        metrics = model.val(data=data_path)
        
        print("\n" + "="*60)
        print("📈 DEĞERLENDİRME SONUÇLARI")
        print("="*60)
        print(f"   mAP50: {metrics.box.map50:.4f}")
        print(f"   mAP50-95: {metrics.box.map:.4f}")
        
        # Sınıf bazlı metrikler
        if hasattr(metrics.box, 'ap_class_index') and hasattr(metrics.box, 'ap50'):
            print("\n   Sınıf Bazlı Performans (mAP50):")
            for i, class_idx in enumerate(metrics.box.ap_class_index):
                if class_idx < len(class_names):
                    class_name = class_names[class_idx]
                    ap50 = metrics.box.ap50[i]
                    print(f"     {class_name}: {ap50:.4f}")
        
        # Modeli export et
        print("\n💾 Model export ediliyor...")
        try:
            model.export(format='onnx', imgsz=640)
            print(f"   ONNX model: {results.save_dir}/weights/best.onnx")
        except Exception as e:
            print(f"   ONNX export hatası: {e}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Eğitim sırasında hata oluştu: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_dataset_structure():
    """Örnek veri seti yapısını göster"""
    print("\n" + "="*60)
    print("📁 ÖRNEK VERİ SETİ YAPISI")
    print("="*60)
    print("""
    dataset/
    ├── images/
    │   ├── train/
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   └── ...
    │   └── val/
    │       ├── image1.jpg
    │       └── ...
    └── labels/
        ├── train/
        │   ├── image1.txt
        │   ├── image2.txt
        │   └── ...
        └── val/
            ├── image1.txt
            └── ...
    
    data.yaml:
        path: ./dataset
        train: images/train
        val: images/val
        nc: 5
        names: 
          - articulated_truck
          - bus
          - long_vehicle
          - motorcycle
          - short_vehicle
    """)
    
    print("\n📝 Label formatı (YOLO format):")
    print("   class_id x_center y_center width height")
    print("   Örnek: 0 0.5 0.5 0.3 0.4")
    print("   (x_center, y_center, width, height normalize edilmiş değerler 0-1 arası)")

def test_model(model_path='runs/train/cctv_vehicle_detection_yolov8/weights/best.pt'):
    """Eğitilmiş modeli test et"""
    try:
        print("\n🔍 Model test ediliyor...")
        if os.path.exists(model_path):
            model = YOLO(model_path)
            print(f"✅ Model başarıyla yüklendi: {model_path}")
            print(f"   Model sınıfları: {model.names}")
            
            # Model özeti
            print(f"   Model tipi: {model.task}")
            return True
        else:
            print(f"❌ Model dosyası bulunamadı: {model_path}")
            return False
    except Exception as e:
        print(f"❌ Model test edilirken hata: {e}")
        return False

if __name__ == "__main__":
    print("🔍 YOLOv8 Eğitim Sistemi - CCTV Araç Tespiti")
    print("="*60)
    
    # Ana eğitim fonksiyonunu çalıştır
    results = main()
    
    if results:
        print("\n✅ Eğitim başarıyla tamamlandı!")
        # Modeli test et
        test_model()
    else:
        print("\n❌ Eğitim başarısız oldu!")
        create_dataset_structure()
        print("\n💡 Lütfen veri setinizi yukarıdaki yapıya göre düzenleyin.")
        print("   Data.yaml dosyasını kontrol edin ve doğru konumda olduğundan emin olun.")