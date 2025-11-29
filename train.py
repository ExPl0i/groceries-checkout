from ultralytics import YOLO

def main():
    # Загружаем предобученную YOLOv11l
    model = YOLO("yolo11l.pt")

    # Запуск обучения
    results = model.train(
        data="data.yaml",         # Путь к вашему YAML
        epochs=50,               # Количество эпох
        imgsz=416,                # Размер изображения
        batch=16,                 # Размер батча
        workers=8,                # Количество потоков
        device=0,                 # GPU 0 (CPU = "cpu")
        optimizer="AdamW",        # Лучше для стабильности
        lr0=1e-3,                 # Начальная скорость обучения
        pretrained=True,          # Использовать предобученные веса
        project="YOLO11_Grocery", # Папка проекта
        name="yolo11l_train",     # Имя эксперимента
        save=True,                # Сохранение весов
        save_period=10,           # Сохранение каждые N эпох
        val=True                  # Запуск валидации
    )

    print("Training completed!")
    print(results)

if __name__ == "__main__":
    main()
