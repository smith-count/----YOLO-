from ultralytics import YOLO

if __name__ == '__main__':

    # Load a model
    model = YOLO(model=r'D:\Python\graduate_design\ultralytics-8.3.2\yolo11n-seg.pt')
    model.predict(source=r'D:\Python\graduate_design\ultralytics-8.3.2\football.jpg',
                  save=True,
                  show=False,
                  imgsz=416,
                  )