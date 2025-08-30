import cv2
from deepface import DeepFace
import numpy as np
import os
import time

# مسیر پوشه تصاویر
folder_path = r"C:\Users\DELL\Desktop\images\project\New folder"

# بررسی وجود پوشه
if not os.path.exists(folder_path):
    print(f"خطا: پوشه {folder_path} وجود ندارد. لطفاً مسیر را بررسی کنید.")
    exit()

# بارگذاری مدل تشخیص چهره OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_cascade.empty():
    print("خطا: فایل Haar Cascade لود نشد. مطمئن شوید OpenCV به درستی نصب شده است.")
    exit()

# گرفتن لیست تصاویر
image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png'))]
if not image_files:
    print(f"خطا: هیچ فایل JPG یا PNG در پوشه {folder_path} پیدا نشد.")
    exit()

# پردازش هر تصویر
for idx, image_file in enumerate(image_files):
    start_time = time.time()  # زمان‌سنج برای دیباگ
    image_path = os.path.join(folder_path, image_file)
    print(f"در حال پردازش تصویر: {image_file}")

    # خواندن تصویر
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"خطا: تصویر {image_file} لود نشد. فایل را بررسی کنید.")
        continue

    # تبدیل به خاکستری برای تشخیص چهره
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        print(f"خطا در تبدیل تصویر {image_file} به خاکستری: {e}")
        continue

    # تشخیص چهره‌ها
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        print(f"هیچ چهره‌ای در تصویر {image_file} تشخیص داده نشد.")
        cv2.imshow(f'Emotion Recognition - {image_file}', frame)
        cv2.imwrite(os.path.join(folder_path, f'output_{image_file}'), frame)
        cv2.waitKey(1000)  # نمایش 1 ثانیه
        cv2.destroyAllWindows()
        continue

    # پردازش چهره‌ها با DeepFace
    for i, (x, y, w, h) in enumerate(faces):
        face = frame[y:y+h, x:x+w]
        try:
            # تشخیص احساس با DeepFace
            result = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']
            confidence = result[0]['emotion'][emotion]  # درصد اطمینان

            # رسم مستطیل و برچسب با فونت بزرگ‌تر
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            label = f"{emotion}: {confidence:.1f}%"
            cv2.putText(frame, label, (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        except Exception as e:
            print(f"خطا در تشخیص احساس برای چهره {i+1} در تصویر {image_file}: {e}")
            continue

    # نمایش و ذخیره تصویر
    cv2.namedWindow(f'Emotion Recognition - {image_file}', cv2.WINDOW_NORMAL)  # پنجره با اندازه قابل تنظیم
    cv2.resizeWindow(f'Emotion Recognition - {image_file}', 800, 600)  # اندازه پنجره
    cv2.imshow(f'Emotion Recognition - {image_file}', frame)
    cv2.imwrite(os.path.join(folder_path, f'output_{image_file}'), frame)
    
    # زمان پردازش
    print(f"زمان پردازش تصویر {image_file}: {time.time() - start_time:.2f} ثانیه")
    
    # نمایش 2 ثانیه یا تا زدن کلید
    cv2.waitKey(2000)
    cv2.destroyAllWindows()

print("پردازش همه تصاویر 완료 شد!")