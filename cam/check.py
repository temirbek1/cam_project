import cv2
import threading

# URL видеопотоков камер (RTSP или HTTP)
camera_urls = [
    'rtsp://test:Realmonitor@192.168.10.63:554/Streaming/Channels/101?transportmode=unicast&profile=Profile_1',
    'rtsp://test:Realmonitor@192.168.10.61:554/cam/realmonitor?channel=1&subtype=0'
]

def display_camera_feed(camera_url):
    cap = cv2.VideoCapture(camera_url)
    
    if not cap.isOpened():
        print(f"Ошибка при подключении к камере: {camera_url}")
        return

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"Не удалось получить кадр из {camera_url}.")
            break

        # Отображение текущего кадра
        cv2.imshow(f"Camera Feed - {camera_url}", frame)

        # Ограничение частоты кадров и проверка на выход ('q')
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Создание потоков для каждой камеры
threads = []
for url in camera_urls:
    thread = threading.Thread(target=display_camera_feed, args=(url,))
    thread.start()
    threads.append(thread)

# Дождаться завершения всех потоков
for thread in threads:
    thread.join()
