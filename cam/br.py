import cv2
import time
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

    start_time = time.time()
    total_bytes = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Не удалось получить кадр из {camera_url}.")
            break
        
        # Рассчитать количество байт для текущего кадра
        total_bytes += frame.nbytes

        # Отображение текущего кадра
        cv2.imshow(f"Camera Feed - {camera_url}", frame)

        # Каждые 5 секунд считать битрейт
        if time.time() - start_time >= 5:
            bitrate = (total_bytes * 8) / (5 * 1024 * 1024)  # Битрейт в Мбит/с
            print(f"Битрейт {camera_url}: {bitrate:.2f} Мбит/с")
            total_bytes = 0
            start_time = time.time()

        # Нажмите 'q' для выхода
        if cv2.waitKey(1) & 0xFF == ord('q'):
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
