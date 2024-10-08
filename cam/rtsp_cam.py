import cv2
import threading

def display_frames(camera_url):
    cap = cv2.VideoCapture(camera_url)

    if not cap.isOpened():
        print(f"Ошибка при подключении к камере: {camera_url}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Не удалось получить кадр из {camera_url}.")
            break

        cv2.imshow(f"Camera Feed - {camera_url}", frame)

        # Добавляем проверку на клавишу 'q' для выхода
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f"Остановка видеопотока для {camera_url}")
            break

    cap.release()
    cv2.destroyAllWindows()

# Пример использования с несколькими камерами
camera_urls = [
    'rtsp://test:Realmonitor@192.168.10.63:554/Streaming/Channels/101?transportmode=unicast&profile=Profile_1',
    'rtsp://test:Realmonitor@192.168.10.61:554/cam/realmonitor?channel=1&subtype=0'
]

threads = []
for url in camera_urls:
    thread = threading.Thread(target=display_frames, args=(url,))
    thread.start()
    threads.append(thread)

for thread in threads:
    thread.join()
