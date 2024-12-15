
# from ultralytics import YOLO
# import cv2
# import time

# # YOLO 모델 로드
# model = YOLO("/home/yunseong/CrowdHuman/coco_tune4/weights/best.pt")

# # 동영상 경로
# video_path = r"/home/yunseong/Downloads/KakaoTalk_20241212_033822418.mp4"
# output_path = r"/home/yunseong/CrowdHuman/tracked_count_v2.mp4"

# # 비디오 캡처 및 출력 설정
# cap = cv2.VideoCapture(video_path)
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fourcc = cv2.VideoWriter_fourcc(*"mp4v")
# out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# # 추적 상태 저장 변수
# active_ids = {}  # 현재 활성화된 ID와 마지막 활성화 시간 저장
# inactive_time_limit = 2.5  # 초 단위로 비활성화 유지 시간

# # 이동 평균 계산용 변수
# frame_count = 0
# count_history = []

# # 객체 추적 및 카운팅
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # YOLO 추적 수행
#     results = model.track(source=frame, persist=True)

#     # 현재 프레임에서 감지된 ID들
#     current_ids = set()

#     # 결과 처리
#     if hasattr(results[0], 'boxes') and results[0].boxes is not None:
#         for box in results[0].boxes:
#             cls = int(box.cls[0])  # 클래스 ID
#             track_id = int(box.id[0]) if box.id is not None else -1  # 추적 ID

#             # "person" 클래스만 카운팅
#             if cls == 0 and track_id != -1:
#                 current_ids.add(track_id)
#                 active_ids[track_id] = time.time()  # 활성화 시간 갱신

#                 # 바운딩 박스 표시
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     # 활성화 ID 유지 시간 확인
#     current_time = time.time()
#     for track_id in list(active_ids.keys()):
#         if track_id not in current_ids and current_time - active_ids[track_id] > inactive_time_limit:
#             del active_ids[track_id]  # 유지 시간이 초과된 ID 제거

#     # 현재 활성 ID 수 (사람 수)
#     current_count = len(active_ids)
#     count_history.append(current_count)

#     # 이동 평균 계산
#     if len(count_history) > 30:  # 최근 10프레임 기준
#         count_history.pop(0)
#     smoothed_count = int(sum(count_history) / len(count_history))

#     # 사람 수 표시
#     cv2.putText(frame, f"People Count: {smoothed_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

#     # 비디오 저장 및 화면 출력
#     out.write(frame)
#     cv2.imshow("Tracked Video", frame)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# # 리소스 정리
# cap.release()
# out.release()
# cv2.destroyAllWindows()






# 실전 코드
# from ultralytics import YOLO
# import cv2
# import time
# import numpy as np

# # YOLO 모델 로드
# model = YOLO("/home/yunseong/CrowdHuman/coco_tune4/weights/best.pt")

# # 동영상 경로
# video_path = r"/home/yunseong/Downloads/KakaoTalk_20241212_033822418.mp4"
# output_path = r"/home/yunseong/CrowdHuman/tracked_count_v1.mp4"

# # 비디오 캡처 및 출력 설정
# cap = cv2.VideoCapture(video_path)
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fourcc = cv2.VideoWriter_fourcc(*"mp4v")
# out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# # 관심 영역(ROI) 설정 (다각형 좌표)
# roi_polygon = np.array([
#    [700,500] ,[1000, 500], [1000, 570], [700, 620]
# ], np.int32)
# # 1 [220, 550]
# # 2 [540, 530]
# # 6 , [220, 660]
# # 추적 상태 저장 변수
# active_ids = {}  # 현재 활성화된 ID와 마지막 활성화 시간 저장
# inactive_time_limit = 2.5  # 초 단위로 비활성화 유지 시간

# # 이동 평균 계산용 변수
# frame_count = 0
# count_history = []

# # 객체 추적 및 카운팅
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # YOLO 추적 수행
#     results = model.track(source=frame, persist=True)

#     # 관심 영역 시각화
#     cv2.polylines(frame, [roi_polygon], isClosed=True, color=(255, 0, 0), thickness=2)

#     # 현재 프레임에서 감지된 ID들
#     current_ids = set()

#     # 결과 처리
#     if hasattr(results[0], 'boxes') and results[0].boxes is not None:
#         for box in results[0].boxes:
#             cls = int(box.cls[0])  # 클래스 ID
#             track_id = int(box.id[0]) if box.id is not None else -1  # 추적 ID

#             # "person" 클래스만 카운팅
#             if cls == 0 and track_id != -1:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 center_x = (x1 + x2) // 2
#                 center_y = (y1 + y2) // 2

#                 # 관심 영역 내부에 있는지 확인
#                 if cv2.pointPolygonTest(roi_polygon, (center_x, center_y), False) >= 0:
#                     current_ids.add(track_id)
#                     active_ids[track_id] = time.time()  # 활성화 시간 갱신

#                     # 바운딩 박스 표시
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                     cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     # 활성화 ID 유지 시간 확인
#     current_time = time.time()
#     for track_id in list(active_ids.keys()):
#         if track_id not in current_ids and current_time - active_ids[track_id] > inactive_time_limit:
#             del active_ids[track_id]  # 유지 시간이 초과된 ID 제거

#     # 현재 활성 ID 수 (사람 수)
#     current_count = len(active_ids)
#     count_history.append(current_count)

#     # 이동 평균 계산
#     if len(count_history) > 10:  # 최근 30프레임 기준
#         count_history.pop(0)
#     smoothed_count = int(sum(count_history) / len(count_history))

#     # 사람 수 표시
#     cv2.putText(frame, f"People Count: {smoothed_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

#     # 비디오 저장 및 화면 출력
#     out.write(frame)
#     cv2.imshow("Tracked Video", frame)

#     # 's' 키를 눌러 멈춤 기능 추가
#     key = cv2.waitKey(1)
#     if key & 0xFF == ord("q"):
#         break
#     elif key & 0xFF == ord("s"):
#         cv2.waitKey(-1)  # 아무 키나 누를 때까지 멈춤

# # 리소스 정리
# cap.release()
# out.release()
# cv2.destroyAllWindows()

# print(f"Processed video saved at: {output_path}")


from ultralytics import YOLO
import cv2
import time
import numpy as np

# YOLO 모델 로드
model = YOLO("/home/yunseong/CrowdHuman/coco_tune4/weights/best.pt")

# 동영상 경로
video_path = r"/home/yunseong/Downloads/KakaoTalk_20241212_033822418.mp4"
output_path = r"/home/yunseong/CrowdHuman/tracked_count_v1.mp4"

# 비디오 캡처 및 출력 설정
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# 관심 영역(ROI) 설정 (다각형 좌표)
roi_polygon = np.array([
   [700,500] ,[1000, 500], [1000, 570], [700, 620]
], np.int32)

# 추적 상태 저장 변수
active_ids = {}  # 현재 활성화된 ID와 마지막 활성화 시간 저장
inactive_time_limit = 2.5  # 초 단위로 비활성화 유지 시간

# 이동 평균 계산용 변수
frame_count = 0
count_history = []

# 15초 평균 계산용 변수
average_count = 0
average_start_time = time.time()
average_count_history = []

# 객체 추적 및 카운팅
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO 추적 수행
    results = model.track(source=frame, persist=True)

    # 관심 영역 시각화
    cv2.polylines(frame, [roi_polygon], isClosed=True, color=(255, 0, 0), thickness=2)

    # 현재 프레임에서 감지된 ID들
    current_ids = set()

    # 결과 처리
    if hasattr(results[0], 'boxes') and results[0].boxes is not None:
        for box in results[0].boxes:
            cls = int(box.cls[0])  # 클래스 ID
            track_id = int(box.id[0]) if box.id is not None else -1  # 추적 ID

            # "person" 클래스만 카운팅
            if cls == 0 and track_id != -1:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # 관심 영역 내부에 있는지 확인
                if cv2.pointPolygonTest(roi_polygon, (center_x, center_y), False) >= 0:
                    current_ids.add(track_id)
                    active_ids[track_id] = time.time()  # 활성화 시간 갱신

                    # 바운딩 박스 표시
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 활성화 ID 유지 시간 확인
    current_time = time.time()
    for track_id in list(active_ids.keys()):
        if track_id not in current_ids and current_time - active_ids[track_id] > inactive_time_limit:
            del active_ids[track_id]  # 유지 시간이 초과된 ID 제거

    # 현재 활성 ID 수 (사람 수)
    current_count = len(active_ids)
    count_history.append(current_count)

    # 이동 평균 계산
    if len(count_history) > 10:  # 최근 30프레임 기준
        count_history.pop(0)
    smoothed_count = int(sum(count_history) / len(count_history))

    # 15초 평균 계산
    elapsed_time = current_time - average_start_time
    if elapsed_time >= 15:
        average_count = int(sum(count_history) / len(count_history)) if count_history else 0
        average_start_time = current_time  # 타이머 초기화
        average_count_history.append(average_count)

    # 사람 수 표시
    cv2.putText(frame, f"People Count: {smoothed_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(frame, f"15s Avg Count: {average_count}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # 비디오 저장 및 화면 출력
    out.write(frame)
    cv2.imshow("Tracked Video", frame)

    # 's' 키를 눌러 멈춤 기능 추가
    key = cv2.waitKey(1)
    if key & 0xFF == ord("q"):
        break
    elif key & 0xFF == ord("s"):
        cv2.waitKey(-1)  # 아무 키나 누를 때까지 멈춤

# 리소스 정리
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed video saved at: {output_path}")




# from ultralytics import YOLO
# import cv2
# import time

# # YOLOv11 모델 로드
# model = YOLO("/home/yunseong/CrowdHuman/coco_tune4/weights/best.pt")

# # # 동영상 경로
# video_path = r"/home/yunseong/P_project_yolo11/IMG_1219.mp4"
# output_path = r"/home/yunseong/CrowdHuman/tracked_count_v3.mp4"

# # 비디오 캡처 및 출력 설정
# cap = cv2.VideoCapture(video_path)
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fourcc = cv2.VideoWriter_fourcc(*"mp4v")
# out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# # 추적 상태 저장 변수
# active_ids = {}  # 현재 활성화된 ID와 마지막 활성화 시간 저장
# queue_ids = {}  # 줄 서 있는 사람 추적용
# inactive_time_limit = 1.0  # 초 단위로 비활성화 유지 시간
# queue_threshold_time = 2.0  # 줄 서 있는 사람 판단 시간 (초)
# movement_threshold = 50  # 픽셀 단위로 이동 허용 거리

# # 이동 평균 계산용 변수
# count_history = []
# queue_count_history = []

# # 객체 추적 및 카운팅
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # YOLO 추적 수행
#     results = model.track(source=frame, persist=True)

#     # 현재 프레임에서 감지된 ID들
#     current_ids = set()

#     # 결과 처리
#     if hasattr(results[0], 'boxes') and results[0].boxes is not None:
#         for box in results[0].boxes:
#             cls = int(box.cls[0])  # 클래스 ID
#             track_id = int(box.id[0]) if box.id is not None else -1  # 추적 ID

#             # "person" 클래스만 카운팅
#             if cls == 0 and track_id != -1:
#                 current_ids.add(track_id)
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 center_x = (x1 + x2) // 2
#                 center_y = (y1 + y2) // 2

#                 # 활성화 상태 갱신
#                 if track_id not in active_ids:
#                     active_ids[track_id] = {"last_position": (center_x, center_y), "last_update": time.time()}
#                 else:
#                     # 이동 거리 계산
#                     last_position = active_ids[track_id]["last_position"]
#                     distance = ((center_x - last_position[0]) ** 2 + (center_y - last_position[1]) ** 2) ** 0.5

#                     # 이동이 없으면 정지 상태 유지 시간 업데이트
#                     if distance < movement_threshold:
#                         time_since_last_update = time.time() - active_ids[track_id]["last_update"]
#                         if time_since_last_update >= queue_threshold_time and track_id not in queue_ids:
#                             # 줄 서는 상태로 전환
#                             queue_ids[track_id] = time.time()
#                     else:
#                         # 이동 상태로 간주
#                         active_ids[track_id]["last_position"] = (center_x, center_y)
#                         active_ids[track_id]["last_update"] = time.time()
#                         if track_id in queue_ids:
#                             del queue_ids[track_id]  # 이동 시작 시 줄 서 있는 상태에서 제외

#                 # 바운딩 박스와 레이블 표시
#                 label = f"ID: {track_id} {'Queue' if track_id in queue_ids else 'Moving'}"
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0) if track_id in queue_ids else (0, 0, 255), 2)
#                 cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     # 활성화 ID 유지 시간 확인
#     current_time = time.time()
#     for track_id in list(active_ids.keys()):
#         if track_id not in current_ids and current_time - active_ids[track_id]["last_update"] > inactive_time_limit:
#             del active_ids[track_id]  # 유지 시간이 초과된 ID 제거

#     # 줄 서 있는 사람 수
#     queue_count = len(queue_ids)
#     queue_count_history.append(queue_count)

#     # 이동 평균 계산
#     if len(queue_count_history) > 10:  # 최근 10프레임 기준
#         queue_count_history.pop(0)
#     smoothed_queue_count = int(sum(queue_count_history) / len(queue_count_history))

#     # 줄 서 있는 사람 수 표시
#     cv2.putText(frame, f"Queue Count: {smoothed_queue_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

#     # 비디오 저장 및 화면 출력
#     out.write(frame)
#     cv2.imshow("Tracked Video", frame)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# # 리소스 정리
# cap.release()
# out.release()
# cv2.destroyAllWindows()

# print(f"Processed video saved at: {output_path}")

# from ultralytics import YOLO
# import cv2
# import time

# # YOLOv11 모델 로드
# model = YOLO("/home/yunseong/CrowdHuman/coco_tune4/weights/best.pt")

# # 동영상 경로
# video_path = r"/home/yunseong/Downloads/KakaoTalk_20241212_033822418.mp4"
# output_path = r"/home/yunseong/CrowdHuman/tracked_count_v3.mp4"

# # 비디오 캡처 및 출력 설정
# cap = cv2.VideoCapture(video_path)
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fourcc = cv2.VideoWriter_fourcc(*"mp4v")
# out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# # 추적 상태 저장 변수
# active_ids = {}  # 현재 활성화된 ID와 마지막 활성화 상태 저장
# queue_ids = {}  # 줄 서 있는 사람 추적용
# inactive_time_limit = 1.0  # 초 단위로 비활성화 유지 시간
# queue_threshold_time = 2.0  # 줄 서 있는 사람 판단 시간 (초)
# movement_threshold = 120  # 픽셀 단위로 이동 허용 거리

# # 이동 평균 계산용 변수
# queue_count_history = []

# # 객체 추적 및 카운팅
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # YOLO 추적 수행
#     results = model.track(source=frame, persist=True)

#     # 현재 프레임에서 감지된 ID들
#     current_ids = set()

#     # 결과 처리
#     if hasattr(results[0], 'boxes') and results[0].boxes is not None:
#         for box in results[0].boxes:
#             cls = int(box.cls[0])  # 클래스 ID
#             track_id = int(box.id[0]) if box.id is not None else -1  # 추적 ID

#             # "person" 클래스만 카운팅
#             if cls == 0 and track_id != -1:
#                 current_ids.add(track_id)
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 center_x = (x1 + x2) // 2
#                 center_y = (y1 + y2) // 2

#                 # 활성화 상태 갱신
#                 if track_id not in active_ids:
#                     active_ids[track_id] = {"last_position": (center_x, center_y), "last_update": time.time()}
#                 else:
#                     # 이동 거리 계산
#                     last_position = active_ids[track_id]["last_position"]
#                     distance = ((center_x - last_position[0]) ** 2 + (center_y - last_position[1]) ** 2) ** 0.5

#                     # 이동이 없으면 정지 상태 유지 시간 업데이트
#                     if distance < movement_threshold:
#                         time_since_last_update = time.time() - active_ids[track_id]["last_update"]
#                         if time_since_last_update >= queue_threshold_time and track_id not in queue_ids:
#                             # 줄 서는 상태로 전환
#                             queue_ids[track_id] = time.time()
#                     else:
#                         # 이동 상태로 간주
#                         active_ids[track_id]["last_position"] = (center_x, center_y)
#                         active_ids[track_id]["last_update"] = time.time()
#                         if track_id in queue_ids:
#                             del queue_ids[track_id]  # 이동 시작 시 줄 서 있는 상태에서 제외

#                 # 바운딩 박스와 레이블 표시
#                 label = f"ID: {track_id} {'Queue' if track_id in queue_ids else 'Moving'}"
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0) if track_id in queue_ids else (0, 0, 255), 2)
#                 cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     # 활성화 ID 유지 시간 확인
#     current_time = time.time()
#     for track_id in list(active_ids.keys()):
#         if track_id not in current_ids and current_time - active_ids[track_id]["last_update"] > inactive_time_limit:
#             del active_ids[track_id]  # 유지 시간이 초과된 ID 제거
#             if track_id in queue_ids:
#                 del queue_ids[track_id]  # 비활성화된 경우 줄 서 있는 상태에서도 제거

#     # 줄 서 있는 사람 수
#     queue_count = len(queue_ids)
#     queue_count_history.append(queue_count)

#     # 이동 평균 계산
#     if len(queue_count_history) > 10:  # 최근 10프레임 기준
#         queue_count_history.pop(0)
#     smoothed_queue_count = int(sum(queue_count_history) / len(queue_count_history))

#     # 줄 서 있는 사람 수 표시
#     cv2.putText(frame, f"Queue Count: {smoothed_queue_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

#     # 비디오 저장 및 화면 출력
#     out.write(frame)
#     cv2.imshow("Tracked Video", frame)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# # 리소스 정리
# cap.release()
# out.release()
# cv2.destroyAllWindows()

# print(f"Processed video saved at: {output_path}")

# from ultralytics import YOLO
# import cv2
# import time

# # IOU 계산 함수
# def calculate_iou(box1, box2):
#     x1, y1, x2, y2 = box1
#     x3, y3, x4, y4 = box2
#     xi1 = max(x1, x3)
#     yi1 = max(y1, y3)
#     xi2 = min(x2, x4)
#     yi2 = min(y2, y4)
#     inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
#     box1_area = (x2 - x1) * (y2 - y1)
#     box2_area = (x4 - x3) * (y4 - y3)
#     union_area = box1_area + box2_area - inter_area
#     return inter_area / union_area if union_area > 0 else 0

# # YOLOv11 모델 로드
# model = YOLO("/home/yunseong/CrowdHuman/coco_tune4/weights/best.pt")

# # 동영상 경로
# video_path = r"/home/yunseong/P_project_yolo11/IMG_1219.mp4"
# output_path = r"/home/yunseong/CrowdHuman/tracked_count_v3.mp4"

# # 비디오 캡처 및 출력 설정
# cap = cv2.VideoCapture(video_path)
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fourcc = cv2.VideoWriter_fourcc(*"mp4v")
# out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# # 추적 상태 저장 변수
# active_ids = {}  # 현재 활성화된 ID와 마지막 활성화 상태 저장
# queue_ids = {}  # 줄 서 있는 사람 추적용
# inactive_time_limit = 1.0  # 초 단위로 비활성화 유지 시간
# queue_threshold_time = 2.0  # 줄 서 있는 사람 판단 시간 (초)
# movement_threshold = 200  # 픽셀 단위로 이동 허용 거리
# iou_threshold = 0.7  # IOU 기준값

# # 이동 평균 계산용 변수
# queue_count_history = []

# # 객체 추적 및 카운팅
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # YOLO 추적 수행
#     results = model.track(source=frame, persist=True)

#     # 현재 프레임에서 감지된 ID들
#     current_ids = set()
#     detected_boxes = []  # 현재 프레임의 바운딩 박스

#     # 결과 처리
#     if hasattr(results[0], 'boxes') and results[0].boxes is not None:
#         for box in results[0].boxes:
#             cls = int(box.cls[0])  # 클래스 ID
#             track_id = int(box.id[0]) if box.id is not None else -1  # 추적 ID
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             detected_boxes.append((track_id, (x1, y1, x2, y2)))

#             # "person" 클래스만 카운팅
#             if cls == 0 and track_id != -1:
#                 current_ids.add(track_id)
#                 center_x = (x1 + x2) // 2
#                 center_y = (y1 + y2) // 2

#                 # 활성화 상태 갱신
#                 if track_id not in active_ids:
#                     active_ids[track_id] = {"last_position": (x1, y1, x2, y2), "last_update": time.time()}
#                 else:
#                     # 이동 거리 계산
#                     last_box = active_ids[track_id]["last_position"]
#                     distance = ((center_x - (last_box[0] + last_box[2]) // 2) ** 2 +
#                                 (center_y - (last_box[1] + last_box[3]) // 2) ** 2) ** 0.5

#                     # 이동이 없으면 정지 상태 유지 시간 업데이트
#                     if distance < movement_threshold:
#                         time_since_last_update = time.time() - active_ids[track_id]["last_update"]
#                         if time_since_last_update >= queue_threshold_time and track_id not in queue_ids:
#                             # 줄 서는 상태로 전환
#                             queue_ids[track_id] = time.time()
#                     else:
#                         # 이동 상태로 간주
#                         active_ids[track_id]["last_position"] = (x1, y1, x2, y2)
#                         active_ids[track_id]["last_update"] = time.time()
#                         if track_id in queue_ids:
#                             del queue_ids[track_id]  # 이동 시작 시 줄 서 있는 상태에서 제외

#                 # 바운딩 박스와 레이블 표시
#                 label = f"ID: {track_id} {'Queue' if track_id in queue_ids else 'Moving'}"
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0) if track_id in queue_ids else (0, 0, 255), 2)
#                 cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     # IOU 기반 매칭
#     for track_id, prev_box in list(active_ids.items()):
#         matched = False
#         for det_id, current_box in detected_boxes:
#             if calculate_iou(prev_box["last_position"], current_box) > iou_threshold:
#                 matched = True
#                 active_ids[track_id]["last_position"] = current_box
#                 active_ids[track_id]["last_update"] = time.time()
#                 break
#         if not matched and time.time() - active_ids[track_id]["last_update"] > inactive_time_limit:
#             del active_ids[track_id]
#             if track_id in queue_ids:
#                 del queue_ids[track_id]

#     # 줄 서 있는 사람 수
#     queue_count = len(queue_ids)
#     queue_count_history.append(queue_count)

#     # 이동 평균 계산
#     if len(queue_count_history) > 30:  # 최근 10프레임 기준
#         queue_count_history.pop(0)
#     smoothed_queue_count = int(sum(queue_count_history) / len(queue_count_history))

#     # 줄 서 있는 사람 수 표시
#     cv2.putText(frame, f"Queue Count: {smoothed_queue_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

#     # 비디오 저장 및 화면 출력
#     out.write(frame)
#     cv2.imshow("Tracked Video", frame)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# # 리소스 정리
# cap.release()
# out.release()
# cv2.destroyAllWindows()

# print(f"Processed video saved at: {output_path}")
