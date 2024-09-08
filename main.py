import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np


depth_model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
depth_model.to(device)
depth_model.eval()


depth_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
depth_transforms = depth_transforms.small_transform


video_file = "C:\\Users\\somra\\Downloads\\sample2.mp4"
video_capture = cv2.VideoCapture(video_file)

fps = video_capture.get(cv2.CAP_PROP_FPS)
frame_delay = int(5000 / fps)


color_map = plt.get_cmap('inferno')

while video_capture.isOpened():
    ret, frame = video_capture.read()

    if not ret:
        break


    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = depth_transforms(rgb_image).to('cpu')

    with torch.no_grad():
        depth_prediction = depth_model(input_batch)
        depth_prediction = torch.nn.functional.interpolate(
            depth_prediction.unsqueeze(1),
            size=rgb_image.shape[:2],
            mode='bicubic',
            align_corners=False
        ).squeeze()

        depth_output = depth_prediction.cpu().numpy()


        min_depth = np.min(depth_output)
        max_depth = np.max(depth_output)
        normalized_depth = (depth_output - min_depth) / (max_depth - min_depth)


        colored_depth = color_map(normalized_depth)
        colored_depth = (colored_depth[:, :, :3] * 255).astype(np.uint8)


    plt.imshow(colored_depth)
    plt.pause(0.0001)

    cv2.imshow('Original Video Frame', frame)

    if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
plt.show()
