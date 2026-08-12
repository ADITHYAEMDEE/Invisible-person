import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# Setup MediaPipe Tasks API modules
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions


def is_hand_open(detection_result):
    """
    Checks if fingers are extended relative to knuckle joints.
    """
    if not detection_result.hand_landmarks:
        return False

    hand_landmarks = detection_result.hand_landmarks[0]
    fingers_extended = 0

    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]

    for tip, pip in zip(tips, pips):
        if hand_landmarks[tip].y < hand_landmarks[pip].y:
            fingers_extended += 1

    return fingers_extended >= 3


def get_person_bbox(mask, frame_shape, threshold=0.5, padding=25):
    """
    Turns a per-pixel confidence mask into a single padded bounding box
    around the person, so we can cut out a rectangle instead of blending
    a soft alpha mask over the whole frame.
    """
    # Some MediaPipe builds return the mask with an extra trailing
    # channel dimension, e.g. (H, W, 1) instead of (H, W) - squeeze it
    # down so np.where gives us exactly (row_indices, col_indices)
    mask = np.squeeze(mask)

    ys, xs = np.where(mask > threshold)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())

    h, w = frame_shape[:2]
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    return (x1, y1, x2, y2)


def smooth_bbox(bbox_history):
    """
    Unions the last few frames' bounding boxes together. If the hand moves
    fast enough that one frame's detection lags behind, the box from a
    frame or two ago still covers that area, so the cutout keeps up
    instead of clipping the hand.
    """
    if not bbox_history:
        return None

    x1 = min(b[0] for b in bbox_history)
    y1 = min(b[1] for b in bbox_history)
    x2 = max(b[2] for b in bbox_history)
    y2 = max(b[3] for b in bbox_history)

    return (x1, y1, x2, y2)


def composite_bbox_region(frame, background, bbox, feather=15):
    """
    Pastes the background over the bbox region, but feathers just the
    edges of that rectangle (a thin soft-blend border) instead of a hard
    cut. This doesn't turn it into a full-body mask blend - it's still
    a box - it just softens the seam so small lighting/exposure
    differences between the live frame and the captured background
    aren't as visible as a sharp line.
    """
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]

    fx1 = max(0, x1 - feather)
    fy1 = max(0, y1 - feather)
    fx2 = min(w, x2 + feather)
    fy2 = min(h, y2 + feather)

    region_mask = np.zeros((fy2 - fy1, fx2 - fx1), dtype=np.float32)
    rx1, ry1 = x1 - fx1, y1 - fy1
    rx2, ry2 = x2 - fx1, y2 - fy1
    region_mask[ry1:ry2, rx1:rx2] = 1.0
    region_mask = cv2.GaussianBlur(region_mask, (feather * 2 + 1, feather * 2 + 1), 0)
    region_mask_3d = np.dstack([region_mask] * 3)

    frame_region = frame[fy1:fy2, fx1:fx2].astype(np.float32)
    bg_region = background[fy1:fy2, fx1:fx2].astype(np.float32)
    blended = frame_region * (1 - region_mask_3d) + bg_region * region_mask_3d

    output = frame.copy()
    output[fy1:fy2, fx1:fx2] = blended.astype(np.uint8)
    return output


def main():
    cap = cv2.VideoCapture(0)
    background = None
    state_buffer = deque(maxlen=5)

    # Debounced "is the hand currently open" state
    current_state_open = False
    # The previous frame's debounced state, used to catch the moment
    # the gesture transitions closed -> open (the toggle trigger)
    previous_state_open = False

    # This is the actual visibility state of the person, and it only
    # flips on a closed->open transition. It does NOT track the hand
    # continuously, which is what makes it a toggle instead of hold.
    is_invisible = False

    # Rolling window of recent bboxes, unioned together in smooth_bbox()
    # to give the cutout a bit of a buffer against fast hand motion
    bbox_history = deque(maxlen=5)

    hand_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1
    )

    seg_options = ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path="selfie_segmenter.tflite"),
        running_mode=VisionRunningMode.IMAGE,
        output_confidence_masks=True
    )

    print("Initializing models...")

    with HandLandmarker.create_from_options(hand_options) as landmarker, \
         ImageSegmenter.create_from_options(seg_options) as segmenter:

        print("Camera ready!")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)

            # Capture background state
            if background is None:
                # Draw instructions on a separate display copy - frame
                # itself must stay clean, since that's what gets saved
                # as the background on 'b' (previously the red text was
                # getting burned permanently into the captured background)
                display = frame.copy()
                cv2.putText(display, "Step out of frame and press 'b' to capture background",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imshow("Magic Vanish", display)

                key = cv2.waitKey(1)
                if key & 0xFF == ord('b'):
                    background = frame.copy()
                    print("Background captured! Step back into frame.")
                elif key & 0xFF == ord('q'):
                    break
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            hand_result = landmarker.detect(mp_image)

            # Raw open/closed reading for this frame
            raw_hand_open = is_hand_open(hand_result) if hand_result.hand_landmarks else current_state_open

            # Debounce to avoid flicker on borderline frames
            state_buffer.append(raw_hand_open)
            current_state_open = sum(state_buffer) > len(state_buffer) // 2

            # Toggle only on the rising edge: closed -> open
            if current_state_open and not previous_state_open:
                is_invisible = not is_invisible

            previous_state_open = current_state_open

            if is_invisible:
                # Only run segmentation when we actually need the cutout,
                # saves a bit of compute while visible
                seg_result = segmenter.segment(mp_image)
                confidence_masks = seg_result.confidence_masks

                output = frame.copy()

                if confidence_masks:
                    mask = confidence_masks[0].numpy_view()
                    bbox = get_person_bbox(mask, frame.shape)

                    if bbox:
                        bbox_history.append(bbox)

                smoothed = smooth_bbox(bbox_history)
                if smoothed:
                    output = composite_bbox_region(frame, background, smoothed)

                cv2.putText(output, "Invisible (toggle open)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                # Clear the trailing bbox history so the next time you
                # vanish, it doesn't start with stale boxes from before
                bbox_history.clear()

                # Plain, untouched webcam frame - no mask, no blending
                output = frame.copy()
                cv2.putText(output, "Visible (toggle open to vanish)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            cv2.imshow("Magic Vanish", output)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()