import cv2


def save_webcam(outPath, fps, mirror=False):
    # Capturing video from webcam:
    cap = cv2.VideoCapture(0)
    currentFrame = 0
    # Get current width of frame
    first_frame = None
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # float
    # Get current height of frame
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # float# Define the codec and create Video Writer object
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(outPath, fourcc, fps, (int(width), int(height)))
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            if mirror == True:
                # Mirror the output video frame
                frame = cv2.flip(frame, 1)
            # Saves for video
            out.write(frame)
            frame_blur = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_blur = cv2.GaussianBlur(frame_blur, (21, 21), 0)
            if first_frame is None:
                first_frame = frame_blur
            diff = cv2.absdiff(first_frame, frame_blur)
            ret, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            thresh = cv2.dilate(thresh, None, iterations=2)
            cv2.imshow("thresh", thresh)
            (_, contours, _) = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                if cv2.contourArea(c) < 5000:
                    continue
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                text = "Occupied"
                cv2.imshow("frame", frame)
                # on indique que la pièce est occupée
                cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        else:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # if 'q' ispressed then quit
            break
        # To stop duplicate images
        currentFrame += 1
    # When everything done, release the capture
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def main():
    save_webcam('output.avi', 30.0, mirror=True)


if __name__ == '__main__':
    main()
# image = cv2.imread(image_path)
