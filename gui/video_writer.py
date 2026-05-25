import cv2


class VideoWriter:
    def __init__(self, filename, frame_width, frame_height, fps):
        trial = [('H264','mp4'), ('avc1','mp4'), ('mp4v','mp4'), ('XVID','avi')]
        for code, ext in trial:
            fourcc = cv2.VideoWriter_fourcc(*code)
            self.video_writer = cv2.VideoWriter(f'{filename}.{ext}', fourcc, fps, (frame_width, frame_height))
            if self.video_writer.isOpened():
                self.frame_width = frame_width
                self.frame_height = frame_height
                break
        if self.video_writer is None or not self.video_writer.isOpened():
            raise Exception('Could not open video writer with any codec!')

    def __del__(self):
        """Automatically release video writer when object is garbage collected."""
        self.end_recording()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatically releases video writer."""
        self.end_recording()
        return False

    def write_frame(self, frame):
        if self.video_writer is None:
            raise Exception('Video writer not initialized, something went wront.')

        if (self.frame_width, self.frame_height) != (frame.shape[1], frame.shape[0]):
            raise ValueError(
                f'Frame size {frame.shape[1], frame.shape[0]} does not match initialized size {(self.frame_width, self.frame_height)}.'
            )

        self.video_writer.write(frame)

    def end_recording(self):
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            self.frame_size = None
