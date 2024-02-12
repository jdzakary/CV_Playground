from app.data.general import DataManager


class StreamDataManager(DataManager):

    def __init__(self):
        self.__streaming = False
        self.__recording = False
        self.__frame_size = (640, 640)
        self.__fps = 20
        self.__file_name = 'video_out.avi'
        self.__video_width = 640
        self.__video_index = 0
        self.__device_loading = False
        mapping = {
            'toggleStream': self.set_streaming,
            'toggleRecording': self.set_recording,
            'setFrameSize': self.set_frame_size,
            'setFPS': self.set_fps,
            'setFileName': self.set_file_name,
            'setVideoWidth': self.set_video_width,
            'setVideoIndex': self.set_video_index,
            'setDeviceLoading': self.set_device_loading,
        }
        super(StreamDataManager, self).__init__(mapping)

    @property
    def streaming(self) -> bool:
        """
        Is the video stream active?
        """
        return self.__streaming

    @property
    def recording(self) -> bool:
        """
        Is the video stream currently being writen to file?
        """
        return self.__recording

    @property
    def frame_size(self) -> tuple[int, int]:
        """
        The natural size of the openCV capture device.
        :return: (width, height) tuple
        """
        return self.__frame_size

    @property
    def fps(self) -> float:
        """
        The frame rate of the openCV capture device
        """
        return self.__fps

    @property
    def file_name(self) -> str:
        """
        Name of the video file being recorded
        """
        return self.__file_name

    @property
    def video_width(self) -> int:
        """
        Width of the video stream as shown in the GUI
        :return:
        """
        return self.__video_width

    @property
    def video_index(self) -> int:
        """
        The index to create the OpenCV capture device
        :return:
        """
        return self.__video_index

    @property
    def device_loading(self) -> bool:
        """
        Is the video capture device being created?
        This is important for cameras that take a long time (>500ms)
        to initialize
        :return:
        """
        return self.__device_loading

    def set_streaming(self, value: bool) -> None:
        self.__streaming = value

    def set_recording(self, value: bool) -> None:
        self.__recording = value

    def set_frame_size(self, value: tuple[int, int]) -> None:
        self.__frame_size = value

    def set_fps(self, value: float) -> None:
        self.__fps = value

    def set_file_name(self, value: str) -> None:
        self.__file_name = value

    def set_video_width(self, value: int) -> None:
        self.__video_width = value

    def set_video_index(self, value: int) -> None:
        self.__video_index = value

    def set_device_loading(self, value: bool) -> None:
        self.__device_loading = value
