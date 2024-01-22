# Computer Vision Playground
Welcome to CV Playground, a GUI toolkit for writing and testing computer vision algorithms.
The GUI is built with PyQt, the Python bindings for the popular Qt C++ framework.
Programmers can add algorithms to the toolkit by implementing the "Operations" abstract class.
These algorithms will be automatically detected and made available in the GUI.

## Capabilities
The capabilities of the toolkit are divided into tabs.
These tabs change the main window to expose the desired functionality.
Currently, there are two tabs available:
* Stream
* Settings

The stream tab allows the user to select a webcam and displays the live video feed.
Through a dockable tool window, users can add real-time CV operations and adjust their settings.

The settings tab allows the user to configure the appearance and behavior of the application.
Settings are loaded from `settings.json` and dumped when the user closes the program.

## Development Plan and New Features
New features will be added as time permits.
The below lists are overall development targets, and are not comprehensive.
For more detailed info on upcoming features, view the issues tagged `enhancement` on the issues tab.
To suggest a new feature or request prioritization of a specific feature, please use the issues tab.

### Video Streaming
* Add multiprocessing support, allowing multiple frames to be processed simultaneously using a thread pool.
* Add metrics to track latency and framerate
* Improve the operation management tool window:
  * Allow the user to re-arrange order of operators by drag-and-drop
  * Make text resizable
  * View the time per frame for each operation

### Settings
* Create setting to remember default video stream size
* Create setting to remember default tool window dock width

## Real-Time CV Operations
Programmers can write new cv operations by implementing the abstract `Operation` class found in `app/video/processing.py`
Any subclass that implements `Operation` should be placed in the folder `app/video/operations` to enable auto-detection.
That folder can contain any number of `.py` files, and each file can contain any number of `Operation` subclasses.

### How it works
The opencv `VideoCapture` class is used to collect raw frames from the camera device.
This is manged by the `CaptureThread` found in `app/video/stream.py`.
The frames are sent to a queue that is managed by the `ProcessThread` found in `app/video/stream.py`.
This process thread contains a list of `Operation` subclasses.
It calls the execute method of each operation in order, passing the output of one operation to the input of the next.
After all operations are finished, the frame is converted to a Qt Image Object and passed to the main thread.
The main thread updates the image onscreen to create the effect of a video.

### Mandatory Subclass Features
All subclasses are required to implement a number of methods and properties.
These features are expected by the toolkit and used to display the operation to the user.
The `GaussianBlur` class in the file `app/video/operations/gaussian.py` is a good example to reference.

```python
class GaussianBlur(Operation):
    name = 'Gaussian Blur'
    description = 'Blurs the frame using a gaussian filter'

    def __init__(self):
        super().__init__()
        self.__sigma = Slider(1, 10, 1)
        self.params.append(self.sigma)

    @property
    def sigma(self) -> Slider:
        return self.__sigma

    def execute(self, frame: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(frame, (0, 0), self.sigma.number)
```

#### Class Attributes
The toolkit expects 2 class attributes for every subclass:
* name
  * The name of the operation. 
  * Must be unique across all operations, or else a cryptic error will be thrown.
* description
  * A short description of the operation. 
  * No current use (v1.0), but future features will leverage this attribute.

#### Instance Methods
The toolkit expects 1 instance method for every subclass:
* execute
  * Called by the processing thread every time a frame is processed.
  * Passed the new frame as an argument
  * Expected to return the processed frame. This return value is passed to the next operation.

The execute method should be very fast. Slow execute methods will result in low framerate.
Support for multiprocessing will help reduce the bottleneck created by slow execute methods.

### Optional Subclass Features
In addition to creating their own methods for internal logic, subclasses can override/leverage optional
methods and properties. 
These optional features will often be hooked into other parts of the toolkit, such as the UI.

#### Parameters
Many CV algorithms have thresholds or parameters that can be adjusted.
The sigma of a gaussian blur is a good example. 
It may be desirable for the user to adjust this value with a slider.\
To enable configuration of CV algorithms at runtime, subclasses can add parameters.
These parameters are linked to a GUI component, such as a slider, that enable the user to change the value.

Parameters are available as subclasses of the abstract `Parameter` class found in `app/video/processing.py`
For example, `Slider` allows the user to adjust a numerical parameter using within a boundary.
This examples creates a slider and registers it with the UI:

```python
def __init__(self):
    super().__init__()
    x = Slider(1, 10, 1)
    self.params.append(x)
```
