from pygrabber.dshow_graph import FilterGraph, FormatTypedDict
from fractions import Fraction


def aspect_ratio(width: int, height: int) -> str:
    frac = Fraction(width, height)
    top, bottom = frac.as_integer_ratio()
    return f'{top}/{bottom}'


# Create Graph
graph = FilterGraph()

# List Cameras
for (i, x) in enumerate(graph.get_input_devices()):
    print(f'{i}: {x}')

# Set Individual Device for Inspection
graph.add_video_input_device(0)
device = graph.get_input_device()

# List Possible Formats
resolutions = sorted(
    device.get_formats(),
    key=lambda r: r['width'] * r['height'],
    reverse=True
)
print(f'\nResolutions for {device.get_name()}')
res: FormatTypedDict
for res in resolutions:
    print(f'Index: {res["index"]:>2}', end=' '*4)
    print(f'Width: {res["width"]:>4}', end=' '*4)
    print(f'Height: {res["height"]:>4}', end=' '*4)
    print(f'Ratio: {aspect_ratio(res["width"], res["height"]):>4}', end=' '*4)
    print(f'Media Type: {res["media_type_str"]:>6}', end=' '*4)
    print(f'FPS: {res["min_framerate"]:.0f} to {res["max_framerate"]:.0f}')

# List Current Format
print(f'\nCurrent Resolution for {device.get_name()}: {device.get_current_format()}')
