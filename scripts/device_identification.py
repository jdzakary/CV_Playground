from pygrabber.dshow_graph import FilterGraph

graph = FilterGraph()
print(graph.get_input_devices())

graph.add_video_input_device(0)
device = graph.get_input_device()
print(device)
print(device.get_current_format())
