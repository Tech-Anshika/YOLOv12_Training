from roboflow import Roboflow

rf = Roboflow(api_key="cNQGjy6gvMLLNMnaaZeT")
project = rf.workspace("image-detection-f0jrc").project("building-detection-qkcv9")
version = project.version(1)
version.download("yolov12")
