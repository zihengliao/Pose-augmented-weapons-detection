from roboflow import Roboflow
rf = Roboflow(api_key="")
project = rf.workspace("").project("40k-dataset-pvktf")
version = project.version(2)
dataset = version.download("coco")
                