from roboflow import Roboflow
rf = Roboflow(api_key="BhgQZacev5LVkLmOUayV")
project = rf.workspace("ailecs-nmbrc").project("40k-dataset-pvktf")
version = project.version(2)
dataset = version.download("coco")
                