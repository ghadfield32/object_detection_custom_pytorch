
from roboflow import Roboflow
rf = Roboflow(api_key="htpcxp3XQh7SsgMfjJns")
project = rf.workspace("ownprojects").project("basketball-w2xcw")
dataset = project.version(2).download("yolov8")

from roboflow import Roboflow
rf = Roboflow(api_key="htpcxp3XQh7SsgMfjJns")
project = rf.workspace("basketball-formations").project("warriors-vs-cavs-2016")
dataset = project.version(10).download("yolov8")
