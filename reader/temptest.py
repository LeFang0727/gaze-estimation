from reader_LBW import *
from reader_LBW import _subject_from_image_path


line = "face/Subject01_1_data/00000190_face.png 378,470 -0.49233252,0.44971914,-3.63335071 -0.1332778,0.12174207,-0.98357303 0.13468339444529132,-0.12204481944248674 493.15,426.71 Subject01_1_data/00000607_face.png -0.00381595923755,-5.04972895e-05,-0.9999852545000001 0.003815996983861851,5.049728952146115e-05 scene/Subject01_1_data/00000190_scene.png "
anno = Decode_LBW(line)
subject = _subject_from_image_path(anno.face)


print(subject)

