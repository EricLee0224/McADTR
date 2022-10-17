# McADTR
'Multi-class Anomaly Detection Transformer with Heterogeneous Knowledge Distillation' code

Modified based on the following code base：   
1.CVPR2020.uninformed students student-teacher anomaly detection with discriminative latent embeddings  
https://github.com/LuyaooChen/uninformed-students-pytorch  
2.ICCV2021.Learning mult-scene absolute pose regression with transformers  
https://github.com/yolish/multi-scene-pose-transformer  
  
model: new_student.py  
train student: new_train_student.py  
train teacher: train_teacher.py  
test: evaluate_1.py  
see pixel infor test: evaluate_~.py  

dataset format:  
~/data/MVtec/  
~/data/MAD1/bottle/.jpeg  
