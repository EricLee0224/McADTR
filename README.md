# McADTR: Multi-class Anomaly Detection Transformer with Heterogeneous Knowledge Distillation
2022 Tsinghua University AIR-DISCOVER summer research project  

official implementation of McADTR  

Modified based on the following code base：   
1.CVPR2020.uninformed students student-teacher anomaly detection with discriminative latent embeddings  
https://github.com/LuyaooChen/uninformed-students-pytorch  
2.ICCV2021.Learning mult-scene absolute pose regression with transformers  
https://github.com/yolish/multi-scene-pose-transformer  

## ps
model: new_student.py  
train student: new_train_student.py  
train teacher: teacher_train.py  
test: evaluate_1.py  
pixel information test: evaluate_~.py  

dataset format:  
.../data/MVtec/  
.../data/MAD1/bottle/.jpeg  

## Details
![Alt text](https://github.com/EricLee0224/McADTR/blob/main/img/1.PNG)
![Alt text](https://github.com/EricLee0224/McADTR/blob/main/img/2.PNG)
![Alt text](https://github.com/EricLee0224/McADTR/blob/main/img/3.PNG)
![Alt text](https://github.com/EricLee0224/McADTR/blob/main/img/4.PNG)
![Alt text](https://github.com/EricLee0224/McADTR/blob/main/img/5.PNG)
![Alt text](https://github.com/EricLee0224/McADTR/blob/main/img/6.PNG)
![Alt text](https://github.com/EricLee0224/McADTR/blob/main/img/7.PNG)
![Alt text](https://github.com/EricLee0224/McADTR/blob/main/img/9.PNG)
![Alt text](https://github.com/EricLee0224/McADTR/blob/main/img/10.PNG)

## Citation
    @misc{weize2022airmcadtr,
    title = {McADTR: Multi-class Anomaly Detection Transformer with Heterogeneous Knowledge Distillation},
    author = {Weize Li, Qiang Zhou, Hao Zhao},
    journal = {AIR-DISCOVER Project},
    url = {https://github.com/EricLee0224/McADTR},
    year = {2022}
    }



