# Table Tennis Train 


## Tasks:
- feature selection - details inside train notebook
- create Network 
	- LSTM of HW + attention - WIP dean
	- dataloader (SEQ len , currently 50, maybe 30 ?) (in the end aug goes here)
- data augmentation + debug - WIP Adi
    - plot landmarks of a frame
    - animate landmarks of a shot
	- create aug data function
- 3 models / experiments:
	all data (both fco fts), just fts, just fco 


## Tips:
- brnn seem good
- not alot of layers
- gru > lstm (lstm is to complicated )
- attention - allow to handle more thrn 30 frames

## Read:
- https://www.tensorflow.org/lite/tutorials/pose_classification
- impl of the above
- https://www.youtube.com/watch?v=aySurynUNAw
- https://web.stanford.edu/class/cs231a/prev_projects_2016/final%20(1).pdf
