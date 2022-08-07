# Table Tennis Train 


## Tasks:
- attention impl
- feature selection - details inside train notebook
- normalize input - see Read below
- data augmentation
- experiments: every experiment should be documented during the process in the pptx
	- all data (both fco fts), just fts, just fco 
    - with/without: attention, feature selection, normalization, augmentation
- false positive analysis

## Tips:
- brnn seem good
- not alot of layers
- gru > lstm (lstm is to complicated )
- attention - allow to handle more thrn 30 frames

## Read:
- https://www.tensorflow.org/lite/tutorials/pose_classification
  - this has normalize_pose_landmarks code that can be usefull
- impl of the above
- https://www.youtube.com/watch?v=aySurynUNAw
- https://web.stanford.edu/class/cs231a/prev_projects_2016/final%20(1).pdf
