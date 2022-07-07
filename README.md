# Table Tennis Train 

## Notice: 
1. labels folder have some old csv's for your to play around - need to re extract those
2. the py files: are the extraction utils, and the notebook is for training. i suggest we have 2 notebooks each.

## Tasks: (by order)
extractor
-  create binary clasification data (convert 0-10 to binary labels) - fully automated labling based on vidoes
-  improve shot extractor 

training:
- create Network
	- LSTM of HW + attention 
	- dataloader (SEQ len , currently 50, maybe 30 ?) (in the end aug goes here)
  
- data augmantation:
	- debug:
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
https://www.tensorflow.org/lite/tutorials/pose_classification
impl of the above
https://www.youtube.com/watch?v=aySurynUNAw

https://web.stanford.edu/class/cs231a/prev_projects_2016/final%20(1).pdf
