
**For more situated intent data and work, please see [Situated Intent](http://situated-intent.net)!**

---

# IUPUI-CSRC Pedestrian Situated Intent (PSI) Dataset 
This repository contains IUPUI-CSRC Pedestrian Situated Intent (PSI) Dataset pre-processing and baseline[to be uploaded].

## Download dataset and extract
Download the dataset from [link](http://situated-intent.net/pedestrian_dataset/), then extract via

```command
unzip Dataset.zip
```

Output: 

```command
Archive:  Dataset.zip
creating: PSI_Intention/Dataset/ 
inflating: PSI_Intention/Dataset/VideoWithIndicator.zip  
inflating: PSI_Intention/Dataset/RawVideos.zip  
inflating: PSI_Intention/Dataset/README.txt  
inflating: PSI_Intention/Dataset/IntentAnnotations.xlsx
inflating: PSI_Intention/Dataset/XmlFiles.zip 
```
Extract videos and spatial annotations:
```command
unzip ./PSI_Intention/Dataset/RawVideos.zip -d ./PSI_Intention/Dataset
unzip ./PSI_Intention/Dataset/XmlFiles.zip -d ./PSI_Intention/Dataset
```

## Video to frames
```python
python split_clips_to_frames.py
```
The splited frames are organized as, e.g.,
```
frames{
    video_0001{
        000.jpg,
        001.jpg,
        ...
    }
}
```
## CV_annotations and NLP_annotations re-organize
```python
python reorganize_annotations.py
```
*Note*: video_0060 and video_0093 are removed due to the missing of spatial segmentation annotations.

## Create database with frames labeled
```python
python pedestrian_intention_database_processing.py
```
Output: 

- **database_*.pkl**: The annotaions of reasoning and intention do not exactly match, i.e., the last several frames only have intention annotations without reasoning, because the reasoning is only for the previous time period before the last annotated time point, while the intention annotation lasts till the end of the video. 
- **database_*_overlap**.pkl: By removing the last frames only with intention labels, the length of annotated reasoning and intention frames are Equal now.

## Train/Val/Test split

- train: [1 ~ 75]
- val: [76 ~ 80]
- test: [81 ~ 110]

*Note*: Due to the missing of spatial segmentation annotations, video_0060 and video_0093 are removed. Besides, video_0003 and video_0028 are ignored as the annotated frame sequences are too short.

In our PSI paper experiments, the observed tracks length is 15, while predicting the 16-th frame intention. The overlap rate is set as 0.8 for both train and test stages. 

# Citing
```
@article{chen2021psi,
title   = {PSI: A Pedestrian Behavior Dataset for Socially Intelligent Autonomous Car},
author  = {Chen, Tina and Tian, Renran and Chen, Yaobin and Domeyer, Joshua and Toyoda, Heishiro and Sherony, Rini and Jing, Taotao and Ding, Zhengming},
journal = {arXiv preprint arXiv:2112.02604},
year    = {2021} }
```