#Validation Plan

Your Name: Rob Straker

Name of your Device: Hippocampal Volume Quantification in Alzheimer's Progression

This Document: This clinical validation plan is meant to prove that our technology performs the way we claim it does. For example, given that we say that our technology can measure hippocampal volume, this validation plan needs to define how we would prove this, and establish these extents.
We assume that we have access to any clinical facility and patient cohorts we need, and that we have all the budget in the world. 

## Algorithm Description 

### 1. General Information

**Intended Use Statement:** This algorithm is intended to assist the radiologist to quantify hippocampal volume for Alzheimer's progression detection or monitoring from 3D medical images of brain MRIs.

### 2. Algorithm Training

**Training Data:** Training data was collected from the "Hippocampus" dataset from the Medical Decathlon competition. This dataset is stored as a collection of NIFTI files, with one file per volume, and one file per corresponding segmentation mask. The original images here are T2 MRI scans of the full brain. However, in this dataset we are using cropped volumes where only the region around the hippocampus has been cut out.

**Training Labels:** Training data was labeled using labels defined in the "Hippocampus" dataset from the Medical Decathlon competition.

### 3. Validation Plan

**Ground Truth:** Ground truth will be defined by the silver standard of radiologist readings, weighted by their years of experience as practitioners.

**Algorithm Accuracy:** Jaccard Similarity Coefficient for 3D volumes and Dice Similarity Coefficient for 3D volumes were used to define and measure the accuracy of the algorithm during training. Real-world performance will also be measured using these coefficients.

**Data:** The algorithm can operate on MRI scans of the hippocampus that are stored in the DICOM format. It will not operate on images stored in other formats, and it will not perform as well if the images are not cropped to include just the region around the hippocampus.
