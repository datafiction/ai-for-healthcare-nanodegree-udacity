# AI for Healthcare Nanodegree

Instructors: 
- Emily Lindemer, Mazen Zawaideh, Ivan Tarapov, Michael DAndrea, and Nikhil Bikhchandani. 
- For biographical information, see [BIO.md][1]  

Offered By: [udacity.com][2]

## Introduction
This repo contains projects and exercises for the four-course AI for Healthcare Nanodegree program offered through Udacity. 

Feel free to use the material for reference purposes or if you get stuck. However, I would encourage you to try to complete all projects and exercises yourself, so that you can maximize your learning and enjoyment of the program.

## Udacity Description
#### About this Nanodegree
Be at the forefront of the revolution of AI in Healthcare, and transform patient outcomes. Enable enhanced medical decision-making powered by machine learning to build the treatments of the future.

Learn to build, evaluate, and integrate predictive models that have the power to transform patient outcomes. Begin by classifying and segmenting 2D and 3D medical images to augment diagnosis and then move on to modeling patient outcomes with electronic health records to optimize clinical trial testing decisions. Finally, build an algorithm that uses data collected from wearable devices to estimate the wearer’s pulse rate in the presence of motion.

#### Overview
Play a critical role in enhancing clinical decision-making with machine learning to build the treatments of the future. Learn to build, evaluate, and integrate predictive models that have the power to transform patient outcomes. Begin by classifying and segmenting 2D and 3D medical images to augment diagnosis and then move on to modeling patient outcomes with electronic health records to optimize clinical trial testing decisions. Finally, build an algorithm that uses data collected from wearable devices to estimate the wearer’s pulse rate in the presence of motion.

A graduate of this program will be able to:
- Recommend appropriate imaging modalities for common clinical applications of 2D medical imaging • Perform exploratory data analysis (EDA) on 2D medical imaging data to inform model training and explain model performance
- Establish the appropriate ‘ground truth’ methodologies for training algorithms to label medical images • Extract images from a DICOM dataset
- Train common CNN architectures to classify 2D medical images
- Translate outputs of medical imaging models for use by a clinician
- Plan necessary validations to prepare a medical imaging model for regulatory approval
- Detect major clinical abnormalities in a DICOM dataset
- Train machine learning models for classification tasks using real-world 3D medical imaging data
- Integrate models into a clinician’s workflow and troubleshoot deployments
- Build machine learning models in a manner that is compliant with U.S. healthcare data security and privacy standards
- Use the TensorFlow Dataset API to scalably extract, transform, and load datasets that are aggregated at the line, encounter, and longitudinal (patient) data levels
- Analyze EHR datasets to check for common issues (data leakage, statistical properties, missing values, high cardinality) by performing exploratory data analysis with TensorFlow Data Analysis and Validation library
- Create categorical features from Key Industry Code Sets (ICD, CPT, NDC) and reduce dimensionality for high cardinality features
- Use TensorFlow feature columns on both continuous and categorical input features to create derived features (bucketing, cross-features, embeddings)
- Use Shapley values to select features for a model and identify the marginal contribution for each selected feature
- Analyze and determine biases for a model for key demographic groups
- Use the TensorFlow Probability library to train a model that provides uncertainty range predictions in order to allow for risk adjustment/prioritization and triaging of predictions
- Preprocess data (eliminate “noise”) collected by IMU, PPG, and ECG sensors based on mechanical, physiology and environmental effects on the signal.
- Create an activity classification algorithm using signal processing and machine learning techniques • Detect QRS complexes using one-dimensional time series processing techniques
- Evaluate algorithm performance without ground truth labels
- Generate a pulse rate algorithm that combines information from the PPG and IMU sensor streams

## Topics Covered
**Course 1: Applying AI to 2D Medical Imaging Data:**  
- 2D Medical Imaging Modalities & Clinical Applications, DICOM Standard, Image Pre-Processing, Image Augmentation, CNN Architecture, Transfer Learning, Model Performance, and FDA Validation Plan.

**Course 2: Applying AI to 3D Medical Imaging Data:**  
- 3D Medical Imaging Modalities, CT & MR Scanner Operation, 3D Medical Image Analysis, DICOM & NIFTI Representations, 3D Medical Image Visualization, Applying Convolutions, U-net Algorithm, Dice & Jaccard Metrics, Clinical Medical Imaging Networks, FDA Medical Device Regulatory Requirements, and HIPAA.

**Course 3: Applying AI to EHR Data:**  
- HIPAA, HITECH, Protected Health Information (PHI), EHR Code Ssets (ICD, CPT, NDC), TensorFlow Dataset API, EHR Transformations, Feature Engineering, Model Bias, Aequitas Framework, TensorFlow Probability Library, and Shapley Values.

**Course 4: Applying AI to Wearable Device Data:**  
- Digital Sampling, Signal Processing (FFT, STFT, Spectrogram), Sensors (IMU, PPG, ECG), Activity Classification, ECG Signal Processing, and QRS Complex Detection.

For further information on topics and technologies covered, see [TOPICS.md][3].

## Syllabus

### Course 1: Applying AI to 2D Medical Imaging Data
2D imaging, such as X-ray, is widely used when making critical decisions about patient care and accessible by most healthcare centers around the world. With the advent of deep learning for non-medical imaging data over the past half decade, the world has quickly turned its attention to how AI could be specifically applied to medical imaging to improve clinical decision-making and to optimize workflows. Learn the fundamental skills needed to work with 2D medical imaging data and how to use AI to derive clinically-relevant insights from data gathered via different types of 2D medical imaging such as x-ray, mammography, and digital pathology. Extract 2D images from DICOM files and apply the appropriate tools to perform exploratory data analysis on them. Build different AI models for different clinical scenarios that involve 2D images and learn how to position AI tools for regulatory approval.

**Lesson 1: Introduction to AI for 2D Medical Imaging.** 
- Outcomes:
  - Explain what AI for 2D medical imaging is and why it is relevant.

**Lesson 2: Clinical Foundations of 2D Medical Imaging.** 
- Outcomes:
  - Learn about different 2D medical imaging modalities and their clinical applications.
  - Understand how different types of machine learning algorithms can be applied to 2D medical imaging.
  - Learn how to statistically assess an algorithm’s performance.
  - Understand the key stakeholders in the 2D medical imaging space.
- Exercises:
  - [1-clinical-applications][11]
  - [2-apply-machine-learning][12] 
  - [3-performance-of-ml][13]

**Lesson 3: 2D Medical Imaging Exploratory Data Analysis.** 
- Outcomes:
  - Learn what the DICOM standard it is and why it exists.
  - Use Python tools to explore images extracted from DICOM files.
  - Apply Python tools to explore DICOM header data.
  - Prepare a DICOM dataset for machine learning.
  - Explore a dataset in preparation for machine learning.
- Exercises:
  - [1-explore-2d-imaging-properties][14] 
  - [2-prepare-dicom-images for-ml][15] 
  - [3-exploring-population-metadata][16] 

**Lesson 4: Classification Models of 2D Medical Images.** 
- Outcomes:
  - Understand architectures of different machine learning and deep learning models, and the differences between them.
  - Split a dataset for training and testing an algorithm.
  - Learn how to define a gold standard.
  - Apply common image pre-processing and augmentation techniques to data.
  - Fine-tune an existing CNN architecture for transfer learning with 2D medical imaging applications.
  - Evaluate a model’s performance and optimize its parameters.
- Exercises:
  - [1-differentiate-between-models][17] 
  - [2-split-dataset-for-model-development][18] 
  - [3-obtaining-a-gold-standard][19] 
  - [4-image-pre-processing-for-model-training][20] 
  - [5-fine-tuning-cnns-for- classification][21] 
  - [6-evaluating-your-model][22] 

**Lesson 5: Translating AI Algorithms for Clinical Settings with the FDA.** 
- Outcomes:
  - Learn about the FDA’s risk categorization for medical devices and how to define an Intended Use statement.
  - Identify and describe algorithmic limitations for the FDA.
  - Translate algorithm performance statistics into clinically meaningful information that can trusted by professionals.
  - Learn how to create an FDA validation plan.
- Exercises:
  - [1-Intended-use-and-clinical-impact][23] 
  - [2-algorithmic-limitations][24] 
  - [3-translate-performance-into-clinical-utility][25] 

**Project: Pneumonia Detection from Chest X-Rays.** 
- Udacity Repo: [AIHCND_C2_Starter][5]
- Chest X-ray exams are one of the most frequent and cost-effective types of medical imaging examinations. Deriving clinical diagnoses from chest X-rays can be challenging, however, even by skilled radiologists. 
- When it comes to pneumonia, chest X-rays are the best available method for point-of-care diagnosis. More than 1 million adults are hospitalized with pneumonia and around 50,000 die from the disease every year in the US alone. 
- The high prevalence of pneumonia makes it a good candidate for the development of a deep learning application for two reasons: 
  1) Data availability in a high enough quantity for training deep learning models for image classification 
  2) Opportunity for clinical aid by providing higher accuracy image reads of a difficult-to-diagnose disease and/or reduce clinical burnout by performing automated reads of very common scans. 
- In this project, you will analyze data from the NIH Chest X-ray dataset and train a CNN to classify a given chest X-ray for the presence or absence of pneumonia. 
  - First, you’ll curate training and testing sets that are appropriate for the clinical question at hand from a large collection of medical images. 
  - Then, you will create a pipeline to extract images from DICOM files that can be fed into the CNN for model training. 
  - Lastly, you’ll write an FDA 501(k) validation plan that formally describes your model, the data that it was trained on, and a validation plan that meets FDA criteria in order to obtain clearance of the software being used as a medical device.
- Project Solution: [Pneumonia Detection from Chest X-Rays][55]

### Course 2: Applying AI to 3D Medical Imaging Data
3D medical imaging exams such as CT and MRI serve as critical decision-making tools in the clinician’s everyday diagnostic armamentarium. These modalities provide a detailed view of the patient’s anatomy and potential diseases, and are a challenging though highly promising data type for AI applications. Learn the fundamental skills needed to work with 3D medical imaging datasets and frame insights derived from the data in a clinically relevant context. Understand how these images are acquired, stored in clinical archives, and subsequently read and analyzed. Discover how clinicians use 3D medical images in practice and where AI holds most potential in their work with these images. Design and apply machine learning algorithms to solve the challenging problems in 3D medical imaging and how to integrate the algorithms into the clinical workflow.

Udacity Repo: [nd320-c3-3d-med-imaging][6]

**Lesson 1: Introduction to AI for 3D Medical Imaging.** 
- Outcomes:
  - Explain what AI for 3D medical imaging is and why it is relevant.

**Lesson 2: 3D Medical Imaging - Clinical Fundamentals.** 
- Outcomes:
  - Identify medical imaging modalities that generate 3D images.
  - List clinical specialties who use 3D images to influence clinical decision making.
  - Describe use cases for 3D medical images.
  - Explain the principles of clinical decision making.
  - Articulate the basic principles of CT and MR scanner operation.
  - Perform some of the common 3D medical image analysis tasks such as windowing, MPR and 3D reconstruction.
- Exercises:
  - [1-ai-problem-definition][26] 
  - [2-ct-backprojection][27] 
  - [3-volume-rendering][28] 

**Lesson 3: 3D Medical Imaging Exploratory Data Analysis.** 
- Outcomes:
  - Describe and use DICOM and NIFTI representations of 3D medical imaging data.
  - Explain specifics of spatial and dimensional encoding of 3D medical images.
  - Use Python-based software packages to load and inspect 3D medical imaging volumes.
  - Use Python-based software packages to explore datasets of 3D medical images and prepare it for machine learning pipelines.
  - Visualize 3D medical images using open software packages.
- Exercises:
  - [1-load-file][29] 
  - [2-volume-mpr][30] 
  - [3-dataset-eda][31] 

**Lesson 4: 3D Medical Imaging - Deep Learning Methods.** 
- Outcomes:
  - Distinguish between classification and segmentation problems as they apply to 3D imaging.
  - Apply 2D, 2.5D and 3D convolutions to a medical imaging volume.
  - Apply U-net algorithm to train an automatic segmentation model of a real-world CT dataset using PyTorch.
  - Interpret results of training, measure efficiency using Dice and Jaccard performance metrics.
- Exercises:
  - [1-convolutions][32] 
  - [2-segmentation-hands-on][33] 
  - [3-performance-metrics][34] 

**Lesson 5: Deploying AI Algorithms in the Real World.** 
- Outcomes:
  - Identify the components of a clinical medical imaging network and integration points as well as DICOM protocol for medical image exchange.
  - Define the requirements for integration of AI algorithms. 
  - Use tools for modeling of clinical environments so that it is possible to emulate and troubleshoot real-world AI deployments.
  - Describe regulatory requirements such as FDA medical device framework and HIPAA required for operating AI for clinical care.
  - Provide input into regulatory process, as a data scientist.
- Exercises:
  - [1-sending-volumes][35] 
  - [2-segmenting-structures][36] 
  - [3-anonymization][37] 

**Project: Hippocampal Volume Quantification in Alzheimer’s Progression.**
- Udacity Repo: [nd320-c3-3d-imaging-starter][7]
- Hippocampus is one of the major structures of the human brain with functions that are primarily connected to learning and memory. The volume of the hippocampus may change over time, with age, or as a result of disease. 
- In order to measure hippocampal volume, a 3D imaging technique with good soft tissue contrast is required. MRI provides such imaging characteristics, but manual volume measurement still requires careful and time consuming delineation of the hippocampal boundary. 
- In this project, you will go through the steps that will have you create an algorithm that will help clinicians assess hippocampal volume in an automated way and integrate this algorithm into a clinician’s working environment. 
  - First, you’ll prepare a hippocampal image dataset to train the U-net based segmentation model, and capture performance on the test data. 
  - Then, you will connect the machine learning execution code into a clinical network, create code that will generate reports based on the algorithm output, and inspect results in a medical image viewer. 
  - Lastly, you’ll write up a validation plan that would help collect clinical evidence of the algorithm performance, similar to that required by regulatory authorities.
- Project Solution: [Hippocampal Volume Quantification in Alzheimer’s Progression][56]

### Course 3: Applying AI to EHR Data
With the transition to electronic health records (EHR) over the last decade, the amount of EHR data has increased exponentially, providing an incredible opportunity to unlock this data with AI to benefit the healthcare system. Learn the fundamental skills of working with EHR data in order to build and evaluate compliant, interpretable machine learning models that account for bias and uncertainty using cutting-edge libraries and tools including TensorFlow Probability, Aequitas, and Shapley. Understand the implications of key data privacy and security standards in healthcare. Apply industry code sets (ICD10-CM, CPT, HCPCS, NDC), transform datasets at different EHR data levels, and use TensorFlow to engineer features.

Udacity Repo: [nd320-c1-emr-data-starter][8]

**Lesson 1: Applying AI to EHR Data Introduction.** 
- Outcomes:
  - Introduction to the EHR Data course and the instructor.

**Lesson 2: EHR Data Security and Analysis.** 
- Outcomes:
  - Understand U.S. healthcare data security and privacy best practices (e.g. HIPAA, HITECH) and how they affect utilizing protected health information (PHI) data and building models.
  - Analyze EHR datasets to check for common issues (data leakage, statistical properties, missing values, high cardinality) by performing exploratory data analysis.
- Exercises:
  - [lesson-2-EHR-Data-Security-and-Analysis][38] 

**Lesson 3: EHR Code Sets.** 
- Outcomes:
  - Understand the usage and structure of key industry code sets (ICD, CPT, NDC).
  - Group and categorize data within EHR datasets using code sets.
- Exercises:
  - [lesson-3-EHR-Code-Sets][39] 

**Lesson 4: EHR Transformations & Feature Engineering.** 
- Outcomes:
  - Use the TensorFlow Dataset API to scalably extract, transform, and load datasets.
  - Build datasets aggregated at the line, encounter, and longitudinal(patient) data levels.
  - Create derived features (bucketing, cross-features, embeddings) utilizing TensorFlow feature columns on both continuous and categorical input features.
- Exercises:
  - [lesson-4-EHR-Data-Transformations-and-Tensorflow-Feature-Engineering][40] 

**Lesson 5: Building, Evaluating, and Interpreting Models.** 
- Outcomes:
  - Analyze and determine biases for a model for key demographic groups by evaluating performance metrics across groups by using the Aequitas framework.
  - Train a model that provides an uncertainty range with the TensorFlow Probability library.
  - Use Shapley values to select features for a model and identify the marginal contribution for each selected feature.
- Exercises:
  - [lesson-5-Building-Evaluating-and-Interpreting-Models-for-Bias-and-Uncertainty][41]

**Project: Patient Selection for Diabetes Drug Testing.**
- EHR data is becoming a key source of real-world evidence (RWE) for the pharmaceutical industry and regulators to make decisions on clinical trials. 
- In this project, you will act as a data scientist for an exciting unicorn healthcare startup that has created a groundbreaking diabetes drug that is ready for clinical trial testing. 
- Your task will be to build a regression model to predict the estimated hospitalization time for a patient in order to help select/filter patients for your study. 
  - First, you will perform exploratory data analysis in order to identify the dataset level and perform feature selection. 
  - Next, you will build necessary categorical and numerical feature transformations with TensorFlow. 
  - Lastly, you will build a model and apply various analysis frameworks, including TensorFlow Probability and Aequitas, to evaluate model bias and uncertainty.
- Project Solution: [Patient Selection for Diabetes Drug Testing][57]

### Course 4: Applying AI to Wearable Device Data
Wearable devices are an emerging source of physical health data. With continuous, unobtrusive monitoring they hold the promise to add richness to a patient’s health information in remarkable ways. Understand the functional mechanisms of three sensors (IMU, PPG, and ECG) that are common to most wearable devices and the foundational signal processing knowledge critical for success in this domain. Attribute physiology and environmental context’s effect on the sensor signal. Build algorithms that process the data collected by multiple sensor streams from wearable devices to surface insights about the wearer’s health.

Udacity Repo: [nd320-c4-wearable-data-starter][9]

**Lesson 1: Intro to Digital Sampling & Signal Processing.** 
- Outcomes:
  - Describe how to digitally sample analog signals.
  - Apply signal processing techniques (eg. filtering, resampling, interpolation) to time series signals.
  - Apply frequency domain techniques (eg. FFT, STFT, spectrogram) to time series signals.
  - Use matplotlib’s plotting functionality to visualize signals.
- Exercises:
  - [1-plotting][42] 
  - [2-interpolation][43] 
  - [3-fourier-transform][44] 
  - [4-spectrograms][45]

**Lesson 2: Introduction to Sensors.** 
- Outcomes:
  - Describe how sensors convert a physical phenomenon into an electrical one.
  - Understand the signal and noise characteristics of the IMU and PPG signals.
- Exercises:
  - [1-step-cadence][46] 
  - [2-ppg-peaks][47] 
  - [3-ppg-snr][48] 

**Lesson 3: Activity Classification.** 
- Outcomes:
  - Perform exploratory data analysis to understand class imbalance and subject imbalance.
  - Gain an intuitive understanding signal characteristics and potential feature performance.
  - Write code to implement features from literature.
  - Recognize the danger overfitting of technique (esp. on small datasets), not simply of model parameters or hyperparameters.
- Exercises:
  - [1-data-exploration][49] 
  - [2-feature-extraction][50] 
  - [3-quirk-in-the-dataset][51] 

**Lesson 4: ECG Signal Processing.** 
- Outcomes:
  - Understand the electrophysiology of the heart at a basic level.
  - Understand the signal and noise characteristics of the ECG.
  - Understand how atrial fibrillation manifests in the ECG.
  - Build a QRS complex detection algorithm.
  - Build an arrhythmia detection algorithm from a wearable ECG signal.
  - Understand how models can be cascaded together to achieve higher-order functionality.
- Exercises:
  - [1-pan-tompkins-algorithm][52] 
  - [2-af-features][53] 
  - [3-atrial-fibrillation][54] 

**Project: Motion Compensated Pulse Rate Estimation.**
- Udacity Repo: [nd320-c4-wearable-data-project-starter][10]
- Wearable devices have multiple sensors all collecting information about the same person at the same time. Combining these data streams allows us to accomplish many tasks that would be impossible from a single sensor. 
- In this project, you will build an algorithm which combines information from two of the sensors that are covered in this course -- the IMU and PPG sensors -- that can estimate the wearer’s pulse rate in the presence of motion. 
  - First, you’ll create and evaluate an activity classification algorithm by building signal processing features and a random forest model. 
  - Then, you will build a pulse rate algorithm that uses the activity classifier and frequency domain techniques, and also produces an associated confidence metric that estimates the accuracy of the pulse rate estimate. 
  - Lastly, you will evaluate algorithm performance and iterate on design until the desired accuracy is achieved.
- Project Solution: [Motion Compensated Pulse Rate Estimation][58]

## License
This project is licensed under the MIT License. See [LICENSE][4] for details.

## Milestones
- 2020-07-28: Completed 4-course Nanodegree program.

[//]: # (Links Section)
[1]:https://github.com/robstraker/ai-for-healthcare-nanodegree-udacity/blob/master/BIO.md
[2]:https://www.udacity.com
[3]:https://github.com/robstraker/ai-for-healthcare-nanodegree-udacity/blob/master/TOPICS.md
[4]:https://github.com/robstraker/ai-for-healthcare-nanodegree-udacity/blob/master/LICENSE

[//]: # (Links to Repos)
[5]:https://github.com/udacity/AIHCND_C2_Starter
[6]:https://github.com/udacity/nd320-c3-3d-med-imaging
[7]:https://github.com/udacity/nd320-c3-3d-imaging-starter
[8]:https://github.com/udacity/nd320-c1-emr-data-starter
[9]:https://github.com/udacity/nd320-c4-wearable-data-starter
[10]:https://github.com/udacity/nd320-c4-wearable-data-project-starter

[//]: # (Links to Exercise Solutions)
[11]:https://github.com/robstraker/ai-for-healthcare-nanodegree-udacity/tree/master/course-1-2d-medical-imaging-data/lesson-2-exercises/1-clinical-applications
[12]:https://github.com/robstraker/ai-for-healthcare-nanodegree-udacity/tree/master/course-1-2d-medical-imaging-data/lesson-2-exercises/2-apply-machine-learning
[13]:https://github.com/robstraker/ai-for-healthcare-nanodegree-udacity/tree/master/course-1-2d-medical-imaging-data/lesson-2-exercises/3-performance-of-ml
[14]:https://github.com/robstraker/ai-for-healthcare-nanodegree-udacity/tree/master/course-1-2d-medical-imaging-data/lesson-3-exercises/1-explore-2d-imaging-properties
[15]:https://github.com/robstraker/ai-for-healthcare-nanodegree-udacity/tree/master/course-1-2d-medical-imaging-data/lesson-3-exercises/2-prepare-dicom-images-for-ml
[16]:https://github.com/robstraker/ai-for-healthcare-nanodegree-udacity/tree/master/course-1-2d-medical-imaging-data/lesson-3-exercises/3-exploring-population-metadata
[17]:https://github.com/robstraker/ai-for-healthcare-nanodegree-udacity/tree/master/course-1-2d-medical-imaging-data/lesson-4-exercises/1-differentiate-between-models
[18]:https://github.com/robstraker/ai-for-healthcare-nanodegree-udacity/tree/master/course-1-2d-medical-imaging-data/lesson-4-exercises/2-split-dataset-for-model-development
[19]:https://github.com/robstraker/ai-for-healthcare-nanodegree-udacity/tree/master/course-1-2d-medical-imaging-data/lesson-4-exercises/3-obtaining-a-gold-standard
[20]:https://github.com/robstraker/ai-for-healthcare-nanodegree-udacity/tree/master/course-1-2d-medical-imaging-data/lesson-4-exercises/4-image-pre-processing-for-model-training
[21]:https://github.com/robstraker/ai-for-healthcare-nanodegree-udacity/tree/master/course-1-2d-medical-imaging-data/lesson-4-exercises/5-fine-tuning-cnns-for-classification
[22]:https://github.com/robstraker/ai-for-healthcare-nanodegree-udacity/tree/master/course-1-2d-medical-imaging-data/lesson-4-exercises/6-evaluating-your-model
[23]:https://github.com/robstraker/ai-for-healthcare-nanodegree-udacity/tree/master/course-1-2d-medical-imaging-data/lesson-5-exercises/1-Intended-use-and-clinical-impact
[24]:https://github.com/robstraker/ai-for-healthcare-nanodegree-udacity/tree/master/course-1-2d-medical-imaging-data/lesson-5-exercises/2-algorithmic-limitations
[25]:https://github.com/robstraker/ai-for-healthcare-nanodegree-udacity/tree/master/course-1-2d-medical-imaging-data/lesson-5-exercises/3-translate-performance-into-clinical-utility
[26]:https://github.com/robstraker/ai-for-healthcare-nanodegree-udacity/tree/master/course-2-3d-medical-imaging-data/lesson-2-clinical-background/exercises/1-ai-problem-definition
[27]:https://github.com/robstraker/ai-for-healthcare-nanodegree-udacity/tree/master/course-2-3d-medical-imaging-data/lesson-2-clinical-background/exercises/2-ct-backprojection
[28]:https://github.com/robstraker/ai-for-healthcare-nanodegree-udacity/tree/master/course-2-3d-medical-imaging-data/lesson-2-clinical-background/exercises/3-volume-rendering
[29]:https://github.com/robstraker/ai-for-healthcare-nanodegree-udacity/tree/master/course-2-3d-medical-imaging-data/lesson-3-3d-imaging-exploratory-data-analysis/exercises/1-load-file
[30]:https://github.com/robstraker/ai-for-healthcare-nanodegree-udacity/tree/master/course-2-3d-medical-imaging-data/lesson-3-3d-imaging-exploratory-data-analysis/exercises/2-volume-mpr
[31]:https://github.com/robstraker/ai-for-healthcare-nanodegree-udacity/tree/master/course-2-3d-medical-imaging-data/lesson-3-3d-imaging-exploratory-data-analysis/exercises/3-dataset-eda
[32]:https://github.com/robstraker/ai-for-healthcare-nanodegree-udacity/tree/master/course-2-3d-medical-imaging-data/lesson-4-3d-imaging-end-to-end-deep-learning-applications/exercises/1-convolutions
[33]:https://github.com/robstraker/ai-for-healthcare-nanodegree-udacity/tree/master/course-2-3d-medical-imaging-data/lesson-4-3d-imaging-end-to-end-deep-learning-applications/exercises/2-segmentation-hands-on
[34]:https://github.com/robstraker/ai-for-healthcare-nanodegree-udacity/tree/master/course-2-3d-medical-imaging-data/lesson-4-3d-imaging-end-to-end-deep-learning-applications/exercises/3-performance-metrics
[35]:https://github.com/robstraker/ai-for-healthcare-nanodegree-udacity/tree/master/course-2-3d-medical-imaging-data/lesson-5-deploy-algorithms-in-real-world/exercises/1-sending-volumes
[36]:https://github.com/robstraker/ai-for-healthcare-nanodegree-udacity/tree/master/course-2-3d-medical-imaging-data/lesson-5-deploy-algorithms-in-real-world/exercises/2-segmenting-structures
[37]:https://github.com/robstraker/ai-for-healthcare-nanodegree-udacity/tree/master/course-2-3d-medical-imaging-data/lesson-5-deploy-algorithms-in-real-world/exercises/3-anonymization
[38]:https://github.com/robstraker/ai-for-healthcare-nanodegree-udacity/tree/master/course-3-ehr-data/lesson-2-EHR-Data-Security-and-Analysis/exercises
[39]:https://github.com/robstraker/ai-for-healthcare-nanodegree-udacity/tree/master/course-3-ehr-data/lesson-3-EHR-Code-Sets/exercises
[40]:https://github.com/robstraker/ai-for-healthcare-nanodegree-udacity/tree/master/course-3-ehr-data/lesson-4-EHR-Data-Transformations-and-Tensorflow-Feature-Engineering/exercises
[41]:https://github.com/robstraker/ai-for-healthcare-nanodegree-udacity/tree/master/course-3-ehr-data/lesson-5-Building-Evaluating-and-Interpreting-Models-for-Bias-and-Uncertainty/exercises
[42]:https://github.com/robstraker/ai-for-healthcare-nanodegree-udacity/tree/master/course-4-wearable-device-data/lesson-2-intro-to-dsp/exercises/1-plotting
[43]:https://github.com/robstraker/ai-for-healthcare-nanodegree-udacity/tree/master/course-4-wearable-device-data/lesson-2-intro-to-dsp/exercises/2-interpolation
[44]:https://github.com/robstraker/ai-for-healthcare-nanodegree-udacity/tree/master/course-4-wearable-device-data/lesson-2-intro-to-dsp/exercises/3-fourier-transform
[45]:https://github.com/robstraker/ai-for-healthcare-nanodegree-udacity/tree/master/course-4-wearable-device-data/lesson-2-intro-to-dsp/exercises/4-spectrograms
[46]:https://github.com/robstraker/ai-for-healthcare-nanodegree-udacity/tree/master/course-4-wearable-device-data/lesson-3-intro-to-sensors/exercises/1-step-cadence
[47]:https://github.com/robstraker/ai-for-healthcare-nanodegree-udacity/tree/master/course-4-wearable-device-data/lesson-3-intro-to-sensors/exercises/2-ppg-peaks
[48]:https://github.com/robstraker/ai-for-healthcare-nanodegree-udacity/tree/master/course-4-wearable-device-data/lesson-3-intro-to-sensors/exercises/3-ppg-snr
[49]:https://github.com/robstraker/ai-for-healthcare-nanodegree-udacity/tree/master/course-4-wearable-device-data/lesson-4-activity-classifier/exercises/1-data-exploration
[50]:https://github.com/robstraker/ai-for-healthcare-nanodegree-udacity/tree/master/course-4-wearable-device-data/lesson-4-activity-classifier/exercises/2-feature-extraction
[51]:https://github.com/robstraker/ai-for-healthcare-nanodegree-udacity/tree/master/course-4-wearable-device-data/lesson-4-activity-classifier/exercises/3-quirk-in-the-dataset
[52]:https://github.com/robstraker/ai-for-healthcare-nanodegree-udacity/tree/master/course-4-wearable-device-data/lesson-5-ecg-processing/exercises/1-pan-tompkins-algorithm
[53]:https://github.com/robstraker/ai-for-healthcare-nanodegree-udacity/tree/master/course-4-wearable-device-data/lesson-5-ecg-processing/exercises/2-af-features
[54]:https://github.com/robstraker/ai-for-healthcare-nanodegree-udacity/tree/master/course-4-wearable-device-data/lesson-5-ecg-processing/exercises/3-atrial-fibrillation

[//]: # (Links to Project Solutions)
[55]:https://github.com/robstraker/ai-for-healthcare-nanodegree-udacity/tree/master/course-1-2d-medical-imaging-data/project-pneumonia-detection-from-chest-x-rays
[56]:https://github.com/robstraker/ai-for-healthcare-nanodegree-udacity/tree/master/course-2-3d-medical-imaging-data/project-hippocampal-volume-quantification-in-alzheimers-progression
[57]:https://github.com/robstraker/ai-for-healthcare-nanodegree-udacity/tree/master/course-3-ehr-data/project-patient-selection-for-diabetes-drug-testing
[58]:https://github.com/robstraker/ai-for-healthcare-nanodegree-udacity/tree/master/course-4-wearable-device-data/project-motion-compensated-pulse-rate-estimation