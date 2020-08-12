# AI for Healthcare Nanodegree Topics

## Topics for Course 1: Applying AI to 2D Medical Imaging Data
Introduction to AI for 2D Medical Imaging:
- Medical Imaging: History of AI.
- Imaging Industry: Key Stakeholders (Clinical, Industry, Regulatory). 

Clinical Foundations for 2D Medical Imaging:
- Medical Imaging: Tools (X-ray, Computed Tomography (CT), Magnetic Resonance Imaging (MRI), Ultrasound), 2D Imaging, 3D Imaging, and Medical Imaging Workflows (PACS, Diagnostic, Screening).
- Imaging Industry: Medical Imaging Clinicians (Radiologists, Diagnosing Clinicians, Pathologists) and Regulatory Landscape.

2D Medical Imaging Exploratory Data Analysis:
- Imaging Standards: DICOM Studies, Series, and Components (Header, Image).
- Data Exploration: Histograms, Scatterplots, Pearson Correlation Coefficient, and Co-Occurrence Matrices.

Classification Models of 2D Medical Images:
- Image Pre-Processing: Otsu's Method, Intensity Normalization (Zero-Meaning, Standardization), Image Augmentation, and Image Resizing.
- Dataset Preparation: Dataset Splitting, Training Dataset, Validation Dataset, Ground Truth, Gold Standard, and Silver Standard.
- Modeling: Convolution Neural Network (CNN), U-Net, 2D Imaging Algorithms (Classification, Segmentation, Localization), Model Training, Model Fine-Tuning (Batch Size, Epoch, Learning Rate, Overfitting), and Model Evaluation (Loss, Accuracy, Overfitting). 
- Model Performance: Confusion Matrix and Performance Metrics (Sensitivity, Specificity, Dice Coefficient).

Translating AI Algorithms for Clinical Settings with the FDA:
- Clinical Performance: False Positives, False Negatives, Precision, Recall, Precision-Recall Curve, Threshold, and F1 Score.
- FDA Approval: FDA Regulatory Process (Class I, Class II, Class III Device), Intended Use, Clinical Impact, and FDA Validation Plan.

## Topics for Course 2: Applying AI to 3D Medical Imaging Data
Introduction to AI for 3D Medical Imaging:
- Medical Imaging - 3D Imaging Historical Context.

3D Medical Imaging - Clinical Fundamentals:
- Medical Imaging: 3D Modalities (CT, MRI, SPECT, Ultrasound), Contrast Resolution, and Spatial Resolution.
- Diagnostic Performance: Likelihood Ratio, Sensitivity, Specificity, and Bayes Theorem.
- Technology: CT Scanners (X-Rays, Sinograms) and MR Scanners (Gradient Fields, RF Pulses, K-space Data).
- Imaging Tasks: Windowing, Multi-planar Reconstruction (MPR), 3D Reconstruction, Registration, and Volumetric Rendering.

3D Medical Imaging Exploratory Data Analysis:
- Standards: DICOM (Patient, Study, Series, Instance, Service-Object pair (SOP), Data Element, Value Representation (VR), Data Element Type, DICOM Informational Object, Information Object Definition (IOD)) and NIFTI.
- Viewers: MicroDicom, 3D Slicer, and Other (Radiant, Asirix, ViewMyScans)
- Image Parameters: Orientation (Image Orientation, Image Position), Physical Spacing (Pixel, Slice), Photometric, Image Size, DICOM Volume Dimensions, and Physical Dimensions.
- DICOM Volume EDA: Voxel Spacing, Data Ranges, Clinical Anomalies, Informatics Anomalies, 

3D Medical Imaging - End-to-End Deep Learning Applications:
- Applications: AI, Machine Learning, Deep Learning, Computer Vision, and Convolutional Neural Networks (CNNs).
- Modeling: Classification, Object Detection, Feature Extraction (2D/2.5D/3D Convolution), Segmentation Architecture (Convolution Encoder-Decoder, U-net), and Segmentation Ground Truth.
- Segmentation Performance: Sensitivity, Specificity, Dice Similarity Coefficient, Jaccard Index, and Hausdorff Distance.
- Clinical Performance: Confusion Matrix adn Likelihood Ratio.

Deploying AI Algorithms in Real World Scenarios:
- DICOM Networking: OSI Model, DICOM Message Service Element (DIMSE), DICOM Web, Application Entity (AE) and Scripting.
- Clinical Networks: Picture Archiving and Communications System (PACS), Vendor Neutral Archive (VNA), Electronic Health Record (EHR), and Radiology Information System (RIS). 
- Standards: Health Level 7 (HL7) and Fast Healthcare Interoperability Resources (FHIR).
- Tools: Open Health Imaging Foundation (OHIF), 3D Slicer, and Other (Cornerstone, DCMTK, Orthanc, Radiant).
- Regulatory: FDA Process, FDA Validation Plan, HIPAA, GDPR, and Anonymization.

## Topics for Course 3: Applying AI to EHR Data
Applying AI to EHR Data:
- Context: Health Record, Electronic Health Record (EHR), HIPAA, and Electronic Medical Record (EMR).

EHR Data Security and Analysis:
- Data Security & Privacy: Regulations (HIPAA, HITECH, GDPR, DPA, PHI), Covered Entities (Payers, Providers, Clearinghouses), PHI Access (De-identifying, Expert Determination, Safe Harbor, Limited Latitude).
- EDA: CRISP-DM, Data Leakage, Model Objectives, Dataset Schema Analysis, Feature Identification (Predictor, Categorical, Numerical), Distributions (Normal/Gaussian, Uniform, Skewed, Bimodal, Poisson), Missing/Null Values (MCAR, MAR, MNAR), and Cardinality. 

EHR Code Sets:
- Context: Medical Encounter, Inpatient vs. Outpatient, Diagnosis, and Key Points.
- Diagnosis Codes: Prioritization (Primary, Principal, Secondary), International Classification of Diseases 10 - Clinical Modification (ICD10-CM), vs. ICD9-CM, and ICD10-CM Code Structure.
- Procecure Codes: ICD10 Procedure Coding Systems (ICD10-PCS), Current Procedural Terminology (CPT), and Healthcare Common Procedure Coding System (HCPCS).
- Medication Codes: National Drug Coce (NDC), RXNorm, and Crosswalk.
- Grouping/Categorizing: Clinical Classification Software (CCS).
- Other Categorization Systems: Medicate Severity-Diagonisis Related Group (MS-DRG) and Systematized Nomenclature of Medicine - Clinical Terms (SNOMED-CT).

EHR Transformations & Feature Engineering:
- EHR Dataset Levels: Line Level, Encounter Level, and Longitudinal Level.
- Dataset Splitting: Representative Spitting, Test and Validation Datasets, and Patient Level.
- Feature Engineering: ETL with Dataset API and TF Feature Column API (Cross Features, Shared Embeddings, Normalizer Function, Numerical Features, Categorical Features).

Building, Evaluating, and Interpreting Models:
- TensorFlow: DenseFeatures (Numeric, Embedding, Bucketized, Indicator Columns)
- Model Evaluation: Classification Metrics (Receiver Operating Characteristic (ROC), Area Under ROC Curve (AUC), F1, Precision, Recall, Brier Score) and Regression Metrics (Root Mean Square Error (RMSE), Mean Absolute Percentage Error (MAPE), Mean Absolute Error (MAE)).
- Demographic Bias Analysis: Unintended Bias, Aequitas, Population, Clinical Trial, Group Bias Analysis, and Fairness Disparity Analysis.
- Uncertainty Estimation: Tensorflow Probability, Bayesian Probability (Prior Distribution, Posterior Probability, Evidence), Uncertainty Estimation, and Uncertainty Types (Aleatoric/Statistical, Epistemic/Systematic).
- Model Interpretability: Black Box, Model Agnostic Methods, Local Interpretable Model-Agnostic Explanations (LIME), Shapley Values, and Integrated Gradients.

## Topics for Course 4: Applying AI to Wearable Device Data
Introduction to Wearable Data:
- Papers: Apple Heart Study (NEJM), Apple Heart Study Response (NEJM), Framingham Study (Stroke), and Wearable Health (JIP).
- Classification Studies: Inclusion Criteria, Exclusion Criteria, Classification Accuracy, Precision, Recall, and Primary Endpoint.

Intro to Digial Sampling & Signal Processing:
- Signals: Sinusoid, Period, Frequency, DC Level, AC Level, Hertz, and Phase Shift. 
- Sampling: Analog Signal, Digital Signal, ADC Converter, Transducers, Bit-Depth, Noise Floor, Dynamic Range, and Sampling Rate. 
- Plotting: Time-Domain, Frequency-Domain, and Packages (Matplotlib, Seaborn, Altair, Plotly).
- Processing: Interpolation (Linear Interpolation, Resampling), Fourier Transform (Frequency Component, Stationarity, Bandwidth, Nyquist Frequency, Aliasing, Passband, Bandpass Filter), Spectrogram (Short-Term Fourier Transform, Quantum Uncertainty Principle, Wavelet Transform), and Harmonics.

Introduction to Sensors:
- Inertial Measurement Unit (IMU): Accelerometer, Gyroscope, Magnetometer, G-Force, Piezoelectric Crystal, Vector Magnitude, and Gait Cycle Segmentation.
- Photoplethysmogram (PPG): Photodetector, Cardiac Cycle (Systole, Diastole), PPG Cycle (Peak, Trough), Noise Sources (Melanin, Arm Motion, Arm Position, Finger Motion, Sensor Displacement, Ambient Light), and Signal-To-Noise Ratio (SNR).
- Electrocardiograms (ECG or EKG): Electrode, Lead, and Holter Monitors.

Activity Classification:
- Algorithm Development Process: Understand Your Data, Understand the Literature, Build Features, Build Model, Optimize Hyperparameters, and Evaluate Model.
- Model Building: Decision Trees and Random Forest Classifier.
- Model Optimization: Hyperparameter Tuning, Regularization, Confusion Matrix, Nested Cross-Validation, and Classification Accuracy, 

ECG Signal Processing:
- Heart Physiology: Atria, Ventricles, Cardiac Conduction (Sinus Node Impulse Generation, Atrial Depolarization, Atrioventricular (AV) Node Impulse Conduction, Ventricular Depolarization, Atrial Repolarization, Ventricular Repolarization), and Cardiac Conduction Sequence (P-wave, Q-wave, T-wave, S-wave).
- QRS Complex Detection: Heart Rate, Heart Rate Variability, Basic Pan-Tompkins Algorithm (Bandpass Filter, 1-Sample Difference, Square, Moving Sum, Peak Detection, Thresholding), and Extending Pan-Tompkins (Refractory Period Blanking, Adaptive Thresholding, T-Wave Rejection).
- Atrial Fibrillation Physiology: Arrhythmia, Sinus Rhythm, Atrial Fibrillation, and Inter-Beat (RR) Interval.
- Arrhythmia Detection: Normal Sinus Rhythm, Atrial Fibrillation, Other Rhythm, Computing in Cardiology Challenge 2017, Data Exploration, Feature Extraction, Modelling

## Technologies

General Tools and Libraries:
- Jupyter Notebooks (Web Application)
- Matplotlib (2D Plotting Library)
- NumPy (Array Computing Library)
- pandas (Data Analysis Library)
- Pillow (Imaging Library)
- seaborn (Statistical Data Visualization Library)
- scikit-image (Image Processing Algorithms)
- SciPy (Scientific Computing Library)
- Shapley (Geometric Analysis Package)

AI Frameworks and Libraries:
- Aequitas (Machine Learning Bias Audit Toolkit)
- Keras (Neural Network Library)
- PyTorch (Machine Learning Framework)
- scikit-learn (Machine Learning Library)
- TensorBoard (TensorFlow Application Library)
- TensorFlow (Machine Learning Library)

Medical Imaging Tools and Libraries:
- 3D Slicer (Medical Imaging Visualization)
- Dcmtk (DICOM Toolkit Library)
- Microdicom (DICOM Processing Application)
- NiBabel (Neuroimaging Processing Package)
- OHIF (Medical Imaging Viewer)
- Pydicom (DICOM Processing Package)
- Radiant (PACS DICOM Viewer)

Deep Neural Network Models:
- U-Net (Convolutional Networks for Biomedical Image Segmentation)
- VGG-16 (Convolutional Network for Classification and Detection)

Databases:
- NIH Chest X-Ray Dataset
- PhysioNet (Complex Physiologic Signals)
- UCI Heart Disease Data Set (Cleveland, Hungary, Switzerland, VA Long Beach)

Programming Languages:
- Python
