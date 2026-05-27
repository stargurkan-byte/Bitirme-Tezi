# ECG-ARRHYTHMIA-DETECTION
The code which can detect arrhythmia
`aritmi_tespiti.py`:The core processing pipeline containing the Pan-Tompkins algorithm, dynamic RR interval calculation, and rule-based pre-assessment logic.
* `mitbih_loader.py`: The data ingestion module responsible for parsing binary ECG signals (`.dat`) and physician annotations (`.atr`) from the MIT-BIH Arrhythmia Database, performing gain normalization and physical unit conversions.
* `test_evaluation.py`: The statistical validation engine that benchmarks the detected R-peaks against the clinical ground truth. It automatically computes performance metrics including True Positives (TP), False Positives (FP), Sensitivity (Se), and Positive Predictive Value (PPV) to evaluate algorithmic accuracy.
