
Project Title: Transformer-Based Prediction of Future Joint Angles using IMU and EMG Signals

#Project Description

This project predicts future knee joint angles using past IMU and EMG sensor data.
A Transformer-based Seq2Seq model is used to learn temporal patterns and generate multi-step future predictions.

The system includes:

* Synthetic data generation
* Data preprocessing and normalization
* Sliding window sequence creation
* Transformer encoder-decoder model with attention mechanisms
* Model training and evaluation
* Visualization of predictions


#Technologies Used

* Python
* PyTorch
* NumPy
* Pandas
* Matplotlib


#Model Architecture

The model is based on a Transformer architecture:

* Embedding Layer → Converts input features to model dimension
* Positional Encoding → Adds time-step information
* Encoder

  * Multi-head self-attention
  * Temporal attention
  * Feature attention

* Decoder

  * Masked self-attention
  * Cross-attention
* Output Layer → Predicts future joint angles


#Dataset

* Synthetic IMU + EMG signals are generated
* Noise and sensor dropout are added
* Target variable: Joint Angle



#Workflow

Data Generation → Preprocessing → Sliding Window → Train/Val/Test Split → Model Training → Evaluation → Visualization → Prediction Saving


#Evaluation Metrics

* MAE (Mean Absolute Error)
* RMSE (Root Mean Squared Error)


#Features

* Multi-step future prediction
* Attention-based temporal modeling
* Feature importance learning
* Real-time latency measurement
* Prediction visualization


#Limitations

* Uses synthetic dataset as of now
* Performance may vary on real-world data


#Future Work

* Using real IMU + EMG datasets
* Improving model architecture
* Deploy for real-time applications


#Execution Steps 

Step 1: Install dependencies

pip install torch numpy pandas matplotlib


Step 2: Generate dataset

python generate_data.py
This will create: data/raw/sensor_data.csv


Step 3: Run the main pipeline

python main.py

This will:

* Create sliding windows
* Train the model
* Evaluate performance
* Plot predictions
* Save outputs


#Output Files

After execution, these files will be generated:
data/processed/X.npy
data/processed/y.npy
data/processed/predictions.csv


#Output Details

predictions.csv contains:

  * sample index
  * time step
  * actual angle
  * predicted angle


#Output (Console)

Sampling Rate: 100 Hz
Past Window: 1280.00 ms
Future Start: 20.00 ms
Future Horizon: 600.00 ms
Saved windows: data\processed\X.npy, data\processed\y.npy

Experiment: IMU + EMG
Train/Val/Test samples: 567/121/123
Input features used: 10
Baseline MAE: 0.440387
Model MAE:    0.040065
Model RMSE:   0.052444

Per-horizon metrics:
Early Horizon: MAE=0.037147 | RMSE=0.045435
Mid Horizon: MAE=0.041854 | RMSE=0.053445
Late Horizon: MAE=0.041193 | RMSE=0.057710
Latency sample 0: 7.74 ms
Latency sample 1: 7.49 ms
Latency sample 2: 7.37 ms
Latency sample 3: 7.50 ms
Latency sample 4: 8.06 ms
Average latency: 7.64 ms
Median latency:  7.50 ms
Max latency:     8.06 ms
Predictions saved to data\processed\predictions_imu_plus_emg.csv

Gait-phase interpretation (predicted):
At 20 ms: for knee joint angle 56.65°, gait phase is Terminal Stance
At 30 ms: for knee joint angle 56.27°, gait phase is Terminal Stance
At 40 ms: for knee joint angle 56.17°, gait phase is Terminal Stance
At 50 ms: for knee joint angle 55.67°, gait phase is Terminal Stance
At 60 ms: for knee joint angle 55.68°, gait phase is Terminal Stance
At 70 ms: for knee joint angle 55.56°, gait phase is Terminal Stance
At 80 ms: for knee joint angle 55.32°, gait phase is Terminal Stance
At 90 ms: for knee joint angle 54.88°, gait phase is Terminal Stance
At 100 ms: for knee joint angle 54.97°, gait phase is Terminal Stance
At 110 ms: for knee joint angle 54.42°, gait phase is Terminal Stance
At 120 ms: for knee joint angle 53.83°, gait phase is Terminal Stance
At 130 ms: for knee joint angle 53.36°, gait phase is Terminal Stance
At 140 ms: for knee joint angle 52.93°, gait phase is Terminal Stance
At 150 ms: for knee joint angle 52.72°, gait phase is Terminal Stance
At 160 ms: for knee joint angle 52.48°, gait phase is Terminal Stance
At 170 ms: for knee joint angle 51.81°, gait phase is Terminal Stance
At 180 ms: for knee joint angle 51.09°, gait phase is Terminal Stance
At 190 ms: for knee joint angle 50.80°, gait phase is Terminal Stance
At 200 ms: for knee joint angle 49.78°, gait phase is Terminal Stance
At 210 ms: for knee joint angle 48.65°, gait phase is Terminal Stance
At 220 ms: for knee joint angle 47.65°, gait phase is Terminal Stance
At 230 ms: for knee joint angle 46.42°, gait phase is Terminal Stance
At 240 ms: for knee joint angle 45.32°, gait phase is Terminal Stance
At 250 ms: for knee joint angle 43.69°, gait phase is Mid Stance
At 260 ms: for knee joint angle 42.11°, gait phase is Mid Stance
At 270 ms: for knee joint angle 40.78°, gait phase is Mid Stance
At 280 ms: for knee joint angle 39.28°, gait phase is Mid Stance
At 290 ms: for knee joint angle 37.91°, gait phase is Mid Stance
At 300 ms: for knee joint angle 35.96°, gait phase is Mid Stance
At 310 ms: for knee joint angle 34.34°, gait phase is Mid Stance
At 320 ms: for knee joint angle 33.06°, gait phase is Mid Stance
At 330 ms: for knee joint angle 31.42°, gait phase is Mid Stance
At 340 ms: for knee joint angle 30.03°, gait phase is Mid Stance
At 350 ms: for knee joint angle 28.37°, gait phase is Initial Contact / Loading Response
At 360 ms: for knee joint angle 26.69°, gait phase is Initial Contact / Loading Response
At 370 ms: for knee joint angle 25.42°, gait phase is Initial Contact / Loading Response
At 380 ms: for knee joint angle 24.47°, gait phase is Initial Contact / Loading Response
At 390 ms: for knee joint angle 23.60°, gait phase is Initial Contact / Loading Response
At 400 ms: for knee joint angle 22.51°, gait phase is Initial Contact / Loading Response
At 410 ms: for knee joint angle 21.96°, gait phase is Initial Contact / Loading Response
At 420 ms: for knee joint angle 21.24°, gait phase is Initial Contact / Loading Response
At 430 ms: for knee joint angle 20.89°, gait phase is Initial Contact / Loading Response
At 440 ms: for knee joint angle 20.77°, gait phase is Initial Contact / Loading Response
At 450 ms: for knee joint angle 20.69°, gait phase is Initial Contact / Loading Response
At 460 ms: for knee joint angle 20.49°, gait phase is Initial Contact / Loading Response
At 470 ms: for knee joint angle 20.22°, gait phase is Initial Contact / Loading Response
At 480 ms: for knee joint angle 20.37°, gait phase is Initial Contact / Loading Response
At 490 ms: for knee joint angle 20.72°, gait phase is Initial Contact / Loading Response
At 500 ms: for knee joint angle 21.18°, gait phase is Initial Contact / Loading Response
At 510 ms: for knee joint angle 21.68°, gait phase is Initial Contact / Loading Response
At 520 ms: for knee joint angle 22.75°, gait phase is Initial Contact / Loading Response
At 530 ms: for knee joint angle 24.01°, gait phase is Initial Contact / Loading Response
At 540 ms: for knee joint angle 24.62°, gait phase is Initial Contact / Loading Response
At 550 ms: for knee joint angle 25.74°, gait phase is Initial Contact / Loading Response
At 560 ms: for knee joint angle 27.53°, gait phase is Initial Contact / Loading Response
At 570 ms: for knee joint angle 28.17°, gait phase is Initial Contact / Loading Response
At 580 ms: for knee joint angle 30.34°, gait phase is Terminal Stance
At 590 ms: for knee joint angle 32.15°, gait phase is Terminal Stance
At 600 ms: for knee joint angle 33.53°, gait phase is Terminal Stance
At 610 ms: for knee joint angle 36.07°, gait phase is Terminal Stance
Figure saved to data\processed\plots\imu_plus_emg\prediction_sample_0.png

Experiment Summary
     name  baseline_mae  model_mae  model_rmse  best_epoch  best_val_loss
IMU + EMG      0.440387   0.040065    0.052444          38       0.057289

#Conclusion

> This project uses a Transformer-based model to predict future joint angles from past IMU and EMG signals using attention mechanisms. The model captures temporal dependencies using attention instead of sequential processing, which improves prediction accuracy and allows parallel computation.

