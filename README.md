## 🦿 Transformer-Based Bilateral Knee Angle Prediction and Gait Phase Recognition

A deep learning project that predicts **future bilateral knee joint angles** and recognizes **gait phases** using wearable IMU sensor data.

The project uses a **Transformer-based Sequence-to-Sequence (Seq2Seq) model** to predict the flexion/extension angles of the **right and left knees** from historical lower-limb motion data.

The work was carried out during an internship at **Defence Bioengineering and Electromedical Laboratory (DEBEL), DRDO, Bengaluru**.

---

## ✨ Features

- 🦿 Bilateral right and left knee angle prediction
- 🔮 Multi-step future prediction
- ⏱️ 1.28-second historical input window
- 📈 0.60-second future prediction horizon
- 🧠 Transformer-based Seq2Seq architecture
- 🚶 Gait phase recognition
- 🏷️ Rule-based gait phase pseudo-labeling
- 📊 MAE and RMSE evaluation
- ⚡ Prediction latency measurement
- 📉 Actual vs predicted knee-angle visualization

---

## 🧠 Model

The project uses a **Transformer-based Sequence-to-Sequence (Seq2Seq) model**.

Historical IMU Features
        ↓
Feature Embedding
        ↓
Positional Encoding
        ↓
Transformer Encoder
        ↓
Transformer Decoder
        ↓
Future Bilateral Knee Angles
        ↓
Right Knee + Left Knee

## Model Configuration

| Parameter              |  Value |
| ---------------------- | -----: |
| Input timesteps        |    128 |
| Prediction timesteps   |     60 |
| Sampling rate          | 100 Hz |
| Embedding dimension    |     96 |
| Encoder layers         |      3 |
| Decoder layers         |      2 |
| Attention heads        |      4 |
| Feed-forward dimension |    192 |
| Dropout                |    0.1 |
| Optimizer              |  AdamW |
| Learning rate          | 0.0005 |
| Epochs                 |     20 |
| Loss                   |    MAE |

The input contains 128 historical timesteps (1.28 s) and the model predicts 60 future timesteps (0.60 s) for both knees.

🚶 Gait Phase Recognition

Gait phases are identified using rule-based pseudo-labeling based on knee angle and its temporal gradient.

The four gait phases are:

- Swing
- Initial Contact / Loading Response
- Stance
- Pre Swing

A Bidirectional GRU (BiGRU) classifier is used for gait phase recognition.

Knee Angle + Temporal Gradient
              ↓
   Rule-Based Pseudo Labels
              ↓
             BiGRU
              ↓
         Gait Phase

## 📂 Dataset

The dataset contains recordings from 25 healthy adult participants performing different locomotion activities.
| Property      | Details                                                        |
| ------------- | -------------------------------------------------------------- |
| Participants  | 25 healthy adults                                              |
| Sensor system | Xsens MVN Awinda                                               |
| IMUs          | 17 wireless IMUs                                               |
| Sampling rate | 100 Hz                                                         |
| Target        | Bilateral knee angles                                          |
| Activities    | Level-ground, inclined, declined walking, stair ascent/descent |

## Locomotion Activities
- Level-ground walking
- Inclined walking
- Declined walking
- Stair ascent
- Stair descent

The sensor system provides measurements including acceleration, angular velocity, orientation, and joint kinematics.

The original experimental dataset is not included in this repository.

## 📊 Results

The reported subject-wise regression results are:
| Metric       |            Result |
| ------------ | ----------------: |
| Overall MAE  | **17.54 ± 6.75°** |
| Overall RMSE | **21.79 ± 8.23°** |
| MAE 95% CI   |  **14.75–20.33°** |
| RMSE 95% CI  |  **18.38–25.20°** |

The evaluation uses five-fold subject-level cross-validation.

MAE and RMSE are the primary regression metrics, while gait phase accuracy is used as a complementary evaluation measure.

## 🚀 How to Run
1. Clone the repository
git clone https://github.com/Vismaya-2244/transformere-based-joint-angle-prediction.git
2. Navigate to the project
cd transformere-based-joint-angle-prediction
3. Install dependencies
pip install -r requirements.txt

On Windows: py -m pip install -r requirements.txt

4. Add the dataset
Place the authorized sensor data inside: data/raw/

Expected files include:
Sensor Free Acceleration.csv
segment_gyro.csv
Sensor Orientation - Euler.csv
Joint Angles XZY.csv
5. Run the project
python run_knee_gait.py

On Windows: py run_knee_gait.py

The pipeline performs preprocessing, sequence generation, model training, bilateral knee-angle prediction, gait phase recognition, evaluation, and visualization.

## 🛠️ Technologies
- Python
- PyTorch
- Transformer Networks
- GRU / BiGRU
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Xsens MVN Awinda

## 🔬 Applications

The project can support research in:
- Gait analysis
- Wearable sensing
- Biomechanics
- Rehabilitation engineering
- Lower-limb motion prediction
- Assistive technology
- Human movement analysis
This project is intended for research and educational purposes and is not a clinically validated medical system.

👩‍💻 Author

Vismaya S
Computer Science Engineering Graduate
Internship Project — DEBEL, DRDO, Bengaluru
