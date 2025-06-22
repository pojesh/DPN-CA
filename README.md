# DPN-CA: Improved Dual Path Network with Coordinate Attention for Multi-Class Image Classification

**DPN-CA** is a deep learning project that implements an enhanced **Dual Path Network (DPN92)** with **Coordinate Attention (CoordAtt)** modules and SiLU activation, designed for robust multi-class image classification. The model is built using TensorFlow/Keras and tested on an Indian Food Classification dataset (20 classes). It features modern architectural improvements for better feature representation and efficient training.

---

## 📂 Repository Structure

- `architecture.py` — Contains the implementation of the Improved DPN92 model with CoordAtt and SiLU.
- `DPN-CA-notebook.ipynb` — Jupyter Notebook with end-to-end training, evaluation, and visualization.
- `README.md` — Project overview and instructions (you’re reading it).

---

## 🚀 Model Highlights

- **Dual Path Block**: Combines residual and dense connections for rich feature propagation.
- **Coordinate Attention (CoordAtt)**: Captures long-range spatial dependencies with precise positional information.
- **SiLU (Swish) Activation**: Improves model expressiveness and convergence.
- **Cosine Learning Rate Decay**: Smooth training and fine-tuned learning rate schedule.
- **Modern Regularization**: L2 weight decay, BatchNorm.

---

## 🗂 Dataset

- **Indian Food Classification** dataset
  - 20 food categories
  - 6269 images used
  - Input size: `(224, 224, 3)`

**Example Classes**:  
`burger`, `butter_naan`, `chai`, `chapati`, `dal_makhani`, `fried_rice`, `idli`, `jalebi`, `kadai_paneer`, `pizza`, `samosa`, and more.

---

## 📈 Training Pipeline

- **Batch size**: 16
- **Epochs**: 100
- **Loss**: Sparse Categorical Crossentropy
- **Optimizer**: Adam with cosine learning rate schedule
- **Augmentations**:
  - Random Rotation
  - Random Flip (horizontal)
  - Random Zoom
  - Random Contrast

---

## 🏆 Results

- Achieved **~60%+ top-1 validation accuracy** on 20-class dataset after 50 epochs.
- Excellent feature interpretability with CoordAtt.
- Robust training stability and gradual convergence.

---

## ⚙️ How to Run

1️⃣ Clone this repository:

```bash
git clone https://github.com/your-repo/DPN-CA.git
cd DPN-CA
```
2️⃣ Install dependencies:

```bash
pip install tensorflow matplotlib scikit-learn pillow opencv-python
```

3️⃣ Run training (in Jupyter Notebook):

```bash
jupyter notebook DPN-CA-notebook.ipynb
```

4️⃣ Run from script:

```python
python architecture.py
```

# Integrate with your own training pipeline
💻 Requirements
Python 3.8+

TensorFlow 2.8+ (tested on GPU with CUDA)

Jupyter Notebook

Basic ML libraries (scikit-learn, matplotlib)

📚 References
Dual Path Networks: Chen et al.

Coordinate Attention: Hou et al., CVPR 2021

Swish/SiLU Activation: Google Brain

🚧 TODOs / Improvements
 Add transfer learning support with pre-trained weights.

 Improve generalization with stronger augmentation (CutMix, MixUp).

 Hyperparameter tuning (learning rate warmup, weight decay scheduling).

 Add multi-GPU training (tf.distribute).

 Add model export to ONNX/TF Lite.

📝 License
This project is released under the MIT License.
