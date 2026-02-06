# **Sarcasm Detection in Reddit Comments (TF‑IDF + Logistic Regression)**  


In this project I built an end-to-end **sarcasm classifier** for Reddit comments using **text only**.   
The goal is to build a **strong, interpretable baseline** with clean ML engineering practices (leakage-safe pipelines, cross-validation, tuning, and diagnostics).


## **Highlights**

- ✅ **Leakage-aware evaluation**: removes explicit sarcasm markers (e.g., `/s`, `#sarcasm`) *before* train/test splitting  
- ✅ **Strong linear baseline**: TF-IDF (uni/bi-grams) + Logistic Regression  
- ✅ **Proper validation**: stratified split + stratified K-fold CV + GridSearchCV  
- ✅ **Diagnostics**: validation curve (regularization) + learning curve (data vs capacity)  
- ✅ **Explainability**: top coefficients + odds ratios  
- ✅ **Error analysis**: high-confidence false positives/negatives + lightweight failure-mode signals


## **Full write-up**

- 📓 **Project notebook:** `sarcasm_detection_detailed.ipynb`  
- 📝 **Medium article:** 
 

At the end of the Medium article, I link back to this repo for full reproducibility.


## **Dataset**

This project uses the dataset from the paper:

**Khodak et al. (2017), “A Large Self-Annotated Corpus for Sarcasm”** containing 1M+ Reddit comments labeled sarcastic vs non-sarcastic.

- Paper: https://arxiv.org/abs/1704.05579  
- Kaggle (processed version): https://www.kaggle.com/danofer/sarcasm  
- File used: `train-balanced-sarcasm.csv`

> ⚠️ **Content warning:** Reddit text can include profanity and offensive language.  
> Examples shown are for analysis only.


## **Problem statement**

Given a comment `text`, predict:

- `label = 1` → sarcastic  
- `label = 0` → non-sarcastic

Sarcasm is challenging because it is often **context-dependent** (requires parent comment, thread history, or world knowledge).


## **Method (what the notebook actually does)**

### **1) Leakage guard (important)**  
Many sarcastic comments explicitly include markers like `/s` or `#sarcasm`.  
If we don’t remove them, the model can “cheat”.

✅ We strip common sarcasm markers **before** splitting so evaluation reflects real generalization.

### **2) Baseline model**
- **TF-IDF Vectorizer** (unigrams)
- **Logistic Regression** (L2 regularization, `saga` solver)

### **3) Focused tuning (GridSearchCV)**
We tune only the highest-impact knobs:

- `ngram_range`: `(1,1)` vs `(1,2)`  
- `min_df`: vocabulary filtering  
- `C`: inverse regularization strength  
- `sublinear_tf`: log-scaled term frequency  

### **4) Generalization diagnostics**
- **Validation curve** over `C` (bias–variance behavior)
- **Learning curve** (is performance data-limited or capacity-limited?)

### **5) Interpretability + error analysis**
- Top positive/negative **coefficients** + **odds ratios**
- Inspect **highest-confidence** false positives/negatives to identify failure modes



## **Results**

Final tuned model on the **held-out test set**:

- **F1 ≈ 0.71**
- **ROC AUC ≈ 0.79**
- **PR AUC ≈ 0.80**
- **Accuracy ≈ 0.72**

Compared to a unigram baseline, tuning (especially **bigrams + regularization**) improves both:
- **thresholded metrics** (F1/accuracy)
- and **ranking quality** (ROC AUC / PR AUC)


## **Repository structure**

```text
.
├── README.md
├── requirements.txt
├── sarcasm_detection_detailed.ipynb
└── data/
    └── train-balanced-sarcasm.csv   # not committed

```

## **Setup**

### **1) Create a virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows
```

### **2) Install dependencies**

```bash
pip install -r requirements.txt
```

## **How to run**

1. Download `train-balanced-sarcasm.csv` from Kaggle
2. Place it at: `./data/train-balanced-sarcasm.csv`
3. Launch Jupyter:

```bash
jupyter notebook sarcasm_detection_logreg_polished_detailed.ipynb
```

4. Run the notebook top-to-bottom


> The notebook includes a `CANDIDATE_PATHS` list so you can keep the dataset anywhere and still run smoothly.


## **Reproducibility notes**

* Fixed seeds via `random_state`
* Stratified split (`train_test_split(..., stratify=y)`)
* Stratified CV (`StratifiedKFold(shuffle=True)`)
* All preprocessing + modeling inside a single `Pipeline` to avoid leakage


## **Limitations & ethical considerations**

* **Context dependence:** Many sarcastic comments require the parent comment or thread to interpret correctly.
* **Domain shift:** Sarcasm cues vary across subreddits and communities.
* **False positives/negatives:** The cost depends on the application (moderation vs analytics).
* **Bias risks:** Misclassification can amplify bias if certain dialects/communities are over-flagged. Avoid using sarcasm classifiers as fully automated decision makers.

## **Next steps**

If extending this project, strong directions include:

1. Add **parent-comment context** (thread-aware modeling)
2. Incorporate lightweight metadata (subreddit, author history) with leakage checks
3. Compare against transformer baselines (e.g., DistilBERT)
4. Threshold tuning and calibrate probabilities for deployment settings


## **Citation**

Khodak, M., Saunshi, N., & Vodrahalli, K. (2017).
*A Large Self‑Annotated Corpus for Sarcasm.* [https://arxiv.org/abs/1704.05579](https://arxiv.org/abs/1704.05579)


