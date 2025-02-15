This repository contains lyrical data, notebooks for analysis of song lyrics with audio features.

- Dataset has 1233 samples which is split in to 70:30 ratio for train and test validation.
- 40 MFCC values are extracted for each audio
- LSTM layers are used to finetune output from BERT based models
- BERT models provide 768 element sentence embeddings
- Fast text pretrained models provide 300 element sentence embeddings
- All fine tuning models are RNN models as feed forward network performance is very poor

## [Machine Learning Models](notebooks/Genre_classification_ML_Models.ipynb)
| Model | Type | Training F1 | Test F1 | Training Precision | Test Precision | Training Recall | Test Recall |
| -- | -- | -- | -- | -- | -- | -- | -- |
| 5 Fold SGD Classifier | Single Modal (Text) | 87% | 84% | 80% | 77% | 94% | 92% |
| 5 Fold Logistic Regression | Single Modal (Text) | 86% | 83% | 79% | 75% | 95% | 93% |
| 5 Fold Multi Nomial Naive Bayes Classification | Single Modal (Text) | 87% | 83% | 80% | 76% | 94% | 90% |
| 10 Fold SGD Classifier | Single Modal (Text) | 87% | 83% | 80% | 77% | 94% | 92% |
| 10 Fold Logistic Regression | Single Modal (Text) | 86% | 83% | 79% | 76% | 94% | 92% |
| 10 Fold Linear SVC | Single Modal (Text) | 87% | 83% | 83% | 77% | 92% | 88% |

## Fast Text Models
| Model | Type | Training F1 | Test F1 | Training Precision | Test Precision | Training Recall | Test Recall |
| -- | -- | -- | -- | -- | -- | -- | -- |
| [Fast Text Supervised Learning](notebooks/Fasttext%20Supervised%20Classification.ipynb) | Single Modal (Text) | 100% | 88% | 100% | 88% | 100% | 88% |
| [Fast Text Embeddings from Pretrained model](notebooks/FT_Vectors_and_MFCC_Fusion.ipynb) | Single Modal (Text) | 94.76% | 94.32% | 94.90% | 94.61% | 94.61% | 94.04% |
| [Fast Text Embeddings from Pretrained model](notebooks/FT_Vectors_and_MFCC_Fusion.ipynb) | Multi Modal (Text + MFCC) | 94.15% | 94.29% | 95.82% | 94.57% | 92.53% | 94.01% |
| [Fast Text Embeddings from Pretrained model](notebooks/FT_Vectors_and_MFCC_Fusion.ipynb) | Multi Modal (Text + MEL) | 92.61% | 94.04% | 92.54% | 93.49% | 92.68% | 94.61% |
| [Indic Fast Text Embeddings from Pretrained model](notebooks/Indic_FT_Simplified.ipynb) | Single Modal (Text) | 88.56% | 93.84% | 91.00% | 92.48% | 86.24% | 95.23% |
| [Indic Fast Text Embeddings from Pretrained model](notebooks/Indic_FT_Simplified.ipynb) | Multi Modal (Text + MFCC) | 91.14% | 93.52% | 92.33% | 92.44% | 89.98% | 94.64% |
| [Indic Fast Text Embeddings from Pretrained model](notebooks/Indic_FT_Simplified.ipynb) | Multi Modal (Text + MEL) | 87.45% | 93.80% | 90.15% | 92.98% | 84.90% | 94.64% |

## LSTM Model from Word Embeddings
| Model | Type | Training F1 | Test F1 | Training Precision | Test Precision | Training Recall | Test Recall |
| -- | -- | -- | -- | -- | -- | -- | -- |
| [LSTM Model](notebooks/LSTM_Simplified.ipynb) | Single Modal (Text) | 82.37% | 82.03% | 87.41% | 74.63% | 77.87% | 91.07% |
| [LSTM Model](notebooks/LSTM_Simplified.ipynb) | Multi Modal (Text + MFCC) | 81.58% | 82.17% | 89.34% | 82.92% | 75.07% | 81.43% |
| [LSTM Model](notebooks/LSTM_Simplified.ipynb) | Multi Modal (Text + MEL) | 82.91% | 81.30% | 87.14% | 81.06% | 79.07% | 81.54% |

## BERT based models with LSTM Layer fine tuning
| Model | Type | Training F1 | Test F1 | Training Precision | Test Precision | Training Recall | Test Recall |
| -- | -- | -- | -- | -- | -- | -- | -- |
| [Indic BERT Finetuning](notebooks/IndicBERT_Simplified.ipynb) | Single Modal (Text) | 96.62% | 85.07% | 94.82% | 80.74% | 98.5% | 89.88% |
| [Indic BERT Finetuning](notebooks/IndicBERT_Simplified.ipynb) | Multi Modal (Text + MFCC) | 95.77% | 84.70% | 95% | 83.72% | 96.56% | 85.71% |
| [Indic BERT Finetuning](notebooks/IndicBERT_Simplified.ipynb) | Multi Modal (Text + MEL) | 95.30% | 85.16% | 92.17% | 79.08% | 98.65% | 92.26% |
| [M-BERT Finetuning](notebooks/MBERT_Simplified.ipynb) | Single Modal (Text) | 87.79% | 86.82% | 86.52% | 86.82% | 89.10% | 86.82% |
| [M-BERT Finetuning](notebooks/MBERT_Simplified.ipynb) | Multi Modal (Text + MFCC) | 90.09% | 83.80% | 88.47% | 89.79% | 91.77% | 78.57% |
| [M-BERT Finetuning](notebooks/MBERT_Simplified.ipynb) | Multi Modal (Text + MEL) | 86.16% | 84.37% | 86.22% | 88.81% | 86.09% | 80.35% |
| [MURIL Finetuning](notebooks/MURIL_Simplified.ipynb) | Single Modal (Text) | 85.43% | 88.76% | 80.71% | 86.03% | 90.73% | 91.66% |
| [MURIL Finetuning](notebooks/MURIL_Simplified.ipynb) | Multi Modal (Text + MFCC) | 86.01% | 89.34% | 87.96% | 88.82% | 84.15% | 89.88% |
| [MURIL Finetuning](notebooks/MURIL_Simplified.ipynb) | Multi Modal (Text + MEL) | 84.55% | 90.54% | 86.54% | 87.29% | 82.66% | 94.04% |
| [XLM-R Finetuning](notebooks/XLMR_Simplified.ipynb) | Single Modal (Text) | 82.30% | 87.00% | 82.55% | 82.79% | 82.06% | 91.66% |
| [XLM-R Finetuning](notebooks/XLMR_Simplified.ipynb) | Multi Modal (Text + MFCC) | 83.93% | 86.40% | 80.80% | 77.88% | 87.31% | 97.00% |
| [XLM-R Finetuning](notebooks/XLMR_Simplified.ipynb) | Multi Modal (Text + MEL) | 85.17% | 88.15% | 85.30% | 82.05% | 85.05% | 95.23% |

