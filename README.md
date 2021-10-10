This repository contains lyrical data, notebooks for analysis of song lyrics with audio features.

- Dataset has 1233 samples which is split in to 70:30 ratio for train and test validation.
- 40 MFCC values are extracted for each audio
- LSTM layers are used to finetune output from BERT based models
- Adding MFCC bought instability in training process
- IndicBERT and MBERT are more stable compared to XLM-R and MURIL
- BERT models provide 768 element sentence embeddings
- Fast text pretrained models provide 300 element sentence embeddings


Following models are experimented:
| Model | Type | Training F1 | Test F1 | Training Precision | Test Precision | Training Recall | Test Recall |
| -- | -- | -- | -- | -- | -- | -- | -- |
| 5 Fold SGD Classifier | Single Modal (Text) | 87% | 84% | 80% | 77% | 94% | 92% |
| 5 Fold Logistic Regression | Single Modal (Text) | 86% | 83% | 79% | 75% | 95% | 93% |
| 5 Fold Multi Nomial Naive Bayes Classification | Single Modal (Text) | 87% | 83% | 80% | 76% | 94% | 90% |
| 10 Fold SGD Classifier | Single Modal (Text) | 87% | 83% | 80% | 77% | 94% | 92% |
| 10 Fold Logistic Regression | Single Modal (Text) | 86% | 83% | 79% | 76% | 94% | 92% |
| 10 Fold Linear SVC | Single Modal (Text) | 87% | 83% | 83% | 77% | 92% | 88% |
| Fast Text | Single Modal (Text) | 100% | 88% | 100% | 88% | 100% | 88% |
| Fast Text Embeddings from Pretrained model * | Single Modal (Text) | 80% | 80% | 80% | 80% | 90% | 90% |
| Fast Text Embeddings from Pretrained model * | Multi Modal (Text + MFCC) | 80% | 80% | 70% | 70% | 100% | 100% |
| Indic Fast Text Embeddings from Pretrained model * | Single Modal (Text) | 80% | 80% | 80% | 80% | 90% | 90% |
| Indic Fast Text Embeddings from Pretrained model * | Multi Modal (Text + MFCC) | 80% | 80% | 70% | 70% | 100% | 100% |
| LSTM Model * | Single Modal (Text) | 86% | 80% | 92% | 79% | 82% | 82% |
| LSTM Model * | Multi Modal (Text + MFCC) | 85% | 80% | 90% | 80% | 79% | 80% |
| Indic BERT Finetuning * | Single Modal (Text) | 90% | 82% | 92% | 78% | 85% | 90% |
| Indic BERT Finetuning * | Multi Modal (Text + MFCC) | 83% | 82% | 85% | 81% | 75% | 83% |
| M-BERT Finetuning * | Single Modal (Text) | 84% | 84% | 90% | 90% | 92% | 84% |
| M-BERT Finetuning * | Multi Modal (Text + MFCC) | 84% | 84% | 90% | 90% | 84% | 83% |
| MURIL Finetuning * | Single Modal (Text) | 85% | 85% | 90% | 80% | 85% | 95% |
| MURIL Finetuning * | Multi Modal (Text + MFCC) | 70% | 85% | 70% | 80% | 75% | 95% |
| XLM-R Finetuning * | Single Modal (Text) | 83% | 83% | 90% | 81% | 80% | 92% |
| XLM-R Finetuning * | Multi Modal (Text + MFCC) | 83% | 85% | 80% | 80% | 95% | 85% |

TODO: * Results are approximately noted from plots. Accurate results to be obtained 
TODO: Run Multi modal experiments with MEL spectrograms
