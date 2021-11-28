Each song is annotated with its Raga name.

### Audio pre-processing
- Each audio is clipped at beginning by a duration of "Offset" seconds
- Audio is resampled to 16KHz
- Stereo audio converted to mono channel audio
- Audio is clipped to 120 seconds

### Audio Feature Extraction
- Mel Spectrogram with 40 Mels for 120 sec duration
- 40 MFCC Mean values
- 40 MFCC Values for 120 sec duration 
- Chromagram Values

### Model Strategy
- 5 fold cross validation performed on all models except for Fast Text Model

## Raga classification using Lyrical features
| Model | Type | Training F1 | Test F1 | Training Precision | Test Precision | Training Recall | Test Recall |
| -- | -- | -- | -- | -- | -- | -- | -- |
| [Fast Text Supervised](Lyric_FT_Raga_Supervised.ipynb) | Single Modal | 79.44% | 10.81% | 79.44% | 10.81% | 79.44% | 10.81% |
| [Random Forest Classifier](Lyric_Raga_Identification.ipynb) | Single Modal | 99.77% | 11.83% | 99.77% | 11.83% | 99.77% | 11.83% |
| [Multi Nomial Naive Bayes Classification](Lyric_Raga_Identification.ipynb) | Single Modal | 70.05% | 11.51% | 70.05% | 11.51% | 70.05% | 11.51% |
| [Logistic Regression](Lyric_Raga_Identification.ipynb) | Single Modal | 49.81% | 11.27% | 49.81% | 11.27% | 49.81% | 11.27% |
| [Bernoulli Naive Bayes Classification](Lyric_Raga_Identification.ipynb) | Single Modal | 99.43% | 11.10% | 99.43% | 11.10% | 99.43% | 11.10% |
| [SGD Classifier](Lyric_Raga_Identification.ipynb) | Single Modal | 44.08% | 10.78% | 44.08% | 10.78% | 44.08% | 10.78% |
| [Nearest Centroid Classification](Lyric_Raga_Identification.ipynb) | Single Modal | 91.58% | 10.21% | 91.58% | 10.21% | 91.58% | 10.21% |
| [Linear SVC](Lyric_Raga_Identification.ipynb) | Single Modal | 99.93% | 10.05% | 99.93% | 10.05% | 99.93% | 10.05% | 
| [Ridge Classifier](Lyric_Raga_Identification.ipynb) | Single Modal | 100% | 9.56% | 100% | 9.56% | 100% | 9.56% |
| [XGB Classifier](Lyric_Raga_Identification.ipynb) | Single Modal | 98.72% | 9.15% | 98.72% | 9.15% | 98.72% | 9.15% |
| [Complement Naive Bayes Classification](Lyric_Raga_Identification.ipynb) | Single Modal | 99.22% | 8.35% | 99.22% | 8.35% | 99.22% | 8.35% |

## Raga Classification using MFCC Mean Levels
| Model | Type | Training F1 | Test F1 | Training Precision | Test Precision | Training Recall | Test Recall |
| -- | -- | -- | -- | -- | -- | -- | -- |
| [Feed Forward 10% Dropout](Raga_Analysis_Experiments.ipynb) | Single Modal | 90.67% | 7.45% | 90.67% | 7.45% | 90.67% | 7.45% |
| [Feed Forward 25% Dropout](Raga_Analysis_Experiments.ipynb) | Single Modal | 81.90% | 7.45% | 81.90% | 7.45% | 81.90% | 7.45% |
| [Feed Forward 40% Dropout](Raga_Analysis_Experiments.ipynb) | Single Modal | 38.37% | 6.14% | 38.37% | 6.14% | 38.37% | 6.14% |
| | | 67.32% | 4.82% | 67.32% | 4.82% | 67.32% | 4.82% |
| [Feed Forward 10% Dropout](Raga_Analysis_Audio_and_Text_Fusion.ipynb) | Multi Modal (FT Vectors) | 99.56% | 11.40% | 99.56% | 11.40% | 99.56% | 11.40% |
| [Multi Nomial Naive Bayes Classification](Raga_Analysis_Experiments.ipynb) | Single Modal | 10.92% | 9.12% | 10.92% | 9.12% | 10.92% | 9.12% |
| [Bernoulli Naive Bayes Classification](Raga_Analysis_Experiments.ipynb) | Single Modal | 12.39% | 8.42% | 12.39% | 8.42% | 12.39% | 8.42% |
| [Random Forest Classifier](Raga_Analysis_Experiments.ipynb) | Single Modal | 98.85% | 8.33% | 98.85% | 8.33% | 98.85% | 8.33% |
| [Ridge Classifier](Raga_Analysis_Experiments.ipynb) | Single Modal | 19.47% | 7.45% | 19.47% | 7.45% | 19.47% | 7.45% |
| [Logistic Regression](Raga_Analysis_Experiments.ipynb) | Single Modal | 17.23% | 7.36% | 17.23% | 7.36% | 17.23% | 7.36% |
| [Linear SVC](Raga_Analysis_Experiments.ipynb) | Single Modal | 22.19% | 6.67% | 22.19% | 6.67% | 22.19% | 6.67% |
| [XGB Classifier](Raga_Analysis_Experiments.ipynb) | Single Modal | 99.75% | 6.49% | 99.75% | 6.49% | 99.75% | 6.49% |
| [Complement Naive Bayes Classification](Raga_Analysis_Experiments.ipynb) | Single Modal | 14.47% | 5.78% | 14.47% | 5.78% | 14.47% | 5.78% |
| [SGD Classifier | Single Modal](Raga_Analysis_Experiments.ipynb) | 10.78% | 4.82% | 10.78% | 4.82% | 10.78% | 4.82% |
| [Nearest Centroid Classification](Raga_Analysis_Experiments.ipynb) | Single Modal | 11.90% | 2.63% | 11.90% | 2.63% | 11.90% | 2.63% |

## Raga Classification using MFCC Matrix for 120sec duration
| Model | Type | Training F1 | Test F1 | Training Precision | Test Precision | Training Recall | Test Recall |
| -- | -- | -- | -- | -- | -- | -- | -- |
| [Feed Forward 10% Dropout](Raga_Analysis_Experiments.ipynb) | Single Modal | 99.89% | 9.69% | 99.89% | 9.69% | 99.89% | 9.69% |
| [Feed Forward 10% Dropout](Raga_Analysis_Audio_and_Text_Fusion.ipynb) | Multi Modal (FT Vectors) | 9.87% | 8.77% | 9.87% | 8.77% | 9.87% | 8.77% |
| [Feed Forward 25% Dropout](Raga_Analysis_Experiments.ipynb) | Single Modal | 66.88% | 7.92% | 66.88% | 7.92% | 66.88% | 7.92% |
| | | 98.02% | 7.92% | 98.02% | 7.92% | 98.02% | 7.92% |
| [Feed Forward 40% Dropout](Raga_Analysis_Experiments.ipynb) | Single Modal | 95.83% | 7.92% | 95.83% | 7.92% | 95.83% | 7.92% |
| [2 Layer LSTM with 10% Dropout](Raga_Analysis_Experiments.ipynb) | Single Modal | 74.86% | 7.89% | 74.86% | 7.89% | 74.86% | 7.89% |
| [2 Layer LSTM with 10% Dropout](Raga_Analysis_Experiments.ipynb) | Multi Modal (FT Vectors) | 44.78% | 7.89% | 44.78% | 7.89% | 44.78% | 7.89% |
| [2 Layer LSTM with 25% Dropout](Raga_Analysis_Experiments.ipynb) | Single Modal | 53.78% | 6.14% | 53.78% | 6.14% | 53.78% | 6.14% |
| [2 Layer LSTM with 40% Dropout](Raga_Analysis_Experiments.ipynb) | Single Modal | 31.50% | 5.70% | 31.50% | 5.70% | 31.50% | 5.70% |
| | | 47.03% | 4.40% | 47.03% | 4.40% | 47.03% | 4.40% |

## Raga Classification using MEL Spectrogram for 120sec duration
| Model | Type | Training F1 | Test F1 | Training Precision | Test Precision | Training Recall | Test Recall |
| -- | -- | -- | -- | -- | -- | -- | -- |
| [Feed Forward 10% Dropout](Raga_Analysis_Experiments.ipynb) | Single Modal | 89.79% | 7.89% | 89.79% | 7.89% | 89.79% | 7.89% |
| | | 99.67% | 7.45% | 99.67% | 7.45% | 99.67% | 7.45% |
| [Feed Forward 25% Dropout](Raga_Analysis_Experiments.ipynb) | Single Modal | 98.68% | 8.33% | 98.68% | 8.33% | 98.68% | 8.33% |
| [Feed Forward 40% Dropout](Raga_Analysis_Experiments.ipynb) | Single Modal | 88.15% | 9.25% | 88.15% | 9.25% | 88.15% | 9.25% |
| [2 Layer LSTM with 10% Dropout](Raga_Analysis_Experiments.ipynb) | Single Modal | 70.80% | 6.57% | 70.80% | 6.57% | 70.80% | 6.57% |
| [2 Layer LSTM with 25% Dropout](Raga_Analysis_Experiments.ipynb) | Single Modal | 50.71% | 6.14% | 50.71% | 6.14% | 50.71% | 6.14% |
| [2 Layer LSTM with 40% Dropout](Raga_Analysis_Experiments.ipynb) | Single Modal | 26.86% | 5.28% | 26.86% | 5.28% | 26.86% | 5.28% |

## Raga Classification using Chromagram for 120sec duration
| Model | Type | Training F1 | Test F1 | Training Precision | Test Precision | Training Recall | Test Recall |
| -- | -- | -- | -- | -- | -- | -- | -- |
| [Feed Forward No Dropout](Raga_Analysis_Experiments.ipynb) | Single Modal | 100% | 9.25% | 100% | 9.25% | 100% | 9.25% |
| [Feed Forward 10% Dropout](Raga_Analysis_Experiments.ipynb) | Single Modal | 100% | 9.25% | 100% | 9.25% | 100% | 9.25% |
| [Feed Forward 40% Dropout](Raga_Analysis_Experiments.ipynb) | Single Modal | 100% | 9.25% | 100% | 9.25% | 100% | 9.25% |
| [2 Layer LSTM with 10% Dropout](Raga_Analysis_Experiments.ipynb) | Single Modal | 67.83% | 8.33% | 67.83% | 8.33% | 67.83% | 8.33% |

