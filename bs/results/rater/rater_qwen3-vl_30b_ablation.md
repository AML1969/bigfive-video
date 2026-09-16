# LLM-оценщик: варианты числа кадров (test200 FIV2)

| Вариант | 16f@640 mACC | 16f@640 mCCC | 8f@640 mACC | 8f@640 mCCC | 28f@640 mACC | 28f@640 mCCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rater_raw | 0.8527 | 0.149 | 0.8560 | 0.185 | 0.8529 | 0.166 |
| rater_calibrated | 0.8851 | 0.130 | 0.8868 | 0.156 | 0.8860 | 0.147 |
| oceanai | 0.9237 | 0.677 | 0.9237 | 0.677 | 0.9237 | 0.677 |
| own5 | 0.9192 | 0.716 | 0.9192 | 0.716 | 0.9192 | 0.716 |
| mean(oceanai,own5) | 0.9256 | 0.724 | 0.9256 | 0.724 | 0.9256 | 0.724 |
| mean(oceanai,own5,rater_calibrated) | 0.9190 | 0.609 | 0.9197 | 0.615 | 0.9193 | 0.611 |
| 0.4*oceanai+0.4*own5+0.2*rater_calibrated | 0.9230 | 0.665 | 0.9234 | 0.669 | 0.9231 | 0.666 |

- 16f@640: корреляция ошибок {'oceanai~own5': 0.777, 'oceanai~rater_calibrated': 0.756, 'own5~rater_calibrated': 0.605}; калибровка (наклон, сдвиг) {'Openness': (0.46, 0.26), 'Conscientiousness': (0.85, 0.05), 'Extraversion': (0.48, 0.19), 'Agreeableness': (0.4, 0.3), 'Non-Neuroticism': (0.23, 0.38)}
- 8f@640: корреляция ошибок {'oceanai~own5': 0.777, 'oceanai~rater_calibrated': 0.754, 'own5~rater_calibrated': 0.603}; калибровка (наклон, сдвиг) {'Openness': (0.46, 0.26), 'Conscientiousness': (0.85, 0.05), 'Extraversion': (0.48, 0.19), 'Agreeableness': (0.4, 0.3), 'Non-Neuroticism': (0.23, 0.38)}
- 28f@640: корреляция ошибок {'oceanai~own5': 0.777, 'oceanai~rater_calibrated': 0.759, 'own5~rater_calibrated': 0.618}; калибровка (наклон, сдвиг) {'Openness': (0.46, 0.26), 'Conscientiousness': (0.85, 0.05), 'Extraversion': (0.48, 0.19), 'Agreeableness': (0.4, 0.3), 'Non-Neuroticism': (0.23, 0.38)}

mACC = 1 − MAE; own5 = своя модель (среднее 5 seed); калибровка — линейная по dev200 (16 кадров).
