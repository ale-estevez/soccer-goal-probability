# Soccer Goal Probability (xG) Model

Predicts the probability that a shot becomes a goal, using free StatsBomb open data.

## Approach
- Data: 2015/16 Premier League, La Liga, Serie A, Ligue 1, plus World Cup 2018 and Bundesliga samples
- Features: distance, angle, header, defenders in the shot cone, goalkeeper distance, pressure, first-time, one-on-one
- Models: logistic regression and LightGBM, compared to a constant-probability baseline
- Validation: trained on earlier matches, tested on later ones, then stress-tested on competitions never seen in training

## Results (log loss, lower is better)
| Model | Held-out league matches | World Cup + Bundesliga |
|---|---|---|
| Baseline | 0.326 | 0.313 |
| Logistic v2 | 0.271 | 0.270 |
| LightGBM | 0.265 | 0.262 |

Calibration plots show predicted and actual goal rates match across the range.

## Limitations
- Bundesliga open data covers only Bayer Leverkusen's matches
- Penalties excluded
- The model measures shot quality, not shooter skill

## Next steps
- Player finishing skill (goals vs. xG, with shrinkage)
- Hockey version to test the multi-sport pipeline
