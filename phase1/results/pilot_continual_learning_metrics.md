# Phase 1: Continual-Learning Metrics Analysis

## 1. Experiment

**Experiment:** Phase 1 B0 - Sequential XNLI Baseline  
**Model:** Meta Llama 3.2 1B  
**Language order:** English -> Hindi -> German  
**Training samples:** 5,000 per language  
**Test samples:** 1,000 per language  
**Epochs:** 1 per language  
**Replay:** None  
**Source:** Google Colab pilot results, `baseline_results.json`

This report analyzes the accuracy and macro-F1 results recorded after each sequential training stage.

## 2. Accuracy Matrix

| Evaluation stage | English |  Hindi | German | Mean accuracy |
| ---------------- | ------: | -----: | -----: | ------------: |
| Initial          |  34.20% | 34.70% | 36.30% |        35.07% |
| After English    |  83.00% | 45.70% | 59.60% |        62.77% |
| After Hindi      |  78.10% | 69.10% | 49.90% |        65.70% |
| After German     |  82.60% | 68.80% | 77.40% |        76.27% |

The mean accuracy increased from 35.07% initially to 76.27% after all three training stages.

## 3. Macro-F1 Matrix

| Evaluation stage | English |  Hindi | German | Mean macro-F1 |
| ---------------- | ------: | -----: | -----: | ------------: |
| Initial          |  21.06% | 25.69% | 35.43% |        27.25% |
| After English    |  83.19% | 40.94% | 52.92% |        59.02% |
| After Hindi      |  78.20% | 69.15% | 44.80% |        64.05% |
| After German     |  82.80% | 68.85% | 77.60% |        76.42% |

Macro-F1 provides a class-balanced view of classification performance. The final mean macro-F1 was 76.42%.

## 4. Stage-to-Stage Accuracy Changes

The following changes are measured in percentage points (pp).

| Training stage | English change | Hindi change | German change |
| -------------- | -------------: | -----------: | ------------: |
| Learn English  |      +48.80 pp |    +11.00 pp |     +23.30 pp |
| Learn Hindi    |       -4.90 pp |    +23.40 pp |      -9.70 pp |
| Learn German   |       +4.50 pp |     -0.30 pp |     +27.50 pp |

The largest observed decline during a training stage was the 9.70 pp drop in German accuracy after Hindi training. German accuracy subsequently increased by 27.50 pp after German training.

These results show asymmetric performance changes across languages in this particular training order.

## 5. Backward Transfer

For a language learned before the final stage, backward transfer is calculated as:

BWT_i = Final accuracy_i - Accuracy immediately after learning language i

| Language                    | Accuracy after own training | Final accuracy |      BWT |
| --------------------------- | --------------------------: | -------------: | -------: |
| English                     |                      83.00% |         82.60% | -0.40 pp |
| Hindi                       |                      69.10% |         68.80% | -0.30 pp |
| Mean over earlier languages |                             |                | -0.35 pp |

Negative BWT indicates a net performance decline following subsequent training. The mean BWT across English and Hindi was -0.35 pp.

German is excluded from this BWT average because it was the final language trained and therefore experienced no subsequent training stage.

## 6. Forgetting

Using the maximum accuracy observed after each language's own training:

Forgetting_i = Maximum post-training accuracy_i - Final accuracy_i

| Language                  | Maximum post-training accuracy | Final accuracy | Forgetting |
| ------------------------- | -----------------------------: | -------------: | ---------: |
| English                   |                         83.00% |         82.60% |    0.40 pp |
| Hindi                     |                         69.10% |         68.80% |    0.30 pp |
| German                    |                         77.40% |         77.40% |    0.00 pp |
| Mean across all languages |                                |                |    0.23 pp |

The mean forgetting across all three languages was approximately 0.23 pp. When considering only English and Hindi, which experienced subsequent training, mean forgetting was 0.35 pp.

The German value is zero under this definition because German was learned last. It does not establish that German would never be forgotten under a different language order.

## 7. Preliminary Interpretation

The experiment shows a substantial improvement in classification performance after sequential training.

The most notable interference event occurred after Hindi training, when both English and German accuracy declined. German subsequently recovered after its own training stage.

The final net decline in accuracy on previously learned languages was small in this run. However, these results should not be interpreted as evidence that catastrophic forgetting is generally absent.

The observed pattern motivates further investigation into language-specific interference and whether replay selection can reduce performance degradation during sequential learning.

## 8. Limitations

1. This is a pilot involving only three languages, not the complete 15-language XNLI benchmark.
2. The experiment uses 5,000 training examples and 1,000 test examples per language.
3. The results represent one training order and one reported run.
4. No replay, joint-training, or alternative replay baseline is compared in this report.
5. The analysis uses rounded accuracy values from the saved performance matrix.
6. Forward-transfer metrics require an explicitly defined baseline and should not be inferred solely from raw accuracy changes.
7. Statistical significance and variability across seeds have not been established.

## 9. Next Experiments

1. Verify the local baseline implementation against the Colab pilot configuration.
2. Compare sequential learning with joint multilingual training and experience replay.
3. Evaluate additional language orders and random seeds.
4. Calculate forward transfer using a documented definition.
5. Evaluate whether interference-aware replay improves accuracy retention compared with suitable baselines.

## 10. Source Artifacts

The original experiment artifacts remain in Google Drive:

`MyDrive/Continual_Learning_Research/Phase1_Baseline_Results/`

Key files include `baseline_results.json`, `final_results_summary.json`, `accuracy_matrix.csv`, `macro_f1_matrix.csv`, plots, and the `final_model/` LoRA adapter directory.

This report contains derived analysis and does not replace the original experimental artifacts.