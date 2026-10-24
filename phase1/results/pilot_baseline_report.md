# Phase 1: XNLI LoRA Sequential Baseline Pilot



## 1. Experiment Overview



**Experiment name:** XNLI LoRA Sequential Baseline Pilot  

**Model:** Meta Llama 3.2 1B  

**Training method:** Sequential fine-tuning with Low-Rank Adaptation (LoRA)  

**Language order:** English -> Hindi -> German  

**Replay:** No replay in this baseline  

**Execution environment:** Google Colab with an NVIDIA Tesla T4 GPU



This pilot evaluates sequential multilingual learning on three languages from the Cross-lingual Natural Language Inference (XNLI) benchmark.



The experiment serves as an initial baseline for studying performance changes when a model learns languages sequentially, without replaying examples from previously learned languages.



## 2. Training Configuration



| Parameter | Setting |

|---|---|

| Base model | meta-llama/Llama-3.2-1B |

| Adaptation method | LoRA |

| LoRA rank | 16 |

| LoRA alpha | 32 |

| Target modules | q_proj, k_proj, v_proj, o_proj |

| Training languages | English, Hindi, German |

| Training examples per language | 5,000 |

| Test examples per language | 1,000 |

| Epochs per language | 1 |

| Training batch size | 4 |

| Maximum sequence length | 256 tokens |

| Learning rate | 0.0002 |

| Random seed | 42 |

| Replay strategy | None |



The experiment used a three-class NLI classification task: entailment, neutral, and contradiction.



## 3. Final Results



### Accuracy



| Language | Final accuracy |

|---|---:|

| English | 82.60% |

| Hindi | 68.80% |

| German | 77.40% |

| **Mean** | **76.27%** |



### Macro-F1



| Language | Final macro-F1 |

|---|---:|

| English | 82.80% |

| Hindi | 68.85% |

| German | 77.60% |

| **Mean** | **76.42%** |



These values are transcribed from the saved `final_results_summary.json` file.



## 4. Preliminary Interpretation



After sequential training on English, Hindi, and German, the model achieved a mean final accuracy of 76.27% and a mean final macro-F1 of 76.42% across the three evaluated languages.



English achieved the highest final accuracy at 82.60%, followed by German at 77.40% and Hindi at 68.80%.



These results establish a preliminary performance reference for the three-language pilot. The final scores alone do not establish whether the model experienced catastrophic forgetting, positive forward transfer, or negative interference.



Those effects should be assessed using the complete accuracy matrix, which records performance on each language before and after the successive training stages.



## 5. Limitations



1\. This is a three-language pilot, not a full evaluation across all 15 XNLI languages.

2\. The experiment uses a limited subset of the training and evaluation data.

3\. Results from one seed do not establish variability across random seeds.

4\. The experiment does not compare sequential learning against joint multilingual training or replay-based methods.

5\. The final scores alone cannot quantify forgetting or transfer.

6\. The results should not be generalized to larger models or other multilingual benchmarks without additional experiments.



## 6. Next Steps



1\. Preserve the original Colab notebook, result files, and trained LoRA adapter in Google Drive.

2\. Inspect the saved accuracy matrix and calculate continual-learning metrics using clearly documented definitions.

3\. Verify that the local Phase 1 implementation reproduces the intended experimental configuration.

4\. Compare sequential training with suitable baselines, including joint multilingual training and experience replay.

5\. Expand the evaluation only after the pilot results and implementation have been verified.



## 7. Artifact Locations



**Colab results folder:**  

`MyDrive/Continual_Learning_Research/Phase1_Baseline_Results/`



**Local report:**  

`phase1/results/pilot_baseline_report.md`



The complete experimental artifacts and trained adapter remain in Google Drive. This report documents the pilot results; it does not replace the original result files or model checkpoint.

