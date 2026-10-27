# Assignment 2 Report

- **Name**: Jaime Fernandez
- **Student ID**: 18632
- **Email**: jfernandezr.ieu2022@student.ie.edu
- **Group**: BBADBA 5A

## 1. Business problem

**Decision:** when a customer returns a car, do we charge for damage?

Today an employee checks the car by eye. The model would check a photo first and only ask the employee when it is unsure.

- **Case** = one photo of a returned car (I assume one photo per return).
- **Positive** = any visible damage (scratch, dent, broken glass, light or hood).
- The model gives each photo a score. The score leads to one of three outcomes: **pass** (no charge), **review** (employee checks) or **charge**.
- The model only sees the return photo, so it cannot know if the damage was already there at pick-up.

**Mistakes have different costs:**

| Item | Cost | What it includes | Source |
|---|---|---|---|
| Missed damage (false negative) | 450 EUR | 350 EUR repair we do not recover + 2 days x 50 EUR lost use | Assumption (350 EUR is the middle of a 120 to 600 EUR range). One cost for all damage types. |
| Charging an undamaged car (false positive) | 107.5 EUR | 25 EUR claim handling + 7.5 EUR refund + 15% chance of losing a 500 EUR customer | All assumptions, no reliable source |
| One human review | 5.17 EUR | 20 minutes x 15.5 EUR per hour | Salary offers of 18,000 to 25,000 EUR per year (InfoJobs, Indeed). To get an hourly cost I assumed 30% employer charges and 1,800 hours per year, which gives 13 to 18 EUR per hour. I used the middle, 15.5. The 20 minutes is my guess. |
| Correct decision | 0 EUR | Only mistakes cost money | Convention |

A missed damage is about 4 times worse than a wrong charge. A review is cheap (about 5 EUR), so in theory almost everything should go to a person. **The real limit is how many people the office has.**

**Other assumptions:** 10% of returns have damage (Auto Rental News says about 10%) and 75 returns per day (no source).

**What I measure:** cost per 1,000 returns and share sent to a person, not mAP. The dataset is 82% damaged, so I reweight all costs to the 10% rate.

## 2. Dataset

| | |
|---|---|
| Workspace | `celebalworkspace` |
| Project | `car-damage-detection-dy5hu` |
| Version | `1` |
| Link | https://universe.roboflow.com/celebalworkspace/car-damage-detection-dy5hu |
| License | CC BY 4.0 (checked on the dataset page). Credit: CelebalWorkspace (2022). *Car-damage-detection Dataset*. Roboflow Universe, Roboflow, October 2022. |
| Export format | yolov8 |

370 images with boxes around damage. Classes: Scratch, Dent, Broken-Glass, Broken-light, Broken-hood and **Undamaged**. Undamaged images carry one large box (median 56.7% of the image). Class `15` has one object and is probably a labelling error, so I ignore it.

| Split | Images | With damage | Undamaged |
|---|---|---|---|
| train | 260 | 215 | 45 |
| valid | 72 | 58 | 14 |
| test | 38 | 31 | 7 |

| Class | Train | Valid | Test | Median object size (% of image) |
|---|---|---|---|---|
| Scratch | 100 | 34 | 12 | 1.65 |
| Dent | 58 | 8 | 17 | 9.63 |
| Broken-Glass | 55 | 10 | 7 | 19.06 |
| Broken-light | 44 | 13 | 10 | 16.35 |
| Undamaged | 45 | 14 | 7 | 56.66 |
| Broken-hood | 39 | 12 | 2 | 13.09 |

Scratches are tiny, which makes them hard to detect.

**Problems with this dataset:**
- **Too few test images.** Only 7 undamaged test images. Even with zero mistakes on them, the real false positive rate could be up to 41%. Broken-hood has 2 test objects, so its results are noise.
- **It does not look like rental returns.** File names show that damaged images are internet search results, and all undamaged images are named after SUV models (hyundai, tata, kia), so they appear to be catalogue pictures. The model may learn photo style instead of damage.
- **Not a realistic mix.** 82% damaged here, 10% assumed in real life.

**So I cannot conclude:** how it works on real rental photos, its real false positive rate, anything about Broken-hood, or anything about damage that existed at pick-up. Costs in euros compare designs; they are not forecasts.

## 3. Model and baselines

- **Model:** YOLO11n object detection, finetuned from COCO weights.
- **Training:** Google Colab (T4 GPU), up to 100 epochs, image size 640, batch 16, patience 30 (never triggered), seed 0. 100 epochs ran and the best was epoch 91. Training took 488 s.
- **Why detection?** The data has boxes, one photo can have several damages, and boxes show the employee where to look. A photo counts as damaged if any damage box passes the threshold.

| Model | mAP50 | Cost per 1,000 returns |
|---|---|---|
| Finetuned model (with review band) | 0.532 | **831 EUR** |
| COCO pretrained (no damage classes) | 0.0 | 45,000 EUR |
| Predict nothing (never charge) | 0.0 | 45,000 EUR |

**Reading this:** mAP 0.532 is mediocre, but the cost drops from 45,000 to 831 EUR. Deciding "damaged or not" is easier than finding every box: one correct box is enough. Image-level AUC is 0.970 (valid) and 1.000 (test).

**Reproducibility:** I ran the training twice on Colab and got different results (mAP50 0.525 vs 0.532). The two runs used different ultralytics versions (8.4.184 and 8.4.190), which is the most likely cause, but I did not prove it. GPU training is also not exactly repeatable. `uv.lock` pins 8.4.173. The report numbers come from the committed `weights/best.pt` and `artifacts/results.json`. Re-running gives similar, not identical, numbers.

## 4. Per-class results (test)

| Class | Test objects | Precision | Recall | AP50 |
|---|---|---|---|---|
| Undamaged | 7 | 0.910 | 1.000 | 0.995 |
| Broken-Glass | 7 | 1.000 | 0.834 | 0.855 |
| Broken-light | 10 | 0.710 | 0.800 | 0.696 |
| Scratch | 12 | 0.477 | 0.417 | 0.338 |
| Dent | 17 | 0.558 | 0.176 | 0.207 |
| Broken-hood | 2 | 0.000 | 0.000 | 0.099 |

- **The 0.532 headline is inflated.** Undamaged (large box) and Glass lift it. The five damage classes average about 0.44.
- **Dent is the worst class with a measurable sample:** the model finds 4 of 17 dents. Broken-hood scores lower (0.099), but with only 2 objects it says nothing.
- **Scratch is weak:** it misses 6 of 12 and invents the most false boxes (8).
- **The model never confuses one damage class with another.** Its mistakes are missed damage and boxes on background. A missed dent just disappears, so "no confusion" is partly an artefact.
- This matters less for the business: one correct damage box is enough to flag the car.

## 5. Choosing the thresholds

Image score = confidence of the most confident damage box (0 if none). I tried every threshold on **validation**, picked the cheapest at a 10% damage rate, and used **test** only for the final figure. I did not use theory because the scores are not true probabilities.

| Design | Thresholds | Cost per 1,000 (valid) | Cost per 1,000 (test) | Sent to a person |
|---|---|---|---|---|
| Single threshold | 0.03 | 8,462 EUR | 2,903 EUR | 0% |
| Pass / review / charge | pass below 0.01, charge from 0.36 | 1,618 EUR | 831 EUR | 16.1% |

- On test, the single threshold misses 2 of 31 damaged images and wrongly charges 0 of 7 undamaged ones.
- **The single threshold is unstable.** In my first training it cost 0 EUR on test; in the second, 2,903 EUR. One missed image is worth about 1,450 EUR, so the test set is too small. I trust the validation figure more.
- **The three-output design is stable** (798 EUR, then 831 EUR). On test it made no mistakes, so all 831 EUR is review time. That is not a promise of zero errors on real photos.
- Caveat: validation was also used to pick `best.pt`, so the thresholds are slightly optimistic.

**What changes the result (test, per 1,000 returns):**

| Assumption changed | Single threshold | Pass/review/charge | Sent to a person |
|---|---|---|---|
| Base case | 2,903 EUR | 831 EUR | 16.1% |
| Missed damage costs 150 EUR | 4,839 EUR | 1,101 EUR | 2.6% |
| Missed damage costs 800 EUR | 5,161 EUR | 831 EUR | 16.1% |
| Damage rate 5% / 15% | 1,452 / 4,355 EUR | 785 / 877 EUR | 15.2% / 17.0% |
| Review takes 10 / 35 minutes | 2,903 EUR | 415 / 1,454 EUR | 16.1% |

The **cost of a missed damage** matters most (only if it drops to about 150 EUR). The damage rate only scales costs. Review time changes workload, not thresholds. I did not test the false positive cost.

## 6. Can the office handle the reviews?

| Item | Value |
|---|---|
| Reviews per day (16.1% of 75) | 12.1 |
| Time per review | 20 minutes |
| Hours per day | 4.0 |
| People needed | **0.50** |
| People available | **1.0** (one employee dedicated to inspections). My assumption, the same as the dashboard default `DEFAULT_PEOPLE = 1.0`. It should be replaced with the real number from the office. |

| Policy | Cost per 1,000 | Sent to a person | People needed |
|---|---|---|---|
| Do nothing (never charge) | 45,000 EUR | 0% | 0 |
| Charge every return | 96,750 EUR | 0% | 0 |
| A person checks every return | 5,167 EUR | 100% | 3.12 |
| Model alone (threshold 0.03) | 2,903 EUR | 0% | 0 |
| **Model + person (two thresholds)** | **831 EUR** | **16.1%** | **0.50** |

- Model + person is about 71% cheaper than the model alone on test (2,903 to 831 EUR). On validation, which I trust more, it is about 81% cheaper (8,462 to 1,618 EUR). It needs half a person.
- Checking everything is cheap per case but needs 3.12 people, so staff is the real limit.

**With 1 person available, it fits with room to spare** (0.50 of the day left). Three warnings:
1. 12.9 of the 16.1 points come from one undamaged test image, so the real share could be much higher or lower.
2. The 20 minutes is a guess. At 35 minutes I need 0.88 people: it still fits, but only just.
3. The model assumes the reviewer is always right and ignores the customer waiting. The plan would overflow at about 150 returns per day (twice my assumption).

This also assumes one person is fully dedicated to inspections. If the same person also serves the counter, there is less time, and the number must come from the office. If the office cannot cope, the review band can be made narrower.

## 7. Limitations and next steps

| Limitation | Next step |
|---|---|
| Only 7 undamaged test images | Collect hundreds of undamaged photos; add bootstrap intervals |
| Photos are not real rental returns; model may learn photo style | Collect real phone photos from returns; run in shadow mode first |
| Dent and Scratch are weak; Broken-hood cannot be measured | Add data, train at larger size or on image tiles, try a bigger model, merge or drop hood |
| Cannot tell new damage from old damage | Compare pick-up and return photos of the same car |
| One photo per return | Group photos by return and recompute costs |
| One cost for every missed damage, although a big dent and a scratch cost different amounts to repair | Use costs per damage type from real workshop data |
| Other cost figures are unsourced assumptions | Use the company's real claims and workshop data; test the false positive cost |
| Reviewer assumed perfect; waiting time ignored; staff available is assumed | Measure real inspection times and staffing at the office |
| Scores are not probabilities; validation used twice | Keep a separate calibration split and calibrate scores |
| Results change between runs and versions | Pin versions, train with several seeds |
| Dataset is small and may have label errors | Clean labels, add more data |

## Generative AI use disclosure

I used Claude (Anthropic) as an assistant only for allowed purposes in this assignment: to compare candidate datasets and help me decide on the Car Damage Detection dataset, to draft code in train.py and app.py, to assist with GitHub, Colab and Visual Studio steps, and to suggest wording and formatting improvements for this report. All substantive decisions have been made on my own. I ran all code locally in Visual Studio Code and verified the outputs myself.
