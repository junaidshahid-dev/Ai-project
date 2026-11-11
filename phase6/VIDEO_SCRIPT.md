# Demo video script (Deliverable 6) — about 5 minutes

Record the screen with sound (Windows: **Win + Alt + R** with Xbox Game Bar, or OBS Studio). Upload the video to
YouTube as *Unlisted* or to Google Drive with "Anyone with the link can view", then paste the link into the README
(`Demo video` line) and into Google Classroom.

**Before recording:** open a terminal in the project folder and make sure the setup in the README has been done
(`pip install -r requirements.txt`, data downloaded, `python scripts/extract_features.py` already run).
Keep 3–4 RAVDESS `.wav` files ready to upload in the app (for example one angry, one sad, one neutral clip).

| Time | Show on screen | What to say |
|---|---|---|
| 0:00–0:30 | Title page of `phase6/F26-11.pdf` | "We are group F26-11 (22L-7786, 23L-3002, 23L-3102). Our project re-tests a 2025 *Scientific Reports* paper that recognises 7 emotions from speech with a lightweight CNN ensemble and reports 97.57% accuracy. We asked: does that accuracy hold for new clips, new speakers, a new dataset, and for both male and female voices?" |
| 0:30–1:15 | GitHub repository page, then the README layout table | "Everything is in this repository. Phase 2 prepares the data and features, Phase 3 builds and trains the models, Phase 4 evaluates them, Phase 5 explains the results, and Phase 6 is the final report. The trained models are in `models/`." |
| 1:15–2:00 | Pipeline figure and architecture figure in the final report | "Each clip is loaded at 22,050 Hz, 2.5 seconds long. We compute ZCR, energy, 20 MFCCs and 12 Chroma values for 108 time frames, so 3,672 numbers per clip. The 1D-CNN and the CNN-BiLSTM each give 7 probabilities, and the ensemble averages them." |
| 2:00–3:15 | Terminal: `streamlit run app/app.py`, browser opens; upload an angry clip, then a sad clip | "This is our demo app. I upload a clip it has never been trained on… it predicts *angry* with this confidence, and the bar chart shows all 7 probabilities. Now a sad clip…" (Say honestly if one is wrong: "this one is confused with neutral, which is the most common mistake we found.") |
| 3:15–4:15 | `phase4/F26-11.ipynb` in Jupyter: run all cells (takes seconds), scroll to the results table and the gender figure | "Our evaluation notebook recomputes every number from the saved predictions. With a clean split the ensemble reaches 81.9%. Only when augmented copies of test clips leak into training do we get 96.5%, close to the paper. For speakers the model has never heard it falls to 58.5%, on another dataset it is near chance, and it is 15.6 points worse for male than for female voices." |
| 4:15–4:45 | `phase5/F26-11.ipynb`: speaker vs emotion figure and speaker-normalisation figure | "Phase 5 explains why: the same features recognise *who* is speaking with 95% accuracy, more than *which emotion* (74%). Normalising each speaker's features raises a simple SVM from 49% to 59% on new speakers and from 24% to 35% on TESS, but the gender gap does not close, so it has another cause." |
| 4:45–5:00 | Conclusion section of the final report | "So the model works, but its headline accuracy depends on how it is tested. Thank you." |

**Tips:** speak slowly, zoom the browser/notebook to 125% so text is readable, and close other windows and
notifications before recording. Each member can present one part.
