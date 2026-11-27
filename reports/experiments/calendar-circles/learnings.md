# Learnings: Calendar circles (days and months)

- The original "N days from D" template was answered correctly only 0–57% of the time per N (the model answered "Wednesday" for 18 of 42 prompts). Centroids built from the true answer then do not reflect the model's representation. Switching to prompts the model solves turned a failing causal test into a partial one.
- Removing the template mean before PCA is essential. Otherwise the top PCs encode the template, not the day.
