# FIFA 19 player-data cache

The independent player-role notebook downloads `fifa19.csv` from a commit-pinned
[public mirror](https://github.com/qualityjacks/Fifa19_Insights) of
[Karan Gadiya's FIFA 19 dataset](https://www.kaggle.com/datasets/karangadiya/fifa19).
The exact URL and SHA-256 are in the notebook and `source.json`.

There are 18,207 source players. The notebook excludes 60 missing-position rows,
retaining 18,147 unique players, including 2,025 goalkeepers. It uses individual
skills to classify a fixed Defender/Midfielder/Attacker/Goalkeeper mapping, not positional
suitability ratings or future match results.

The CSV is downloaded automatically on first run and checked
against the expected hash on every run. After download, the notebook works offline.
Respect the original dataset's terms and attribution when sharing the data;
the mirror's code license is not a substitute for the original data terms.
