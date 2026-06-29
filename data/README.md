# Data

The **PaySim** dataset (`paysim.csv`, ~493 MB) is **not committed** to this
repository because it exceeds GitHub's 100 MB file limit.

## Get the dataset

1. Download **PaySim** from Kaggle:
   https://www.kaggle.com/datasets/ealaxi/paysim1
2. Place the file here as:

   ```
   data/paysim.csv
   ```

The app also auto-detects `paysim.csv` in the project root, or you can upload it
from the sidebar at runtime. When running with Docker, `./data` is mounted as a
volume (see `docker-compose.yml`).
