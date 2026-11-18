# Crop Yield Anomaly Detection

INT396 Unsupervised Learning CA project. Finds unusual crop yield records in the India district-wise Crop Area Production Yield dataset (about 4.5 lakh rows).

## Syllabus topics used
- Unit I: unsupervised problem formulation, Euclidean distance
- Unit II: feature scaling (StandardScaler), K-Means, choosing k with the silhouette coefficient

## Files
- `data/crop_yield.csv` : dataset
- `train.py` : cleaning, features, scaling, K-Means, distance threshold, saves `model/artifacts.joblib`
- `app.py` : Streamlit app
- `.streamlit/config.toml` : theme

## Run
```
pip install -r requirements.txt
python train.py
streamlit run app.py
```

## How it works
1. Each record becomes 3 numbers: log of area, log of yield, and log of yield compared with the typical (median) yield of the same crop in the same state. They are standardised.
2. K-Means groups all records into k clusters. k is picked from 2 to 6 by the best silhouette score.
3. A record's anomaly score is its distance to the nearest cluster centre. Records beyond the 99th percentile of training distances are anomalies.
4. The app takes a state, crop, year, area and production, and shows the verdict, the distance against the limit, and the yield trend with the entry marked as a dot.
