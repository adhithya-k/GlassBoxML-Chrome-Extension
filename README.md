
# GlassBoxML in the Browser (Chrome Extension)

**What it does**
- Load data from **HTML tables** on any web page or **CSV uploads**.
- Train simple **Classification**, **Regression**, and **Clustering (k-means)** models **fully in-browser**.
- See **interpretable outputs**: feature importances (weights + permutation), and simple **decision rules** (OneR).
- Make **predictions** on custom inputs and **export** your trained model to JSON.

> Note: This build is dependency-free and uses compact, hand-written ML in pure JS for easy review and portability. If desired, you can swap the training functions with TensorFlow.js equivalents (MV3 allows bundling local scripts).

## How to load
1. Download the `MLExtension.zip` and extract the `MLExtension/` folder.
2. Open **chrome://extensions** → toggle **Developer mode** → **Load unpacked** → select the extracted `MLExtension` folder.
3. Pin **GlassBoxML** from the extensions toolbar.

## Quick demo
1. Open any page with a data table (e.g., Wikipedia table).
2. Open the extension → **Import from page tables**.
3. Choose **Task** (Classification/Regression/Clustering), select **target** for supervised tasks.
4. Click **Train Model** → see metrics, importances, rules.
5. Enter a JSON object and click **Predict**.
6. **Export Model** to JSON for reuse.

## Files
- `manifest.json` — MV3 manifest
- `popup.html/css/js` — UI + ML logic
- `content.js` — scrapes tables from the active page
- `background.js` — install hook
- `icons/` — minimal icons

## Interpretable bits
- **Feature importance** via (a) absolute weights (for linear/logistic) and (b) **permutation importance** (metric drop when shuffling a feature).
- **Decision rules**: **OneR** over discretized bins to yield human-readable if-then rules.
- **Clustering explainability**: variance across centroid coordinates → which features separate clusters.
