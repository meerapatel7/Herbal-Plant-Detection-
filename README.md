# 🌿 Herbal Plant Identifier

An AI project that **identifies medicinal plants from a photo** and then shows information about the detected plant: its traditional uses, how it is used, and safety precautions. Users can also **ask questions** about the plant (for example, "Is it good for fever?" or "Is it safe in pregnancy?") and get answers taken only from the stored plant information.

> **Educational project. Not medical advice.** See the [Disclaimer](#disclaimer).

---

## Features

- Identifies **40 medicinal plants** from an uploaded photo or a camera capture
- Shows the **top 3 predictions** with confidence scores
- Refuses to name a plant when the model is not confident enough
- Lets the user **correct the result** manually with a dropdown
- Displays scientific name, traditional uses, how it is used, and precautions
- **Question answering (NLP)** that answers only from the verified plant database and says "I don't have verified information" when it cannot find an answer
- Detects emergency words (for example "poison" or "overdose") and tells the user to seek medical help
- Simple web interface built with **Streamlit**

## Demo

Add your screenshots to a `screenshots/` folder and keep the file names below, or change them.

| Home screen | Prediction | Question box |
|---|---|---|
| ![Home](screenshots/home.png) | ![Prediction](screenshots/prediction.png) | ![Question box](screenshots/question_box.png) |

## How it works

```
Photo  ->  Image model (MobileNetV2)  ->  plant name + confidence
                                              |
                              plant_info.json (uses, how to use, precautions)
                                              |
        User question  ->  nlp_helper.py  ->  answer from stored text only
```

1. **Image classification:** the photo is resized to 160 x 160 and passed to a MobileNetV2 model fine-tuned on 40 plant classes.
2. **Confidence check:** if the top confidence is below the chosen threshold, the app does not name a plant.
3. **Information lookup:** the predicted plant name is used as a key in `plant_info.json`.
4. **Question answering:** a three-step process (safety check, intent detection, TF-IDF retrieval) finds the best matching stored sentence.

## Dataset

| Item | Value |
|---|---|
| Plants (classes) | 40 |
| Total images | 5,809 |
| Images per plant | about 146 (Ganike has 115) |
| Split | 70% train / 15% validation / 15% test |
| Image size used | 160 x 160 |

The dataset is organised as one folder per plant. **The images are not included in this repository** because of their size.

Dataset source: *add the name and link of your dataset here.*

## Model and training

- **Architecture:** MobileNetV2 (pretrained on ImageNet) with data augmentation (flip, rotation, zoom, brightness), global average pooling, dropout (0.3) and a 40-way softmax layer
- **Stage 1:** train the new top layer with the base network frozen (early stopping on validation accuracy)
- **Stage 2:** fine-tune the last 30 layers of the base network with a low learning rate (1e-5)
- **Framework:** TensorFlow / Keras, trained on CPU in a Jupyter notebook (batch size 16)

## Results

| Metric | Value |
|---|---|
| Validation accuracy (after fine-tuning) | about 90% |
| **Test accuracy** | **87.76%** (915 unseen images) |
| Macro F1-score | 0.87 |

Strongest plants (about 95% or more of test photos correct): Avacado, Arali, Pomegranate, Lemon_grass, Ashoka, Bamboo, Betel_Nut, Curry_Leaf.

Weakest plants: Henna, Amruta_Balli, Betel, Mango, Tulasi. Common confusions include Rose as Hibiscus, Nagadali as Ganike, and Tulasi as Insulin.

The full per-plant report is in `results/test_report.txt` and the confusion matrix in `results/confusion_matrix.png`.

## Question answering (NLP)

`nlp_helper.py` answers questions about the detected plant without inventing anything:

1. **Safety check:** emergency words trigger a message to contact a doctor or emergency services.
2. **Intent detection:** keywords decide whether the user wants precautions, how to use the plant, or its scientific name.
3. **Retrieval:** otherwise, TF-IDF and cosine similarity find the stored "uses" sentence that best matches the question. If no sentence is similar enough, the app says it has no verified information.

## Safety design

- Confidence threshold: low-confidence photos are not named
- Manual correction dropdown
- Precautions are always shown with the plant information
- Toxic plants have explicit warnings (for example Castor, Ekka, Nagadali, Ganike, Nithyapushpa, Betel Nut)
- Uses are worded as "traditionally used for...", not as cures
- Disclaimer shown in the app

## Project structure

```
herbal-plant-identifier/
├── app.py                    # Streamlit web app
├── nlp_helper.py             # Question answering
├── plant_info.json           # Plant information used by the app
├── plant_info_template.csv   # Editable spreadsheet version of the information
├── herbal_detection.ipynb    # Training and evaluation notebook (rename to match yours)
├── requirements.txt
├── models/
│   ├── herbal_model.keras    # Trained model (see the note below if it is not included)
│   └── classes.json          # Plant names in the model's output order
├── results/
│   ├── test_report.txt
│   └── confusion_matrix.png
└── screenshots/
```

## Installation

Python 3.10 or 3.11 is recommended for TensorFlow.

```bash
git clone https://github.com/YOUR-USERNAME/herbal-plant-identifier.git
cd herbal-plant-identifier

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

## Usage

**Run the app** (needs `models/herbal_model.keras` and `models/classes.json`):

```bash
python -m streamlit run app.py
```

The app opens at `http://localhost:8501`.

**Retrain the model:** put your images in `herbal_data/<plant_name>/`, open the notebook, and run the cells in order (clean, split, load, build, train, fine-tune, evaluate, save).

**Add or edit plant information:** edit `plant_info_template.csv` (separate several uses with `;`) or `plant_info.json`. Plant names must match the folder names used in training exactly.

> If the trained model file is not in this repository (for example because of its size), download it from *add your link here* and place it in the `models/` folder.

## Supported plants

Aloevera, Amla, Amruta_Balli, Arali, Ashoka, Ashwagandha, Avacado, Bamboo, Basale, Betel, Betel_Nut, Brahmi, Castor, Curry_Leaf, Doddapatre, Ekka, Ganike, Gauva, Geranium, Henna, Hibiscus, Honge, Insulin, Jasmine, Lemon, Lemon_grass, Mango, Mint, Nagadali, Neem, Nithyapushpa, Nooni, Pappaya, Pepper, Pomegranate, Raktachandini, Rose, Sapota, Tulasi, Wood_sorel.

## Limitations

- **Closed set:** the model can only choose among its 40 plants. A plant that is not in the list may still be matched to one of them, sometimes with moderate confidence.
- Accuracy is lower for look-alike plants and for photos with cluttered backgrounds, poor light, or no leaf visible.
- The question answering matches **words, not meaning**, so it can miss paraphrased questions.
- The plant information was compiled from public references and should be **verified against authoritative sources** (for example IMPPAT, the Ministry of AYUSH, the National Medicinal Plants Board, and WHO monographs) before any real-world use.

## Future work

- Handle plants outside the database: stricter confidence and margin rules, and a "not in database" class
- Add more plants and more real phone photos for weak classes
- Smarter NLP with sentence embeddings and multilingual support (Hindi, Gujarati)
- Voice input and a mobile version (TensorFlow Lite)
- A feedback option so users can submit photos of unrecognised plants

## Disclaimer

This project is for **educational purposes only**. It is not medical advice and must not be used to diagnose, treat, or prevent any condition, or to decide whether a wild plant is safe to eat or use. Some plants in this project are toxic. Always consult a doctor or a qualified practitioner before using any plant medicinally.

## Author

**Meera Patel**, MSc Data Science

## License

Choose a licence (for example MIT) and add a `LICENSE` file, or remove this section.
