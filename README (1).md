# UPI Scam Detector

An API-based machine-learning project that analyzes SMS messages for
potential UPI-related scams and returns a prediction score with risk
indicators.

> **Important:** This is an educational project, not a guaranteed
> fraud-detection system. An external benchmark during development
> showed weak scam detection, so predictions must not be treated as
> definitive proof that a message is safe or fraudulent.

## Features

-   **SMS classification:** Uses a trained scikit-learn model to
    classify messages as `SCAM` or `SAFE`.
-   **Text and behavioral features:** Combines TF-IDF text features with
    handcrafted indicators for suspicious message patterns.
-   **Explainable responses:** Returns a score and human-readable risk
    indicators when available.
-   **FastAPI API:** Provides HTTP endpoints for submitting messages and
    retrieving predictions.
-   **Privacy-conscious design:** Includes PII masking and prediction
    audit logging.
-   **Docker support:** Runs the API in a container.
-   **Tests:** Includes API and security tests in the `tests/`
    directory.

## Technology Stack

Python, FastAPI, Uvicorn, scikit-learn, TF-IDF, NumPy, pandas, Joblib,
Pydantic, pytest, HTTPX, Docker.

## Project Structure

``` text
upi-scam-detector/
├── app/
│   └── main.py             # FastAPI application
├── data/                   # Dataset files
├── models/
│   └── model.joblib        # Trained model, if included
├── src/
│   ├── features.py         # Text normalization and feature extraction
│   ├── predict.py          # Prediction logic
│   └── train.py            # Model training pipeline
├── tests/                  # API and security tests
├── requirements.txt
├── pytest.ini
├── Dockerfile
├── .dockerignore
└── README.md
```

File names may vary slightly by repository version.

## Requirements

Choose either Docker or a local Python environment.

## Run with Docker

From the project root, with Docker running:

``` bash
docker build -t upi-scam-detector .
docker run --rm -p 8000:8000 upi-scam-detector
```

Open: - Swagger UI: <http://localhost:8000/docs> - OpenAPI schema:
<http://localhost:8000/openapi.json>

Keep the terminal running while using the API. If the container fails to
start, inspect the build output and container logs.

## Run Locally

### Windows PowerShell

``` powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Start the API from the project root:

``` powershell
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open <http://localhost:8000/docs> to inspect the available endpoints. If
your app uses a different entry point, adjust the Uvicorn import path
accordingly.

## Test the API

Check that the OpenAPI schema responds:

``` powershell
Invoke-WebRequest http://localhost:8000/openapi.json
```

A successful response normally has HTTP status `200`.

Use Swagger UI to inspect the prediction route, click **Try it out**,
enter a request body matching the displayed schema, and execute the
request. The exact endpoint and request fields are defined by the
running API.

## Run Tests

From the project root:

``` powershell
python -m pytest -v
```

The production Docker image may be configured for running the API only
and may not contain test files.

## Model and Evaluation Notes

The model combines TF-IDF text representations with handcrafted message
indicators and a scikit-learn classification pipeline. Prediction
responses may include a score and risk indicators.

During development, an external dataset benchmark evaluated 120 rows.
The model classified every row as `SAFE`, resulting in **0% recall for
scam messages** at the configured threshold. This indicates that the
model needs improvement before it can be relied upon for scam detection.
A low scam score is not proof that a message is safe.

Recommended improvements: 1. Add more representative, properly labelled
scam and legitimate messages in English, Hindi, and Hinglish. 2.
Evaluate on a deduplicated holdout set not used for training or
threshold selection. 3. Report precision, recall, F1 score, confusion
matrix, and false-positive rate. 4. Choose a decision threshold using a
validation set and document trade-offs. 5. Re-test privacy masking,
audit logging, and API behavior after changes.

## Security and Privacy

-   Never commit API keys, credentials, `.env` files, or private data.
-   Keep local virtual environments and caches out of version control.
-   Review datasets before publishing them; ensure you have permission
    to redistribute them and remove personal information where
    appropriate.
-   Treat predictions as advisory. Verify suspicious messages through
    official bank or payment-app channels rather than using links or
    phone numbers in the message.

## Roadmap

-   Improve scam recall and generalization to multilingual messages.
-   Expand evaluation using representative, independently labelled test
    data.
-   Add health checks and structured API error handling if not already
    implemented.
-   Add automated CI tests and image publishing when useful.

## License

No license is specified. Add a `LICENSE` file before granting others
permission to reuse, modify, or distribute this project.

## Author

**Suyash Verma**

GitHub: <https://github.com/suyashverma16/upi-scam-detector>
