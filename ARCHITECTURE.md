Resume_Screener/
│
├── .github/workflows/      # CI/CD: Automated testing and deployment config
├── artifacts/              # STORAGE: Generated files (Train/Test data, .pkl models)
├── config/                 # CONFIG: YAML files for paths and hyperparameters
├── logs/                   # OBSERVABILITY: Runtime log files
│
├── notebooks/              # LAB: Experimental Sandbox
│   ├── data/               # Raw datasets for experimentation
│   ├── 1_EDA.ipynb         # Exploratory Data Analysis
│   └── 2_Model_Train.ipynb # Prototyping model logic
│
├── src/                    # PRODUCTION CODE
│   ├── __init__.py         # Makes 'src' a Python package
│   ├── components/         # CORE LOGIC (The "Workers")
│   │   ├── __init__.py
│   │   ├── data_ingestion.py       # Splits data -> artifacts
│   │   ├── data_transformation.py  # Cleaning & Vectorization pipeline
│   │   └── model_trainer.py        # Training & Model Saving
│   │
│   ├── pipeline/           # ORCHESTRATION (The "Conductors")
│   │   ├── __init__.py
│   │   ├── train_pipeline.py       # Triggers Ingestion -> Transform -> Train
│   │   └── predict_pipeline.py     # Triggers Preprocessing -> Inference
│   │
│   ├── logger.py           # INFRA: Centralized logging config
│   ├── exception.py        # INFRA: Custom exception handling
│   └── utils.py            # INFRA: Helpers (save_object, load_object)
│
├── templates/              # FRONTEND: HTML files
├── static/                 # FRONTEND: CSS/JS assets
├── app.py                  # API: Flask Entry point
├── setup.py                # SETUP: Installs project as a local package
├── requirements.txt        # DEPENDENCIES
└── README.md               # DOCUMENTATION