todo, tested on single machine with 4 T4 cards (three of them are used)

```
pip install -r requirements.txt
python import_models.py
BENTOML_CONFIG=configuration.yml bentoml serve service:svc --api-workers=3
```
