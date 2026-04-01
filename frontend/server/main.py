from backend.app.main import app
from backend.app.services.inference import MODELS, prepare_image_obj, startup_models
from backend.app.services.risk import compute_clinical_score as _clinical_score

_load_models = startup_models
