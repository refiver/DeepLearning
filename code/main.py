import model_administration
import environments


loaded_model, initial_epoch = model_administration.load_model(environments.CURRENT_MODEL)

model_administration.fit_and_evaluate_model(
    loaded_model,
    initial_epoch,
    *model_administration.create_image_generators(),
    environments.CURRENT_MODEL
)
