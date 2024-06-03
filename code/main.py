import model_administration
import environments


for current_model in range(environments.CURRENT_MODEL, 11):

    loaded_model, initial_epoch = model_administration.load_model(current_model)

    model_administration.fit_and_evaluate_model(
        loaded_model,
        initial_epoch,
        *model_administration.create_image_generators(),
        current_model
    )
