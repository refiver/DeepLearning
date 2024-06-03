import model_administration
from models import model5
from models import model6
from models import model7
import environments


for current_model in range(environments.CURRENT_MODEL, 11):

    loaded_model, initial_epoch = model_administration.load_model(current_model)
    match current_model:
        case 5:
            model_administration.fit_and_evaluate_model(
                loaded_model,
                initial_epoch,
                *model5.create_image_generators(),
                current_model
            )
        case 6:
            model_administration.fit_and_evaluate_model(
                loaded_model,
                initial_epoch,
                *model6.create_image_generators(),
                current_model
            )
        case 7:
            model_administration.fit_and_evaluate_model(
                loaded_model,
                initial_epoch,
                *model7.create_image_generators(),
                current_model
            )
        case _:
            model_administration.fit_and_evaluate_model(
                loaded_model,
                initial_epoch,
                *model_administration.create_image_generators(),
                current_model
            )

