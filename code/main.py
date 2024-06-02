
import model_administration as model_administration

loaded_model, initial_epoch = model_administration.load_model()

model_administration.fit_and_evaluate_model(
    loaded_model,
    initial_epoch,
    *model_administration.create_image_generators())
