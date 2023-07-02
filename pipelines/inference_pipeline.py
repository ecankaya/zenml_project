from zenml import pipeline
from zenml.integrations.evidently.steps import evidently_profile_step
from steps.flower_predictions import infer, predict
from steps.sklearn_dataloader import training_data_loader


@pipeline(enable_cache=False)
def inference_pipeline():
    x_train, x_test, y_train, y_test = infer()
    x_train2, x_test2, y_train2, y_test2 = training_data_loader()
    evidently_profile_step(reference_dataset=x_train2, comparison_dataset=x_train,
                           profile_sections=["datadrift"])
    predictions = predict(x_train)
    evidently_profile_step(reference_dataset=y_train, comparison_dataset=predictions,
                           profile_sections=["datadrift"])


inference_pipeline()
