from zenml import pipeline
from zenml.integrations.evidently.steps import evidently_profile_step
from steps.sklearn_dataloader import training_data_loader


@pipeline(enable_cache=False)
def data_report_pipeline():
    x_train, x_test, y_train, y_test = training_data_loader()
    evidently_profile_step(reference_dataset=x_train, comparison_dataset=x_test,
                           profile_sections=["datadrift"])
    evidently_profile_step(reference_dataset=y_train, comparison_dataset=y_test,
                           profile_sections=["datadrift"])


data_report_pipeline()
