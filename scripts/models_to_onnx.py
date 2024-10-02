# this script transforms all the models in savedmodels/models.json into
# deliverable onnx files. All the models must have an assigned values to the
# attribute self.example_input_array, or else the lightning module method
# to_onnx requires an example of an input sample.

from poregen.models.loader import load_model, list_models


def main():
    config_path = "savedmodels/production"

    model_indentifiers = [
        model_key for model_key in list_models(config_path).keys()
    ]

    for model_indentifier in model_indentifiers:
        model, config_class = load_model(
            config_path=config_path,
            model_identifier=model_indentifier
        )

        filepath = config_path + "/" + model_indentifier[:-4] + "onnx"
        model.to_onnx(filepath, export_params=True)


if __name__ == "__main__":

    main()
