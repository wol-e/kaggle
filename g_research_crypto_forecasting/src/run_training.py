from train import run

save_model = True
model_name = "light_gbm_benchmark"
validate = False  # Set to True for scoring model on validatino set
run(model_name=model_name, fold=0, save_model=save_model, validate=validate)
run(model_name=model_name, fold=1, save_model=save_model, validate=validate)
run(model_name=model_name, fold=2, save_model=save_model, validate=validate)
run(model_name=model_name, fold=3, save_model=save_model, validate=validate)
run(model_name=model_name, fold=4, save_model=save_model, validate=validate)
