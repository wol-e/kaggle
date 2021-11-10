from train import run

save_model=True
model_name="random_forest"
run(model_name="random_forest", fold=0, save_model=save_model)
run(model_name="random_forest", fold=1, save_model=save_model)
run(model_name="random_forest", fold=2, save_model=save_model)
run(model_name="random_forest", fold=3, save_model=save_model)
run(model_name="random_forest", fold=4, save_model=save_model)