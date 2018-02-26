with open('tiramisu_fc_dense_model.json') as model_file:
    tiramisu = model_from_json(model_file.read())
    tiasmisu.load_weights('weights/tiramisu_weights.best.hdf5')
    tiramisu.predict(x, batch_size=1)
