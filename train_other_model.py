from desc_predic import UFODescriptorPredictor, get_lang_splits

predictor = UFODescriptorPredictor()
predictor.train_model(save_path="./outpts/text_predictor_model")
