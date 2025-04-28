from desc_predict import UFODescriptorPredictor, get_lang_splits

train_df, val_df, _ = get_lang_splits()
predictor = UFODescriptorPredictor()
predictor.train(train_df, val_df, save_path="./outpts/text_predictor_model")
