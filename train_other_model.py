from desc_predict import UFODescriptorPredictor, get_lang_splits, precompute_embeddings

train_df, val_df, test_df = get_lang_splits()
predictor = UFODescriptorPredictor()
predictor.train(train_df, val_df, save_path="./outpts/text_predictor_model")

precompute_embeddings(test_df)
