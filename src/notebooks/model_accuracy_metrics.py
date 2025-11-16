# # %%
# y_pred = clf.predict(X_test)
# pred_series = pd.Series(y_pred, index=X_test.index, name="win_pred")

# df_with_preds = X_test.join(pred_series, how="left")
# df_with_preds["win_true"] = y_test