def precision_at(threshold, iou):
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) == 1   # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Missed objects
    false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
    tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
    return tp, fp, fn
  
  
for i in range(len(X_test)): 
  precision = []

  true_objects = len(np.unique(Y_test[i]))
  pred_objects = len(np.unique(preds_test_t_1[i]))

  intersection = np.histogram2d(Y_test[i].flatten(), preds_test_t_1[i].flatten(), bins=(true_objects, pred_objects))[0]

  # Compute areas (needed for finding the union between all objects)
  area_true = np.histogram(Y_test[i], bins = true_objects)[0]
  area_pred = np.histogram(preds_test_t_1[i], bins = pred_objects)[0]
  area_true = np.expand_dims(area_true, -1)
  area_pred = np.expand_dims(area_pred, 0)

  # Compute union
  union = area_true + area_pred - intersection

  # Exclude background from the analysis
  intersection = intersection[1:,1:]
  union = union[1:,1:]
  union[union == 0] = 1e-9

  # Compute the intersection over union
  iou = (intersection / union)
  prec = []
  for t in np.arange(0.5, 1.0, 0.05):
      tp, fp, fn = precision_at(t, iou)
      p = (1/t) * (tp / (tp + fp + fn))
      prec.append(p)



avg_precision = np.mean(prec)
print(avg_precision)
print(iou)
dice_coefficient = (2*intersection)/((area_true + area_pred)[1:,1:])
print(dice_coefficient)
  
 
