# @Time    : 2019/5/22 11:26
# @Author  : shakespere
# @FileName: sum.py
import pandas as pd

submission_1 = pd.read_csv("./data/merge_0.8550913438849271_predictions.csv")
submission_2 = pd.read_csv("./data/merge_0.8551243481873769_predictions.csv")
submission_3 = pd.read_csv("./data/merge_0.8571411176454415_predictions.csv")
submission_4 = pd.read_csv("./data/merge_0.8582128855527719_predictions.csv")
submission_5 = pd.read_csv("./data/merge_0.8585647873963975_predictions.csv")
submission_6 = pd.read_csv("./data/merge_0.8599225290804536_predictions.csv")
submission_7 = pd.read_csv("./data/merge_0.860564284049377_predictions.csv")
submission_8 = pd.read_csv("./data/merge_0.8606908440533374_predictions.csv")

submission = pd.DataFrame.from_dict({
    'ID': submission_1['ID'],
    'Pred': (submission_1.Pred.values * 0.125) + (submission_2.Pred.values * 0.125) + (submission_3.Pred.values * 0.125) + (submission_4.Pred.values * 0.125)+ (submission_5.Pred.values * 0.125)+ (submission_6.Pred.values * 0.125)+ (submission_7.Pred.values * 0.125)+ (submission_8.Pred.values * 0.125)
})

submission.to_csv('./data/submission.csv', index=False)