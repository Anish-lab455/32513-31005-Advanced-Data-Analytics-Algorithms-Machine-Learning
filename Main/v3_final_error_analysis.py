import warnings; warnings.filterwarnings("ignore")
import sys,subprocess,importlib.util
req={"numpy":"numpy","pandas":"pandas","matplotlib":"matplotlib","seaborn":"seaborn","sklearn":"scikit-learn","openpyxl":"openpyxl","xgboost":"xgboost"}
miss=[p for m,p in req.items() if importlib.util.find_spec(m) is None]
if miss: subprocess.check_call([sys.executable,"-m","pip","install","-q",*miss])
from pathlib import Path
import numpy as np,pandas as pd,matplotlib.pyplot as plt,seaborn as sns
from scipy.spatial.distance import pdist,squareform
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,classification_report,confusion_matrix,brier_score_loss
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder,StandardScaler
from xgboost import XGBClassifier

RANDOM_STATE=42; N_SPLITS=5; N_REPEATS=5; np.random.seed(RANDOM_STATE); sns.set_theme(style="whitegrid")
root=Path.cwd().parent if (Path.cwd().parent/"Data").exists() else Path.cwd()
df=pd.read_excel(root/"Data"/"dummy_dga_dataset.xlsx")
T="fault_by_rogers"; raw=["H2","C2H2","C2H4","C2H6","CH4","CO","CO2"]; ratios=["R_C2H2_C2H4","R_CH4_H2","R_C2H4_C2H6"]; features=raw+ratios
encoder=LabelEncoder(); y=encoder.fit_transform(df[T]); classes=list(encoder.classes_); labels=np.arange(len(classes)); X=df[features].copy()
print("V3 final fixed-model evaluation and error analysis")
print("The V1/V2 XGBoost configuration is used without tuning. Scores are repeated out-of-fold predictions only; asset/time are not predictive features.")

def fixed_xgb():
 return Pipeline([("imputer",SimpleImputer(strategy="median")),("model",XGBClassifier(objective="multi:softprob",num_class=len(classes),n_estimators=150,max_depth=3,learning_rate=.05,subsample=.9,colsample_bytree=.9,reg_lambda=1.0,eval_metric="mlogloss",random_state=RANDOM_STATE,n_jobs=1,tree_method="hist"))])

# Every row is held out once per repeat. Aggregate its five independent OOF probability vectors.
cv=RepeatedStratifiedKFold(n_splits=N_SPLITS,n_repeats=N_REPEATS,random_state=RANDOM_STATE)
prob_sum=np.zeros((len(df),len(classes))); n_oof=np.zeros(len(df),dtype=int)
fold_rows=[]
for fold,(tr,te) in enumerate(cv.split(X,y),start=1):
 m=fixed_xgb().fit(X.iloc[tr],y[tr]); p=m.predict_proba(X.iloc[te]); pr=p.argmax(axis=1)
 prob_sum[te]+=p; n_oof[te]+=1
 fold_rows.append([fold,f1_score(y[te],pr,average="macro",labels=labels,zero_division=0),accuracy_score(y[te],pr),balanced_accuracy_score(y[te],pr)])
assert np.all(n_oof==N_REPEATS), "Each observation must have one OOF prediction per repeat."
oof_prob=prob_sum/n_oof[:,None]; oof_pred=oof_prob.argmax(axis=1); confidence=oof_prob.max(axis=1); correct=oof_pred==y
fold_scores=pd.DataFrame(fold_rows,columns=["validation_fold","macro_f1","accuracy","balanced_accuracy"])
summary=pd.DataFrame([["OOF repeated-CV aggregate",accuracy_score(y,oof_pred),balanced_accuracy_score(y,oof_pred),f1_score(y,oof_pred,average="macro",labels=labels,zero_division=0),f1_score(y,oof_pred,average="weighted",labels=labels,zero_division=0),fold_scores.macro_f1.mean(),fold_scores.macro_f1.std(ddof=1)]],columns=["evaluation","accuracy","balanced_accuracy","macro_f1","weighted_f1","mean_fold_macro_f1","fold_macro_f1_sd"])
print("\n1. Repeated out-of-fold aggregate performance")
display(summary.round(4))
print("This aggregate combines five held-out probability predictions per observation. Mean fold macro-F1 remains the primary V2 comparison quantity.")
plt.figure(figsize=(10,4)); sns.stripplot(data=fold_scores,y="macro_f1",jitter=.15); plt.axhline(fold_scores.macro_f1.mean(),color="crimson",ls="--",label="mean"); plt.legend(); plt.title("25 fixed-model held-out fold macro-F1 values"); plt.tight_layout(); plt.show()

# Per-class report and error matrix.
report=pd.DataFrame(classification_report(y,oof_pred,target_names=classes,output_dict=True,zero_division=0)).T
per_class=report.loc[classes,["precision","recall","f1-score","support"]].rename(columns={"f1-score":"f1"})
print("\n2. Per-class held-out error analysis")
display(per_class.round(4))
cm=confusion_matrix(y,oof_pred,labels=labels); cm_norm=confusion_matrix(y,oof_pred,labels=labels,normalize="true")
fig,ax=plt.subplots(1,2,figsize=(16,6))
sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=classes,yticklabels=classes,ax=ax[0]); ax[0].set(title="OOF confusion matrix: counts",xlabel="Predicted",ylabel="Actual")
sns.heatmap(cm_norm,annot=True,fmt=".2f",cmap="Blues",xticklabels=classes,yticklabels=classes,ax=ax[1]); ax[1].set(title="OOF confusion matrix: row-normalised",xlabel="Predicted",ylabel="Actual")
for z in ax:
 z.tick_params(axis="x",rotation=30); z.tick_params(axis="y",rotation=0)
plt.tight_layout(); plt.show()
errors=[]
for i,a in enumerate(classes):
 for j,pred in enumerate(classes):
  if i!=j and cm[i,j]>0: errors.append([a,pred,int(cm[i,j]),cm_norm[i,j]])
error_table=pd.DataFrame(errors,columns=["actual_class","predicted_class","count","within_actual_class_rate"]).sort_values(["count","within_actual_class_rate"],ascending=False)
print("Most frequent off-diagonal errors")
display(error_table.round(4))

# Confidence is descriptive: assess whether probability concentration distinguishes correct from incorrect OOF predictions.
result=df[["row_id","asset","ts_H2",T]].copy()
result["oof_prediction"]=encoder.inverse_transform(oof_pred); result["correct"]=correct; result["max_predicted_probability"]=confidence
for i,c in enumerate(classes): result["p_"+c.replace(" ","_").replace("-","_")]=oof_prob[:,i]
confidence_summary=result.groupby("correct")["max_predicted_probability"].agg(["count","mean","median","min","max"])
bins=np.linspace(.25,.70,6); result["confidence_bin"]=pd.cut(result["max_predicted_probability"],bins=bins,include_lowest=True)
confidence_bins=result.groupby("confidence_bin",observed=False).agg(n=("correct","size"),accuracy=("correct","mean"),mean_confidence=("max_predicted_probability","mean")).reset_index()
print("\n3. Probability-confidence diagnostic")
display(confidence_summary.round(4)); display(confidence_bins.round(4))
plt.figure(figsize=(10,5)); sns.histplot(data=result,x="max_predicted_probability",hue="correct",bins=18,stat="density",common_norm=False,element="step"); plt.title("OOF maximum predicted probability: correct vs incorrect predictions"); plt.tight_layout(); plt.show()
plt.figure(figsize=(8,5)); sns.barplot(data=confidence_bins,x="confidence_bin",y="accuracy",color="#4c78a8"); plt.ylim(0,1); plt.xticks(rotation=25); plt.title("Observed OOF accuracy by confidence bin"); plt.tight_layout(); plt.show()
# Multiclass Brier score, averaged across one-vs-all targets.
brier=np.mean([brier_score_loss((y==i).astype(int),oof_prob[:,i]) for i in labels])
print(f"Mean one-vs-rest Brier score: {brier:.4f} (descriptive probability-quality measure; not a tuning target).")

# V2.5 qualitative evidence: closest pairs with different labels in exactly the same standardised feature space.
Z=StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(X)); D=squareform(pdist(Z)); iu=np.triu_indices(len(df),1); cross=y[iu[0]]!=y[iu[1]]; positions=np.where(cross)[0]; close=positions[np.argsort(D[iu][positions])[:15]]
pair_rows=[]
for pos in close:
 i,j=iu[0][pos],iu[1][pos]
 pair_rows.append([int(df.row_id.iloc[i]),df[T].iloc[i],result.oof_prediction.iloc[i],round(float(confidence[i]),3),int(df.row_id.iloc[j]),df[T].iloc[j],result.oof_prediction.iloc[j],round(float(confidence[j]),3),round(float(D[i,j]),4)])
close_pairs=pd.DataFrame(pair_rows,columns=["row_id_a","actual_a","OOF_pred_a","confidence_a","row_id_b","actual_b","OOF_pred_b","confidence_b","standardised_DGA_distance"])
print("\n4. Closest cross-label DGA pairs from V2.5 feature geometry")
display(close_pairs)
print("These are qualitative failure evidence only: a pair may not itself be misclassified, but close different labels demonstrate target overlap in observed feature space.")

# Representative high-confidence errors are particularly important when the representation is weak.
wrong=result.loc[~result.correct,["row_id","asset","ts_H2",T,"oof_prediction","max_predicted_probability"]].sort_values("max_predicted_probability",ascending=False).head(15)
print("\n5. Fifteen most confident incorrect OOF predictions")
display(wrong.round(4))

print("="*78)
print("V3 FINAL ERROR-ANALYSIS CONCLUSION")
print("The fixed XGBoost model produces substantial, class-specific held-out confusion. Its probability concentration and high-confidence errors should be interpreted alongside V2's failed permutation gate and V2.5's absent local label consistency. These results characterise failure; they do not justify tuning.")

