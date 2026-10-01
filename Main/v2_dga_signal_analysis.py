import warnings; warnings.filterwarnings("ignore")
import sys, subprocess, importlib.util
_required = {"numpy":"numpy", "pandas":"pandas", "matplotlib":"matplotlib", "seaborn":"seaborn", "scipy":"scipy", "sklearn":"scikit-learn", "openpyxl":"openpyxl", "xgboost":"xgboost"}
_missing = [package for module, package in _required.items() if importlib.util.find_spec(module) is None]
if _missing:
    print("Installing missing V2 dependencies:", _missing)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *_missing])
from pathlib import Path
import numpy as np,pandas as pd,matplotlib.pyplot as plt,seaborn as sns
from scipy.stats import chi2_contingency,kruskal
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score,accuracy_score,balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold,cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder,StandardScaler
from xgboost import XGBClassifier
RANDOM_STATE=42; N_PERMUTATIONS=100; np.random.seed(RANDOM_STATE); sns.set_theme(style="whitegrid")
root=Path.cwd().parent if (Path.cwd().parent/"Data").exists() else Path.cwd()
df=pd.read_excel(root/"Data"/"dummy_dga_dataset.xlsx")
T,A,DT="fault_by_rogers","asset","ts_H2"; raw=["H2","C2H2","C2H4","C2H6","CH4","CO","CO2"]; ratios=["R_C2H2_C2H4","R_CH4_H2","R_C2H4_C2H6"]; features=raw+ratios
df[DT]=pd.to_datetime(df[DT]); classes=sorted(df[T].unique()); y=LabelEncoder().fit_transform(df[T]); labels=np.arange(len(classes)); X=df[features]
print(f"Rows={len(df)}; time range: {df[DT].min()} to {df[DT].max()}"); display(pd.DataFrame({"count":df[T].value_counts(),"proportion":df[T].value_counts(normalize=True)}).sort_index())

# 1. Target relation checks: diagnostic fields remain excluded from models.
print("\n1. Synthetic target-generation / label relationship checks")
for c in ["status","abnormal","triggers"]:
 print(f"\n{c} by target"); display(pd.crosstab(df[c].fillna("<missing>"),df[T],normalize="index").round(3))
mi=mutual_info_classif(X,y,random_state=RANDOM_STATE); out=[]
for c in features:
 h,p=kruskal(*[g[c].dropna() for _,g in df.groupby(T)]); out.append([c,h,p,max(0,(h-len(classes)+1)/(len(df)-len(classes)))])
tbl=pd.DataFrame(out,columns=["feature","Kruskal_H","p_unadjusted","epsilon_squared"]); tbl["mutual_information"]=mi; display(tbl.sort_values("epsilon_squared",ascending=False).round(4))
print("The data cannot prove its hidden generator. Provider question: which DGA fields/rules/thresholds/transformations/randomisation generated fault_by_rogers, and were labels independent of DGA?")

# 2. Separability plots and PCA, never a model.
print("\n2. Class separability and feature distributions")
fig,ax=plt.subplots(2,4,figsize=(18,9))
for z,c in zip(ax.flat,raw): sns.boxplot(data=df,x=T,y=c,order=classes,showfliers=False,ax=z); z.set_title(c+" by class"); z.tick_params(axis="x",rotation=25)
ax.flat[-1].axis("off"); plt.tight_layout(); plt.show()
fig,ax=plt.subplots(2,2,figsize=(13,11))
for z,(x1,x2) in zip(ax.flat,[("H2","CH4"),("H2","C2H2"),("C2H4","C2H6"),("C2H2","C2H4")]): sns.scatterplot(data=df,x=x1,y=x2,hue=T,hue_order=classes,alpha=.75,ax=z); z.set_title(f"{x2} vs {x1}")
plt.tight_layout(); plt.show()
xx=Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler())]).fit_transform(X); pc=PCA(n_components=2,random_state=42); p=pc.fit_transform(xx)
print(f"PCA PC1+PC2 variance explained: {pc.explained_variance_ratio_.sum():.1%}")
plt.figure(figsize=(10,7)); sns.scatterplot(data=pd.DataFrame({"PC1":p[:,0],"PC2":p[:,1],T:df[T]}),x="PC1",y="PC2",hue=T,hue_order=classes,alpha=.8); plt.title("PCA visual diagnostic only"); plt.tight_layout(); plt.show()

# 3. Asset diagnostic; asset is never in X.
print("\n3. Asset-target relationship")
ac=pd.crosstab(df[A],df[T]).reindex(columns=classes,fill_value=0); chi,pv,_,_=chi2_contingency(ac); v=np.sqrt((chi/ac.to_numpy().sum())/min(ac.shape[0]-1,ac.shape[1]-1)); display(ac); print(f"chi-square p={pv:.4g}; Cramer's V={v:.3f}")
ac.div(ac.sum(axis=1),axis=0).plot(kind="bar",stacked=True,figsize=(10,5),colormap="tab10"); plt.ylabel("within-asset class proportion"); plt.tight_layout(); plt.show()

# 4. Temporal diagnostics; timestamp is never in X.
print("\n4. Temporal structure"); display(df.groupby(A)[DT].agg(["count","min","max"]))
q=df[[DT,A,T]].sort_values(DT).copy(); q["day"]=q[DT].dt.date
fig,ax=plt.subplots(2,1,figsize=(13,9),sharex=True); q.groupby("day").size().plot(ax=ax[0],marker="o"); ax[0].set_title("Rows by day"); pd.crosstab(q["day"],q[T],normalize="index").reindex(columns=classes,fill_value=0).plot(kind="bar",stacked=True,ax=ax[1],colormap="tab10"); ax[1].set_title("Class composition by day"); plt.tight_layout(); plt.show()
a=q[A].to_numpy(); print(f"Chronologically adjacent rows from same asset: {(a[1:]==a[:-1]).mean():.1%}")

# Exact V1 XGBoost configuration; intentionally no tuning.
def mod():
 return Pipeline([("i",SimpleImputer(strategy="median")),("m",XGBClassifier(objective="multi:softprob",num_class=len(classes),n_estimators=150,max_depth=3,learning_rate=.05,subsample=.9,colsample_bytree=.9,reg_lambda=1.,eval_metric="mlogloss",random_state=42,n_jobs=1,tree_method="hist"))])
cv=RepeatedStratifiedKFold(n_splits=5,n_repeats=5,random_state=42); scoring={"f1":"f1_macro","accuracy":"accuracy","balanced":"balanced_accuracy"}

# 5. Repeated CV, including train-validation gap.
print("\n5. Repeated stratified CV: 5 folds x 5 repeats")
real=cross_validate(mod(),X,y,cv=cv,scoring=scoring,return_train_score=True,n_jobs=-1); base=cross_validate(DummyClassifier(strategy="most_frequent"),X,y,cv=cv,scoring=scoring,return_train_score=True,n_jobs=-1)
def sm(n,z): return [n,z["test_f1"].mean(),z["test_f1"].std(ddof=1),z["train_f1"].mean(),z["train_f1"].mean()-z["test_f1"].mean(),z["test_accuracy"].mean(),z["test_balanced"].mean()]
cvt=pd.DataFrame([sm("Fixed XGBoost",real),sm("Majority baseline",base)],columns=["model","validation_macro_f1","fold_sd","train_macro_f1","gap","accuracy","balanced_accuracy"]); display(cvt.round(4))
plt.figure(figsize=(10,4)); sns.stripplot(data=pd.DataFrame({"validation":real["test_f1"],"training":real["train_f1"]}).melt(),x="variable",y="value",jitter=.18); plt.title("25 repeated-CV macro-F1 scores"); plt.tight_layout(); plt.show()

# 6. Repeated exact same CV under permuted labels.
print("\n6. Repeated permutation-label control: 100 null experiments")
pm=[]
for seed in range(42,142): pm.append(cross_validate(mod(),X,np.random.default_rng(seed).permutation(y),cv=cv,scoring="f1_macro",n_jobs=-1)["test_score"].mean())
pm=np.asarray(pm); rm=real["test_f1"].mean(); p95=np.quantile(pm,.95); ep=(1+(pm>=rm).sum())/101
display(pd.DataFrame([[rm,pm.mean(),pm.std(ddof=1),p95,ep]],columns=["real_mean","null_mean","null_sd","null_95th_pct","empirical_one_sided_p"]).round(4))
plt.figure(figsize=(10,5)); sns.histplot(pm,bins=15,kde=True); plt.axvline(rm,color="crimson",ls="--",label="real"); plt.axvline(p95,color="black",ls=":",label="null 95th"); plt.legend(); plt.title("Repeated permutation control"); plt.tight_layout(); plt.show()

# 7. Representation, leave-one-asset-out and chronological checks.
print("\n7. Robustness checks")
rr=[]
for n,c in {"Raw DGA":raw,"Raw + ratios":features,"Ratios only":ratios}.items():
 z=cross_validate(mod(),df[c],y,cv=cv,scoring="f1_macro",return_train_score=True,n_jobs=-1); rr.append([n,z["test_score"].mean(),z["train_score"].mean()-z["test_score"].mean()])
rt=pd.DataFrame(rr,columns=["representation","repeated_cv_macro_f1","gap"]).sort_values("repeated_cv_macro_f1",ascending=False); display(rt.round(4))
ar=[]
for hold in sorted(df[A].unique()):
 tr=df[A].ne(hold); te=~tr; m=mod().fit(X[tr],y[tr]); d=DummyClassifier(strategy="most_frequent").fit(X[tr],y[tr]); ar.append([hold,tr.sum(),te.sum(),f1_score(y[tr],m.predict(X[tr]),average="macro",labels=labels,zero_division=0),f1_score(y[te],m.predict(X[te]),average="macro",labels=labels,zero_division=0),f1_score(y[te],d.predict(X[te]),average="macro",labels=labels,zero_division=0)])
at=pd.DataFrame(ar,columns=["held_out_asset","train_n","test_n","train_macro_f1","test_macro_f1","majority_test_macro_f1"]); print("Leave-one-asset-out"); display(at.round(4))
o=df.sort_values(DT).index.to_numpy(); k=int(.7*len(o)); tr,te=o[:k],o[k:]; m=mod().fit(X.loc[tr],y[tr]); d=DummyClassifier(strategy="most_frequent").fit(X.loc[tr],y[tr])
ct=pd.DataFrame([[len(tr),len(te),f1_score(y[tr],m.predict(X.loc[tr]),average="macro",labels=labels,zero_division=0),f1_score(y[te],m.predict(X.loc[te]),average="macro",labels=labels,zero_division=0),f1_score(y[te],d.predict(X.loc[te]),average="macro",labels=labels,zero_division=0),accuracy_score(y[te],m.predict(X.loc[te])),balanced_accuracy_score(y[te],m.predict(X.loc[te]))]],columns=["train_n","test_n","train_macro_f1","test_macro_f1","majority_test_macro_f1","test_accuracy","test_balanced_accuracy"]); print("Chronological 70/30 holdout"); display(ct.round(4))

# 8. Conservative all-pass tuning gate.
bm=base["test_f1"].mean(); gap=real["train_f1"].mean()-rm; xa=at.test_macro_f1.mean(); ba=at.majority_test_macro_f1.mean()
gate=pd.DataFrame([["Real exceeds permutation",rm>p95 and ep<.05,f"real={rm:.4f}; null p95={p95:.4f}; p={ep:.4f}"],["Meaningfully exceeds majority",rm-bm>=.05,f"margin={rm-bm:.4f}; threshold=.0500"],["Some representation exceeds null",rt.iloc[0].repeated_cv_macro_f1>p95,f"best={rt.iloc[0].representation}: {rt.iloc[0].repeated_cv_macro_f1:.4f}"],["No substantial memorisation",gap<=.20,f"gap={gap:.4f}; threshold <=.2000"],["Asset and time robustness beat baselines",xa>ba and ct.loc[0,"test_macro_f1"]>ct.loc[0,"majority_test_macro_f1"],f"asset={xa:.4f}/{ba:.4f}; time={ct.loc[0,'test_macro_f1']:.4f}/{ct.loc[0,'majority_test_macro_f1']:.4f}"]],columns=["condition","pass","evidence"])
print("\n8. XGBoost tuning gate — all conditions must pass"); display(gate)
if gate["pass"].all(): verdict="GO — isolated/nested XGBoost tuning is justified in V3."
elif rm>pm.mean() and rm>bm: verdict="INVESTIGATE — weak/unstable evidence; review target construction, overlap, asset and temporal effects before tuning."
else: verdict="NO-GO / DATA–TARGET REVIEW — no reproducible DGA to fault-class signal demonstrated; tuning is not scientifically justified."
print("="*72+"\nV2 CONCLUSION\n"+verdict)
