import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np,pandas as pd,matplotlib.pyplot as plt,seaborn as sns
from scipy.spatial.distance import cdist,pdist,squareform
from scipy.stats import mannwhitneyu
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler,LabelEncoder
RANDOM_STATE=42; N_PERMUTATIONS=100; np.random.seed(RANDOM_STATE); sns.set_theme(style="whitegrid")
root=Path.cwd().parent if (Path.cwd().parent/"Data").exists() else Path.cwd()
df=pd.read_excel(root/"Data"/"dummy_dga_dataset.xlsx")
T="fault_by_rogers"; raw=["H2","C2H2","C2H4","C2H6","CH4","CO","CO2"]; ratios=["R_C2H2_C2H4","R_CH4_H2","R_C2H4_C2H6"]; features=raw+ratios
classes=sorted(df[T].unique()); y=LabelEncoder().fit_transform(df[T]); labeler=LabelEncoder().fit(df[T])
X=StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(df[features]))
print(f"V2.5 local-signal diagnostic: {len(df)} observations, {len(features)} supplied DGA features.")
print("Features are median-imputed then standardised only so distance is not dominated by CO/CO2 scale. No data values are changed.")
display(pd.DataFrame({"count":df[T].value_counts(),"proportion":df[T].value_counts(normalize=True)}).sort_index())

# Find the same fixed ten nearest neighbours for every observation, excluding itself.
nn=NearestNeighbors(n_neighbors=11,metric="euclidean").fit(X)
dist,ind=nn.kneighbors(X); dist,ind=dist[:,1:],ind[:,1:]
neighbour_labels=y[ind]
def purity(labels):
 return {f"{k}-neighbour purity":(neighbour_labels[:,:k]==labels[:,None]).mean() for k in (1,3,5,10)}
observed=purity(y)
# Permuting labels retains geometry and class balance, testing whether local labels are more coherent than chance.
null={k:[] for k in observed}
for seed in range(RANDOM_STATE,RANDOM_STATE+N_PERMUTATIONS):
 p=np.random.default_rng(seed).permutation(y)
 for k,v in purity(p).items(): null[k].append(v)
rows=[]
for k,v in observed.items():
 a=np.asarray(null[k]); rows.append([k,v,a.mean(),a.std(ddof=1),np.quantile(a,.95),(1+(a>=v).sum())/(N_PERMUTATIONS+1),v-a.mean()])
purity_table=pd.DataFrame(rows,columns=["metric","observed","permutation_mean","permutation_sd","permutation_95th_pct","empirical_one_sided_p","observed_minus_null_mean"])
print("\n1. Local label consistency: are nearest DGA neighbours assigned the same target more often than chance?")
display(purity_table.round(4))
plt.figure(figsize=(10,5)); sns.barplot(data=purity_table.melt(id_vars="metric",value_vars=["observed","permutation_mean"],var_name="source",value_name="purity"),x="metric",y="purity",hue="source")
plt.title("Observed nearest-neighbour label purity versus permuted labels"); plt.ylim(0,1); plt.tight_layout(); plt.show()

# Class-specific neighbour purity identifies which class has locally consistent geometry.
class_rows=[]
for ci,c in enumerate(classes):
 mask=y==ci
 for k in (1,3,5,10):
  class_rows.append([c,k,(neighbour_labels[mask,:k]==ci).mean(),mask.sum()])
class_purity=pd.DataFrame(class_rows,columns=["class","k","same_class_neighbour_purity","n"])
print("\nClass-specific local purity")
display(class_purity.pivot(index="class",columns="k",values="same_class_neighbour_purity").round(3))
plt.figure(figsize=(10,5)); sns.lineplot(data=class_purity,x="k",y="same_class_neighbour_purity",hue="class",marker="o")
plt.title("Local consistency by fault class"); plt.ylim(0,1); plt.tight_layout(); plt.show()

# Same-vs-different class distances and close conflicting pairs.
D=squareform(pdist(X,metric="euclidean")); iu=np.triu_indices(len(df),k=1)
pair_distance=D[iu]; same=(y[iu[0]]==y[iu[1]])
same_d=pair_distance[same]; diff_d=pair_distance[~same]
u,p=mannwhitneyu(same_d,diff_d,alternative="less")
distance_summary=pd.DataFrame([["same class",len(same_d),same_d.mean(),np.median(same_d),np.quantile(same_d,.05),np.quantile(same_d,.95)],["different class",len(diff_d),diff_d.mean(),np.median(diff_d),np.quantile(diff_d,.05),np.quantile(diff_d,.95)]],columns=["pair_type","n_pairs","mean_distance","median_distance","p05","p95"])
print("\n2. Standardised DGA distance: same-label versus different-label observation pairs")
display(distance_summary.round(4))
print(f"One-sided Mannâ€“Whitney test, same-class distances smaller: p={p:.4g}. This is descriptive evidence, not a predictive-model result.")
plt.figure(figsize=(10,5)); sns.kdeplot(same_d,label="same class",fill=True); sns.kdeplot(diff_d,label="different class",fill=True); plt.legend(); plt.title("Distance overlap: same-class vs different-class pairs"); plt.xlabel("standardised Euclidean DGA distance"); plt.tight_layout(); plt.show()

# Each class pair: standardized centroid separation relative to pooled within-class spread.
pair_rows=[]
for i,c1 in enumerate(classes):
 for j,c2 in enumerate(classes):
  if j<=i: continue
  a,b=X[y==i],X[y==j]; centre=np.linalg.norm(a.mean(0)-b.mean(0)); pooled=np.sqrt((a.var(0,ddof=1).mean()+b.var(0,ddof=1).mean())/2)
  pair_rows.append([c1,c2,centre,pooled,centre/pooled])
sep=pd.DataFrame(pair_rows,columns=["class_1","class_2","centroid_distance","pooled_feature_sd","standardised_centroid_separation"]).sort_values("standardised_centroid_separation")
print("\n3. Pairwise class separation: centroid distance divided by pooled within-class feature SD")
display(sep.round(4))
plt.figure(figsize=(10,5)); sns.barplot(data=sep,x="standardised_centroid_separation",y=sep.class_1+" vs "+sep.class_2,color="#4c78a8"); plt.title("Pairwise DGA class separation"); plt.tight_layout(); plt.show()

# The nearest cross-label observation pairs offer concrete report-ready failure examples.
cross=np.where(~same)[0]; order=np.argsort(pair_distance[cross])[:15]
records=[]
for pos in cross[order]:
 i,j=iu[0][pos],iu[1][pos]
 r={"row_a":int(df.index[i]),"class_a":df[T].iloc[i],"row_b":int(df.index[j]),"class_b":df[T].iloc[j],"standardised_distance":pair_distance[pos]}
 for f in raw: r[f]=round((df[f].iloc[i]+df[f].iloc[j])/2,3)
 records.append(r)
close_cross=pd.DataFrame(records)
print("\n4. Fifteen closest cross-label pairs. Gas columns are pair means for compact display; inspect original row IDs if needed.")
display(close_cross.round(4))

# Decision is deliberately a diagnostic checkpoint, not a route to tuning.
p1=purity_table.loc[purity_table.metric=="1-neighbour purity"].iloc[0]
p5=purity_table.loc[purity_table.metric=="5-neighbour purity"].iloc[0]
local_signal=(p1.observed>p1.permutation_95th_pct and p1.empirical_one_sided_p<.05 and p5.observed>p5.permutation_95th_pct and p5.empirical_one_sided_p<.05)
if local_signal:
 verdict="LOCAL SIGNAL DETECTED: nearest-neighbour consistency is clearly above the permutation control. This supports a carefully isolated V3 study, but does not require it."
else:
 verdict="LOCAL SIGNAL NOT ESTABLISHED: nearest DGA neighbours are not more label-coherent than the permutation control. Prioritise target-generation review before V3 tuning."
gate=pd.DataFrame([["1-neighbour purity exceeds null 95th percentile",p1.observed>p1.permutation_95th_pct,f"{p1.observed:.4f} vs {p1.permutation_95th_pct:.4f}; p={p1.empirical_one_sided_p:.4f}"],["5-neighbour purity exceeds null 95th percentile",p5.observed>p5.permutation_95th_pct,f"{p5.observed:.4f} vs {p5.permutation_95th_pct:.4f}"],["Same-class distances are lower than different-class distances",p<.05,f"Mann-Whitney p={p:.4g}"]],columns=["diagnostic criterion","pass","evidence"])
print("\n5. V2.5 local-signal gate"); display(gate)
print("="*78+"\nV2.5 CONCLUSION\n"+verdict)
print("Provider question: Was fault_by_rogers generated directly from supplied DGA values/ratios, from hidden inputs, or independently? Please provide the target-generation rule and any withheld variables.")

