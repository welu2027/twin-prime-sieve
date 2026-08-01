import math
import class_conditional_gaps as m

# empirical data
emp = {
11:(1.014089,0.002024),17:(1.017468,0.002004),29:(1.047188,0.002009),
41:(1.036196,0.001977),59:(1.034974,0.002023),71:(1.026757,0.001993),
101:(0.989765,0.002008),107:(1.022417,0.002013),137:(0.963056,0.002016),
149:(0.955327,0.001988),167:(0.957303,0.002023),179:(0.962563,0.002030),
191:(0.967870,0.002025),197:(0.995225,0.002017),209:(1.009803,0.002024)}

# density-weighted average of predictions over the window [1e5, 1e9], weight ~ e^u/u^2
us = [11.5 + i*(math.log(1e9)-11.5)/12 for i in range(13)]
wts = [math.exp(u)/u**2 for u in us]
W = sum(wts)

pred_mean = {r:0.0 for r in m.CLASSES}
pred_wheel = {r:0.0 for r in m.CLASSES}
grand_acc = 0.0
for u,w in zip(us,wts):
    mf = {r: m.class_mean_gap(r,u) for r in m.CLASSES}
    mw = {r: m.class_mean_gap(r,u,wheel_only=True) for r in m.CLASSES}
    for r in m.CLASSES:
        pred_mean[r]  += w*mf[r]
        pred_wheel[r] += w*mw[r]
    grand_acc += w*sum(mf.values())/15

for r in m.CLASSES:
    pred_mean[r]/=W; pred_wheel[r]/=W
grand = sum(pred_mean.values())/15
grandw = sum(pred_wheel.values())/15
pred = {r: pred_mean[r]/grand for r in m.CLASSES}
predw = {r: pred_wheel[r]/grandw for r in m.CLASSES}
print(f"predicted grand mean gap (weighted over window): {grand:.1f}  (empirical: 292.0)")

# comparison stats
import statistics
print(f"\n{'r':>5} {'emp':>9} {'pred':>9} {'resid':>9} {'z':>7} {'pred_wheelonly':>15}")
chi2=0; n=0
xs=[];ys=[]
for r in m.CLASSES:
    e,se = emp[r]; p=pred[r]
    z=(e-p)/se; chi2+=z*z; n+=1
    xs.append(p-1); ys.append(e-1)
    print(f"{r:>5} {e:>9.4f} {p:>9.4f} {e-p:>+9.4f} {z:>+7.1f} {predw[r]:>15.4f}")
mx=sum(xs)/n; my=sum(ys)/n
sxx=sum((x-mx)**2 for x in xs); syy=sum((y-my)**2 for y in ys)
sxy=sum((x-mx)*(y-my) for x,y in zip(xs,ys))
r2 = sxy*sxy/(sxx*syy)
slope = sxy/sxx
# rank correlation
def ranks(v):
    s=sorted(range(len(v)),key=lambda i:v[i]); rk=[0]*len(v)
    for j,i in enumerate(s): rk[i]=j
    return rk
rx,ry=ranks(xs),ranks(ys)
d2=sum((a-b)**2 for a,b in zip(rx,ry))
rho_s = 1-6*d2/(n*(n*n-1))
print(f"\nchi2 = {chi2:.0f}  (dof=15)   r^2 = {r2:.4f}   slope = {slope:.3f}   Spearman = {rho_s:.4f}")
print(f"deviation spread: empirical {max(e for e,_ in emp.values())-min(e for e,_ in emp.values()):.4f}, predicted {max(pred.values())-min(pred.values()):.4f}")
resid = {r: emp[r][0]-pred[r] for r in m.CLASSES}
print("largest residuals:", sorted(resid.items(), key=lambda kv:-abs(kv[1]))[:4])
