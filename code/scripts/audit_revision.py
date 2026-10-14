#!/usr/bin/env python3
"""Independent exact identities, high-precision constants and nonuniform-load checks."""
from pathlib import Path
import sys,json
import numpy as np
import sympy as sp
import mpmath as mp
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
from sgbeam import green as gr,solvers as so
x,L,a=sp.symbols('x L a',positive=True)
phi=x**2*(L-x)**2/(24*a**2)+(6*x*x-6*L*x+L*L)/(12*a**4)+1/a**6-(L*L/(12*a**4)+1/a**6)*sp.cosh(a*(x-L/2))/sp.cosh(a*L/2)
checks={'uniform-load differential equation':-sp.diff(phi,x,6)+a*a*sp.diff(phi,x,4)-1}
for end in [0,L]:
 for label,expr in [('u',phi),('u_second',sp.diff(phi,x,2)),('boundary_combination',sp.diff(phi,x,3)-a*a*sp.diff(phi,x))]:
  checks[label+' at '+str(end)]=expr.subs(x,end)
out={}
for key,e in checks.items():
 assert sp.simplify(e)==0,key
 out[key]=True
 print(key+': exact zero',flush=True)
k=sp.symbols('k',positive=True)
D=1-16/k**2+384/k**4-(32/k**2+384/k**4)/sp.cosh(k/2)
assert sp.limit(D/k**2,k,0)==sp.Rational(11,120)
assert sp.limit(k**2*(1-D),k,sp.oo)==16
out['size_asymptotics']={'small':'11/120','large':'16'}
# Translate the analytic coefficient representation to independent mpmath arithmetic.
s=sp.symbols('s',positive=True)
A1=-(1-s)**2*s/a**4;B1=(1-3*s*s+2*s**3)/a**4
C1=-(1-s)**2*s/(2*a*a);D1=(1-3*s*s+2*s**3)/(6*a*a)
den=2*a**5*sp.sinh(a)
E1=-(a*s*(1-s)*((1-s)*sp.exp(-a)-s)+sp.sinh(a*(1-s)))/den
F1=(a*s*(1-s)*((1-s)*sp.exp(a)-s)+sp.sinh(a*(1-s)))/den
A2=(a*a*s+12-6*s)*s*s/(6*a**4);B2=-(a*a+6-4*s)*s*s/(2*a**4)
C2=(2-s)*s*s/(2*a*a);D2=-(3-2*s)*s*s/(6*a*a)
E2=-(a*s*(1-s)*((1-s)*sp.exp(-a)-s)-sp.exp(-a)*sp.sinh(a*s))/den
F2=(a*s*(1-s)*((1-s)*sp.exp(a)-s)-sp.exp(a)*sp.sinh(a*s))/den
kl=sp.lambdify((x,s,a),-(A1+B1*x+C1*x*x+D1*x**3+E1*sp.exp(a*x)+F1*sp.exp(-a*x)),'mpmath')
kr=sp.lambdify((x,s,a),-(A2+B2*x+C2*x*x+D2*x**3+E2*sp.exp(a*x)+F2*sp.exp(-a*x)),'mpmath')
mp.mp.dps=65
constants={}
for alpha in [1,5]:
 kernel=lambda xx,ss:kl(xx,ss,alpha) if xx<=ss else kr(xx,ss,alpha)
 av=mp.quad(lambda ss:kernel(mp.mpf('.5'),ss),[0,mp.mpf('.5'),1])
 bv=mp.quad(lambda ss:kernel(mp.mpf('.25'),ss),[mp.mpf('.25'),mp.mpf('.75')])
 assert abs(float(av)-gr.A_const(alpha))<1e-16
 assert abs(float(bv)-gr.B_const(.25,alpha))<1e-15
 constants[str(alpha)]={'A':mp.nstr(av,40),'B':mp.nstr(bv,40),'eta_H':mp.nstr(mp.sinh(mp.mpf(alpha)/4)/mp.sinh(alpha),40)}
 print('High-precision constants '+str(alpha)+': '+str(constants[str(alpha)]),flush=True)
out['high_precision_constants']=constants
# Exact asymmetric solution with positive, nonconstant load on a nonunit interval.
p=x*x*(L-x)**2*(1+sp.Rational(1,5)*x/L)
v=p/a**2+sp.diff(p,x,2)/a**4+sp.diff(p,x,4)/a**6
u=v-v.subs(x,0)*sp.sinh(a*(L-x))/sp.sinh(a*L)-v.subs(x,L)*sp.sinh(a*x)/sp.sinh(a*L)
h=sp.diff(p,x,4)
assert sp.simplify(-sp.diff(u,x,6)+a*a*sp.diff(u,x,4)-h)==0
for end in [0,L]:
 for expr in [u,sp.diff(u,x,2),sp.diff(u,x,3)-a*a*sp.diff(u,x)]:assert sp.simplify(expr.subs(x,end))==0
exact=sp.lambdify(x,u.subs({L:sp.Rational(17,10),a:2}),'numpy');load=sp.lambdify(x,h.subs(L,sp.Rational(17,10)),'numpy')
f=lambda t,z:load(t);fu=lambda t,z:np.zeros_like(z)
xx=np.linspace(0,1.7,801);uu=exact(xx);scale=max(uu)
gn=so.GreenNystrom(2,L=1.7,n=48);z=gn.solve(f,fu)
err=float(max(abs(gn.evaluate(xx,f,z)-uu))/scale)
fd=[];fe=[]
for n in [25,50,100]:
 t,z=so.solve_fd(f,fu,2,L=1.7,N=n);fd.append(float(max(abs(z-exact(t)))/scale))
for n in [8,16,32]:
 fem=so.MixedFEM(2,L=1.7,ne=n);fem.solve(f,fu);fe.append(float(max(abs(fem.evaluate(xx)-uu))/scale))
ofd=np.log2(np.array(fd[:-1])/np.array(fd[1:]));ofe=np.log2(np.array(fe[:-1])/np.array(fe[1:]))
assert err<1e-9 and np.all((ofd>1.9)&(ofd<2.1)) and np.all(ofe>3.5)
out['manufactured_nonconstant_positive_load']={'L':1.7,'alpha':2,'kernel_relative_error':err,'fd_errors':fd,'fd_orders':ofd.tolist(),'fe_errors':fe,'fe_orders':ofe.tolist()}
print('Manufactured solution: '+str(out['manufactured_nonconstant_positive_load']),flush=True)
r=json.loads((ROOT/'code/results/results.json').read_text());folds=[];f=lambda x,u:np.exp(u)
for row in r['example2']['rows']:
 alpha=row['alpha'];gn=so.GreenNystrom(alpha,n=64,m=52);shape=gr.uniform_load_response(gn.nodes,alpha)
 uf,muf=so.fold_point(gn,f,f,f,1.4*shape/shape.max(),row['mu_fold'])
 residual=float(max(abs(uf-muf*gn.W@np.exp(uf))));relative=abs(muf/row['mu_fold']-1)
 assert residual<1e-10 and relative<1e-9
 folds.append({'alpha':alpha,'fold_48':row['mu_fold'],'fold_64':muf,'relative_difference':relative,'nodal_residual':residual})
out['fold_refinement']=folds
print('Fold refinement: '+str(folds),flush=True)
(ROOT/'code/results/audit.json').write_text(json.dumps(out,indent=2))
print('ALL INDEPENDENT AUDIT CHECKS PASSED',flush=True)
