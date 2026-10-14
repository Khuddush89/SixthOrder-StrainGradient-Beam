#!/usr/bin/env python3
"""Original, model-derived GIFs for GitHub. Motion is a parameter sweep.

Run from any directory. All axes and canvases are white. Figures are rebuilt
from the verified kernel, solver and JSON results rather than stock imagery.
"""
from pathlib import Path
import json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
from sgbeam import green as gr,solvers as so
OUT=ROOT/'assets/animations';PRE=ROOT/'assets/previews'
OUT.mkdir(parents=True,exist_ok=True);PRE.mkdir(parents=True,exist_ok=True)
C=['#0078d4','#e64c69','#009b8b','#df9500','#7654d8']
plt.rcParams.update({'font.family':'DejaVu Sans','mathtext.fontset':'stix',
    'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white',
    'axes.spines.top':False,'axes.spines.right':False,'axes.labelsize':11,
    'font.size':10,'axes.titlesize':12,'axes.titleweight':'bold',
    'axes.edgecolor':'#8793a8','xtick.color':'#4b5870','ytick.color':'#4b5870',
    'lines.linewidth':2.4,'legend.frameon':False,'axes.grid':False})
N=64
phase=(1-np.cos(2*np.pi*np.arange(N)/N))/2
r=json.loads((ROOT/'code/results/results.json').read_text())
x=np.linspace(0,1,301)
manifest={}


def curve(ax,x,y,color,label=None,**kw):
    return ax.plot(x,y,color=color,label=label,**kw)[0]


def setup(ax,xlabel,ylabel,title):
    ax.set(xlabel=xlabel,ylabel=ylabel,title=title)
    ax.tick_params(labelsize=9)


def footer(fig,words):
    fig.text(.055,.024,words,fontsize=9,color='#52617c')
    fig.text(.945,.024,'M. KHUDDUSH  /  VERIFIED MODEL',ha='right',fontsize=8,color='#52617c')


def canvas(title):
    fig,axs=plt.subplots(1,2,figsize=(13,6.6),dpi=110)
    fig.subplots_adjust(left=.08,right=.97,bottom=.17,top=.80,wspace=.31)
    fig.text(.055,.93,title,fontsize=21,fontweight='bold',color='#172842')
    fig.text(.055,.873,'STATIC BOUNDARY VALUE PROBLEM  /  ANIMATED PARAMETER EXPLORATION',fontsize=9,color=C[0])
    return fig,axs


def write_gif(fig,name,update,description):
    frames=[]
    for i,t in enumerate(phase):
        update(i,float(t));fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()))
    atlas=Image.new('RGB',(320*4,160),'white')
    for k,j in enumerate([0,N//8,N//4,N//2]):
        im=frames[j].copy();im.thumbnail((320,160));atlas.paste(im,(k*320,0))
    palette=atlas.quantize(colors=256)
    quant=[im.quantize(palette=palette,dither=Image.Dither.NONE) for im in frames]
    quant[0].save(OUT/(name+'.gif'),save_all=True,append_images=quant[1:],
                  duration=100,loop=0,disposal=2,optimize=True)
    frames[N//4].save(PRE/(name+'.png'))
    with Image.open(OUT/(name+'.gif')) as check:
        saved_frames=check.n_frames
    manifest[name]={'file':f'assets/animations/{name}.gif','frames':saved_frames,
                    'nominal_frame_duration_ms':100,'pixels':list(frames[0].size),
                    'meaning':description,'source':'code/scripts/make_animations.py'}
    plt.close(fig)
    print('Rendered '+name,flush=True)


def source_scan():
    fig,(a,b)=canvas('A moving source, a positive response')
    q=np.linspace(0,1,101);X,S=np.meshgrid(q,q);Z=1e4*gr.Gt(X,S)
    a.contour(X,S,Z,levels=np.linspace(.25,4,11),cmap='viridis',linewidths=1.4)
    horizontal=curve(a,[0,1],[.15,.15],C[1],ls='--',lw=1.8)
    diagonal=curve(a,[0,1],[0,1],'#b4bdca',ls=':',lw=1)
    a.set(xlim=(0,1),ylim=(0,1));setup(a,'evaluation position x','source position s','Positive kernel contours')
    a.set_aspect('equal')
    for s in [.15,.5,.85]:curve(b,x,1e4*gr.Gt(x,s),'#c9d5e5',lw=1.2)
    response=curve(b,x,1e4*gr.Gt(x,.15),C[0]);marker,=b.plot([],[],'o',color=C[1],ms=8)
    b.set(xlim=(0,1),ylim=(0,4.5));setup(b,'x',r'$10^4\widetilde G(x,s)$','Kernel section at the selected source')
    status=fig.text(.66,.84,'',color=C[1],fontsize=12,fontweight='bold')
    footer(fig,'L = 1, alpha = 1. Source position varies; these are static influence functions.')
    def update(i,t):
        s=.15+.70*t;y=1e4*gr.Gt(x,s)
        horizontal.set_ydata([s,s]);response.set_ydata(y)
        j=int(np.argmax(y));marker.set_data([x[j]],[y[j]])
        status.set_text(f's = {s:.3f}')
    write_gif(fig,'source_scan',update,'Source position s sweeps from 0.15 to 0.85 and back; alpha=L=1.')


def material_length():
    fig,(a,b)=canvas('Material length changes the deflection')
    classical=16*x*x*(1-x)**2
    curve(a,x,classical,'#8190a8','clamped classical limit',ls='--',lw=1.5)
    current=curve(a,x,np.zeros_like(x),C[2],'strain-gradient response')
    a.legend(loc='upper left',fontsize=9);a.set(xlim=(0,1),ylim=(0,1.1))
    setup(a,'x/L',r'$w(x)/w_{\max}^{\rm EB}$','Uniform-load profile')
    k=np.geomspace(.1,200,400);curve(b,k,gr.size_effect_factor(k),C[4])
    marker,=b.plot([],[],'o',color=C[1],ms=9)
    b.set(xscale='log',xlim=(.1,200),ylim=(0,1.05))
    setup(b,r'$\kappa=L/g$',r'$\mathcal{D}(\kappa)$','Maximal-deflection size factor')
    status=fig.text(.66,.84,'',color=C[2],fontsize=12,fontweight='bold')
    footer(fig,'Fixed uniform physical loading. The imposed constraints approach the clamped classical limit.')
    def update(i,t):
        kk=.1*2000**t;current.set_ydata(384*kk*kk*gr.uniform_load_response(x,kk))
        marker.set_data([kk],[gr.size_effect_factor(kk)])
        status.set_text(f'kappa = {kk:.3f}    g/L = {1/kk:.4f}')
    write_gif(fig,'material_length',update,'The static length ratio kappa=L/g sweeps from 0.1 to 200 and back.')


def kernel_surface():
    fig=plt.figure(figsize=(13,6.6),dpi=110)
    fig.text(.055,.93,'The geometry of the positive Green kernel',fontsize=21,fontweight='bold',color='#172842')
    fig.text(.055,.873,'ROTATING VIEW  /  FIXED MODEL: ALPHA = L = 1',fontsize=9,color=C[0])
    a=fig.add_subplot(121,projection='3d');b=fig.add_subplot(122)
    q=np.linspace(0,1,61);X,S=np.meshgrid(q,q);Z=1e4*gr.Gt(X,S)
    a.plot_wireframe(X,S,Z,rstride=3,cstride=3,color=C[0],linewidth=.65)
    a.set(xlabel='x',ylabel='s',zlabel=r'$10^4\widetilde G$',zlim=(0,4.5))
    a.set_box_aspect((1,1,.7));a.tick_params(labelsize=8)
    for axis in [a.xaxis,a.yaxis,a.zaxis]:
        axis.pane.set_facecolor((1,1,1,1));axis.pane.set_edgecolor('white')
        axis._axinfo['grid']['color']=(1,1,1,0)
    b.contour(X,S,Z,levels=np.linspace(.3,4,11),cmap='viridis',linewidths=1.5)
    b.set(xlim=(0,1),ylim=(0,1));b.set_aspect('equal')
    setup(b,'x','s','Unfilled contours on a white canvas')
    fig.subplots_adjust(left=.04,right=.96,bottom=.18,top=.80,wspace=.26)
    footer(fig,'Camera angle changes; the mathematical kernel remains fixed. No time dynamics are represented.')
    def update(i,t):a.view_init(elev=24+7*np.sin(2*np.pi*i/N),azim=-140+360*i/N)
    write_gif(fig,'kernel_surface',update,'A rotating camera displays the fixed positive response kernel at alpha=L=1.')


def cone_estimates():
    fig,(a,b)=canvas('Cone geometry and admissible bounds')
    alphas=[.5,1,2,5,10,20];ss=np.linspace(.001,.999,61)
    ratios=[]
    for alpha in alphas:
        values=gr.eta_ratio(ss,.25,alpha);end=gr.eta_endpoint_limit(.25,alpha)
        ratios.append(np.r_[end,values,end])
    xx=np.r_[0,ss,1]
    for j,values in enumerate(ratios):curve(a,xx,values,'#d5deeb',lw=1)
    line=curve(a,xx,ratios[0],C[4]);dot,=a.plot([],[],'o',color=C[1],ms=8)
    a.set(xlim=(0,1),ylim=(.28,.82));setup(a,'source position s','optimized ratio r(s)','Sampled ratio curves, delta = 1/4')
    aa=np.geomspace(.5,30,200);hh=np.array([gr.eta_helmholtz(.25,v) for v in aa])
    curve(b,aa,hh,C[2],r'analytical $\eta_H$')
    data=r['constants']['eta_vs_alpha'];numerical=np.asarray(data['delta=0.25'])
    curve(b,data['alpha'],numerical,C[4],r'estimated $\eta_\ast$',marker='o',ms=5)
    marker,=b.plot([],[],'o',color=C[1],ms=9,mec='white',mew=1.5)
    b.set(xscale='log',yscale='log',xlim=(.45,33),ylim=(1e-11,1));b.legend(loc='lower left',fontsize=10)
    setup(b,r'$\alpha L$','cone constant','Guaranteed bound and numerical estimate')
    status=fig.text(.66,.84,'',color=C[4],fontsize=12,fontweight='bold')
    footer(fig,'The analytical eta_H is guaranteed. Optimized ratios and eta_star are numerical diagnostics.')
    def update(i,t):
        j=min(int(t*len(alphas)),len(alphas)-1);alpha=alphas[j]
        line.set_ydata(ratios[j]);pos=.001+.998*t
        dot.set_data([pos],[np.interp(pos,xx,ratios[j])])
        marker.set_data([alpha],[gr.eta_helmholtz(.25,alpha)])
        status.set_text(f'alpha = {alpha:g}    delta = 1/4')
    write_gif(fig,'cone_estimates',update,'Six computed ratio curves and an analytical Helmholtz lower bound; numerical minima are not certified constants.')
    (ROOT/'code/results/animation_cone_data.json').write_text(json.dumps({'alpha':alphas,'s':xx.tolist(),'ratios':[v.tolist() for v in ratios]},indent=2)+'\n')


def nonlinear_branches():
    fig,(a,b)=canvas('Trace the nonlinear equilibrium branches')
    for j,(key,v) in enumerate(r['example2']['branches'].items()):
        curve(a,np.asarray(v['mu'])/float(key)**2,v['amp'],C[j],fr'$\alpha={float(key):g}$',lw=1.7)
    a.legend(loc='lower left',ncol=2,fontsize=9)
    a.set(xscale='log',yscale='log',xlim=(.08,3000),ylim=(5e-5,20))
    setup(a,r'$\mu/\alpha^2$',r'$u(1/2)$','Computed positive branches')
    gn=so.GreenNystrom(1,n=48);f=lambda xx,u:np.exp(u)
    amps=np.geomspace(.001,12,80)
    amp,mu,U=so.amplitude_branch(gn,f,f,amps,mu0=amps[0]/gr.A_const(1))
    if len(amp)!=len(amps):raise RuntimeError('Animation continuation did not reach the requested amplitudes')
    residual=max(float(np.max(np.abs(u-m*gn.W@np.exp(u)))) for u,m in zip(U,mu))
    assert residual<1e-9
    W=gn.matrix(x);profiles=np.asarray([m*W@np.exp(u) for u,m in zip(U,mu)])
    marker,=a.plot([],[],'o',color=C[0],mec='white',mew=1.5,ms=10)
    profile=curve(b,x,profiles[0]/amp[0],C[0],'normalized deflection')
    load=curve(b,x,np.exp(profiles[0]-amp[0]),C[1],'normalized exponential load',ls='--',lw=1.8)
    b.set(xlim=(0,1),ylim=(0,1.07));b.legend(loc='lower left',fontsize=9)
    setup(b,'x','normalized values','Selected equilibrium and load, alpha = 1')
    status=fig.text(.66,.84,'',color=C[0],fontsize=11,fontweight='bold')
    footer(fig,'Numerical continuation through a fold. A computed branch does not exclude other positive solutions.')
    def update(i,t):
        k=int(round(t*(len(amp)-1)));marker.set_data([mu[k]],[amp[k]])
        profile.set_ydata(profiles[k]/amp[k]);load.set_ydata(np.exp(profiles[k]-amp[k]))
        status.set_text(f'mu = {mu[k]:.4f}    u(1/2) = {amp[k]:.4f}')
    write_gif(fig,'nonlinear_branches',update,'Computed equilibrium continuation, not time dynamics or a proof of global nonexistence.')
    manifest['nonlinear_branches']['max_discrete_residual']=residual


def convergence():
    fig,(a,b)=canvas('Refine the grid, resolve the beam')
    for j,key in enumerate(['fd','fe']):
        v=r['verification']['1.0'][key];nn=np.array(list(v),int);err=np.array(list(v.values()))
        curve(a,nn,err,C[j],['finite differences','mixed finite elements'][j],marker=['o','s'][j],ms=5)
    a.set(xscale='log',yscale='log',xlim=(3,240),ylim=(1e-8,.04))
    setup(a,'N (FD), elements (FE)','sampled relative maximum error','Uniform-load convergence, alpha = 1')
    a.legend(loc='lower left',fontsize=9)
    exact=gr.uniform_load_response(x,1);scale=float(exact.max())
    curve(b,x,exact/scale,'#8290a7','exact response',ls='--',lw=1.6)
    points,=b.plot([],[],'o',color=C[0],ms=5,label='finite-difference nodes')
    node_cursor,=b.plot([],[],'o',color=C[2],ms=9,mec='white',mew=1.5)
    b.set(xlim=(0,1),ylim=(0,1.07));setup(b,'x',r'$u(x)/\max u_{\rm exact}$','Direct sixth-order discretization')
    b.legend(loc='upper left',fontsize=9)
    states=[];one=lambda xx,u:np.ones_like(u);zero=lambda xx,u:np.zeros_like(u)
    for n in [25,50,100,200]:
        xx,u=so.solve_fd(one,zero,1,N=n);states.append((n,xx,u/scale))
    marker,=a.plot([],[],'o',color=C[2],ms=10,mec='white',mew=1.5)
    status=fig.text(.66,.84,'',color=C[0],fontsize=12,fontweight='bold')
    footer(fig,'Reported errors use sampled maxima. Finite differences approach order 2; finite elements approach order 4.')
    def update(i,t):
        k=min(int(t*4),3);n,xx,u=states[k];points.set_data(xx,u)
        j=int(round(t*(len(xx)-1)));node_cursor.set_data([xx[j]],[u[j]])
        e=r['verification']['1.0']['fd'][str(n)];marker.set_data([n],[e])
        status.set_text(f'N = {n}    relative error = {e:.3e}')
    write_gif(fig,'convergence',update,'Four actually computed finite-difference grids cycle from N=25 to 200 and back.')


def hero():
    fig,axs=plt.subplots(2,3,figsize=(16,8),dpi=100)
    fig.subplots_adjust(left=.055,right=.97,bottom=.13,top=.72,hspace=.65,wspace=.36)
    fig.text(.055,.91,'STRAIN-GRADIENT BEAMS',fontsize=34,fontweight='bold',color='#172842')
    fig.text(.055,.846,'Green kernels  /  positive solutions  /  material length effects',fontsize=15,color='#52617c')
    fig.text(.97,.918,'ANIMATED RESEARCH TILES',ha='right',fontsize=10,fontweight='bold',color=C[0])
    fig.text(.97,.865,'static model, parameter sweeps',ha='right',fontsize=10,color='#52617c')
    # Coloured tile tabs on an entirely white canvas.
    for j,ax in enumerate(axs.flat):
        pos=ax.get_position();fig.patches.append(FancyBboxPatch((pos.x0-.011,pos.y0-.055),
            pos.width+.022,pos.height+.11,boxstyle='round,pad=.007,rounding_size=.013',
            transform=fig.transFigure,facecolor='white',edgecolor='#dce4f0',linewidth=1,zorder=-5))
        fig.patches.append(FancyBboxPatch((pos.x0-.006,pos.y1+.038),.045,.007,
            boxstyle='round,pad=0',transform=fig.transFigure,facecolor=C[j%5],edgecolor='none'))
        ax.tick_params(labelsize=7);ax.set_xlabel('x',fontsize=8)
    a,b,c,d,e,f=axs.flat
    a.set_title('01  POSITIVE RESPONSE',loc='left',fontsize=10,color=C[0]);a.set(xlim=(0,1),ylim=(0,1.08))
    kernel=curve(a,x,np.zeros_like(x),C[0]);a.set_ylabel('normalized kernel',fontsize=8)
    q=np.linspace(0,1,81);X,S=np.meshgrid(q,q)
    b.contour(X,S,1e4*gr.Gt(X,S),levels=np.linspace(.5,4,8),cmap='viridis',linewidths=.9)
    scan=curve(b,[0,1],[.15,.15],C[1],ls='--',lw=1.5)
    b.set_title('02  SOURCE SCAN',loc='left',fontsize=10,color=C[1]);b.set(xlim=(0,1),ylim=(0,1),ylabel='s')
    curve(c,x,16*x*x*(1-x)**2,'#acb7c8',ls='--',lw=1)
    profile=curve(c,x,np.zeros_like(x),C[2]);c.set(xlim=(0,1),ylim=(0,1.06))
    c.set_title('03  UNIFORM LOAD',loc='left',fontsize=10,color=C[2]);c.set_ylabel('relative deflection',fontsize=8)
    kk=np.geomspace(.1,200,250);curve(d,kk,gr.size_effect_factor(kk),C[4],lw=2)
    dotd,=d.plot([],[],'o',color=C[1],ms=7);d.set(xscale='log',xlim=(.1,200),ylim=(0,1.05))
    d.set_title('04  MATERIAL LENGTH',loc='left',fontsize=10,color=C[4]);d.set_xlabel('kappa = L/g',fontsize=8)
    d.set_ylabel('size factor',fontsize=8)
    branch=r['example1']['branch'];amp=np.asarray(branch['amp']);mu=np.asarray(branch['mu'])
    curve(e,mu,amp,C[1],lw=2);dote,=e.plot([],[],'o',color=C[0],ms=7)
    e.set(xscale='log',yscale='log',xlim=(.3,2500),ylim=(1e-4,20))
    e.set_title('05  NONLINEAR BRANCH',loc='left',fontsize=10,color=C[1]);e.set_xlabel('mu',fontsize=8);e.set_ylabel('u(1/2)',fontsize=8)
    vv=r['verification']['1.0'];dots=[]
    for j,key in enumerate(['fd','fe']):
        nn=np.array(list(vv[key]),int);err=np.array(list(vv[key].values()))
        curve(f,nn,err,C[j],key.upper(),marker=['o','s'][j],ms=3,lw=1.6)
        dots.append((nn,err,f.plot([],[],'o',color=C[j],ms=7)[0]))
    f.set(xscale='log',yscale='log',xlim=(3,240),ylim=(1e-8,.04));f.legend(fontsize=7)
    f.set_title('06  RESOLUTION CHECK',loc='left',fontsize=10,color=C[3]);f.set_xlabel('resolution',fontsize=8);f.set_ylabel('relative error',fontsize=8)
    footer(fig,'ORIGINAL MODEL-DERIVED GRAPHICS   /   OPEN COMPUTATIONS   /   WHITE CANVAS')
    status=fig.text(.055,.785,'',fontsize=10,fontweight='bold',color=C[0])
    def update(i,t):
        alpha=.5*40**t;s=.15+.7*t;k=.1*2000**t
        v=gr.Gt(x,.5,alpha);kernel.set_ydata(v/v.max());scan.set_ydata([s,s])
        profile.set_ydata(384*k*k*gr.uniform_load_response(x,k));dotd.set_data([k],[gr.size_effect_factor(k)])
        target=1e-4*(14/1e-4)**t;j=int(np.argmin(abs(np.log(amp/target))))
        dote.set_data([mu[j]],[amp[j]])
        for nn,err,dot in dots:
            q=min(int(t*len(nn)),len(nn)-1);dot.set_data([nn[q]],[err[q]])
        status.set_text(f'alpha = {alpha:.2f}       source s = {s:.2f}       kappa = {k:.2f}')
    write_gif(fig,'hero',update,'Six tiles show kernel shape, source position, static deflection, size factor, computed branch and mesh refinement.')


if __name__=='__main__':
    funcs={'source_scan':source_scan,'kernel_surface':kernel_surface,'cone_estimates':cone_estimates,
           'material_length':material_length,'nonlinear_branches':nonlinear_branches,
           'convergence':convergence,'hero':hero}
    selected=sys.argv[1:] or list(funcs)
    existing=ROOT/'assets/animations/manifest.json'
    prior=json.loads(existing.read_text()) if existing.exists() else {}
    for name in selected:
        funcs[name]()
        prior.update(manifest);existing.write_text(json.dumps(prior,indent=2)+'\n')
