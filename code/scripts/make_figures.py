#!/usr/bin/env python3
"""Regenerate the six figures as 1000-dpi PNGs and vector SVGs on pure white backgrounds."""
from __future__ import annotations
import json,os,sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator,NullFormatter,MaxNLocator
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.join(HERE,'..'))
from sgbeam import green as gr,solvers as so,beams
from sgbeam.plotting import COLORS,STYLES,use_style,label,legend_below,save
use_style()


def axes(ax,xlim=None,ylim=None):
    if xlim is not None:ax.set_xlim(xlim)
    if ylim is not None:ax.set_ylim(ylim)
    ax.set_facecolor('white')
    ax.tick_params(which='both',direction='in')
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))


def logticks(ax,which='both'):
    for axis in ([ax.xaxis,ax.yaxis] if which=='both' else [ax.xaxis] if which=='x' else [ax.yaxis]):
        axis.set_minor_locator(LogLocator(base=10,subs=np.arange(2,10)))
        axis.set_minor_formatter(NullFormatter())


def fig1(r):
    fig,axs=plt.subplots(1,2,figsize=(7.5,3.4))
    fig.subplots_adjust(wspace=.36,bottom=.26)
    x=np.linspace(0,1,1201)
    for i,s in enumerate([.1,.3,.5,.7,.9]):
        axs[0].plot(x,-gr.Gt(x,s)*1e4,color=COLORS[i],ls=STYLES[i],label=fr'$s={s:g}$')
    axes(axs[0],(0,1));axs[0].set_xlabel('$x$');axs[0].set_ylabel(r'$10^{4}G_0(x,s)$')
    label(axs[0],'a');legend_below(axs[0],3)
    for i,a in enumerate([1,3,10,30]):
        g=-gr.Gt(x,.5,a)
        axs[1].plot(x,g/np.max(abs(g)),color=COLORS[i],ls=STYLES[i],label=fr'$\alpha={a}$')
    q=gr.G4(x,.5)
    axs[1].plot(x,-q/q.max(),color='black',lw=1.1,ls=(0,(2,2)),label=r'$-G_4$')
    axes(axs[1],(0,1),(-1.05,.03));axs[1].set_xlabel('$x$');axs[1].set_ylabel(r'$G_0(x,1/2)/\max_x|G_0(x,1/2)|$')
    label(axs[1],'b');legend_below(axs[1],3)
    save(fig,'fig1_green_sections')


def fig2(r):
    fig=plt.figure(figsize=(7.5,3.8))
    ax=fig.add_subplot(121,projection='3d')
    x=np.linspace(0,1,181);X,S=np.meshgrid(x,x);Z=-gr.Gt(X,S)*1e4
    ax.plot_wireframe(X,S,Z,rstride=9,cstride=9,color=COLORS[0],linewidth=.5)
    ax.set_xlabel('$x$',labelpad=3);ax.set_ylabel('$s$',labelpad=3);ax.set_zlabel('')
    ax.view_init(elev=24,azim=-126);ax.set_box_aspect((1,1,.65));ax.tick_params(labelsize=8,pad=1)
    ax.set_zlim(Z.min()*1.05,.02)
    for aa in [ax.xaxis,ax.yaxis,ax.zaxis]:
        aa.pane.set_facecolor((1,1,1,1));aa.pane.set_edgecolor('white')
        aa._axinfo['grid']['color']=(1,1,1,0)
    ax.text2D(.06,1.0,r'(a) $10^4G_0(x,s)$',transform=ax.transAxes,fontweight='bold',fontsize=10)
    ax=fig.add_subplot(122)
    lev=np.arange(-4,-.2,.5)
    cs=ax.contour(X,S,Z,levels=lev,colors=COLORS[0],linewidths=.85,linestyles='solid')
    ax.clabel(cs,cs.levels[::2],fmt='%.1f',fontsize=8,inline=True,inline_spacing=4)
    ax.plot([0,1],[0,1],color='.45',lw=.75,ls='--')
    axes(ax,(0,1),(0,1));ax.set_aspect('equal');ax.set_xlabel('$x$');ax.set_ylabel('$s$');label(ax,'b')
    ax.text(.02,-.27,r'Contour values: $10^4G_0(x,s)$',transform=ax.transAxes,fontsize=9)
    fig.subplots_adjust(wspace=.28,bottom=.22,top=.90)
    save(fig,'fig2_green_surface')


def fig3(r):
    fig,axs=plt.subplots(1,2,figsize=(7.5,3.5));fig.subplots_adjust(wspace=.4,bottom=.28)
    s=np.linspace(.001,.999,151)
    for i,a in enumerate([1,2,5,10]):
        ratio=gr.eta_ratio(s,.25,a);end=gr.eta_endpoint_limit(.25,a)
        axs[0].plot(np.r_[0,s,1],np.r_[end,ratio,end],color=COLORS[i],ls=STYLES[i],label=fr'$\alpha={a}$')
        axs[0].plot([0,1],[end,end],marker='o',ms=3,mfc='white',mec=COLORS[i],ls='none',clip_on=False)
    axes(axs[0],(0,1),(.30,.81));label(axs[0],'a');axs[0].set_xlabel('$s$');axs[0].set_ylabel(r'$r(s)$')
    legend_below(axs[0],2)
    e=r['constants']['eta_vs_alpha'];a=np.array(e['alpha'])
    for i,(key,lab,m) in enumerate([('delta=0.125',r'$\delta=1/8$','o'),('delta=0.25',r'$\delta=1/4$','s'),('delta=0.375',r'$\delta=3/8$','^')]):
        axs[1].plot(a,e[key],color=COLORS[i],ls=STYLES[i],marker=m,ms=4,mfc='white',mec=COLORS[i],label=lab)
    axs[1].set_xscale('log');axs[1].set_xlim(.45,35);axs[1].set_ylim(0,.92)
    axs[1].set_xlabel(r'$\alpha L$');axs[1].set_ylabel(r'$\widehat\eta_\ast$');label(axs[1],'b');logticks(axs[1],'x');legend_below(axs[1],3)
    save(fig,'fig3_eta')


def fig4(r):
    fig=plt.figure(figsize=(7.5,5.1));gs=fig.add_gridspec(2,2,wspace=.37,hspace=.85,width_ratios=[1,1])
    ax=fig.add_subplot(gs[0,0]);x=np.linspace(0,1,1001);f=lambda x,u:np.exp(u)
    gn=so.GreenNystrom(1,n=48);e=r['example1']
    for i,(key,lab) in enumerate([('u_minus_nodes','$u_1$'),('u_plus_nodes','$u_2$')]):
        v=gn.evaluate(x,f,np.array(e[key]));ax.plot(x,np.exp(v-v.max()),color=COLORS[i],ls=STYLES[i],label=lab)
    axes(ax,(0,1),(0,1.05));label(ax,'a');ax.set_xlabel('$x$');ax.set_ylabel(r'$e^{u_i}/\max_x e^{u_i}$');legend_below(ax,2,y=-.32)
    ax=fig.add_subplot(gs[:,1]);e2=r['example2']
    for i,(key,branch) in enumerate(e2['branches'].items()):
        a=float(key);p=np.array(branch['mu'])/a**2;amp=np.array(branch['amp']);row=e2['rows'][i]
        ax.plot(p,amp,color=COLORS[i],ls=STYLES[i],label=fr'$\alpha={a:g}$')
        gn_fold=so.GreenNystrom(a,n=48)
        shape=gr.uniform_load_response(gn_fold.nodes,a);u0=1.4*shape/shape.max()
        uf,muf=so.fold_point(gn_fold,f,f,f,u0,row['mu_fold'])
        af=gn_fold.evaluate([.5],f,uf,muf)[0]
        ax.plot([muf/a**2],[af],marker='o',mfc='white',mec=COLORS[i],ms=4,ls='none')
    ax.plot([1,1],[e['norm_minus'],e['norm_plus']],marker='D',mfc='white',mec='black',ms=4,ls='none')
    ax.set_xscale('log');ax.set_yscale('log');ax.set_xlim(.3,3000);ax.set_ylim(5e-5,20)
    ax.set_xlabel(r'$\mu/\alpha^2$');ax.set_ylabel(r'$u(1/2)$');label(ax,'b');logticks(ax)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.14),ncol=3)
    ax=fig.add_subplot(gs[1,0]);e3=r['example3'];gn5=so.GreenNystrom(5,n=48)
    f3=lambda x,u:e3['lam']*(1+x)*u**2;v=gn5.evaluate(x,f3,np.array(e3['u_nodes']))
    ax.plot(x,v,color=COLORS[0],label='$u$')
    ax.plot([.25,.75],[e3['eta']*v.max()]*2,color=COLORS[1],ls='--',label=r'$\eta\|u\|_\infty$')
    axes(ax,(0,1),(0,1.5));label(ax,'c');ax.set_xlabel('$x$');ax.set_ylabel('$u(x)$');legend_below(ax,2,y=-.34)
    fig.subplots_adjust(bottom=.15,top=.91)
    save(fig,'fig4_examples')


def fig5(r):
    fig=plt.figure(figsize=(7.5,5.1));gs=fig.add_gridspec(2,2,wspace=.41,hspace=.82)
    ax=fig.add_subplot(gs[:,0]);v=r['verification']
    for i,a in enumerate(['1.0','10.0']):
        for j,key in enumerate(['fd','fe']):
            xx=np.array([int(k) for k in v[a][key]])
            ax.plot(xx,list(v[a][key].values()),color=COLORS[i],ls=STYLES[j],marker=['o','s'][j],
                    ms=4,mfc='white',mec=COLORS[i],label=fr'{key.upper()}, $\alpha={float(a):g}$')
    ax.set_xscale('log');ax.set_yscale('log');ax.set_xlim(3,240);ax.set_ylim(1e-8,.05)
    nn=np.array([30.,180]);ax.plot(nn,8/nn**2,color='.35',lw=.8,ls=':');ax.text(70,8/70**2*1.5,r'$h^2$',fontsize=9)
    nn=np.array([6.,45]);ax.plot(nn,.25/nn**4,color='.35',lw=.8,ls=':');ax.text(12,.25/12**4/2,r'$h^4$',fontsize=9)
    ax.set_xlabel(r'$N$ (FD), $n_e$ (FE)');ax.set_ylabel('Sampled relative maximum error');label(ax,'a');logticks(ax)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.14),ncol=2)
    f=lambda x,u:np.exp(u);gn=so.GreenNystrom(1,n=48);up=np.array(r['example1']['u_plus_nodes'])
    x=np.linspace(0,1,2001);v=gn.evaluate(x,f,up);dv=np.gradient(v,x)
    initial=lambda fem:fem.nodal_initial_guess(lambda t:np.interp(t,x,v),lambda t:np.interp(t,x,dv))
    ax=fig.add_subplot(gs[0,1]);ax.plot(x,v,color='black',label='Kernel')
    xx,u=so.solve_fd(f,f,1,N=40,u0=lambda t:np.interp(t,x,v));ax.plot(xx[::2],u[::2],ls='none',marker='o',mfc='white',mec=COLORS[0],ms=3.5,label='FD, $N=40$')
    fem=so.MixedFEM(1,ne=10);xx,u=fem.solve(f,f,U0=initial(fem));ax.plot(xx,u,ls='none',marker='s',mfc='white',mec=COLORS[1],ms=3.5,label='FE, $n_e=10$')
    axes(ax,(0,1),(0,12.5));ax.set_xlabel('$x$');ax.set_ylabel('$u_2(x)$');label(ax,'b');legend_below(ax,3,y=-.32)
    ax=fig.add_subplot(gs[1,1])
    for j,N in enumerate([50,200]):
        xx,u=so.solve_fd(f,f,1,N=N,u0=lambda t:np.interp(t,x,v));err=abs(u-gn.evaluate(xx,f,up))/v.max()
        ax.plot(xx[1:-1],err[1:-1],color=COLORS[0],ls=STYLES[j],label=fr'FD, ${N}$')
    for j,ne in enumerate([16,64]):
        fem=so.MixedFEM(1,ne=ne);fem.solve(f,f,U0=initial(fem));err=[]
        for k in range(ne):
            t=np.linspace(fem.xn[k],fem.xn[k+1],41)
            err.append(np.max(abs(fem.evaluate(t)-gn.evaluate(t,f,up)))/v.max())
        ax.stairs(err,fem.xn,color=COLORS[1],ls=STYLES[j],label=fr'FE, ${ne}$')
    ax.set_xlim(0,1);ax.set_yscale('log');ax.set_ylim(1e-9,.03);ax.set_xlabel('$x$');ax.set_ylabel('Relative difference');label(ax,'c');logticks(ax,'y')
    legend_below(ax,2,y=-.34)
    fig.subplots_adjust(bottom=.15,top=.91)
    save(fig,'fig5_verification')


def fig6(r):
    fig=plt.figure(figsize=(7.5,5.2));gs=fig.add_gridspec(2,2,wspace=.37,hspace=.85)
    ax=fig.add_subplot(gs[0,0]);k=np.geomspace(.1,1000,1001)
    ax.plot(k,gr.size_effect_factor(k),color=COLORS[0],label='Exact')
    ks=np.geomspace(.1,1.2,100);ax.plot(ks,11/120*ks**2,color='.35',ls=':',lw=1,label=r'Small-$\kappa$')
    ks=np.geomspace(8,1000,200);ax.plot(ks,1-16/ks**2,color=COLORS[1],ls='--',lw=1,label=r'Large-$\kappa$')
    one=(lambda x,u:np.ones_like(u),lambda x,u:np.zeros_like(u));kf=[.5,1,2,5,10,20,50,200];df=[]
    for a in kf:
        fem=so.MixedFEM(a,ne=64);fem.solve(*one);df.append(fem.evaluate([.5])[0]*384*a*a)
    ax.plot(kf,df,ls='none',marker='o',mfc='white',mec='black',ms=3,label='FE')
    ax.set_xscale('log');ax.set_xlim(.1,1000);ax.set_ylim(0,1.05);ax.set_xlabel(r'$\kappa=L/g$');ax.set_ylabel(r'$\mathcal{D}(\kappa)$');label(ax,'a');logticks(ax,'x');legend_below(ax,2,y=-.30)
    ax=fig.add_subplot(gs[0,1]);hg=np.geomspace(.1,100,600)
    ax.axhline(beams.shear_ratio(10,.3),color=COLORS[0],ls='--',label='Cowper')
    ax.axhline(beams.shear_ratio(10,.3,5/6),color=COLORS[1],ls=':',label='$k_s=5/6$')
    ax.axhline(1,color='black',ls='-.',lw=1,label='Classical')
    ax.plot(hg,gr.size_effect_factor(10*hg),color=COLORS[2],label='Gradient')
    ax.set_xscale('log');ax.set_xlim(.1,100);ax.set_ylim(0,1.25);ax.set_xlabel('$h/g$');ax.set_ylabel(r'$w_{\max}/w^{\mathrm{EB}}_{\max}$');label(ax,'b');logticks(ax,'x');legend_below(ax,2,y=-.30)
    ax=fig.add_subplot(gs[1,:]);x=np.linspace(0,1,1201)
    for i,g in enumerate([.5,.2,.1,.02]):
        a=1/g;ax.plot(x,384*a*a*gr.uniform_load_response(x,a),color=COLORS[i],ls=STYLES[i],label=fr'$g/L={g:g}$')
    ax.plot(x,16*x*x*(1-x)**2,color='black',lw=1.1,ls=(0,(2,2)),label='Classical limit')
    axes(ax,(0,1),(0,1.04));ax.set_xlabel('$x/L$');ax.set_ylabel(r'$w(x)/w^{\mathrm{EB}}_{\max}$');label(ax,'c');legend_below(ax,5,y=-.31)
    fig.subplots_adjust(top=.91,bottom=.15)
    save(fig,'fig6_size_effect')

if __name__=='__main__':
    with open(os.path.join(HERE,'..','results','results.json')) as fh:r=json.load(fh)
    selected={int(i) for i in sys.argv[1:]} if len(sys.argv)>1 else set(range(1,7))
    for i,fun in enumerate([fig1,fig2,fig3,fig4,fig5,fig6],1):
        if i in selected:
            fun(r);print(f'Figure {i}: 1000-dpi PNG + vector SVG',flush=True)
