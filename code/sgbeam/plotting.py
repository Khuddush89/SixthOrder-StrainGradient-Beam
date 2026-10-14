"""Journal figure style: white canvas, unfilled curves, serif labels and vector exports."""
from __future__ import annotations
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

COLORS=['#0078d4','#e64c69','#009b8b','#df9500','#7654d8']
STYLES=['-', '--', '-.', ':', (0,(6,2,1,2))]

def use_style():
    mpl.rcParams.update({
        'font.family':'serif', 'font.serif':['STIX Two Text','STIXGeneral','DejaVu Serif'],
        'mathtext.fontset':'stix','font.size':10,'axes.labelsize':11,
        'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9,
        'axes.titlesize':10,'axes.linewidth':0.75,'axes.edgecolor':'black',
        'axes.grid':False,'axes.facecolor':'white','figure.facecolor':'white',
        'savefig.facecolor':'white','savefig.edgecolor':'white','savefig.transparent':False,
        'xtick.direction':'in','ytick.direction':'in','xtick.top':True,'ytick.right':True,
        'xtick.major.size':4,'ytick.major.size':4,'xtick.minor.size':2,'ytick.minor.size':2,
        'lines.linewidth':1.5,'lines.markersize':4,'legend.frameon':False,
        'legend.handlelength':2.7,'legend.borderaxespad':0.3,'svg.fonttype':'none', 'pdf.fonttype':42,
        'ps.fonttype':42,'axes.formatter.use_mathtext':True,'axes.unicode_minus':False,
        'savefig.dpi':1000,'figure.dpi':110,'axes.prop_cycle':mpl.cycler(color=COLORS)
    })

def label(ax,letter):
    ax.set_title(f'({letter})',loc='left',fontweight='bold',pad=8)

def legend_below(ax,ncol=2,y=-.28):
    return ax.legend(loc='upper center',bbox_to_anchor=(.5,y),ncol=ncol,columnspacing=1.2)

def save(fig,name):
    root=os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..'))
    for fmt in ['png','svg']:
        target=os.path.join(root,'figures',fmt)
        os.makedirs(target,exist_ok=True)
        options={'dpi':1000} if fmt=='png' else {'metadata':{'Title':name,'Creator':'Mahammad Khuddush','Date':None}}
        fig.savefig(os.path.join(target,name+'.'+fmt),facecolor='white',edgecolor='white',
                    transparent=False,bbox_inches='tight',pad_inches=.08,**options)
    plt.close(fig)
