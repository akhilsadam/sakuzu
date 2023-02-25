import matplotlib.pyplot as plt
from matplotlib.patches import Patch as Patch
from matplotlib.patches import ConnectionPatch
from matplotlib import colors
import math
import numpy as np
import jpcm
from dataclasses import dataclass
from typing import List
import inspect

plt.rcParams['text.usetex'] = True

def figure(width = 5,height = 5, c=jpcm.maps.hakushi, ec=jpcm.maps.kokushoku, lw=0.5):
    """
        Creates a figure with a white background and no axes.
        fc: facecolor, c: color, lw: linewidth
    """
    fig, ax = plt.subplots(1,1,figsize=(width, height), tight_layout=True)
    ax.set_xlim(-0.5,0.5)
    ax.set_ylim(-0.5,0.5)
    ax.tick_params(bottom=False, top=False, left=False, right=False)
    ax.tick_params(labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    ax.set_facecolor(c)
    for t in ['top', 'bottom', 'left', 'right']:
        ax.spines[t].set_color(ec)
        ax.spines[t].set_linewidth(lw)
    return fig, ax

def clean_axes(ax, c2=(0,0,0,0)):
    c = (0,0,0,0)
    ax.tick_params(bottom=False, top=False, left=False, right=False)
    ax.tick_params(labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    ax.set_aspect('equal')
    ax.set_facecolor(c)
    for t in ['top', 'bottom', 'left', 'right']:
        ax.spines[t].set_color(c2)
        ax.spines[t].set_linewidth(0.5)
    return ax # not necessary, but makes it easier to chain

def clean_axes2(ax, c2=(0,0,0,1)):
    c = (0,0,0,0)
    ax.tick_params(bottom=False, top=False, left=False, right=False)
    ax.tick_params(labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    ax.set_aspect('equal')
    ax.set_facecolor(c)
    for t in ['top', 'bottom', 'left', 'right']:
        ax.spines[t].set_color(c2)
        ax.spines[t].set_linewidth(0.5)
    return ax # not necessary, but makes it easier to chain

def out(fig, filename,  **kwargs):
    """
        Saves the figure as a file.
    """
    default = {'bbox_inches': 'tight', 'pad_inches': 0.0, 'transparent': True, 'dpi': 300}
    default.update(kwargs)
    fig.savefig(filename,**default)

@dataclass
class Block:
    x: float
    y: float
    sx: float
    sy: float
    tf: List[float]
    width: float
    height: float

    rcenter: List[float]
    rdim: List[float]

    item: Patch
    shape: str
    subblocks: List
    arrows: List
    label: str
    kwargs: dict

    def __init__(self, x, y, width, height,**kwargs):
        if 'tf' not in kwargs:
            self.tf = [0,0,1,1]
        else:
            self.tf = kwargs['tf']
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        if 'item' in kwargs:
            self.item = kwargs['item']
        else:
            self.shape = kwargs.get('shape', 'Rectangle')
            self.item = None
        self.kwargs = kwargs
        if 'sx' in kwargs:
            self.sx = kwargs['sx']
            self.sy = kwargs['sy']
        else:
            self.sx = 0
            self.sy = 0
        
        if 'label' in kwargs:
            self.label = kwargs['label']
        else:
            self.label = ''

        if 'tc' in kwargs:
            self.tc = kwargs['tc']
        else:
            self.tc = jpcm.maps.kokushoku
        
        self.subblocks = []
        self.arrows = []

        self.rcenter=[self.sx + self.tf[0] + self.tf[2]*self.x, self.sy + self.tf[1] + self.tf[3]*self.y]
        self.rdim = [self.tf[2]*self.width, self.tf[3]*self.height]

    def draw(self, fig):
        """
            Draws the block.
        """
        color=self.kwargs.get('c', (0,0,0,0))
        ec=self.kwargs.get('ec', (0,0,0,0))
        axc = self.kwargs.get('axc', (0,0,0,0))
        fs=self.kwargs.get('fs', 60)
        mfs=self.kwargs.get('mfs', 20)
        lfs = self.kwargs.get('lfs', 16)

        def color_filter(kwargs):
            kw2 = {}
            for key in kwargs:
                if key in ['ec']:
                    kw2[key] = kwargs.get(key, (0,0,0,0))
            return kw2

        if self.label:
            px = self.kwargs.get('text_x',0.5)
            py = 0.5
            fz = math.sqrt(self.width*self.height)
            if len(self.subblocks) > 0 or self.item is not None:
                py = 1.2
                lfs/=math.sqrt(len(self.label))

            if 0 <= py <= 1:
                # TC adaptation for close colors
                if type(self.tc) ==str:
                    self.tc = colors.hex2color(colors.cnames[self.tc])
                if type(color) ==str:
                    color = colors.hex2color(colors.cnames[color])
                alpha = color[3] if len(color) == 4 else 1
                delta = np.linalg.norm((self.tc[:3] - color[:3]))
                if alpha > 0.5 and delta < 0.5*math.sqrt(3):
                    self.tc = (1-self.tc[:3])

            textkwargs = {'ha': 'center', 'va': 'center'}
            for key in textkwargs.keys():
                if key in self.kwargs:
                    textkwargs[key] = self.kwargs[key]


        self.ax = clean_axes(fig.add_axes((self.rcenter[0]-self.rdim[0]/2, self.rcenter[1]-self.rdim[1]/2, self.rdim[0], self.rdim[1])),axc)
        if self.item is not None and type(self.item) == plt.plot:
                self.ax.add_patch(self.item)
        elif self.item is not None and type(self.item) == type(lambda x: x):
            self.item(self.ax)
        else:
            args = [(0, 0), 1, 1]
            if 'shape' not in self.kwargs:
                self.shape = 'Rectangle'
            if self.shape == 'Circle':
                args = [(0.5,0.5),0.5]
            elif self.shape == 'Rectangle':
                fsc = min(self.width, self.height)
                args[1] = self.width/fsc
                args[2] = self.height/fsc
                args[0] = (0.5-args[1]/2, 0.5-args[2]/2)
            elif self.shape == 'Square':
                self.shape = 'Rectangle'

            if self.item is not None:
                fcolor = color
                color = (0,0,0,0)

            fpatch = plt.__dict__[self.shape]
            self.ax.add_patch(fpatch(*args, color=color, **color_filter(self.kwargs), clip_on=False))
            if self.item is not None and type(self.item) == np.ndarray:
                ax2 = clean_axes2(fig.add_axes((self.rcenter[0]-self.rdim[0]/2, self.rcenter[1]-self.rdim[1]/2, self.rdim[0], self.rdim[1])))
                ax2.set_aspect('equal')
                ax2.plot(self.item[0], self.item[1], color=fcolor, clip_on=False)


        if self.label:
            self.ax.text(px, py, self.label,**textkwargs, fontsize=max(min(fs*fz-mfs,0)+mfs,lfs), color=self.tc)

        for a in self.arrows:
            self.draw_arrow(self.ax,a)

        for b in self.subblocks:
            b.draw(fig)
        

    def block(self,*args,**kwargs):
        """
            Adds a block to the diagram.
        """
        if 'tf' not in kwargs:
            kwargs.update({'tf':[self.x-self.width/2, self.y-self.height/2, self.width, self.height]})
        b = Block(*args,**kwargs)
        self.subblocks.append(b)
        return b

    def arrow(self,b1,b2,**kwargs):
        self.arrows.append((b1,b2,kwargs))

    def draw_arrow(self, ax, ar):
        """
            Draws an arrow from b1 to b2.
        """
        default = {'fc': jpcm.maps.kokushoku, 'ec': jpcm.maps.kokushoku, 'lw': 0.5, 'bent': True}
        b1,b2,kwargs = ar
        default.update(kwargs)
        fc = default['fc']
        ec = default['ec']

        w1,h1 = b1.width, b1.height
        w2,h2 = b2.width, b2.height
        x1,y1 = b1.x -w1/2, b1.y-h1/2
        x2,y2 = b2.x -w2/2, b2.y-h2/2
        
        # Compute box centers
        cx1 = x1 + w1 / 2
        cy1 = y1 + h1 / 2
        cx2 = x2 + w2 / 2
        cy2 = y2 + h2 / 2
        
        # Compute distance and angle between box centers
        dx = cx2 - cx1
        dy = cy2 - cy1
        # dist = (dx**2 + dy**2)**0.5

        start_face = ''
        end_face = ''
        if cx1 > cx2 and cy1 > cy2:
            start_face = 'left'
            end_face = 'top'
        elif cx1 > cx2 and cy1 < cy2:
            start_face = 'left'
            end_face = 'bottom'
        elif cx1 < cx2 and cy1 > cy2:
            start_face = 'right'
            end_face = 'top'
        elif cx1 < cx2 and cy1 < cy2:
            start_face = 'right'
            end_face = 'bottom'
        elif abs(cx1 - cx2) < min(w1,w2) and cy1 > cy2:
            start_face = 'bottom'
            end_face = 'top'
        elif abs(cx1 - cx2) < min(w1,w2) and cy1 < cy2:
            start_face = 'top'
            end_face = 'bottom'
        elif cx1 > cx2 and abs(cy1 - cy2) < min(h1,h2):
            start_face = 'left'
            end_face = 'right'
        elif cx1 < cx2 and abs(cy1 - cy2) < min(h1,h2):
            start_face = 'right'
            end_face = 'left'
        # calculate the start and end points of the arrow based on the start and end faces
        if start_face == 'bottom':
            start_point = (cx1, y1)
        elif start_face == 'top':
            start_point = (cx1, y1 + h2)
        elif start_face == 'left':
            start_point = (x1, cy1)
        else:
            start_point = (x1 + w1, cy1)
        if end_face == 'bottom':
            end_point = (cx2, y2)
        elif end_face == 'top':
            end_point = (cx2, y2 + h2)
        elif end_face == 'left':
            end_point = (x2, cy2)
        else:
            end_point = (x2 + w2, cy2)



        x1_int, y1_int = start_point
        x2_int, y2_int = end_point

        if default['bent'] and x1_int != x2_int and y1_int != y2_int:
            if start_face in ['left','right']:
                x1_int2 = x2_int
                y1_int2 = y1_int
            else:
                x1_int2 = x1_int
                y1_int2 = y2_int
            arrow = ConnectionPatch((x1_int, y1_int), (x1_int2, y1_int2), "data", "data", fc=fc,ec=ec)
            ax.add_patch(ConnectionPatch((x1_int2, y1_int2), (x2_int, y2_int), "data", "data", fc=fc,ec=ec, arrowstyle="-|>"))
        else:
            arrow = ConnectionPatch((x1_int, y1_int), (x2_int, y2_int), "data", "data", fc=fc,ec=ec, arrowstyle="-|>")
        ax.add_patch(arrow)



class Diagram:
    def __init__(self, width, height, **kwargs):
        self.width = width
        self.height = height
        self.fig, self.ax = figure(width, height, **kwargs)
        default = {'shape': 'Rectangle', 'c': jpcm.maps.hakushi, 'ec': 'k', 'lw':'2'}
        default.update(kwargs)
        self.main = Block(0.5,0.5,1.0,1.0, tf = [0.0,0.0,1.0,1.0], **default)

    def block(self, *args, **kwargs):
        return self.main.block(*args, **kwargs)
    
    def arrow(self, b1, b2, **kwargs):
        self.main.arrow(b1, b2, **kwargs)

    def render(self,**kwargs):
        self.main.draw(self.fig)
        default = {'filename':'test.png', 'transparent':True}
        default.update(kwargs)
        out(self.fig, **default)

if __name__ == '__main__':
    # Test
    d = Diagram(4,4)
    B1 = d.block(0.1, 0.1, 0.2, 0.2, shape='Rectangle', c=jpcm.maps.chigusa_iro, label='B1')
    B2 = B1.block(0.25, 0.50, 0.2, 0.2, shape='Rectangle', c=jpcm.maps.benimidori)
    B3 = B1.block(0.55, 0.25, 0.2, 0.2, shape='Rectangle', c=jpcm.maps.azuki_iro)
    x = np.arange(-1,1,0.1)
    relu = lambda x: np.maximum(x, 0)
    B4 = d.block(0.4, 0.7, 0.05, 0.05, item = np.array([x, relu(x)]), axc='k', c=jpcm.maps.ginshu, label=r'$\sigma(x)$')
    B5 = d.block(0.6, 0.7, 0.1, 0.1, shape='Circle', c=jpcm.maps.kokimurasaki, label='-', tc=np.array([0,0,1.0]))
    B6 = d.block(0.9, 0.7, 0.2, 0.1, shape='Rectangle', c=jpcm.maps.rurikon)
    d.arrow(B1, B4)
    B1.arrow(B2, B3)
    d.arrow(B5, B6)
    d.arrow(B4, B5)

    q = lambda ax: ax.annotate('Nx', xy=(0.5, 0.775), xytext=(0.5, 0.82), xycoords='axes fraction', 
        fontsize=12, ha='center', va='center',
        arrowprops=dict(arrowstyle='-[, widthB=4.5, lengthB=1.0', lw=0.5))

    ann = d.block(0.5, 0.5, 1.0, 1.0, item = q)


    d.render(filename='out.pdf')