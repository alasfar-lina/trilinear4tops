# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.ticker import AutoMinorLocator
import arviz as az
######
#     This is a modified version of the "EffectMeasurePlot" class from zepid package
#        (https://github.com/pzivich)
#       adapted for the use in this analysis
######






class ForestPlot:
    """Used to generate effect measure plots. effectmeasure plot accepts four list type objects.
    effectmeasure_plot is initialized with the associated names for each line, the point estimate,
    the lower confidence limit, and the upper confidence limit.
    Plots will resemble the following form:
        _____________________________________________      Measure     % CI
        |                                           |
    1   |        --------o-------                   |       x        n, 2n
        |                                           |
    2   |                   ----o----               |       w        m, 2m
        |                                           |
        |___________________________________________|
        #           #           #           #
    The following functions (and their purposes) live within effectmeasure_plot
    labels(**kwargs)
        Used to change the labels in the plot, as well as the center and scale. Inputs are
        keyword arguments
        KEYWORDS:
            -effectmeasure  + changes the effect measure label
            -conf_int       + changes the confidence interval label
            -scale          + changes the scale to either log or linear
            -center         + changes the reference line for the center
    colors(**kwargs)
        Used to change the color of points and lines. Also can change the shape of points.
        Valid colors and shapes for matplotlib are required. Inputs are keyword arguments
        KEYWORDS:
            -errorbarcolor  + changes the error bar colors
            -linecolor      + changes the color of the reference line
            -pointcolor     + changes the color of the points
            -pointshape     + changes the shape of points
    plot(t_adjuster=0.01,decimal=3,size=3)
        Generates the effect measure plot of the input lists according to the pre-specified
        colors, shapes, and labels of the class object
        Arguments:
            -t_adjuster     + used to refine alignment of the table with the line graphs.
                              When generate plots, trial and error for this value are usually
                              necessary
            -decimal        + number of decimal places to display in the table
            -size           + size of the plot to generate
    Example)
    >>>lab = ['One','Two'] #generating lists of data to plot
    >>>emm = [1.01,1.31]
    >>>lcl = ['0.90',1.01]
    >>>ucl = [1.11,1.53]
    >>>
    >>>x = zepid.graphics.effectmeasure_plot(lab,emm,lcl,ucl) #initializing effectmeasure_plot with the above lists
    >>>x.labels(effectmeasure='RR') #changing the table label to 'RR'
    >>>x.colors(pointcolor='r') #changing the point colors to red
    >>>x.plot(t_adjuster=0.13) #generating the effect measure plot
    """

    def __init__(self, label, effect_measure,lcl,ucl):
        """Initializes effectmeasure_plot with desired data to plot. All lists should be the same
        length. If a blank space is desired in the plot, add an empty character object (' ') to
        each list at the desired point.
        Inputs:
        label
            -list of labels to use for y-axis
        effect_measure
            -list of numbers for point estimates to plot. If point estimate has trailing zeroes,
             input as a character object rather than a float
        lcl
            -list of numbers for upper confidence limits to plot. If point estimate has trailing
             zeroes, input as a character object rather than a float
        ucl
            -list of numbers for upper confidence limits to plot. If point estimate has
             trailing zeroes, input as a character object rather than a float
        """
        self.df = pd.DataFrame()
        self.df['study'] = label
        self.df['OR'] = effect_measure
        self.df['LCL'] = lcl
        self.df['UCL'] = ucl
        self.df['OR2'] = self.df['OR'].astype(str).astype(float)
        if (all(isinstance(item, float) for item in lcl)) & (all(isinstance(item, float) for item in effect_measure)):
            self.df['LCL_dif'] = self.df['OR'] - self.df['LCL']
        else:
            self.df['LCL_dif'] = (pd.to_numeric(self.df['OR'])) - (pd.to_numeric(self.df['LCL']))
        if (all(isinstance(item, float) for item in ucl)) & (all(isinstance(item, float) for item in effect_measure)):
            self.df['UCL_dif'] = self.df['UCL'] - self.df['OR']
        else:
            self.df['UCL_dif'] = (pd.to_numeric(self.df['UCL'])) - (pd.to_numeric(self.df['OR']))
        self.em = r'$\mu$'
        self.ci = r'95\% CI'
        self.scale = 'linear'
        self.center = 1
        self.errc = 'dimgrey'
        self.shape = 'o'
        self.pc = 'k'
        self.linec = 'gray'

    def labels(self, **kwargs):
        """Function to change the labels of the outputted table. Additionally, the scale and reference
        value can be changed.
        Accepts the following keyword arguments:
        effectmeasure
            -changes the effect measure label
        conf_int
            -changes the confidence interval label
        scale
            -changes the scale to either log or linear
        center
            -changes the reference line for the center
        """
        if 'effectmeasure' in kwargs:
            self.em = kwargs['effectmeasure']
        if 'ci' in kwargs:
            self.ci = kwargs['conf_int']
        if 'scale' in kwargs:
            self.scale = kwargs['scale']
        if 'center' in kwargs:
            self.center = kwargs['center']

    def colors(self, **kwargs):
        """Function to change colors and shapes.
        Accepts the following keyword arguments:
        errorbarcolor
            -changes the error bar colors
        linecolor
            -changes the color of the reference line
        pointcolor
            -changes the color of the points
        pointshape
            -changes the shape of points
        """
        if 'errorbarcolor' in kwargs:
            self.errc = kwargs['errorbarcolor']
        if 'pointshape' in kwargs:
            self.shape = kwargs['pointshape']
        if 'linecolor' in kwargs:
            self.linec = kwargs['linecolor']
        if 'pointcolor' in kwargs:
            self.pc = kwargs['pointcolor']

    def plot(self, figsize=(3, 3), t_adjuster=0.01, decimal=3, size=3, max_value=None, min_value=None):
        """Generates the matplotlib effect measure plot with the default or specified attributes.
        The following variables can be used to further fine-tune the effect measure plot
        t_adjuster
            -used to refine alignment of the table with the line graphs. When generate plots, trial
             and error for this value are usually necessary. I haven't come up with an algorithm to
             determine this yet...
        decimal
            -number of decimal places to display in the table
        size
            -size of the plot to generate
        max_value
            -maximum value of x-axis scale. Default is None, which automatically determines max value
        min_value
            -minimum value of x-axis scale. Default is None, which automatically determines min value
        """
        tval = []
        ytick = []
        for i in range(len(self.df)):
            if (np.isnan(self.df['OR2'][i]) == False):
                if ((isinstance(self.df['OR'][i], float)) & (isinstance(self.df['LCL'][i], float)) & (
                isinstance(self.df['UCL'][i], float))):
                    tval.append([round(self.df['OR2'][i], decimal), (
                                '[' + str(round(self.df['LCL'][i], decimal)) + ', ' + str(
                            round(self.df['UCL'][i], decimal)) + ']')])
                else:
                    tval.append(
                        [self.df['OR'][i], ('[' + str(self.df['LCL'][i]) + ', ' + str(self.df['UCL'][i]) + ']')])
                ytick.append(i)
            else:
                tval.append([' ', ' '])
                ytick.append(i)
        if max_value is None:
            if pd.to_numeric(self.df['UCL']).max() < 1:
                maxi = round(((pd.to_numeric(self.df['UCL'])).max() + 0.05),
                             2)  # setting x-axis maximum for UCL less than 1
            if (pd.to_numeric(self.df['UCL']).max() < 9) and (pd.to_numeric(self.df['UCL']).max() >= 1):
                maxi = round(((pd.to_numeric(self.df['UCL'])).max() + 1),
                             0)  # setting x-axis maximum for UCL less than 10
            if pd.to_numeric(self.df['UCL']).max() > 9:
                maxi = round(((pd.to_numeric(self.df['UCL'])).max() + 10),
                             0)  # setting x-axis maximum for UCL less than 100
        else:
            maxi = max_value
        if min_value is None:
            if pd.to_numeric(self.df['LCL']).min() > 0:
                mini = round(((pd.to_numeric(self.df['LCL'])).min() - 0.1), 1)  # setting x-axis minimum
            if pd.to_numeric(self.df['LCL']).min() < 0:
                mini = round(((pd.to_numeric(self.df['LCL'])).min() - 0.05), 2)  # setting x-axis minimum
        else:
            mini = min_value
        fig=plt.figure(figsize=figsize)  # blank figure
        gspec = gridspec.GridSpec(1, 6)  # sets up grid
        plot = plt.subplot(gspec[0, 0:4])  # plot of data
        tabl = plt.subplot(gspec[0, 4:])  # table of OR & CI
        plot.set_ylim(-1, (len(self.df)))  # spacing out y-axis properly
        ###### some effects #####
        n_lines = 10
        diff_linewidth = 1.05
        alpha_value = 0.03
       # plot.set_facecolor('#EFEFF1')
        if self.scale == 'log':
            try:
                plot.set_xscale('log')
            except:
                raise ValueError('For the log scale, all values must be positive')
        plot.axvline(self.center, color=self.linec, zorder=1)
        for pos, y, err1,err2, color in zip(self.df.OR2, self.df.index, self.df.LCL_dif, self.df.UCL_dif, self.errc):
            plot.errorbar(pos , y, xerr=[[err1], [err2]], lw=0, capsize=5, capthick=2, 
                          ecolor=color,elinewidth=2.*(size / size),marker='None', zorder=2)
            #### add glow
          #  for n in range(1, n_lines+1):
           #     plot.errorbar(pos , y, xerr=[[err1], [err2]], lw=0, capsize=0, capthick=0, 
            #              ecolor=color,elinewidth=1+(diff_linewidth*n),marker='None', zorder=1,alpha=alpha_value)
                


          
        scatter = plot.scatter(self.df.OR2, self.df.index, c=self.pc, s=(size * 25), marker=self.shape, zorder=3,
                     edgecolors='None')
        plot.xaxis.set_ticks_position('both')
        plot.yaxis.set_ticks_position('both')
        plot.get_yaxis().set_minor_locator(AutoMinorLocator())
        plot.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        plot.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        plot.set_yticks(ytick)
        plot.set_xlim([mini, maxi])
        plot.set_yticklabels(self.df.study)
        plot.yaxis.set_ticks_position('none')
        plot.invert_yaxis()  # invert y-axis to align values properly with table
        tb = tabl.table(cellText=tval, cellLoc='center', loc='right', colLabels=[self.em, self.ci],
                        bbox=[0, t_adjuster, 1, 1])
        tabl.axis('off')
        tb.auto_set_font_size(False)
        tb.set_fontsize(20)
        for key, cell in tb.get_celld().items():
            cell.set_linewidth(0)

        return plot
    
