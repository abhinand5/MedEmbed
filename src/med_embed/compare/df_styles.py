import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def style_dataframe(df, colormap='viridis'):
    def color_scale(s):
        norm = (s - s.min()) / (s.max() - s.min())
        colors = plt.get_cmap(colormap)(norm)
        return ['background-color: rgba({}, {}, {}, 0.7); color: {}'.format(
            int(r*255), int(g*255), int(b*255), 
            'black' if r * 0.299 + g * 0.587 + b * 0.114 > 0.729 else 'white') 
            for r, g, b, _ in colors]

    return df.style.apply(color_scale, axis=0, subset=df.columns[2:])

def create_custom_colormap():
    colors = ['#FFCCCB', '#FFFFFF', '#006400']  # light red, white, dark green
    
    n_bins = 100
    return LinearSegmentedColormap.from_list('custom_light_red_dark_green', colors, N=n_bins)
