import matplotlib.pyplot as plt
import pandas as pd
import math
import torch

def plot_strokes(df, ascii=None, save_path=None):
    line_groups = [l_g.reset_index(drop=True) for _, l_g in df.groupby("line")]
    n_lines = len(line_groups)

    n_cols = 1
    n_rows = math.ceil(n_lines / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), constrained_layout=True)
    
    if n_lines == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, l_g in enumerate(line_groups):
        ax = axes[i]
        if ascii is not None:
            ax.set_title(ascii['text'].iloc[i], fontsize=12)

        l_g['x'] = l_g['delta_x'].cumsum()
        l_g['y'] = -l_g['delta_y'].cumsum()
        l_g['stroke'] = [0] + l_g['lift_point'].cumsum().to_list()[:-1]

        stroke_groups = [s_g.reset_index(drop=True) for _, s_g in l_g.groupby("stroke")]
        for s_g in stroke_groups:
            ax.plot(s_g['x'], s_g['y'], color="black", antialiased=True, linewidth=0.5)

        ax.set_aspect('equal')
        padding = 10
        ax.set_ylim(l_g['y'].min() - padding, l_g['y'].max() + padding)
        ax.axis("off")

    # Hide unused subplots if any
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    if save_path is not None:
        plt.savefig(save_path, format='svg', bbox_inches='tight')

    plt.show()

def tensor_to_df(output, denormalize_stats=None):
    df = pd.DataFrame(output.squeeze().detach().cpu().numpy(), columns=['delta_x', 'delta_y', 'lift_point'])
    df['line'] = 0

    if denormalize_stats is not None:
        stats = pd.read_csv(denormalize_stats).set_index('stat')['value']
        mu_dx = stats['mu_dx']
        sd_dx = stats['sd_dx']
        mu_dy = stats['mu_dy']
        sd_dy = stats['sd_dy']

        df['delta_x'] = df['delta_x'] * sd_dx + mu_dx
        df['delta_y'] = df['delta_y'] * sd_dy + mu_dy

    return df

def plot_tensor(output, denormalize_stats=None, ascii = None, save_path=None):
    ascii = pd.DataFrame({'text':[ascii]})
    plot_strokes(tensor_to_df(output, denormalize_stats=denormalize_stats),ascii=ascii,save_path=save_path)

if __name__ == "__main__":
    import os
    import pandas as pd
    from utils.parse_strokes import parse_strokes
    from utils.get_ascii import get_ascii
    from constants import TEST_RESULTS_PATH

    
    code = 'a01-000u'
    df = parse_strokes(code)
    ascii = get_ascii(code)

    os.makedirs(TEST_RESULTS_PATH,exist_ok=True)
    save_path = os.path.join(TEST_RESULTS_PATH,"strokeTest.svg")

    plot_strokes(df,ascii,save_path)
