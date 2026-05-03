import json

with open('graficos_dataset_03.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = cell['source']
        if any('plt.subplots(' in line for line in source) or any('sns.barplot' in line for line in source) or any('sns.heatmap' in line for line in source):
            cell['source'].append('\nplt.savefig("graficos_informe/ds02_plot_' + str(i) + '.png", bbox_inches="tight", dpi=150)\n')

with open('graficos_dataset_03_modified.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f)
