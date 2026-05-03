import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Data_set_03
df = pd.read_csv('./Dataset/Data_set_03.csv')

# Configure style
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.dpi'] = 150

# Figure 1: AI use vs Final score
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=df, x='uses_ai', y='final_score', ax=ax, palette=['#e74c3c', '#3498db'])
ax.set_title('Distribución de Puntaje Final por Uso de IA', fontweight='bold')
ax.set_xlabel('Usa IA (0 = No, 1 = Sí)')
ax.set_ylabel('Puntaje Final')
plt.tight_layout()
fig.savefig('graficos_informe/ds03_rendimiento_vs_ia.png')

# Figure 2: Dependence score vs final score
fig2, ax2 = plt.subplots(figsize=(8, 5))
df_uses_ai = df[df['uses_ai'] == 1]
sns.scatterplot(data=df_uses_ai, x='ai_dependency_score', y='final_score', alpha=0.5, color='#2ecc71', ax=ax2)
ax2.set_title('Puntaje Final vs Nivel de Dependencia de la IA', fontweight='bold')
ax2.set_xlabel('Puntaje de Dependencia de IA')
ax2.set_ylabel('Puntaje Final')
plt.tight_layout()
fig2.savefig('graficos_informe/ds03_dependencia_rendimiento.png')
