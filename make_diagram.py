import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(1, 1, figsize=(20, 14))
ax.set_xlim(0, 20)
ax.set_ylim(0, 14)
ax.axis('off')
fig.patch.set_facecolor('#0F1117')
ax.set_facecolor('#0F1117')

C_AGENT    = '#7C3AED'
C_TOOL     = '#1D4ED8'
C_PIPELINE = '#065F46'
C_DATA     = '#92400E'
C_CONFIG   = '#BE185D'
C_TEXT     = '#F8FAFC'
C_MUTED    = '#94A3B8'


def box(ax, x, y, w, h, color, text, fontsize=9, radius=0.3, alpha=0.92,
        text_color='#F8FAFC', bold=False):
    fancy = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.05,rounding_size={radius}",
        facecolor=color, edgecolor='white', linewidth=0.8, alpha=alpha, zorder=3)
    ax.add_patch(fancy)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, color=text_color, weight=weight,
            zorder=4, multialignment='center')


def arrow(ax, x1, y1, x2, y2, color='#64748B', lw=1.5, label=''):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                connectionstyle='arc3,rad=0.0'),
                zorder=5)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx+0.1, my, label, fontsize=7, color='#94A3B8', zorder=6)


def dashed_arrow(ax, x1, y1, x2, y2, color='#64748B', lw=1.2, label=''):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                linestyle='dashed',
                                connectionstyle='arc3,rad=0.0'),
                zorder=5)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx+0.1, my, label, fontsize=7, color='#94A3B8', zorder=6)


# TITLE
ax.text(10, 13.4, 'AI Agent Workflow  —  Insights Generation',
        ha='center', va='center', fontsize=15, color=C_TEXT, weight='bold', zorder=6)
ax.text(10, 13.0, 'BOCPD & MMM exposed as agent-callable tools, orchestrated by an LLM',
        ha='center', va='center', fontsize=9, color=C_MUTED, zorder=6)

# ROW 1 — Inputs
box(ax, 0.4, 11.2, 2.8, 1.0, C_DATA, 'Gold Table\n(Raw HCP data)', fontsize=8, bold=True)
box(ax, 3.6, 11.2, 3.2, 1.0, C_DATA,
    'Task Description\n"Detect & explain sales anomalies"', fontsize=8, bold=True)
box(ax, 14.5, 11.0, 5.0, 1.4, C_CONFIG,
    'DatasetConfig  (dataset_config.json)\ntarget_col  |  channel groups  |  decay rates\norganic_cps  |  true_effects',
    fontsize=8, bold=True)

# ROW 2 — AI AGENT
box(ax, 2.2, 9.0, 8.4, 1.6, C_AGENT,
    'AI Agent  (LLM Orchestrator)\nDecides: which tools to call  |  in what order  |  how to interpret results',
    fontsize=10, bold=True, radius=0.4)

arrow(ax, 1.8, 11.2, 4.0, 10.6, color='#94A3B8', lw=1.5)
arrow(ax, 5.2, 11.2, 6.5, 10.6, color='#94A3B8', lw=1.5)
dashed_arrow(ax, 14.5, 11.7, 10.6, 10.4, color='#F472B6', lw=1.4, label='schema context')

# ROW 3 — TOOLS
tools = [
    (0.3,  'Tool 1\ndetect_changepoints()',  'wraps  bocpd()'),
    (5.2,  'Tool 2\nfit_mmm()',              'wraps  mmm_data_prep()\n+ mmm_fit()'),
    (10.1, 'Tool 3\nclassify_shifts()',      'wraps  integration()'),
    (15.0, 'Tool 4\nvalidate()',             'wraps  validation()'),
]
tool_centres = []
for x, label, sub in tools:
    box(ax, x, 6.9, 4.4, 1.7, C_TOOL, f'{label}\n{sub}', fontsize=8)
    tool_centres.append(x + 2.2)

for tc in tool_centres:
    arrow(ax, tc, 9.0, tc, 8.6, color='#818CF8', lw=1.6)

# ROW 4 — PIPELINE STAGES
pipeline_groups = [
    (0.3,  ['data_prep()', 'BOCPD algorithm', 'CP candidates']),
    (5.2,  ['MMM feature prep', 'PyMC NUTS sampling', 'Contribution decomp']),
    (10.1, ['Pre/post windows', 'Field vs broadcast', 'Root-cause rules']),
    (15.0, ['MAPE checks', 'CP detection checks', 'Coeff. recovery']),
]
for gx, stages in pipeline_groups:
    sy = 5.85
    for stage in stages:
        box(ax, gx + 0.15, sy - 0.70, 4.1, 0.60, C_PIPELINE,
            stage, fontsize=7.5, radius=0.15, alpha=0.85)
        sy -= 0.75

for tc in tool_centres:
    arrow(ax, tc, 6.9, tc, 5.9, color='#34D399', lw=1.2)

# HORIZONTAL FLOW between tools
inter_y = 7.75
flow_labels = ['CP candidates', 'contributions', 'classified CPs']
for i in range(3):
    x1 = tool_centres[i] + 2.0
    x2 = tool_centres[i+1] - 2.0
    arrow(ax, x1, inter_y, x2, inter_y, color='#FCD34D', lw=1.4, label=flow_labels[i])

# ROW 5 — OUTPUTS
outputs = [
    (0.4,  'CP_PROBS\nCP_CANDIDATES\n(parquet / UC table)'),
    (5.3,  'MMM_TRACE (.nc)\nCONTRIBUTIONS\n(parquet / UC table)'),
    (10.2, 'integration_report.csv\n(classified CPs\nwith root causes)'),
    (15.1, 'validation_report.csv\n(PASS / FAIL / SKIP)'),
]
out_xs = [2.55, 7.45, 12.35, 17.25]
for (ox, olabel), cx in zip(outputs, out_xs):
    box(ax, ox, 2.9, 4.3, 1.1, C_DATA, olabel, fontsize=7.5, radius=0.2, alpha=0.9)
    arrow(ax, cx, 3.65, cx, 4.0, color='#F87171', lw=1.2)

# NL REPORT
box(ax, 4.5, 1.0, 11.0, 1.5, C_AGENT,
    'Agent Output  —  Natural Language Report\n'
    '"3 changepoints detected. TREMFYA launch (Jul-2017) driven by broadcast (+18% TV GRPs).\n'
    'INC-001 flagged as artifact (residual z=3.8). In-sample MAPE 6.2%  [PASS]"',
    fontsize=8.5, radius=0.4)
for cx in out_xs[1:3]:
    arrow(ax, cx, 2.9, 10.0, 2.5, color='#C084FC', lw=1.2)

# LEGEND
legend_items = [
    (C_AGENT,    'AI Agent / LLM'),
    (C_TOOL,     'Tool (agent-callable)'),
    (C_PIPELINE, 'Pipeline stage inside tool'),
    (C_DATA,     'Data artefact'),
    (C_CONFIG,   'DatasetConfig'),
]
lx, ly = 0.3, 2.5
for color, label in legend_items:
    rect = FancyBboxPatch((lx, ly), 0.35, 0.28,
                          boxstyle='round,pad=0.02,rounding_size=0.05',
                          facecolor=color, edgecolor='white', lw=0.5, zorder=6)
    ax.add_patch(rect)
    ax.text(lx + 0.5, ly + 0.14, label, fontsize=7.5, color=C_TEXT, va='center', zorder=7)
    lx += 2.2

plt.tight_layout(pad=0.3)
plt.savefig('agent_architecture.png', dpi=160, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print("saved agent_architecture.png")
