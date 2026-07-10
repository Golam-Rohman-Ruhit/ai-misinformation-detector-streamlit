import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.patches as mpatches


# ১. Dataset Distribution Pie Chart
def plot_pie_chart():
    # যেহেতু আমাদের কাছে রিয়েল ডেটা কাউন্ট আছে (Kaggle ডেটাসেট অনুযায়ী প্রায় ৫০-৫০)
    labels = ['Reliable (Real)', 'Unreliable (Fake)']
    sizes = [10387, 10413]  # Kaggle ডেটাসেটের আনুমানিক অনুপাত
    colors = ['#66b3ff', '#ff9999']
    explode = (0.05, 0.05)

    plt.figure(figsize=(7, 7))
    plt.pie(sizes, colors=colors, labels=labels, autopct='%1.1f%%', startangle=90, pctdistance=0.85, explode=explode,
            shadow=True)

    # মাঝখানে সাদা বৃত্ত (Donut chart বানানোর জন্য)
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    plt.title('Dataset Distribution: Real vs Fake', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('dataset_distribution.png', dpi=300)
    print("✅ Pie Chart Saved!")
    plt.show()


# ২. System Architecture Diagram (Workflow)
def plot_architecture():
    fig, ax = plt.subplots(figsize=(12, 6))

    # বক্সের স্টাইল
    box_style = dict(boxstyle="round,pad=0.5", fc="#f0f0f0", ec="#333", lw=2)
    arrow_props = dict(facecolor='black', shrink=0.05, width=2)

    # বক্সের অবস্থান
    steps = ["Input News", "Preprocessing\n(Cleaning/Stemming)", "Feature Extraction\n(Doc2Vec / Tokenization)",
             "AI Models\n(Bi-LSTM / SVM)", "Web App\n(Streamlit)", "Prediction\n(Real/Fake)"]
    x_pos = [0, 2, 4, 6, 8, 10]
    y_pos = [0.5] * 6

    # বক্স আঁকা
    for i, step in enumerate(steps):
        ax.text(x_pos[i], 0.5, step, ha="center", va="center", size=10, bbox=box_style, fontweight='bold')

    # তীর (Arrows) আঁকা
    for i in range(len(steps) - 1):
        ax.annotate("", xy=(x_pos[i + 1] - 0.8, 0.5), xytext=(x_pos[i] + 0.8, 0.5), arrowprops=arrow_props)

    ax.set_xlim(-1, 11)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.title('System Workflow Architecture', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig('system_architecture.png', dpi=300)
    print("✅ Architecture Diagram Saved!")
    plt.show()


if __name__ == "__main__":
    plot_pie_chart()
    plot_architecture()