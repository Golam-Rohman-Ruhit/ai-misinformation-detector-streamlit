import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# আপনার প্রজেক্টের প্রাপ্ত ফলাফল
results = {
    'Model': [
        'Bi-LSTM (Best)',
        'Keras Neural Net',
        'TensorFlow NN',
        'Simple LSTM',
        'SVM',
        'Naive Bayes'
    ],
    'Accuracy': [93.45, 93.16, 92.61, 91.60, 90.18, 71.88],
    'Color': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
}

# ডেটাফ্রেমে রূপান্তর
df = pd.DataFrame(results)

# গ্রাফের সাইজ এবং স্টাইল
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

# বার চার্ট তৈরি
ax = sns.barplot(x='Accuracy', y='Model', data=df, palette='viridis')

# টাইটেল এবং লেবেল
plt.title('Fake News Detection Model Comparison', fontsize=16, fontweight='bold')
plt.xlabel('Accuracy (%)', fontsize=12)
plt.ylabel('Model Name', fontsize=12)
plt.xlim(60, 100)  # X-axis রেঞ্জ (সুন্দর দেখানোর জন্য)

# বারের পাশে পার্সেন্টেজ লেখা যোগ করা
for i, v in enumerate(df['Accuracy']):
    ax.text(v + 0.5, i + 0.1, str(v) + '%', color='black', fontweight='bold', fontsize=11)

# গ্রাফটি দেখানো
plt.tight_layout()
plt.savefig('model_comparison_graph.png', dpi=300) # গ্রাফটি সেভ হবে
plt.show()