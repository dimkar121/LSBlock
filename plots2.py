import matplotlib.pyplot as plt

# Scholar DBLP
methods = ['BG', 'DBD', 'SPK']

#Scholar
query_time = [0.0002, 0.00056, 0.002]
recall = [0.96, 0.7256451791, 0.928276]
precision = [0.76, 0.13, 0.09]
name="scholar"

#ACM
#query_time = [0.0001, 0.00032, 0.001]
#recall = [0.98, 0.890798251791, 0.9628276]
#precision = [0.71496, 0.313, 0.2309]
#name="acm"

#restaurants
#query_time = [0.0001, 0.00036, 0.001]
#recall = [0.9933, 0.848246890798251791, 0.92628276]
#precision = [0.74807096, 0.2313, 0.12309]
#name = "restaurants"

#abt
query_time = [0.0001, 0.0003356, 0.001]
recall = [0.856, 0.879217256451791, 0.8154580928276]
precision = [0.2726, 0.03, 0.162]
name="abt"



#amazon
query_time = [0.0001, 0.0004356, 0.001]
recall = [0.7169856, 0.759217256451791, 0.6654580928276]
precision = [0.22726, 0.01, 0.162]
name="amazon"



#imdb
#vectorization_time = [0.007625, 0.0934, 0.178, 0.21]
#recall = [0.81, 0.71791, 0.61, 0.71]
#precision = [0.415, 0.383, 0.322, 0.293]




# Compute F1-Measure
f1_measure = [2 * (p * r) / (p + r) if (p + r) > 0 else 0 for p, r in zip(precision, recall)]

# Create a single figure with 4 subplots
plt.figure(figsize=(7.6, 1.83))  # Large figure size # 3.5, 1.23)

# Recall Plot
plt.subplot(1, 4, 1)  # 2 rows, 2 columns, second subplot
bars = plt.bar(methods, recall, color='green', alpha=0.8,zorder=3)
#plt.title("Recall", fontsize=8, weight='bold')
plt.ylabel("Recall", fontsize=8)
plt.xticks(fontsize=8, rotation=45)
plt.yticks([0.5,0.7,0.9],fontsize=8)

# Precision Plot
plt.subplot(1, 4, 2)  # 2 rows, 2 columns, third subplot
bars = plt.bar(methods, precision, color='orange', alpha=0.8,zorder=3)
#plt.title("Precision", fontsize=8, weight='bold')
plt.ylabel("Precision", fontsize=8)
plt.xticks(fontsize=8, rotation=45)
plt.yticks(fontsize=8)

# F1-Measure Plot
plt.subplot(1, 4, 3)  # 2 rows, 2 columns, fourth subplot
bars = plt.bar(methods, f1_measure, color='purple', alpha=0.8,zorder=3)
#plt.title("F1-Score", fontsize=8, weight='bold')
plt.ylabel("F1-Score", fontsize=8)
plt.xticks(fontsize=8, rotation=45)
plt.yticks(fontsize=8)

# Query Time Plot
plt.subplot(1, 4, 4)  # 2 rows, 2 columns, first subplot
bars = plt.bar(methods, query_time, color='blue', alpha=0.8,zorder=3)
#plt.title("Query Time", fontsize=8, weight='bold')
plt.ylabel("Query Time (secs)", fontsize=8)
plt.xticks(fontsize=8, rotation=45)
plt.yticks(fontsize=8)


# Adjust layout
plt.tight_layout()

# Save the plot as a high-quality PDF
plt.savefig(f"c://plots//{name}.pdf", format='pdf', dpi=600)
#plt.savefig("c://plots//imdb.pdf", format='pdf', dpi=600)





# Data
methods = ['MH', 'GloVe', 'DBERT', 'SGT5']
vectorization_time = [0.0025, 0.04, 0.08, 0.1]
recall = [0.97, 0.851791, 0.62, 0.76]
precision = [0.645, 0.37, 0.29, 0.250]

#abt
#vectorization_time = [0.0325, 0.119, 0.695, 0.97]
#recall = [0.7087, 0.67851791, 0.7351, 0.792816]
#precision = [0.4645, 0.237, 0.329, 0.4250]


# Compute F1-Measure
f1_measure = [2 * (p * r) / (p + r) if (p + r) > 0 else 0 for p, r in zip(precision, recall)]

# Create a single figure with 4 subplots
plt.figure(figsize=(7.6, 1.83))  # Large figure size #18,12

# Vectorization Time Plot
plt.subplot(1 , 4, 1)  # 2 rows, 2 columns, first subplot
bars = plt.bar(methods, vectorization_time, color='blue', alpha=0.8)
#plt.title("Vectorization Time", fontsize=36, weight='bold')
plt.ylabel("Time (s)", fontsize=8)
plt.xticks(fontsize=8,  rotation=45)
plt.yticks(fontsize=8)

# Recall Plot
plt.subplot(1, 4, 2)  # 2 rows, 2 columns, second subplot
bars = plt.bar(methods, recall, color='green', alpha=0.8)
#plt.title("Recall", fontsize=8)
plt.ylabel("Recall", fontsize=8)
plt.xticks(fontsize=8, rotation=45)
plt.yticks(fontsize=8)

# Precision Plot
plt.subplot(1, 4, 3)  # 2 rows, 2 columns, third subplot
bars = plt.bar(methods, precision, color='orange', alpha=0.8)
#plt.title("Precision", fontsize=8)
plt.ylabel("Precision", fontsize=8)
plt.xticks(fontsize=8,  rotation=45)
plt.yticks(fontsize=8)

# F1-Measure Plot
plt.subplot(1, 4, 4)  # 2 rows, 2 columns, fourth subplot
bars = plt.bar(methods, f1_measure, color='purple', alpha=0.8)
#plt.title("F1-Score", fontsize=36)
plt.ylabel("F1-Score", fontsize=8)
plt.xticks(fontsize=8,  rotation=45)
plt.yticks(fontsize=8)

# Adjust layout
plt.tight_layout()

# Save the plot as a high-quality PDF
#plt.savefig("c://plots//dblp.pdf", format='pdf', dpi=600)
#plt.savefig("c://plots//abt.pdf", format='pdf', dpi=600)
