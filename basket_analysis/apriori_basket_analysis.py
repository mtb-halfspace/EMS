from load_dataset import load_dataset
from helpers import unpack_frozenset_columns

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

df = load_dataset()

# Decide which level to do the basket analysis on, e.g., "Item Name", "Retail Product Name", or "Item Category Code"
analysis_level = "Retail Product Name"
basket_analysis_cols = ["Store No.", "Transaction No."] + [analysis_level]
df_basket_analysis = df[basket_analysis_cols]

basket = (
    df_basket_analysis.groupby(["Store No.", "Transaction No."])[analysis_level]
    .apply(list)
    .reset_index()
)
transactions = basket[analysis_level].tolist()

te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_array, columns=te.columns_)

frequent_itemsets = apriori(df_encoded, min_support=0.01, use_colnames=True, max_len=2)

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)

rules = unpack_frozenset_columns(df=rules)

rule_cols = ["antecedents", "consequents", "support", "confidence", "lift"]
rules = rules[rule_cols].sort_values("support", ascending=False).reset_index(drop=True)


# Visualize support as heatmap
support = rules.pivot(index="antecedents", columns="consequents", values="support")

items = sorted(set(rules["antecedents"]).union(rules["consequents"]))
support_sym = support.reindex(index=items, columns=items)
support_sym = support.combine_first(support_sym.T).fillna(0)
# support_sym.values[np.diag_indices_from(support_sym)] = 1.0

fig, ax = plt.subplots(figsize=(16, 12))
sns.heatmap(support_sym, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
plt.tight_layout()
plt.show()
