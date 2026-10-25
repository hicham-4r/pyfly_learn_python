import numpy as np


DATA = "artifacts/features/token_visual_features.npz"

data = np.load(DATA, allow_pickle=True)

X = data["X"].astype(np.float64)
labels = data["labels"].astype(str)
seeds = data["seeds"]


def distance(a, b):
    return np.linalg.norm(a - b)


within_token = []
between_token_same_seed = []


# --------------------------------------------------
# Same token, different neural seeds
# --------------------------------------------------

for token in np.unique(labels):

    idx = np.where(labels == token)[0]

    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):

            within_token.append(
                distance(
                    X[idx[i]],
                    X[idx[j]],
                )
            )


# --------------------------------------------------
# Different tokens, same neural seed
# --------------------------------------------------

for seed in np.unique(seeds):

    idx = np.where(seeds == seed)[0]

    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):

            if labels[idx[i]] != labels[idx[j]]:

                between_token_same_seed.append(
                    distance(
                        X[idx[i]],
                        X[idx[j]],
                    )
                )


within_token = np.asarray(within_token)
between_token_same_seed = np.asarray(
    between_token_same_seed
)


print()
print("====================================")
print("      PyFly SIGNAL ANALYSIS")
print("====================================")

print()
print(
    "Same token / different seed:"
)
print(
    f"Mean distance: "
    f"{within_token.mean():.4f}"
)

print()
print(
    "Different token / same seed:"
)
print(
    f"Mean distance: "
    f"{between_token_same_seed.mean():.4f}"
)


ratio = (
    between_token_same_seed.mean()
    /
    within_token.mean()
)

print()
print(
    f"Signal / noise distance ratio: "
    f"{ratio:.4f}"
)


if ratio > 1.5:

    print()
    print(
        "✅ Token identity is stronger "
        "than neural-seed variation."
    )

elif ratio > 1.0:

    print()
    print(
        "⚠️ Weak token signal detected, "
        "but noise is substantial."
    )

else:

    print()
    print(
        "❌ Neural-seed variation is "
        "stronger than token separation."
    )