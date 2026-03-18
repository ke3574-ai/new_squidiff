import numpy as np
import anndata
import os

def generate_synthetic_dataset():
    np.random.seed(42)

    # -----------------------------
    # Parameters (small + fast)
    # -----------------------------
    n_cells = 1200
    n_genes = 50
    branch_time = 0.5

    # -----------------------------
    # 1. Generate pseudotime
    # -----------------------------
    t = np.random.rand(n_cells)

    # assign branch after split
    branch = np.zeros(n_cells)
    mask = t > branch_time
    branch[mask] = np.random.choice([0,1], mask.sum())

    # -----------------------------
    # 2. Latent trajectory (2D)
    # -----------------------------
    z = np.zeros((n_cells,2))
    z[:,0] = t

    for i in range(n_cells):
        if t[i] <= branch_time:
            z[i,1] = np.random.normal(0, 0.05)
        else:
            direction = 1 if branch[i] == 1 else -1
            z[i,1] = direction*(t[i]-branch_time) + np.random.normal(0,0.05)

    # -----------------------------
    # 3. Simple gene programs
    # -----------------------------
    X = np.zeros((n_cells,n_genes))

    # split genes into programs
    n_trunk = 20
    n_branchA = 15
    n_branchB = 15

    for i in range(n_cells):

        trunk = np.exp(-3 * max(0, t[i]-branch_time))

        branchA = (t[i]-branch_time) if (branch[i]==1 and t[i]>branch_time) else 0
        branchB = (t[i]-branch_time) if (branch[i]==0 and t[i]>branch_time) else 0

        X[i,:n_trunk] = trunk + np.random.normal(0,0.2,n_trunk)
        X[i,n_trunk:n_trunk+n_branchA] = branchA + np.random.normal(0,0.2,n_branchA)
        X[i,n_trunk+n_branchA:] = branchB + np.random.normal(0,0.2,n_branchB)

    # keep nonnegative
    X = np.clip(X,0,None)

    # -----------------------------
    # 4. Convert to discrete time (Group)
    # -----------------------------
    bins = [0, 0.25, 0.5, 0.75, 1.0]
    Group = np.digitize(t, bins) - 1

    # -----------------------------
    # 5. Create AnnData
    # -----------------------------
    adata = anndata.AnnData(X)
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.obs["Group"] = Group

    # -----------------------------
    # 6. Train/test split (Squidiff style)
    # -----------------------------
    train_mask = np.isin(Group, [0,3])
    test_mask  = np.isin(Group, [1,2])

    train_adata = adata[train_mask].copy()
    test_adata  = adata[test_mask].copy()

    # -----------------------------
    # 7. Save in Squidiff format
    # -----------------------------
    print(f"Creating directory in: {os.getcwd()}")
    os.makedirs("datasets", exist_ok=True)

    train_out = anndata.AnnData(train_adata.X)
    train_out.var_names = train_adata.var_names
    train_out.obs_names = train_adata.obs_names
    train_out.obs["Group"] = train_adata.obs["Group"]
    train_out.write("datasets/synth_train.h5ad")

    test_out = anndata.AnnData(test_adata.X)
    test_out.var_names = test_adata.var_names
    test_out.obs_names = test_adata.obs_names
    test_out.obs["Group"] = test_adata.obs["Group"]
    test_out.write("datasets/synth_test.h5ad")

    print("Saved synthetic dataset:")
    print("Train:", train_out.shape)
    print("Test :", test_out.shape)

if __name__ == "__main__": 
    generate_synthetic_dataset()